import torch
import torch.nn as nn
from pose.utils.bbox import point2box_xywh, bbox_iou
from pose.utils import DISTRIBUTED

class ComputeLoss:
    def __init__(self, model, use_focal=True, use_vari_focal=True, kpt_loss_type='hybrid'):
        self.device = next(model.parameters()).device
        # self.nc = model.head.nc
        # self.nk = model.head.nk
        
        self.nc = model.module.head.nc if DISTRIBUTED else model.head.nc
        self.nk = model.module.head.nk if DISTRIBUTED else model.head.nk
        # 모델 구조에 따라 kpt_shape[1]이 2(xy)인지 3(xyc)인지 확인 필요. 여기선 3으로 가정
        self.num_kpts = self.nk // 3 
        
        # 1. Loss Functions
        if use_focal:
            if use_vari_focal:
                from pose.core.loss_fn.classify.pose import VarifocalLoss
                self.cls_loss_fn = VarifocalLoss(gamma=2.0, alpha=0.75)
            else:
                # FocalLoss 구현체가 따로 있다고 가정
                self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='mean') 
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Keypoint Sigmas (COCO 17 points 기준)
        if self.num_kpts == 17:
            kpt_sigmas = torch.tensor([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 
                1.07, 1.07, .87, .87, .89, .89
            ], device=self.device) / 10.0
        else:
            # Custom keypoints인 경우 기본값
            kpt_sigmas = torch.ones(self.num_kpts, device=self.device) * 0.05
        
        
        if kpt_loss_type == 'oks':
            from pose.core.loss_fn.pose.pose import KeypointLoss
            self.kpt_loss_fn = KeypointLoss(kpt_sigmas)
        elif kpt_loss_type == 'improved':
            from pose.core.loss_fn.pose.pose import ImprovedKeypointLoss
            self.kpt_loss_fn = ImprovedKeypointLoss(kpt_sigmas)
        elif kpt_loss_type == 'robust':
            from pose.core.loss_fn.pose.pose import RobustKeypointLoss
            self.kpt_loss_fn = RobustKeypointLoss(kpt_sigmas)
        else:  # 'hybrid'
            from pose.core.loss_fn.pose.pose import HybridKeypointLoss   
            self.kpt_loss_fn = HybridKeypointLoss(kpt_sigmas)

        
        self.kpt_vis_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        
        # Weights
        self.cls_weight = 5.0
        self.kpt_weight = 10.0 
        
    def set_train_loss(self):
        self.cls_loss_sum = 0.0
        self.kpt_loss_sum = 0.0
        self.obj_loss_sum = 0.0

    def add_loss(self, loss_items):
        l_cls = loss_items[0].item() if torch.is_tensor(loss_items[0]) else loss_items[0]
        l_kpt = loss_items[1].item() if torch.is_tensor(loss_items[1]) else loss_items[1]
        l_obj = loss_items[2].item() if torch.is_tensor(loss_items[2]) else loss_items[2]
        self.cls_loss_sum += l_cls
        self.kpt_loss_sum += l_kpt
        self.obj_loss_sum += l_obj
        return l_cls, l_kpt, l_obj


    def __call__(self, preds, targets):
        loss_cls = torch.zeros(1, device=self.device)
        loss_kpt = torch.zeros(1, device=self.device)
        loss_kpt_vis = torch.zeros(1, device=self.device)
        
        num_pos_total = 0
        
        for i, pred in enumerate(preds):
            B, C, H, W = pred.shape
            
            # (B, H, W, C)
            pred = pred.permute(0, 2, 3, 1)
            
            # 1. 배경을 포함한 전체 Target 초기화 (0으로 채움)
            target_cls = torch.zeros(B, H, W, self.nc, device=self.device)
            target_mask = torch.zeros(B, H, W, device=self.device, dtype=torch.bool)
            
            # 2. 정답(Positive) 데이터 준비
            if targets.shape[0] > 0:
                gt_box = targets[:, 2:6].clone()
                gt_kpts = targets[:, 6:].clone()
                
                # Scaling
                gt_box[:, [0, 2]] *= W
                gt_box[:, [1, 3]] *= H
                gt_kpts[:, 0::3] *= W
                gt_kpts[:, 1::3] *= H
                
                b_idx = targets[:, 0].long()
                c_idx = targets[:, 1].long()

                gx = gt_box[:, 0].long().clamp(0, W-1)
                gy = gt_box[:, 1].long().clamp(0, H-1)

                # Positive Mask 마킹
                target_mask[b_idx, gy, gx] = True
                
                num_pos = target_mask.sum()
                num_pos_total += num_pos

                if num_pos > 0:
                    # -----------------------------------------------------------
                    # A. Positive Sample에 대해서만 IoU 및 Keypoint Loss 계산
                    # -----------------------------------------------------------
                    
                    # Offset 계산
                    gt_box_offset = gt_box.clone()
                    gt_box_offset[:, 0] -= gx.float()
                    gt_box_offset[:, 1] -= gy.float()
                    gt_kpts[:, 0::3] -= gx.unsqueeze(1).float()
                    gt_kpts[:, 1::3] -= gy.unsqueeze(1).float()
                    
                    pred_pos = pred[b_idx, gy, gx]
                    idx_cls_end = self.nc
                    
                    pred_kpts_raw = pred_pos[:, idx_cls_end:]
                    pred_kpts = pred_kpts_raw.reshape(-1, self.num_kpts, 3)
                    pred_kpts_xy = pred_kpts[..., :2]
                    pred_kpts_conf = pred_kpts[..., 2]

                    gt_kpts_pos = gt_kpts.reshape(-1, self.num_kpts, 3)
                    gt_kpts_xy = gt_kpts_pos[..., :2]
                    kpt_mask = (gt_kpts_pos[..., 2] > 0).float()
                    area = gt_box[:, 2] * gt_box[:, 3]

                    # IoU 계산
                    pred_box_from_kpts = point2box_xywh(pred_kpts_xy, kpt_mask)
                    pred_iou = bbox_iou(
                        pred_box_from_kpts, 
                        gt_box_offset, 
                        CIoU=True
                    )
                    
                    
                    loss_kpt += self.kpt_loss_fn(pred_kpts_xy, gt_kpts_xy, kpt_mask, area)
                    loss_kpt_vis += self.kpt_vis_loss_fn(pred_kpts_conf, kpt_mask)

                    pred_iou_score = pred_iou.detach().clamp(0, 1)
                    target_cls[b_idx, gy, gx, c_idx] = pred_iou_score

            pred_cls_all = pred[..., :self.nc].reshape(-1, self.nc)
            target_cls_all = target_cls.reshape(-1, self.nc)
            
            target_label_all = (target_cls_all > 0).float()

            # 전체 픽셀에 대해 Loss 수행
            loss_cls += self.cls_loss_fn(pred_cls_all, target_cls_all, target_label_all)

        # Normalize Loss
        num_pos_total = max(num_pos_total, 1)
        
        # Class Loss는 전체 픽셀 합이므로 보통 num_pos로 나누거나 H*W로 나눔.
        # YOLO 계열은 보통 num_pos로 나눕니다.
        loss_cls = (loss_cls / num_pos_total) * 1.0
        loss_kpt = (loss_kpt / num_pos_total) * 10.0
        loss_kpt_vis = (loss_kpt_vis / num_pos_total) * 5.0
        
        total_loss = loss_cls + loss_kpt + loss_kpt_vis
        
        return total_loss, (loss_cls.item(), loss_kpt.item(), loss_kpt_vis.item())