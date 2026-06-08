"""
6DoF Head를 위한 결합 Loss (Direction B - Depth + Rotation Regression).

구성:
    L_total = λ_box  * L_CIoU          # 2D bbox 회귀
            + λ_dfl  * L_DFL           # box 거리 분포(DFL)
            + λ_obj  * L_obj  (BCE)    # objectness
            + λ_cls  * L_cls  (BCE)    # class score
            + λ_rot  * L_geodesic      # 3D 회전 (6D/quat → R, 측지거리)
            + λ_size * L_size (SmoothL1)# 3D 크기
            + λ_dep  * L_depth (L1)    # dense depth map
            + λ_tr   * L_trans (L1)    # unprojection translation

라벨 할당은 anchor-free TaskAlignedAssigner(경량판)를 사용한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.geometry import (quaternion_to_matrix, rotation_6d_to_matrix,
                       rotation_cosine_loss, sample_depth, unproject_translation)
from ..core.head import dist2bbox, make_anchors


# ----------------------------------------------------------------------------- #
#  IoU / CIoU
# ----------------------------------------------------------------------------- #
def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """CIoU 계산. box1, box2: [...,4] (x1,y1,x2,y2). 반환 [...] CIoU(-1~1)."""
    (b1x1, b1y1, b1x2, b1y2) = box1.unbind(-1)
    (b2x1, b2y1, b2x2, b2y2) = box2.unbind(-1)

    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    w1, h1 = (b1x2 - b1x1), (b1y2 - b1y1)
    w2, h2 = (b2x2 - b2x1), (b2y2 - b2y1)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2x1 + b2x2 - b1x1 - b1x2) ** 2 + (b2y1 + b2y2 - b1y1 - b1y2) ** 2) / 4
    v = (4 / (torch.pi ** 2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


# ----------------------------------------------------------------------------- #
#  DFL Loss
# ----------------------------------------------------------------------------- #
def df_loss(pred_dist: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """
    Distribution Focal Loss. 연속 타깃을 좌/우 정수 bin의 CE 가중합으로.
    pred_dist: [N, 4, reg_max], target: [N, 4] (0~reg_max-1)
    """
    tl = target.long()
    tr = tl + 1
    wl = tr - target
    wr = 1 - wl
    pred = pred_dist.view(-1, reg_max)
    loss = (F.cross_entropy(pred, tl.view(-1), reduction="none") * wl.view(-1)
            + F.cross_entropy(pred, tr.clamp(max=reg_max - 1).view(-1), reduction="none") * wr.view(-1))
    return loss.view(target.shape[0], 4).mean(-1, keepdim=True)


# ----------------------------------------------------------------------------- #
#  Task-Aligned Assigner (경량판)
# ----------------------------------------------------------------------------- #
class TaskAlignedAssigner(nn.Module):
    """metric = score^alpha * iou^beta 로 각 GT에 top-k anchor를 양성 배정."""

    def __init__(self, topk: int = 10, num_classes: int = 80,
                 alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.nc = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        pred_scores:[B,A,nc], pred_bboxes:[B,A,4] xyxy px, anc_points:[A,2] px,
        gt_labels:[B,M,1], gt_bboxes:[B,M,4], mask_gt:[B,M,1]
        반환: target_labels[B,A], target_bboxes[B,A,4], target_scores[B,A,nc],
              fg_mask[B,A], target_gt_idx[B,A]
        """
        B, A = pred_scores.shape[:2]
        M = gt_bboxes.shape[1]
        if M == 0:
            dev = pred_scores.device
            return (torch.zeros(B, A, dtype=torch.long, device=dev),
                    torch.zeros(B, A, 4, device=dev),
                    torch.zeros(B, A, self.nc, device=dev),
                    torch.zeros(B, A, dtype=torch.bool, device=dev),
                    torch.zeros(B, A, dtype=torch.long, device=dev))

        in_gts = self._in_gt_mask(anc_points, gt_bboxes)                       # [B,M,A]
        align_metric, overlaps = self._alignment(pred_scores, pred_bboxes,
                                                  gt_labels, gt_bboxes, in_gts * mask_gt)
        mask_topk = self._select_topk(align_metric)
        mask_pos = mask_topk * in_gts * mask_gt                                # [B,M,A]

        fg_mask = mask_pos.sum(1)                                              # [B,A]
        if (fg_mask > 1).any():
            multi = (fg_mask.unsqueeze(1) > 1).expand(-1, M, -1)
            max_gt = overlaps.argmax(1)
            is_max = F.one_hot(max_gt, M).permute(0, 2, 1)
            mask_pos = torch.where(multi, is_max, mask_pos)
            fg_mask = mask_pos.sum(1)
        target_gt_idx = mask_pos.argmax(1)                                    # [B,A]

        batch_idx = torch.arange(B, device=gt_labels.device).unsqueeze(-1)
        flat_idx = target_gt_idx + batch_idx * M
        target_labels = gt_labels.long().flatten()[flat_idx]                  # [B,A]
        target_bboxes = gt_bboxes.view(-1, 4)[flat_idx]                       # [B,A,4]

        target_scores = F.one_hot(target_labels.clamp(0), self.nc).float()
        fg = fg_mask.bool()
        align_metric *= mask_pos
        pos_align = align_metric.amax(-1, keepdim=True)
        pos_overlap = (overlaps * mask_pos).amax(-1, keepdim=True)
        norm = (align_metric * pos_overlap / (pos_align + self.eps)).amax(1)
        target_scores = target_scores * norm.unsqueeze(-1) * fg.unsqueeze(-1)
        return target_labels, target_bboxes, target_scores, fg, target_gt_idx

    def _in_gt_mask(self, anc_points, gt_bboxes):
        lt = gt_bboxes[..., :2].unsqueeze(2)
        rb = gt_bboxes[..., 2:].unsqueeze(2)
        ap = anc_points.view(1, 1, -1, 2)
        deltas = torch.cat([ap - lt, rb - ap], -1)
        return deltas.amin(-1) > 0

    def _alignment(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask):
        B, A, _ = pred_scores.shape
        M = gt_bboxes.shape[1]
        idx = gt_labels.long().squeeze(-1).clamp(0)
        bs = torch.arange(B).view(-1, 1).expand(-1, M)
        scores = pred_scores[bs, :, idx] * mask
        iou = self._pair_iou(gt_bboxes, pred_bboxes).clamp(0) * mask
        align = scores.pow(self.alpha) * iou.pow(self.beta)
        return align, iou

    @staticmethod
    def _pair_iou(gt, pred):
        gt = gt.unsqueeze(2)
        pred = pred.unsqueeze(1)
        lt = torch.max(gt[..., :2], pred[..., :2])
        rb = torch.min(gt[..., 2:], pred[..., 2:])
        wh = (rb - lt).clamp(0)
        inter = wh[..., 0] * wh[..., 1]
        area_g = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
        area_p = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        return inter / (area_g + area_p - inter + 1e-9)

    def _select_topk(self, metric):
        topk_val, topk_idx = metric.topk(self.topk, dim=-1)
        mask = torch.zeros_like(metric)
        mask.scatter_(-1, topk_idx, (topk_val > 0).float())
        return mask


# ----------------------------------------------------------------------------- #
#  결합 Loss 모듈
# ----------------------------------------------------------------------------- #
class Pose6DoFLoss(nn.Module):
    """
    모델 출력 + GT를 받아 6DoF 학습 손실을 계산.

    model_out: {"det": head raw dict, "depth": [B,1,Hd,Wd]}
    targets(batch padding):
        gt_labels   : [B, M, 1]
        gt_bboxes   : [B, M, 4]   (x1,y1,x2,y2 픽셀)
        gt_rot      : [B, M, 3, 3] 회전행렬
        gt_size     : [B, M, 3]   (dx,dy,dz)
        gt_trans    : [B, M, 3]   (X,Y,Z 카메라좌표)
        mask_gt     : [B, M, 1]
        K           : [B, 3, 3]   intrinsics
        img_size    : (H, W)
        gt_depth    : [B, 1, Hd, Wd] (옵션, dense depth GT)
        depth_mask  : [B, 1, Hd, Wd] (옵션, 유효 깊이 마스크)
    """

    def __init__(self, nc: int = 80, reg_max: int = 16,
                 strides: tuple[int, ...] = (8, 16, 32), rot_repr: str = "6d",
                 weights: dict | None = None):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.rot_repr = rot_repr
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=nc)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.proj = torch.arange(reg_max, dtype=torch.float)
        self.w = {"box": 7.5, "dfl": 1.5, "obj": 1.0, "cls": 0.5,
                  "rot": 2.0, "size": 1.0, "depth": 1.0, "trans": 2.0}
        if weights:
            self.w.update(weights)

    def _rot_to_matrix(self, raw: torch.Tensor) -> torch.Tensor:
        """raw rotation(6D/quat) → 회전행렬."""
        return rotation_6d_to_matrix(raw) if self.rot_repr == "6d" else quaternion_to_matrix(raw)

    def _dfl_decode(self, pred_box):
        B, A, _ = pred_box.shape
        x = pred_box.view(B, A, 4, self.reg_max).softmax(-1)
        return x @ self.proj.to(x.device)

    def forward(self, model_out: dict, targets: dict):
        head_out = model_out["det"]
        depth_map = model_out["depth"]
        feats = head_out["feats"]
        device = feats[0].device
        B = feats[0].shape[0]
        H, W = targets["img_size"]

        # 1) 레벨별 raw → [B, A, C]
        box = torch.cat([b.flatten(2) for b in head_out["box"]], 2).permute(0, 2, 1)    # [B,A,4*rm]
        cls = torch.cat([c.flatten(2) for c in head_out["cls"]], 2).permute(0, 2, 1)    # [B,A,nc+1]
        rot = torch.cat([r.flatten(2) for r in head_out["rot"]], 2).permute(0, 2, 1)    # [B,A,rot_dim]
        size = torch.cat([s.flatten(2) for s in head_out["size"]], 2).permute(0, 2, 1)  # [B,A,3]
        size = F.softplus(size)
        obj_logits, cls_logits = cls[..., :1], cls[..., 1:]

        anchors, stride_t = make_anchors(feats, self.strides)   # [A,2](셀), [A,1]
        anchors_px = anchors * stride_t

        # 2) 예측 box 디코딩(픽셀) — assigner & translation 용
        dist = self._dfl_decode(box)                            # [B,A,4]
        pred_xyxy = dist2bbox(dist.permute(0, 2, 1), anchors).permute(0, 2, 1)
        pred_xyxy_px = pred_xyxy * stride_t.view(1, -1, 1)      # [B,A,4] px
        pred_scores = cls_logits.sigmoid()

        # 3) 라벨 할당
        gt_labels = targets["gt_labels"].to(device)
        gt_bboxes = targets["gt_bboxes"].to(device)
        mask_gt = targets["mask_gt"].to(device)
        (t_labels, t_bboxes, t_scores, fg_mask, t_gt_idx) = self.assigner(
            pred_scores, pred_xyxy_px.detach(), anchors_px, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(t_scores.sum(), 1.0)

        # 4) 분류 손실
        #    objectness: 전체 anchor (배경 검출 담당) — 배경 다수라 mean으로 정규화
        obj_target = fg_mask.float().unsqueeze(-1)
        loss_obj = self.bce(obj_logits, obj_target).mean()

        # 누적 손실 초기화
        zero = torch.zeros(1, device=device)
        loss_cls = loss_box = loss_dfl = loss_rot = loss_size = loss_trans = zero.clone()

        if fg_mask.any():
            # class: 양성 anchor에만 (배경 cls는 objectness가 게이팅 → cls 폭발/발산 방지)
            loss_cls = (self.bce(cls_logits[fg_mask], t_scores[fg_mask]).sum()
                        / target_scores_sum)

            weight = t_scores.sum(-1)[fg_mask]                  # 정렬 가중치 [Npos]
            idx_pos = (t_gt_idx + torch.arange(B, device=device).view(-1, 1)
                       * gt_bboxes.shape[1])[fg_mask]           # GT flat index [Npos]

            # --- 2D BBox: CIoU ---
            pb, tb = pred_xyxy_px[fg_mask], t_bboxes[fg_mask]
            ciou = bbox_ciou(pb, tb)
            loss_box = ((1.0 - ciou) * weight).sum() / target_scores_sum

            # --- DFL ---
            tb_cell = tb / stride_t.view(1, -1, 1).expand(B, -1, 4)[fg_mask]
            anc_pos = anchors.unsqueeze(0).expand(B, -1, 2)[fg_mask]
            tgt_ltrb = torch.cat([anc_pos - tb_cell[:, :2],
                                  tb_cell[:, 2:] - anc_pos], 1).clamp(0, self.reg_max - 1.01)
            pred_dist_pos = box[fg_mask].view(-1, 4, self.reg_max)
            loss_dfl = (df_loss(pred_dist_pos, tgt_ltrb, self.reg_max).squeeze(-1)
                        * weight).sum() / target_scores_sum

            # --- 3D Rotation: geodesic loss ---
            R_pred = self._rot_to_matrix(rot[fg_mask])          # [Npos,3,3]
            R_gt = targets["gt_rot"].to(device).view(-1, 3, 3)[idx_pos]
            loss_rot = (rotation_cosine_loss(R_pred, R_gt) * weight).sum() / target_scores_sum

            # --- 3D Size: Smooth L1 ---
            sz_pred = size[fg_mask]                             # [Npos,3]
            sz_gt = targets["gt_size"].to(device).view(-1, 3)[idx_pos]
            loss_size = (F.smooth_l1_loss(sz_pred, sz_gt, reduction="none").mean(-1)
                         * weight).sum() / target_scores_sum

            # --- Translation: unprojection(예측 depth at bbox center) L1 ---
            centers = pred_xyxy_px[fg_mask]                     # px xyxy
            centers = torch.stack([(centers[:, 0] + centers[:, 2]) / 2,
                                   (centers[:, 1] + centers[:, 3]) / 2], -1)  # [Npos,2]
            # 배치 인덱스별로 depth 샘플 위해 per-sample 처리
            t_gt = targets["gt_trans"].to(device).view(-1, 3)[idx_pos]       # [Npos,3]
            t_pred = self._unproject_pos(depth_map, centers, fg_mask,
                                         targets["K"].to(device), (H, W))    # [Npos,3]
            loss_trans = (F.smooth_l1_loss(t_pred, t_gt, reduction="none").mean(-1)
                          * weight).sum() / target_scores_sum

        # 5) Dense Depth Loss (옵션, GT depth map이 있을 때)
        loss_depth = zero.clone()
        if targets.get("gt_depth") is not None:
            gt_d = targets["gt_depth"].to(device)
            # 예측 depth를 GT 해상도로 맞춤
            pd = F.interpolate(depth_map, size=gt_d.shape[-2:], mode="bilinear",
                               align_corners=False)
            dmask = targets.get("depth_mask")
            if dmask is not None:
                dmask = dmask.to(device)
                denom = dmask.sum().clamp(min=1)
                loss_depth = (F.l1_loss(pd, gt_d, reduction="none") * dmask).sum() / denom
            else:
                loss_depth = F.l1_loss(pd, gt_d)

        # 6) 가중 결합
        loss = (self.w["box"] * loss_box + self.w["dfl"] * loss_dfl
                + self.w["obj"] * loss_obj + self.w["cls"] * loss_cls
                + self.w["rot"] * loss_rot + self.w["size"] * loss_size
                + self.w["depth"] * loss_depth + self.w["trans"] * loss_trans)

        items = {k: float(v.detach()) for k, v in {
            "box": loss_box, "dfl": loss_dfl, "obj": loss_obj, "cls": loss_cls,
            "rot": loss_rot, "size": loss_size, "depth": loss_depth,
            "trans": loss_trans, "total": loss}.items()}
        # 손실은 이미 각 항이 평균/정규화되어 있으므로 batch 곱 없이 그대로 반환.
        # (× B 증폭은 AMP fp16 그래디언트와 곱해져 오버플로/발산을 유발)
        return loss, items

    @staticmethod
    def _unproject_pos(depth_map, centers, fg_mask, K, img_size):
        """
        양성 anchor들의 bbox 중심에서 예측 depth를 샘플 → translation 복원.
        배치마다 anchor 수가 달라, 배치 인덱스를 따라 묶어서 처리.
        """
        B = depth_map.shape[0]
        device = depth_map.device
        # fg_mask:[B,A] → 각 양성 샘플이 어느 배치인지
        batch_of_pos = torch.arange(B, device=device).view(-1, 1).expand_as(fg_mask)[fg_mask]
        out = torch.zeros(centers.shape[0], 3, device=device)
        for b in range(B):
            sel = batch_of_pos == b
            if not sel.any():
                continue
            uv = centers[sel].unsqueeze(0)                         # [1,n,2]
            z = sample_depth(depth_map[b:b + 1], uv, img_size)     # [1,n]
            out[sel] = unproject_translation(uv, z, K[b:b + 1]).squeeze(0)
        return out
