"""
6DoF Head를 위한 결합 Loss (Direction B - Depth + Rotation Regression).

구성:
    L_total = λ_box  * L_CIoU          # 2D bbox 회귀 (l,t,r,b 직접)
            + λ_obj  * L_obj  (BCE)    # objectness
            + λ_cls  * L_cls  (BCE)    # class score
            + λ_rot  * L_geodesic      # 3D 회전 (6D/quat → R, 측지거리)
            + λ_size * L_size (SmoothL1)# 3D 크기
            + λ_dep  * L_depth (L1)    # dense depth map
            + λ_tr   * L_trans (L1)    # unprojection translation

라벨 할당은 anchor-free center-sampling(FCOS/YOLOX 방식, anchors.py)을 사용한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.geometry import (quaternion_to_matrix, rotation_6d_to_matrix,
                       rotation_cosine_loss, sample_depth, unproject_translation)
from ..core.anchors import assign_centers, grid_points


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

    def __init__(self, nc: int = 80, strides: tuple[int, ...] = (8, 16, 32),
                 rot_repr: str = "6d", radius: float = 2.5,
                 weights: dict | None = None):
        super().__init__()
        self.nc = nc
        self.strides = strides
        self.rot_repr = rot_repr
        self.radius = radius                       # center-sampling 반경(×stride)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.w = {"box": 7.5, "obj": 1.0, "cls": 0.5,
                  "rot": 2.0, "size": 1.0, "depth": 1.0, "trans": 2.0}
        if weights:
            self.w.update(weights)

    def _rot_to_matrix(self, raw: torch.Tensor) -> torch.Tensor:
        """raw rotation(6D/quat) → 회전행렬."""
        return rotation_6d_to_matrix(raw) if self.rot_repr == "6d" else quaternion_to_matrix(raw)

    def forward(self, model_out: dict, targets: dict):
        head_out = model_out["det"]
        depth_map = model_out["depth"]
        feats = head_out["feats"]
        device = feats[0].device
        B = feats[0].shape[0]
        H, W = targets["img_size"]

        # 1) 레벨별 raw → [B, A, C]
        box = torch.cat([b.flatten(2) for b in head_out["box"]], 2).permute(0, 2, 1)    # [B,A,4]
        cls = torch.cat([c.flatten(2) for c in head_out["cls"]], 2).permute(0, 2, 1)    # [B,A,nc+1]
        rot = torch.cat([r.flatten(2) for r in head_out["rot"]], 2).permute(0, 2, 1)    # [B,A,rot_dim]
        size = torch.cat([s.flatten(2) for s in head_out["size"]], 2).permute(0, 2, 1)  # [B,A,3]
        size = F.softplus(size)
        obj_logits, cls_logits = cls[..., :1], cls[..., 1:]

        points, stride_t = grid_points(feats, self.strides)     # [A,2](셀), [A,1]
        points_px = points * stride_t                           # [A,2] px
        strides_px = stride_t.squeeze(1)                        # [A]
        A = points.shape[0]

        # 2) 예측 box 디코딩(픽셀, l/t/r/b 직접) — IoU 손실 & translation 용
        dist = F.softplus(box)                                  # [B,A,4] ≥ 0 (셀단위)
        px, py = points[:, 0], points[:, 1]
        pred_xyxy = torch.stack([px - dist[..., 0], py - dist[..., 1],
                                 px + dist[..., 2], py + dist[..., 3]], -1)  # [B,A,4] 셀
        pred_xyxy_px = pred_xyxy * stride_t.view(1, -1, 1)      # [B,A,4] px

        # 3) center-sampling 라벨 할당 (이미지별)
        gt_labels = targets["gt_labels"].to(device)
        gt_bboxes = targets["gt_bboxes"].to(device)
        mask_gt = targets["mask_gt"].to(device)
        M = gt_bboxes.shape[1]
        fg_mask = torch.zeros(B, A, dtype=torch.bool, device=device)
        gt_flat = torch.zeros(B, A, dtype=torch.long, device=device)   # 양성의 GT flat idx
        for b in range(B):
            valid = mask_gt[b, :, 0] > 0
            orig = valid.nonzero(as_tuple=False).squeeze(1)     # [m] 원본 GT 인덱스
            if orig.numel() == 0:
                continue
            local = assign_centers(points_px, strides_px, gt_bboxes[b][valid], self.radius)
            pos = local >= 0
            fg_mask[b] = pos
            gt_flat[b, pos] = orig[local[pos]] + b * M
        npos = fg_mask.sum().clamp(min=1)

        # 4) objectness: 전체 anchor (배경 다수라 mean 정규화)
        loss_obj = self.bce(obj_logits, fg_mask.float().unsqueeze(-1)).mean()

        zero = torch.zeros(1, device=device)
        loss_cls = loss_box = loss_rot = loss_size = loss_trans = zero.clone()

        if fg_mask.any():
            idx_pos = gt_flat[fg_mask]                          # [Npos] flat GT idx

            # class: 양성 anchor에만 (배경 cls는 objectness가 게이팅)
            labels = gt_labels.long().view(-1)[idx_pos]
            cls_t = F.one_hot(labels.clamp(0, self.nc - 1), self.nc).float()
            loss_cls = self.bce(cls_logits[fg_mask], cls_t).sum() / npos

            # --- 2D BBox: CIoU ---
            pb = pred_xyxy_px[fg_mask]
            tb = gt_bboxes.view(-1, 4)[idx_pos]
            loss_box = (1.0 - bbox_ciou(pb, tb)).sum() / npos

            # --- 3D Rotation: geodesic loss ---
            R_pred = self._rot_to_matrix(rot[fg_mask])          # [Npos,3,3]
            R_gt = targets["gt_rot"].to(device).view(-1, 3, 3)[idx_pos]
            loss_rot = rotation_cosine_loss(R_pred, R_gt).sum() / npos

            # --- 3D Size: Smooth L1 ---
            sz_pred = size[fg_mask]                             # [Npos,3]
            sz_gt = targets["gt_size"].to(device).view(-1, 3)[idx_pos]
            loss_size = F.smooth_l1_loss(sz_pred, sz_gt, reduction="none").mean(-1).sum() / npos

            # --- Translation: unprojection(예측 depth at bbox center) L1 ---
            centers = pred_xyxy_px[fg_mask]                     # px xyxy
            centers = torch.stack([(centers[:, 0] + centers[:, 2]) / 2,
                                   (centers[:, 1] + centers[:, 3]) / 2], -1)  # [Npos,2]
            t_gt = targets["gt_trans"].to(device).view(-1, 3)[idx_pos]       # [Npos,3]
            t_pred = self._unproject_pos(depth_map, centers, fg_mask,
                                         targets["K"].to(device), (H, W))    # [Npos,3]
            loss_trans = F.smooth_l1_loss(t_pred, t_gt, reduction="none").mean(-1).sum() / npos

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
        loss = (self.w["box"] * loss_box
                + self.w["obj"] * loss_obj + self.w["cls"] * loss_cls
                + self.w["rot"] * loss_rot + self.w["size"] * loss_size
                + self.w["depth"] * loss_depth + self.w["trans"] * loss_trans)

        items = {k: float(v.detach()) for k, v in {
            "box": loss_box, "obj": loss_obj, "cls": loss_cls,
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
