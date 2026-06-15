"""
2D 멀티태스크 Loss — Detection + Segmentation + Pose.

  L = λ_box·CIoU + λ_cls·BCE                       (Detect, l/t/r/b 직접 회귀)
    + λ_seg·MaskBCE(coeff·proto, box-crop)         (Segment)
    + λ_kpt·(keypoint L1 + visibility BCE)          (Pose)

라벨 할당은 anchor-free center-sampling(FCOS/YOLOX 방식, anchors.py)을 사용한다.

targets(dict):
    gt_labels : [B, M, 1]
    gt_bboxes : [B, M, 4]  (x1,y1,x2,y2 픽셀, imgsz 기준)
    mask_gt   : [B, M, 1]
    gt_masks  : [B, M, Hm, Wm]  (segment, proto 해상도 이진 마스크)  — segment 시
    gt_kpts   : [B, M, nk, D]   (D=2 xy / 3 xy+vis, 픽셀)            — pose 시
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.anchors import assign_centers, grid_points
from .loss import bbox_ciou


def crop_mask(masks, boxes):
    """proto 해상도 마스크를 박스 밖은 0으로 crop. masks:[N,H,W], boxes:[N,4](xyxy, proto좌표)."""
    n, h, w = masks.shape
    x1, y1, x2, y2 = boxes.chunk(4, 1)                       # 각 [N,1]
    r = torch.arange(w, device=masks.device).view(1, 1, w)
    c = torch.arange(h, device=masks.device).view(1, h, 1)
    return masks * ((r >= x1[:, :, None]) & (r < x2[:, :, None]) &
                    (c >= y1[:, :, None]) & (c < y2[:, :, None]))


class MultiTaskLoss(nn.Module):
    def __init__(self, nc=80, strides=(8, 16, 32),
                 tasks=("detect", "segment", "pose"), kpt_shape=(4, 3),
                 nm=32, radius=2.5, weights=None):
        super().__init__()
        self.nc = nc
        self.strides = strides
        self.tasks = tuple(tasks)
        self.segment = "segment" in self.tasks
        self.pose = "pose" in self.tasks
        self.nk, self.kdim = kpt_shape
        self.nm = nm
        self.radius = radius                       # center-sampling 반경(×stride)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.w = {"box": 7.5, "cls": 0.5, "seg": 2.5, "kpt": 12.0, "vis": 1.0}
        if weights:
            self.w.update(weights)

    def forward(self, out: dict, targets: dict):
        feats = out["feats"]
        device = feats[0].device
        B = feats[0].shape[0]

        box = torch.cat([b.flatten(2) for b in out["box"]], 2).permute(0, 2, 1)   # [B,A,4]
        cls = torch.cat([c.flatten(2) for c in out["cls"]], 2).permute(0, 2, 1)   # [B,A,nc]
        points, stride_t = grid_points(feats, self.strides)     # [A,2](셀), [A,1]
        points_px = points * stride_t
        strides_px = stride_t.squeeze(1)
        A = points.shape[0]

        # 예측 box 디코딩(px, l/t/r/b 직접)
        dist = F.softplus(box)                                  # [B,A,4] ≥ 0
        px, py = points[:, 0], points[:, 1]
        pred_xyxy = torch.stack([px - dist[..., 0], py - dist[..., 1],
                                 px + dist[..., 2], py + dist[..., 3]], -1)
        pred_xyxy_px = pred_xyxy * stride_t.view(1, -1, 1)

        # center-sampling 라벨 할당 (이미지별)
        gt_labels = targets["gt_labels"].to(device)
        gt_bboxes = targets["gt_bboxes"].to(device)
        mask_gt = targets["mask_gt"].to(device)
        M = gt_bboxes.shape[1]
        fg_mask = torch.zeros(B, A, dtype=torch.bool, device=device)
        gt_flat = torch.zeros(B, A, dtype=torch.long, device=device)
        for b in range(B):
            valid = mask_gt[b, :, 0] > 0
            orig = valid.nonzero(as_tuple=False).squeeze(1)
            if orig.numel() == 0:
                continue
            local = assign_centers(points_px, strides_px, gt_bboxes[b][valid], self.radius)
            pos = local >= 0
            fg_mask[b] = pos
            gt_flat[b, pos] = orig[local[pos]] + b * M
        npos = fg_mask.sum().clamp(min=1)

        # --- 분류 (전체 anchor, 양성에만 one-hot) ---
        cls_t = torch.zeros(B, A, self.nc, device=device)
        if fg_mask.any():
            labels = gt_labels.long().view(-1)[gt_flat[fg_mask]]
            cls_t[fg_mask] = F.one_hot(labels.clamp(0, self.nc - 1), self.nc).float()
        loss_cls = self.bce(cls, cls_t).sum() / npos

        zero = torch.zeros(1, device=device)
        loss_box = loss_seg = loss_kpt = loss_vis = zero.clone()

        if fg_mask.any():
            idx_pos = gt_flat[fg_mask]                          # GT flat index [Npos]

            # --- Detect: CIoU ---
            pb = pred_xyxy_px[fg_mask]
            tb = gt_bboxes.view(-1, 4)[idx_pos]
            loss_box = (1 - bbox_ciou(pb, tb)).sum() / npos

            # --- Segment: mask = coeff·proto, box-crop BCE ---
            if self.segment:
                proto = out["proto"]                            # [B,nm,Hm,Wm]
                Hm, Wm = proto.shape[-2:]
                coef = torch.cat([m.flatten(2) for m in out["mask_coef"]], 2
                                 ).permute(0, 2, 1)             # [B,A,nm]
                gt_masks = targets["gt_masks"].to(device)       # [B,M,Hm,Wm]
                batch_of_pos = torch.arange(B, device=device).view(-1, 1).expand_as(fg_mask)[fg_mask]
                gm_flat = gt_masks.view(-1, Hm, Wm)[idx_pos]    # [Npos,Hm,Wm]
                # box를 proto 좌표로 (px → proto 해상도)
                sx = Wm / (feats[0].shape[-1] * self.strides[0])
                boxes_p = tb * sx
                segl = 0.0
                for b in range(B):
                    sel = batch_of_pos == b
                    if not sel.any():
                        continue
                    pm = torch.einsum("nc,chw->nhw", coef[b][fg_mask[b]], proto[b])  # [n,Hm,Wm]
                    pm = crop_mask(pm, boxes_p[sel])
                    l = F.binary_cross_entropy_with_logits(pm, gm_flat[sel], reduction="none")
                    area = (boxes_p[sel, 2] - boxes_p[sel, 0]).clamp(1) * \
                           (boxes_p[sel, 3] - boxes_p[sel, 1]).clamp(1)
                    segl = segl + (l.mean((1, 2)) / area).sum()
                loss_seg = segl / npos

            # --- Pose: keypoint L1 + visibility BCE ---
            if self.pose:
                kpt = torch.cat([k.flatten(2) for k in out["kpt"]], 2).permute(0, 2, 1)  # [B,A,nk*D]
                # 키포인트는 픽셀 좌표라 제곱오차가 fp16(max 65504)에서 overflow→NaN 될 수
                # 있다. 좌표·정규화항을 fp32로 계산해 수치 안정성 확보(AMP 무관).
                kp = kpt[fg_mask].view(-1, self.nk, self.kdim).float()
                st = stride_t.view(1, -1, 1).expand(B, -1, 1)[fg_mask].float()   # [Npos,1]
                ap = points.unsqueeze(0).expand(B, -1, 2)[fg_mask].float()       # [Npos,2]
                pxk = (kp[:, :, 0] * 2.0 + ap[:, 0:1]) * st              # [Npos,nk]
                pyk = (kp[:, :, 1] * 2.0 + ap[:, 1:2]) * st
                gk = targets["gt_kpts"].to(device).view(-1, self.nk, self.kdim)[idx_pos].float()
                # 박스 크기로 정규화한 L1 (가시 키포인트만)
                bw = (tb[:, 2] - tb[:, 0]).clamp(1)[:, None].float()
                bh = (tb[:, 3] - tb[:, 1]).clamp(1)[:, None].float()
                vis = (gk[:, :, 2] > 0).float() if self.kdim == 3 else torch.ones_like(pxk)
                d = (((pxk - gk[:, :, 0]) / bw) ** 2 + ((pyk - gk[:, :, 1]) / bh) ** 2)
                loss_kpt = ((d * vis).sum(1) / vis.sum(1).clamp(1)).sum() / npos
                if self.kdim == 3:
                    loss_vis = self.bce(kp[:, :, 2], vis).mean(1).sum() / npos

        loss = (self.w["box"] * loss_box
                + self.w["cls"] * loss_cls + self.w["seg"] * loss_seg
                + self.w["kpt"] * loss_kpt + self.w["vis"] * loss_vis)
        items = {k: float(v.detach()) for k, v in {
            "box": loss_box, "cls": loss_cls, "seg": loss_seg,
            "kpt": loss_kpt, "vis": loss_vis, "total": loss}.items()}
        return loss, items
