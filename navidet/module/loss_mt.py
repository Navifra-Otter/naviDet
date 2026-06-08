"""
2D 멀티태스크 Loss — Detection + Segmentation + Pose.

  L = λ_box·CIoU + λ_dfl·DFL + λ_cls·BCE          (Detect)
    + λ_seg·MaskBCE(coeff·proto, box-crop)         (Segment)
    + λ_kpt·(keypoint L1 + visibility BCE)          (Pose)

라벨 할당은 기존 TaskAlignedAssigner를 그대로 재사용한다(학습 방식 유지).

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

from ..core.head import dist2bbox, make_anchors
from .loss import TaskAlignedAssigner, bbox_ciou, df_loss


def crop_mask(masks, boxes):
    """proto 해상도 마스크를 박스 밖은 0으로 crop. masks:[N,H,W], boxes:[N,4](xyxy, proto좌표)."""
    n, h, w = masks.shape
    x1, y1, x2, y2 = boxes.chunk(4, 1)                       # 각 [N,1]
    r = torch.arange(w, device=masks.device).view(1, 1, w)
    c = torch.arange(h, device=masks.device).view(1, h, 1)
    return masks * ((r >= x1[:, :, None]) & (r < x2[:, :, None]) &
                    (c >= y1[:, :, None]) & (c < y2[:, :, None]))


class MultiTaskLoss(nn.Module):
    def __init__(self, nc=80, reg_max=16, strides=(8, 16, 32),
                 tasks=("detect", "segment", "pose"), kpt_shape=(4, 3),
                 nm=32, weights=None):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.tasks = tuple(tasks)
        self.segment = "segment" in self.tasks
        self.pose = "pose" in self.tasks
        self.nk, self.kdim = kpt_shape
        self.nm = nm
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=nc)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.proj = torch.arange(reg_max, dtype=torch.float)
        self.w = {"box": 7.5, "dfl": 1.5, "cls": 0.5, "seg": 2.5, "kpt": 12.0, "vis": 1.0}
        if weights:
            self.w.update(weights)

    def _dfl_decode(self, box):
        B, A, _ = box.shape
        return (box.view(B, A, 4, self.reg_max).softmax(-1) @ self.proj.to(box.device))

    def forward(self, out: dict, targets: dict):
        feats = out["feats"]
        device = feats[0].device
        B = feats[0].shape[0]

        box = torch.cat([b.flatten(2) for b in out["box"]], 2).permute(0, 2, 1)   # [B,A,4rm]
        cls = torch.cat([c.flatten(2) for c in out["cls"]], 2).permute(0, 2, 1)   # [B,A,nc]
        anchors, stride_t = make_anchors(feats, self.strides)
        anchors_px = anchors * stride_t

        dist = self._dfl_decode(box)
        pred_xyxy = dist2bbox(dist.permute(0, 2, 1), anchors).permute(0, 2, 1)
        pred_xyxy_px = pred_xyxy * stride_t.view(1, -1, 1)
        pred_scores = cls.sigmoid()

        gt_labels = targets["gt_labels"].to(device)
        gt_bboxes = targets["gt_bboxes"].to(device)
        mask_gt = targets["mask_gt"].to(device)
        t_labels, t_bboxes, t_scores, fg_mask, t_gt_idx = self.assigner(
            pred_scores, pred_xyxy_px.detach(), anchors_px, gt_labels, gt_bboxes, mask_gt)
        tss = max(t_scores.sum(), 1.0)

        # --- 분류 (전체 anchor, soft label) ---
        loss_cls = self.bce(cls, t_scores).sum() / tss
        zero = torch.zeros(1, device=device)
        loss_box = loss_dfl = loss_seg = loss_kpt = loss_vis = zero.clone()

        if fg_mask.any():
            weight = t_scores.sum(-1)[fg_mask]
            bidx = torch.arange(B, device=device).view(-1, 1) * gt_bboxes.shape[1]
            idx_pos = (t_gt_idx + bidx)[fg_mask]                # GT flat index [Npos]

            # --- Detect: CIoU + DFL ---
            pb, tb = pred_xyxy_px[fg_mask], t_bboxes[fg_mask]
            loss_box = ((1 - bbox_ciou(pb, tb)) * weight).sum() / tss
            tb_cell = tb / stride_t.view(1, -1, 1).expand(B, -1, 4)[fg_mask]
            anc_pos = anchors.unsqueeze(0).expand(B, -1, 2)[fg_mask]
            tgt_ltrb = torch.cat([anc_pos - tb_cell[:, :2], tb_cell[:, 2:] - anc_pos], 1
                                 ).clamp(0, self.reg_max - 1.01)
            loss_dfl = (df_loss(box[fg_mask].view(-1, 4, self.reg_max), tgt_ltrb,
                                self.reg_max).squeeze(-1) * weight).sum() / tss

            # --- Segment: mask = coeff·proto, box-crop BCE ---
            if self.segment:
                proto = out["proto"]                            # [B,nm,Hm,Wm]
                Hm, Wm = proto.shape[-2:]
                coef = torch.cat([m.flatten(2) for m in out["mask_coef"]], 2
                                 ).permute(0, 2, 1)             # [B,A,nm]
                gt_masks = targets["gt_masks"].to(device)       # [B,M,Hm,Wm]
                batch_of_pos = torch.arange(B, device=device).view(-1, 1).expand_as(fg_mask)[fg_mask]
                gm_flat = gt_masks.view(-1, Hm, Wm)[idx_pos]    # [Npos,Hm,Wm]
                # box를 proto 좌표로 (px → /stride_proto)
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
                    segl = segl + (l.mean((1, 2)) / area * weight[sel]).sum()
                loss_seg = segl / tss

            # --- Pose: keypoint L1 + visibility BCE ---
            if self.pose:
                kpt = torch.cat([k.flatten(2) for k in out["kpt"]], 2).permute(0, 2, 1)  # [B,A,nk*D]
                kp = kpt[fg_mask].view(-1, self.nk, self.kdim)
                st = stride_t.view(1, -1, 1).expand(B, -1, 1)[fg_mask]
                ap = anchors.unsqueeze(0).expand(B, -1, 2)[fg_mask]
                px = (kp[:, :, 0] * 2.0 + ap[:, None, 0]) * st          # [Npos,nk]
                py = (kp[:, :, 1] * 2.0 + ap[:, None, 1]) * st
                gk = targets["gt_kpts"].to(device).view(-1, self.nk, self.kdim)[idx_pos]
                # 박스 크기로 정규화한 L1 (가시 키포인트만)
                bw = (tb[:, 2] - tb[:, 0]).clamp(1)[:, None]
                bh = (tb[:, 3] - tb[:, 1]).clamp(1)[:, None]
                vis = (gk[:, :, 2] > 0).float() if self.kdim == 3 else torch.ones_like(px)
                d = (((px - gk[:, :, 0]) / bw) ** 2 + ((py - gk[:, :, 1]) / bh) ** 2)
                loss_kpt = ((d * vis).sum(1) / vis.sum(1).clamp(1) * weight).sum() / tss
                if self.kdim == 3:
                    vl = self.bce(kp[:, :, 2], vis).mean(1)
                    loss_vis = (vl * weight).sum() / tss

        loss = (self.w["box"] * loss_box + self.w["dfl"] * loss_dfl
                + self.w["cls"] * loss_cls + self.w["seg"] * loss_seg
                + self.w["kpt"] * loss_kpt + self.w["vis"] * loss_vis)
        items = {k: float(v.detach()) for k, v in {
            "box": loss_box, "dfl": loss_dfl, "cls": loss_cls, "seg": loss_seg,
            "kpt": loss_kpt, "vis": loss_vis, "total": loss}.items()}
        return loss, items
