"""
Anchor-free 검출용 grid/박스 유틸 — direct(l,t,r,b) 거리 회귀 방식.

표준 anchor-free 검출(FCOS / YOLOX, Apache-2.0 설계)의 공개 수식을 PyTorch
기본 연산만으로 구현한 것입니다. 외부 라이브러리 의존성은 없습니다.

  · grid_points    : 각 FPN 레벨의 grid-center(셀 단위) 좌표와 per-point stride
  · dist_to_xyxy   : (l,t,r,b) 거리 → (x1,y1,x2,y2)
  · dist_to_xywh   : (l,t,r,b) 거리 → (cx,cy,w,h)
  · bbox_iou       : IoU / CIoU (박스 회귀 손실용)
  · assign_centers : center-sampling 라벨 할당 (grid point ↔ GT)
"""

from __future__ import annotations

import torch


def grid_points(feats, strides, offset: float = 0.5):
    """각 레벨 feature → grid-center(셀 단위) 좌표 + per-point stride.

    반환: points[A,2] (셀 단위 중심), strides[A,1].  A = Σ Hi*Wi
    """
    pts, strs = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for f, s in zip(feats, strides):
        _, _, h, w = f.shape
        xs = torch.arange(w, device=device, dtype=dtype) + offset
        ys = torch.arange(h, device=device, dtype=dtype) + offset
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pts.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), -1))      # [h*w,2]
        strs.append(torch.full((h * w, 1), float(s), device=device, dtype=dtype))
    return torch.cat(pts), torch.cat(strs)


def dist_to_xyxy(ltrb: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """(l,t,r,b) 거리 → (x1,y1,x2,y2). ltrb:[B,4,A] (셀 단위), points:[A,2]."""
    l, t, r, b = ltrb.unbind(1)
    px, py = points[:, 0], points[:, 1]
    return torch.stack((px - l, py - t, px + r, py + b), dim=1)            # [B,4,A]


def dist_to_xywh(ltrb: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """(l,t,r,b) 거리 → (cx,cy,w,h). ltrb:[B,4,A] (셀 단위), points:[A,2]."""
    x1, y1, x2, y2 = dist_to_xyxy(ltrb, points).unbind(1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=1)


def bbox_iou(pred: torch.Tensor, target: torch.Tensor, ciou: bool = True,
             eps: float = 1e-7) -> torch.Tensor:
    """elementwise IoU(또는 CIoU). pred/target: [...,4] xyxy. 반환 [...] (IoU/CIoU)."""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    # 교집합
    ix1 = torch.maximum(px1, tx1); iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2); iy2 = torch.minimum(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    pw = (px2 - px1).clamp(min=0); ph = (py2 - py1).clamp(min=0)
    tw = (tx2 - tx1).clamp(min=0); th = (ty2 - ty1).clamp(min=0)
    union = pw * ph + tw * th - inter + eps
    iou = inter / union
    if not ciou:
        return iou
    # CIoU: 중심거리 + 종횡비 패널티
    cw = torch.maximum(px2, tx2) - torch.minimum(px1, tx1)
    ch = torch.maximum(py2, ty2) - torch.minimum(py1, ty1)
    c2 = cw * cw + ch * ch + eps
    pcx = (px1 + px2) / 2; pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2; tcy = (ty1 + ty2) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    import math
    v = (4 / math.pi ** 2) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


def assign_centers(points: torch.Tensor, strides: torch.Tensor,
                   gts_xyxy: torch.Tensor, radius: float = 2.5) -> torch.Tensor:
    """center-sampling 라벨 할당 (FCOS/YOLOX 방식).

    grid point가 GT 박스 내부 + 중심 반경(radius*stride) 안에 있으면 후보로 보고,
    여러 GT의 후보면 면적이 가장 작은 GT에 할당한다.

    points : [A,2] (픽셀 좌표 grid center)
    strides: [A]   (per-point stride, 픽셀)
    gts_xyxy: [M,4] (픽셀 xyxy). M=0이면 모두 음성(-1).
    반환    : gt_idx[A]  — 할당된 GT 인덱스(0..M-1) 또는 음성 -1.
    """
    A = points.shape[0]
    M = gts_xyxy.shape[0]
    if M == 0:
        return points.new_full((A,), -1, dtype=torch.long)
    px = points[:, 0:1]; py = points[:, 1:2]                  # [A,1]
    x1, y1, x2, y2 = gts_xyxy.unbind(1)                       # 각 [M]
    inside = (px > x1) & (px < x2) & (py > y1) & (py < y2)    # [A,M]
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    r = (radius * strides).unsqueeze(1)                       # [A,1]
    near = ((px - cx).abs() < r) & ((py - cy).abs() < r)      # [A,M]
    cand = inside & near                                      # [A,M]
    area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)    # [M]
    big = area.max() * 2 + 1
    cost = torch.where(cand, area.unsqueeze(0).expand(A, M), big)
    gt_idx = cost.argmin(1)                                   # [A]
    gt_idx[~cand.any(1)] = -1
    return gt_idx
