"""
6DoF pose 정확도 평가 (재사용 모듈).

예측을 GT와 2D IoU로 매칭한 뒤 회전·translation·크기 오차와 recall/precision을
집계한다. eval.py(독립 평가)와 학습 루프(epoch별 검증)가 함께 사용한다.
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.geometry import geodesic_loss
from ..utils.nms import nms, xywh2xyxy


def box_iou_np(a, b):
    """a[N,4], b[M,4] xyxy → IoU[N,M]."""
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@torch.no_grad()
def _predict(model, s, nc, rd, imgsz, conf, iou, device, use_sensor_depth=False):
    img = s["img"].unsqueeze(0).to(device)
    K = s["K"].unsqueeze(0).to(device)
    out = model(img)
    out = {"det": out["det"].float(), "depth": out["depth"].float()}
    det = out["det"][0]
    boxes, scores = det[:4].T, det[4:4 + nc]
    rot = det[4 + nc:4 + nc + rd].T
    size = det[4 + nc + rd:].T
    cf, _ = scores.max(0)
    idx = (cf > conf).nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return np.zeros((0, 4)), np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    sel = idx[nms(xywh2xyxy(boxes[idx]), cf[idx], iou)]
    dm = (s["gt_depth"].unsqueeze(0).to(device)
          if use_sensor_depth and "gt_depth" in s else None)
    pose = model.decode_pose(out, K, (imgsz, imgsz), boxes[sel].unsqueeze(0),
                             rot[sel].unsqueeze(0), size[sel].unsqueeze(0), depth_map=dm)
    return (xywh2xyxy(boxes[sel]).cpu().numpy(), pose["R"][0].cpu().numpy(),
            pose["t"][0].cpu().numpy(), pose["size"][0].cpu().numpy())


@torch.no_grad()
def evaluate_pose(model, ds, conf=0.25, iou=0.5, iou_match=0.5,
                  device=None, max_frames=0, use_sensor_depth=False) -> dict:
    """ds 전체(또는 max_frames)에 대해 pose 지표 dict 반환."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    nc, rd, imgsz = model.nc, model.head.rot_dim, ds.imgsz

    rot_err, t_err, w_err = [], [], []
    n_gt = n_pred = n_tp = 0
    N = min(max_frames, len(ds)) if max_frames else len(ds)
    for i in range(N):
        s = ds[i]
        ng = s["gt_labels"].shape[0]
        n_gt += ng
        pb, R, t, sz = _predict(model, s, nc, rd, imgsz, conf, iou, device, use_sensor_depth)
        n_pred += len(pb)
        if ng == 0 or len(pb) == 0:
            continue
        gb, gR = s["gt_bboxes"].numpy(), s["gt_rot"].numpy()
        gt, gs = s["gt_trans"].numpy(), s["gt_size"].numpy()
        iou_m = box_iou_np(pb, gb)
        used_p, used_g = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-iou_m, axis=None), iou_m.shape))[0]
        for pi, gi in order:
            pi, gi = int(pi), int(gi)
            if iou_m[pi, gi] < iou_match:
                break
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi); n_tp += 1
            Rp = torch.from_numpy(R[pi])[None].float()
            Rg = torch.from_numpy(gR[gi])[None].float()
            rot_err.append(float(geodesic_loss(Rp, Rg)) * 180 / np.pi)
            t_err.append(float(np.linalg.norm(t[pi] - gt[gi])))
            w_err.append(float(abs(sz[pi, 0] - gs[gi, 0])))

    model.train(was_training)
    med = lambda v: float(np.median(v)) if v else float("nan")
    return {
        "recall": n_tp / max(n_gt, 1), "precision": n_tp / max(n_pred, 1),
        "n_gt": n_gt, "n_pred": n_pred, "n_tp": n_tp,
        "rot_deg": med(rot_err), "trans_mm": med(t_err) * 1000,
        "width_mm": med(w_err) * 1000,
    }


def format_metrics(m: dict) -> str:
    return (f"recall={m['recall']:.3f} prec={m['precision']:.3f} | "
            f"rot={m['rot_deg']:.2f}deg trans={m['trans_mm']:.1f}mm "
            f"width={m['width_mm']:.1f}mm")
