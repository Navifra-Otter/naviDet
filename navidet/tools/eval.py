"""
검증셋 6DoF 정확도 평가.

예측을 GT와 2D IoU로 매칭한 뒤, 매칭쌍에 대해 회전·translation·크기 오차와
검출 recall/precision을 집계한다.

사용 예
-------
  python -m navidet.tools.eval --set predict.ckpt=runs/exp_light/best.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from navidet.core.model import Det6DoF
from navidet.module.dataset import PoseDataset
from navidet.utils.config import load_config
from navidet.utils.geometry import geodesic_loss
from navidet.utils.nms import nms, xywh2xyxy


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
def predict(model, s, nc, rd, imgsz, conf, iou, device):
    """단일 샘플 → (pred_box_xyxy[N,4], R[N,3,3], t[N,3], size[N,3])."""
    img = s["img"].unsqueeze(0).to(device)
    K = s["K"].unsqueeze(0).to(device)
    out = model(img)
    out = {"det": out["det"].float(), "depth": out["depth"].float()}
    det = out["det"][0]
    boxes = det[:4].T
    scores = det[4:4 + nc]
    rot = det[4 + nc:4 + nc + rd].T
    size = det[4 + nc + rd:].T
    cf, _ = scores.max(0)
    mask = cf > conf
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return np.zeros((0, 4)), np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    keep = nms(xywh2xyxy(boxes[idx]), cf[idx], iou)
    sel = idx[keep]
    pose = model.decode_pose(out, K, (imgsz, imgsz),
                             boxes[sel].unsqueeze(0), rot[sel].unsqueeze(0), size[sel].unsqueeze(0))
    pb = xywh2xyxy(boxes[sel]).cpu().numpy()
    return pb, pose["R"][0].cpu().numpy(), pose["t"][0].cpu().numpy(), pose["size"][0].cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--iou-match", type=float, default=0.5, help="GT 매칭 IoU 임계값")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d, p = cfg.data, cfg.predict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(p.ckpt, map_location=device, weights_only=False)
    if ck.get("task", "6dof") != "6dof":
        raise SystemExit(
            f"eval.py는 6DoF 정량평가(회전°/translation/depth) 전용입니다. "
            f"체크포인트 task='{ck.get('task')}' (pose/detect)는 미지원입니다. "
            f"pose 시각화는 `python -m navidet.tools.predict`를 사용하세요.")
    nc, rd, imgsz = ck["nc"], (6 if ck["rot_repr"] == "6d" else 4), ck["imgsz"]
    model = Det6DoF(nc=nc, scale=ck["scale"], rot_repr=ck["rot_repr"],
                     light_head=ck.get("light_head", True)).to(device)
    model.load_state_dict(ck["model"]); model.eval().fuse()
    print(f"ckpt={p.ckpt}  light_head={ck.get('light_head', True)}  "
          f"params={sum(x.numel() for x in model.parameters())/1e6:.2f}M")

    ds = PoseDataset(d.val_root, d.ini, imgsz=imgsz, depth_scale=d.depth_scale)

    rot_err, t_err, txy_err, tz_err, w_err = [], [], [], [], []
    n_gt = n_pred = n_tp = 0
    for i in range(len(ds)):
        s = ds[i]
        ng = s["gt_labels"].shape[0]
        n_gt += ng
        pb, R, t, sz = predict(model, s, nc, rd, imgsz, p.conf, p.iou, device)
        n_pred += len(pb)
        if ng == 0 or len(pb) == 0:
            continue
        gb = s["gt_bboxes"].numpy()
        gR = s["gt_rot"].numpy(); gt = s["gt_trans"].numpy(); gs = s["gt_size"].numpy()
        iou = box_iou_np(pb, gb)
        # greedy 매칭 (IoU 최대 우선)
        used_p, used_g = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for pi, gi in order:
            pi, gi = int(pi), int(gi)
            if iou[pi, gi] < args.iou_match:
                break
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            n_tp += 1
            Rp = torch.from_numpy(R[pi])[None].float()
            Rg = torch.from_numpy(gR[gi])[None].float()
            rot_err.append(float(geodesic_loss(Rp, Rg)) * 180 / np.pi)
            dt = t[pi] - gt[gi]
            t_err.append(float(np.linalg.norm(dt)))
            txy_err.append(float(np.linalg.norm(dt[[0, 1]])))
            tz_err.append(float(abs(dt[2])))
            w_err.append(float(abs(sz[pi, 0] - gs[gi, 0])))

    def stat(v, scale=1.0, unit=""):
        if not v:
            return "  (없음)"
        v = np.array(v) * scale
        return f"mean {v.mean():.2f}{unit} | median {np.median(v):.2f}{unit} | p90 {np.percentile(v,90):.2f}{unit}"

    rec = n_tp / max(n_gt, 1); prec = n_tp / max(n_pred, 1)
    print("=" * 60)
    print(f"GT {n_gt} | pred {n_pred} | matched(TP) {n_tp}")
    print(f"detection  recall={rec:.3f}  precision={prec:.3f}")
    print(f"rotation err  : {stat(rot_err, 1.0, 'deg')}")
    print(f"translation   : {stat(t_err, 1000, 'mm')}")
    print(f"  └ XY        : {stat(txy_err, 1000, 'mm')}")
    print(f"  └ Z(depth)  : {stat(tz_err, 1000, 'mm')}")
    print(f"width err     : {stat(w_err, 1000, 'mm')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
