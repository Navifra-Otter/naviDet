"""
검증셋 추론 + 시각화 (config 기반). 체크포인트 meta의 task로 분기한다.

  · 6dof : 예측(청록)·GT(초록) 전면 쿼드 + 좌표축 시각화, 6DoF 결과 JSONL 발행
  · pose : 박스 + 키포인트 오버레이 시각화

사용 예
-------
  python -m navidet.tools.predict --config navidet/config/default_6dof.yaml --set predict.ckpt=runs/exp/best.pt
  python -m navidet.tools.predict --config navidet/config/default_pose.yaml --set predict.ckpt=runs/exp_s/best.pt
"""

from __future__ import annotations

import argparse
import os
import random

import torch
from PIL import Image, ImageDraw

from navidet.core.model import Det6DoF, MultiTaskDet, TASK_PRESET
from navidet.messaging.publisher import PosePublisher
from navidet.module.dataset import PoseDataset
from navidet.module.infer import render_sample
from navidet.utils.config import load_config
from navidet.utils.nms import nms, xywh2xyxy

_PALETTE = [(0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 255, 0),
            (255, 0, 0), (0, 128, 255)]


@torch.no_grad()
def render_pose_sample(model, sample, names, conf=0.25, iou=0.5):
    """pose 모델 추론 → 박스+키포인트 오버레이 PIL 이미지 반환."""
    device = next(model.parameters()).device
    was_training = model.training                    # 학습 직후 호출돼도 디코딩 보장
    model.eval()
    prev_raw = model.head.return_raw
    model.head.return_raw = False
    x = sample["img"].unsqueeze(0).to(device)
    dec = model(x)                                   # eval dict
    model.head.return_raw = prev_raw
    model.train(was_training)
    boxes = dec["boxes"][0].transpose(0, 1)          # [A,4] xywh px
    scores = dec["scores"][0]                        # [nc,A]
    cls_score, cls_id = scores.max(0)                # [A]
    keep_conf = cls_score > conf
    xyxy = xywh2xyxy(boxes)
    has_kpt = "kpt" in dec and getattr(model, "kpt_shape", None)
    if has_kpt:
        nk, D = model.kpt_shape
        kpt = dec["kpt"][0].view(nk, D, -1).permute(2, 0, 1)  # [A,nk,D] (px, +vis)

    # uint8 이미지 복원
    img = (sample["img"].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    pil = Image.fromarray(img); draw = ImageDraw.Draw(pil)

    idx = keep_conf.nonzero(as_tuple=False).squeeze(1)
    if idx.numel():
        keep = nms(xyxy[idx], cls_score[idx], iou_thr=iou)
        sel = idx[torch.tensor(keep, device=device)] if keep else idx[:0]
        for i in sel.tolist():
            c = _PALETTE[int(cls_id[i]) % len(_PALETTE)]
            x1, y1, x2, y2 = xyxy[i].tolist()
            draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
            nm = names[int(cls_id[i])] if int(cls_id[i]) < len(names) else str(int(cls_id[i]))
            draw.text((x1, max(0, y1 - 10)), f"{nm} {cls_score[i]:.2f}", fill=c)
            if has_kpt:
                for k in range(nk):
                    kx, ky = kpt[i, k, 0].item(), kpt[i, k, 1].item()
                    v = kpt[i, k, 2].item() if D == 3 else 1.0
                    if v > 0.3:
                        draw.ellipse([kx - 3, ky - 3, kx + 3, ky + 3], fill=c)
    return pil, int(idx.numel())


def render_pose_gt(sample, names) -> Image.Image:
    """모델 없이 2D pose GT(박스+키포인트)만 그린 PIL 이미지. (한 번만 저장용)

    sample: PoseDataset(2D)["gt_bboxes"=xyxy px, "gt_kpts"=[N,nk,D] px(+vis)].
    """
    img = (sample["img"].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    pil = Image.fromarray(img); draw = ImageDraw.Draw(pil)
    boxes = sample["gt_bboxes"]
    labels = sample["gt_labels"]
    kpts = sample.get("gt_kpts")                      # detect/segment는 없음
    for i in range(boxes.shape[0]):
        ci = int(labels[i, 0])
        c = _PALETTE[ci % len(_PALETTE)]
        x1, y1, x2, y2 = boxes[i].tolist()
        draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
        nm = names[ci] if ci < len(names) else str(ci)
        draw.text((x1, max(0, y1 - 10)), nm, fill=c)
        if kpts is not None:
            for k in range(kpts.shape[1]):
                kx, ky = kpts[i, k, 0].item(), kpts[i, k, 1].item()
                v = kpts[i, k, 2].item() if kpts.shape[2] == 3 else 1.0
                if v > 0.3:
                    draw.ellipse([kx - 3, ky - 3, kx + 3, ky + 3], fill=c)
    return pil


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d, p = cfg.data, cfg.predict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(p.out, exist_ok=True)

    ckpt = torch.load(p.ckpt, map_location=device, weights_only=False)
    task = ckpt.get("task", "6dof")
    is_2d = task in TASK_PRESET

    if is_2d:
        kpt_shape = tuple(ckpt.get("kpt_shape", (4, 3)))
        model = MultiTaskDet(nc=ckpt["nc"], scale=ckpt["scale"], tasks=TASK_PRESET[task],
                         nm=ckpt.get("nm", 32), kpt_shape=kpt_shape).to(device)
    else:
        model = Det6DoF(nc=ckpt["nc"], scale=ckpt["scale"], rot_repr=ckpt["rot_repr"],
                         light_head=ckpt.get("light_head", True)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    ds = PoseDataset(d.val_root, d.ini, imgsz=ckpt["imgsz"], depth_scale=d.depth_scale,
                     task=task, kpt_shape=tuple(ckpt.get("kpt_shape", (4, 3))))
    rng = random.Random(p.get("seed", 0))
    idxs = rng.sample(range(len(ds)), min(p.num, len(ds)))
    names = list(d.class_names)

    if is_2d:
        for idx in idxs:
            s = ds[idx]
            pil, n = render_pose_sample(model, s, names, conf=p.conf, iou=p.iou)
            pil.save(os.path.join(p.out, f"{s['stem']}_pred.png"))
            print(f"saved {p.out}/{s['stem']}_pred.png  (pred {n}개)")
        return

    # ── 6DoF 경로 ──
    pub = PosePublisher(os.path.join(p.out, "preds.jsonl")) if p.get("publish_json") else None
    if pub and os.path.exists(pub.out_path):
        os.remove(pub.out_path)
    for idx in idxs:
        s = ds[idx]
        pil, msgs = render_sample(model, s, names, conf=p.conf, iou=p.iou)
        pil.save(os.path.join(p.out, f"{s['stem']}_pred.png"))
        if pub:
            pub.publish(s["stem"], msgs)
        print(f"saved {p.out}/{s['stem']}_pred.png  (pred {len(msgs)}개)")
    if pub:
        print(f"발행: {pub.out_path}")


if __name__ == "__main__":
    main()
