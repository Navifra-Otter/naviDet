"""
COCO val2017 person keypoint AP/AR 평가 (pycocotools COCOeval, iouType='keypoints').

모델을 COCO val 전체(기본 5000장)에 돌려 OKS 기반 AP/AR을 산출하고, published
방법론 수치와 한 표로 비교한다. YOLO 변환셋이 아니라 COCO GT json + 원본 이미지를
직접 읽으므로 published 와 동일 조건(전체 val)으로 평가된다.

사용 예
-------
  python -m navidet.tools.eval_coco_pose \
      --config navidet/config/coco_pose.yaml \
      --set predict.ckpt=runs/pose/coco/best.pt \
      --ann /home/otter/fiftyone/coco-2017/raw/person_keypoints_val2017.json \
      --img-dir /home/otter/fiftyone/coco-2017/validation/data
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from navidet.core.model import MultiTaskDet, TASK_PRESET
from navidet.module.dataset import letterbox_rgb
from navidet.utils.config import load_config
from navidet.utils.nms import nms, xywh2xyxy

# COCO val2017 person keypoint AP (single-scale). ⚠ published 수치 — 원 논문/레포에서
# 재확인 권장. navidet pose 는 YOLO 계열 single-stage 라 같은 계열이 가장 직접 비교군.
PUBLISHED = [
    # (method, input, AP, source)
    ("YOLOv8n-pose", "640", 50.4, "Ultralytics docs (val2017)"),
    ("YOLOv8s-pose", "640", 60.0, "Ultralytics docs (val2017)"),
    ("YOLOv8m-pose", "640", 65.0, "Ultralytics docs (val2017)"),
    ("YOLOv8l-pose", "640", 67.6, "Ultralytics docs (val2017)"),
    ("YOLOv8x-pose", "640", 69.2, "Ultralytics docs (val2017)"),
    ("RTMPose-m (top-down)", "256x192", 75.3, "RTMPose paper (val2017)"),
    ("HRNet-W32 (top-down)", "256x192", 74.4, "HRNet paper (val2017)"),
    ("ViTPose-B (top-down)", "256x192", 75.8, "ViTPose paper (val2017)"),
]


@torch.no_grad()
def infer_image(model, pil, imgsz, conf, iou, device, nk, D):
    """단일 PIL 이미지 → (boxes_xyxy_lb, scores, kpts_lb[N,nk,D]) (letterbox 좌표).
    추가로 (r, left, top) 역변환 파라미터 반환."""
    rgb, r, left, top = letterbox_rgb(pil, imgsz)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    dec = model(x)
    boxes = dec["boxes"][0].transpose(0, 1)              # [A,4] xywh px
    scores = dec["scores"][0]                            # [nc,A]
    cls_score, _ = scores.max(0)                         # [A]  (person 단일 클래스)
    kpt = dec["kpt"][0].view(nk, D, -1).permute(2, 0, 1)  # [A,nk,D] px(+vis)

    idx = (cls_score > conf).nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, nk, D)), (r, left, top)
    xyxy = xywh2xyxy(boxes[idx])
    keep = nms(xyxy, cls_score[idx], iou_thr=iou)
    sel = idx[torch.tensor(keep, device=device)] if len(keep) else idx[:0]
    return (xywh2xyxy(boxes[sel]).cpu().numpy(),
            cls_score[sel].cpu().numpy(),
            kpt[sel].cpu().numpy(), (r, left, top))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ann", required=True, help="person_keypoints_val2017.json")
    ap.add_argument("--img-dir", required=True, help="원본 val 이미지 디렉토리(data/)")
    ap.add_argument("--max", type=int, default=0, help="평가 이미지 수 제한(0=전체)")
    ap.add_argument("--out", default="_pred/coco_kpt_results.json")
    args = ap.parse_args()

    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cfg = load_config(args.config, args.set)
    p = cfg.predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(p.ckpt, map_location=device, weights_only=False)
    task = ckpt.get("task", "pose")
    if task not in TASK_PRESET or "pose" not in TASK_PRESET[task]:
        raise SystemExit(f"pose 체크포인트가 아닙니다 (task={task}).")
    nk, D = tuple(ckpt.get("kpt_shape", (17, 3)))
    imgsz = ckpt["imgsz"]
    model = MultiTaskDet(nc=ckpt["nc"], scale=ckpt["scale"], tasks=TASK_PRESET[task],
                         nm=ckpt.get("nm", 32), kpt_shape=(nk, D)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"ckpt={p.ckpt}  scale={ckpt['scale']}  kpt_shape={(nk, D)}  imgsz={imgsz}  "
          f"params={sum(x.numel() for x in model.parameters()) / 1e6:.2f}M")
    if nk != 17:
        print(f"⚠ kpt_shape={nk} ≠ 17. COCO person keypoint AP는 17 키포인트 기준입니다.")

    coco_gt = COCO(args.ann)
    img_ids = sorted(coco_gt.getImgIds(catIds=coco_gt.getCatIds(catNms=["person"])))
    if args.max:
        img_ids = img_ids[:args.max]

    results = []
    for n, img_id in enumerate(img_ids):
        info = coco_gt.loadImgs(img_id)[0]
        pil = Image.open(os.path.join(args.img_dir, info["file_name"])).convert("RGB")
        boxes, sc, kpts, (r, left, top) = infer_image(
            model, pil, imgsz, p.conf, p.iou, device, nk, D)
        for i in range(len(boxes)):
            kp = kpts[i].copy()
            if not (np.isfinite(boxes[i]).all() and np.isfinite(kp[:, :2]).all()):
                continue                                # 불안정 예측(NaN/inf) 제외
            kp[:, 0] = (kp[:, 0] - left) / r            # letterbox → 원본 px
            kp[:, 1] = (kp[:, 1] - top) / r
            flat = []
            for k in range(nk):
                flat += [float(kp[k, 0]), float(kp[k, 1]), 1.0]  # v=1 (COCOeval 무시)
            results.append({"image_id": int(img_id), "category_id": 1,
                            "keypoints": flat, "score": float(sc[i])})
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(img_ids)} images, {len(results)} dets")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"))
    print(f"결과 저장: {args.out}  (dets={len(results)})")
    if not results:
        raise SystemExit("검출 0개 — conf 임계값/체크포인트를 확인하세요.")

    coco_dt = coco_gt.loadRes(args.out)
    ev = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    ap_main = ev.stats[0]                                 # AP @ OKS=.50:.95

    # ── 비교표 ──
    print("\n" + "=" * 64)
    print(f"{'method':32}{'input':>9}{'AP(val)':>10}")
    print("-" * 64)
    print(f"{'navidet pose (ours)':32}{str(imgsz):>9}{ap_main * 100:>9.1f}*")
    for name, inp, ap_v, _ in PUBLISHED:
        print(f"{name:32}{inp:>9}{ap_v:>10.1f}")
    print("-" * 64)
    print("* 본 평가에서 측정. 나머지는 published 수치(원문 재확인 권장).")
    for name, inp, ap_v, src in PUBLISHED:
        print(f"  - {name}: {src}")
    print("=" * 64)


if __name__ == "__main__":
    main()
