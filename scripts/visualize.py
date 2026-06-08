#!/usr/bin/env python3
"""navidet 추론 결과 시각화 — 2D pose / 6DoF 를 체크포인트 task로 자동 분기.

임의의 이미지(파일 또는 폴더)에 대해 추론하고 오버레이 이미지를 저장한다.
GT/라벨이 없어도 동작하므로(추론 전용) tools/predict.py(검증셋+GT 시각화)와 달리
새 프레임/실사용 이미지를 바로 확인할 때 쓴다.

  · 6dof : 전면 쿼드(청록) + 좌표축(R) 재투영. 카메라 K가 필요하지만 --ini 가
           없으면 프레임 크기로 근사한다(축/쿼드 스케일이 대략적).
  · pose : 박스 + 키포인트 오버레이 (K 불필요).

사용 예:
    # repo 루트에서 실행
    python scripts/visualize.py --ckpt runs/exp_s/best.pt --source some.png
    python scripts/visualize.py --ckpt runs/exp_s/best.pt --source images/ --out _viz
    python scripts/visualize.py --ckpt runs/6dof/best.pt --source f.png --ini cam.ini
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# repo 루트를 import path 에 추가 (scripts/ 하위에서 실행돼도 navidet 패키지 인식).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navidet.core.model import YOLO6DoF, YOLOPose, TASK_PRESET
from navidet.module.dataset import letterbox_rgb
from navidet.module.infer import render_sample
from navidet.tools.predict import render_pose_sample
from navidet.utils.camera import load_orbbec_ini

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="체크포인트 .pt (task 자동 판별)")
    p.add_argument("--source", required=True, help="이미지 파일 또는 폴더")
    p.add_argument("--out", default="_viz", help="결과 저장 폴더")
    p.add_argument("--ini", default=None, help="[6dof] 카메라 intrinsics .ini (없으면 근사)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence 임계값")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU 임계값")
    p.add_argument("--max", type=int, default=0, help="처리할 최대 이미지 수 (0=전체)")
    p.add_argument("--device", default=None, help="cuda | cpu (기본 자동)")
    return p.parse_args()


def list_images(source: str) -> list[str]:
    sp = Path(source)
    if sp.is_dir():
        files = sorted(f for f in glob(str(sp / "*")) if f.lower().endswith(IMG_EXTS))
        if not files:
            raise FileNotFoundError(f"이미지 없음: {source}/*{IMG_EXTS}")
        return files
    if not sp.exists():
        raise FileNotFoundError(f"경로 없음: {source}")
    return [str(sp)]


def letterbox_K(K0: np.ndarray, native_wh, frame_wh, r, left, top) -> np.ndarray:
    """원본 K0(native 해상도) → 프레임 스케일 → letterbox 보정된 K (predictor와 동일식)."""
    W0, H0 = frame_wh
    K = K0.copy()
    if native_wh is not None:
        K[0] *= W0 / native_wh[0]
        K[1] *= H0 / native_wh[1]
    K[0] *= r; K[1] *= r
    K[0, 2] += left; K[1, 2] += top
    return K


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    task = ck.get("task", "6dof")
    is_2d = task in TASK_PRESET
    imgsz = ck["imgsz"]
    nc = ck["nc"]
    cfg = ck.get("cfg", {}) or {}
    names = (cfg.get("data", {}) or {}).get("class_names") or [str(i) for i in range(nc)]

    if is_2d:
        kpt_shape = tuple(ck.get("kpt_shape", (4, 3)))
        model = YOLOPose(nc=nc, scale=ck["scale"], tasks=TASK_PRESET[task],
                         nm=ck.get("nm", 32), kpt_shape=kpt_shape).to(device)
    else:
        model = YOLO6DoF(nc=nc, scale=ck["scale"], rot_repr=ck["rot_repr"],
                         light_head=ck.get("light_head", True)).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    # 6dof intrinsics (native 해상도 K + 원본 해상도). 없으면 프레임마다 근사.
    K0 = native_wh = None
    if not is_2d and args.ini:
        intr = load_orbbec_ini(args.ini)
        K0, native_wh = intr.K, (intr.width, intr.height)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    files = list_images(args.source)
    if args.max:
        files = files[:args.max]

    print(f"task={task}  imgsz={imgsz}  nc={nc}  device={device}  images={len(files)}")
    for fp in files:
        img = Image.open(fp).convert("RGB")
        W0, H0 = img.size
        rgb, r, left, top = letterbox_rgb(img, imgsz)          # letterbox 입력 공간
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        sample = {"img": x, "gt_labels": torch.zeros(0, 1)}

        if is_2d:
            pil, n = render_pose_sample(model, sample, names, conf=args.conf, iou=args.iou)
        else:
            if K0 is not None:
                K = letterbox_K(K0, native_wh, (W0, H0), r, left, top)
            else:                                              # intrinsics 미제공 → 근사
                f = float(max(W0, H0))
                K = letterbox_K(np.array([[f, 0, W0 / 2], [0, f, H0 / 2], [0, 0, 1]]),
                                None, (W0, H0), r, left, top)
            sample["K"] = torch.from_numpy(K).float()
            pil, msgs = render_sample(model, sample, names, conf=args.conf, iou=args.iou,
                                      draw_gt=False)
            n = len(msgs)

        stem = Path(fp).stem
        dst = out_dir / f"{stem}_pred.png"
        pil.save(dst)
        print(f"  saved {dst}  (pred {n}개)")


if __name__ == "__main__":
    main()
