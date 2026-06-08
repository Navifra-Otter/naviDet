"""
생성된 6DoF GT(.npz) 시각 검증 — 좌표축/쿼드 재투영.

build_labels로 만든 pose_labels/*.npz 를 원본 이미지에 그려 저장한다.

사용 예
-------
  python -m navidet.tools.viz_gt --root ".../train" --out _viz --num 20
"""

from __future__ import annotations

import argparse
import os
import random
from glob import glob

import numpy as np
from PIL import Image, ImageDraw

from navidet.utils.visualize import draw_axes, draw_quad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pose-name", default="pose_labels")
    ap.add_argument("--out", default="_viz")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None,
                    help="랜덤 시드 (미지정이면 매번 다른 샘플)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_files = sorted(glob(os.path.join(args.root, args.pose_name, "*.npz")))
    rng = random.Random(args.seed)
    npz_files = rng.sample(all_files, min(args.num, len(all_files)))

    for fp in npz_files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        img_path = os.path.join(args.root, "images", stem + ".png")
        if not os.path.exists(img_path):
            continue
        data = np.load(fp)
        K = data["K"]
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for i in range(len(data["cls"])):
            if not data["valid"][i]:
                continue
            draw_quad(draw, data["kpts3d"][i], K)
            draw_axes(draw, data["R"][i], data["t"][i], K)
        out_path = os.path.join(args.out, stem + "_pose.png")
        img.save(out_path)
        print("saved", out_path)


if __name__ == "__main__":
    main()
