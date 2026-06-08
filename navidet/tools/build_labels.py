"""
YOLO keypoint 데이터셋 → 6DoF GT 라벨 생성 (배치, 검수용).

프레임당 .npz 저장 + 품질 통계. 학습은 on-the-fly로 GT를 만들므로 필수는 아니며,
GT 검수/시각화(viz_gt) 용도다.

사용 예
-------
  python -m navidet.tools.build_labels \
      --root ".../6dof2/test/train" --ini ".../CameraParam_....ini"
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import numpy as np
from PIL import Image

from navidet.module.pose_label import keypoints_to_pose, parse_yolo_kpt_label
from navidet.utils.camera import load_orbbec_ini


def process_frame(label_path, depth_path, K, img_wh, depth_scale, num_kpts, mode):
    lines = [l for l in open(label_path).read().strip().split("\n") if l.strip()]
    depth_m = np.asarray(Image.open(depth_path)).astype(np.float64) * depth_scale
    poses = []
    for line in lines:
        cls, bbox, kpts = parse_yolo_kpt_label(line, num_kpts)
        poses.append(keypoints_to_pose(cls, bbox, kpts, depth_m, K, img_wh, mode=mode))
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ini", required=True)
    ap.add_argument("--intrinsic-section", default="ColorIntrinsic")
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--num-kpts", type=int, default=4)
    ap.add_argument("--mode", default="face",
                    choices=["face", "full", "bottom", "corners"],
                    help="face=테두리 띠 평면+GT코너 복원(full 6DoF, 권장), "
                         "full=쿼드 전체(팔레트), bottom=바닥 변(얇은 단면), "
                         "corners=4코너 Kabsch(cart 등 각진 객체)")
    ap.add_argument("--out-name", default="pose_labels")
    ap.add_argument("--rmse-thresh", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    intr = load_orbbec_ini(args.ini, args.intrinsic_section)
    K, img_wh = intr.K, (intr.width, intr.height)
    out_dir = os.path.join(args.root, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    label_files = sorted(glob(os.path.join(args.root, "labels", "*.txt")))
    if args.limit:
        label_files = label_files[:args.limit]

    n_frames = n_obj = n_valid = n_bad = 0
    rmse_all, size_all, z_all = [], [], []
    for lp in label_files:
        stem = os.path.splitext(os.path.basename(lp))[0]
        dp = os.path.join(args.root, "depth", stem + ".png")
        if not os.path.exists(dp):
            continue
        poses = process_frame(lp, dp, K, img_wh, args.depth_scale, args.num_kpts, args.mode)
        n_frames += 1
        N = len(poses)
        np.savez(
            os.path.join(out_dir, stem + ".npz"),
            cls=np.array([p.cls for p in poses], dtype=np.int64),
            bbox_xyxy=np.stack([p.bbox_xyxy for p in poses]) if N else np.zeros((0, 4)),
            R=np.stack([p.R for p in poses]) if N else np.zeros((0, 3, 3)),
            t=np.stack([p.t for p in poses]) if N else np.zeros((0, 3)),
            size=np.stack([p.size for p in poses]) if N else np.zeros((0, 3)),
            kpts3d=np.stack([p.kpts3d for p in poses]) if N else np.zeros((0, 4, 3)),
            rmse=np.array([p.fit_rmse for p in poses]),
            n_inliers=np.array([p.n_inliers for p in poses], dtype=np.int64),
            valid=np.array([p.valid for p in poses], dtype=bool),
            K=K, img_wh=np.array(img_wh),
        )
        for p in poses:
            n_obj += 1
            if p.valid:
                n_valid += 1
                rmse_all.append(p.fit_rmse); size_all.append(p.size[:2]); z_all.append(p.t[2])
                if p.fit_rmse > args.rmse_thresh:
                    n_bad += 1

    print("\n================ 요약 ================")
    print(f"프레임 {n_frames} | 오브젝트 {n_obj} | 유효 {n_valid} "
          f"({100*n_valid/max(n_obj,1):.1f}%) | 깊이실패 {n_obj-n_valid}")
    if rmse_all:
        rmse_all = np.array(rmse_all); size_all = np.array(size_all); z_all = np.array(z_all)
        print(f"RANSAC RMSE: mean={rmse_all.mean()*1000:.1f}mm med={np.median(rmse_all)*1000:.1f}mm "
              f"max={rmse_all.max()*1000:.1f}mm | RMSE>{args.rmse_thresh*1000:.0f}mm: {n_bad}")
        print(f"가로 mean={size_all[:,0].mean():.3f} | 세로 mean={size_all[:,1].mean():.3f} "
              f"| Z mean={z_all.mean():.3f}")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
