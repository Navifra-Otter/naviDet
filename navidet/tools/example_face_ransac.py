"""
[예제] 전체 면 다운샘플 + 3D RANSAC 평면 → full 6DoF GT 추출.

기존 방식과의 차이
------------------
현재 `keypoints_to_pose(..., mode="bottom")` 은 4키포인트 중 **바닥 변 부근만**
샘플해서 X–Z(top-down) 평면에서 **2D RANSAC 직선**을 피팅한다. 즉 점 한 줄(line)만
보고 yaw 하나만 추정하며, roll/pitch 는 "팔레트/카트는 직립" 이라는 가정으로 0 으로
박아 버린다. 카트가 앞으로 기울거나(피치) 좌우로 기울면(롤) 그 정보를 못 잡는다.

이 예제(`pose_from_face_ransac`)는 다르게 한다.
  1) 4점이 이루는 **쿼드 전체 면**을 가로×세로 격자로 다운샘플 (한 줄이 아니라 면 전체)
  2) 유효 depth 점만 3D 로 언프로젝션 → 면 전체의 점군(point cloud)
  3) **3D RANSAC 평면** 피팅 (2D 직선이 아니라 3D 평면) → 면의 법선까지 강건 추정
  4) 4코너 픽셀 레이를 그 평면에 교차시켜 3D 코너 4점 복원
  5) Kabsch 로 정준 쿼드를 맞춰 **회전 전체(roll·pitch·yaw) R, 위치 t, 크기 size** 추출

→ depth 구멍이 군데군데 있어도 면 전체에서 다수 점을 보므로 강건하고,
   평면의 기울기(법선)를 그대로 쓰므로 6DoF 가 전부 살아난다.

실행
----
  # repo 루트에서
  python -m navidet.tools.example_face_ransac \
      --root /media/otter/otterHD/mando_aug/train \
      --ini  "/media/otter/otterHD/mando_aug/CameraParam_Orbbec Femto BoltCL8855300X5_Color1920x1080_Depth640x576.ini" \
      --stem augmented_frame_0 --out _face_ransac.png

  # --stem 생략 시 라벨 디렉토리에서 첫 프레임 자동 선택
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import numpy as np
from PIL import Image, ImageDraw

# 핵심 헬퍼는 pose_label 에 정식 반영됨 → 여기서 import (단일 출처).
# 이 예제는 그 위에 '복원 점 시각화'만 덧붙인 데모다.
from navidet.module.pose_label import (
    TR, BR, BL, TL,
    ObjectPose,
    sample_face_grid,
    sample_face_border,
    rgbd_unproject,
    canonical_quad,
    kabsch,
    ray_plane_intersect,
    fit_plane,
    ransac_plane_full,
    R_from_plane_corners,
    keypoints_to_pose,
    parse_yolo_kpt_label,
)
from navidet.utils.camera import load_orbbec_ini
from navidet.utils.visualize import draw_axes, draw_quad, project


def pose_from_face_ransac(cls: int, bbox_norm: np.ndarray, kpts_norm: np.ndarray,
                          depth_m: np.ndarray, K: np.ndarray, img_wh: tuple[int, int],
                          max_depth: float = 10.0, num_w: int = 25, num_h: int = 15,
                          offset: float = 0.10, ransac_iters: int = 400,
                          ransac_thresh: float = 0.03, thickness: float = 0.0,
                          border_band: float = 0.12, border_expand: float = 0.04,
                          rigid: bool = False, return_debug: bool = False):
    """
    [테두리 띠로 평면 먼저 → GT 점으로 남은 영역 복원]

      1) 쿼드 **테두리 띠**(좌·우·하, 상단 제외)를 모양 유지한 채 살짝 확장해 샘플
         → 프레임은 내부(트레이/구멍)보다 depth 가 robust → 3D RANSAC 평면이 깨끗
      2) 그 평면에 GT 키포인트(4코너) 레이를 쏴 코너 3D 복원
         — Femto Bolt 가 사이드에 depth 를 못 줘도, 코너는 평면 위에서 복원됨
      3) 쿼드 전 영역을 평면에 재투영해 'depth 없던 사이드'까지 면을 채움(복원)
      4) 평면 법선 + 복원 코너로 R 구성, 코너 그대로를 kpts3d 로(=edge 에 붙음)

    rigid=True 면 복원 코너에 직사각형(Kabsch)을 강제(기존 방식, 비교용).
    kpts_norm 순서: TR,BR,BL,TL.
    return_debug=True → (ObjectPose, dict) 추가 반환.
    """
    W, H = img_wh
    cx, cy, bw, bh = bbox_norm
    bbox_xyxy = np.array([(cx - bw / 2) * W, (cy - bh / 2) * H,
                          (cx + bw / 2) * W, (cy + bh / 2) * H], dtype=np.float64)
    kpts_px = kpts_norm * np.array([W, H])

    def fail():
        pose = ObjectPose(cls, bbox_xyxy, np.eye(3), np.zeros(3), np.zeros(3),
                          np.full((4, 3), np.nan), np.inf, 0, valid=False)
        return (pose, {}) if return_debug else pose

    # 1) 테두리 띠(좌·우·하; 상단 제외)를 모양 유지 확장해 샘플 — 프레임이 robust
    grid_px = sample_face_border(kpts_px, num_long=max(num_w, num_h) + 9,
                                 band=border_band, expand=border_expand)

    # 2) 유효 depth 점만 3D 로 (depth 구멍/사이드는 자동 탈락)
    P = rgbd_unproject(grid_px, depth_m, K, max_depth)
    if len(P) < 10:
        return fail()

    # 3) 3D RANSAC 평면 — 중앙의 다수 점만으로도 강건하게 면 평면 추정
    plane = ransac_plane_full(P, iters=ransac_iters, thresh=ransac_thresh)
    if plane is None:
        return fail()
    c, n, inliers, rmse = plane

    # 4) 4코너 픽셀 레이를 평면에 교차 → 공면화된 3D 코너 (사이드 depth 없어도 복원)
    P4 = np.zeros((4, 3))
    for i, (u, v) in enumerate(kpts_px):
        x = ray_plane_intersect(u, v, K, c, n)
        if x is None:                       # 레이가 평면과 평행(거의 없음)
            return fail()
        P4[i] = x

    # 5) '남은 영역' 복원: 쿼드 전 영역(코너까지, offset=0)을 평면에 재투영.
    #    depth 가 없던 셀까지 평면 위 3D 점으로 메워 사이드를 복원한다.
    fill_px = sample_face_grid(kpts_px, num_w=num_w, num_h=num_h, offset=0.0)
    recov, was_measured = [], []
    Hd, Wd = depth_m.shape
    for (u, v) in fill_px:
        x = ray_plane_intersect(u, v, K, c, n)
        if x is None:
            continue
        recov.append(x)
        ui = int(np.clip(round(u), 0, Wd - 1)); vi = int(np.clip(round(v), 0, Hd - 1))
        z = depth_m[vi, ui]
        was_measured.append(bool(0.1 < z < max_depth))   # 원래 depth 있었나?
    recov = np.asarray(recov); was_measured = np.asarray(was_measured, dtype=bool)

    # 6) 회전: 평면 법선 + 복원 코너. 크기: 변 길이.
    R = R_from_plane_corners(n, P4)
    t = P4.mean(0)
    width = 0.5 * (np.linalg.norm(P4[TR] - P4[TL]) + np.linalg.norm(P4[BR] - P4[BL]))
    height = 0.5 * (np.linalg.norm(P4[TR] - P4[BR]) + np.linalg.norm(P4[TL] - P4[BL]))
    size = np.array([width, height, thickness], dtype=np.float64)

    # kpts3d: 기본은 '복원 코너'(edge 에 붙음). rigid=True 면 직사각형 강제(비교용).
    if rigid:
        Q = canonical_quad(width, height)
        R, t, _ = kabsch(Q, P4)
        kpts3d = (Q @ R.T) + t
    else:
        kpts3d = P4

    pose = ObjectPose(cls, bbox_xyxy, R, t, size, kpts3d,
                      fit_rmse=rmse, n_inliers=int(inliers.sum()), valid=True)
    if return_debug:
        return pose, dict(grid_px=grid_px, P=P, inliers=inliers, c=c, n=n,
                          recov=recov, was_measured=was_measured, P4=P4)
    return pose


# --------------------------------------------------------------------------- #
#  비교용 유틸
# --------------------------------------------------------------------------- #
def euler_zyx_deg(R: np.ndarray):
    """R(camera_R_object) → (roll_x, pitch_y, yaw_z) 근사 도(°). 직관 비교용."""
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1]); y = np.arctan2(-R[2, 0], sy); z = 0.0
    return np.degrees([x, y, z])


def tilt_from_vertical_deg(R: np.ndarray) -> float:
    """오브젝트 z축(면 법선)이 수평면에서 벗어난 각(°). 2D-line 방식은 항상 0."""
    z_axis = R[:, 2]
    horiz = np.hypot(z_axis[0], z_axis[2])
    return float(np.degrees(np.arctan2(abs(z_axis[1]), horiz)))


def describe(tag: str, p: ObjectPose):
    if not p.valid:
        return f"  [{tag}] INVALID (depth 부족/피팅 실패)"
    r, pi, ya = euler_zyx_deg(p.R)
    return (f"  [{tag}]\n"
            f"      t (m)        : [{p.t[0]:+.3f}, {p.t[1]:+.3f}, {p.t[2]:+.3f}]\n"
            f"      size W×H (m)  : {p.size[0]:.3f} × {p.size[1]:.3f}\n"
            f"      euler r/p/y(°): [{r:+6.1f}, {pi:+6.1f}, {ya:+6.1f}]"
            f"   (면 기울기 {tilt_from_vertical_deg(p.R):4.1f}°)\n"
            f"      fit RMSE / inliers: {p.fit_rmse*1000:5.1f} mm / {p.n_inliers}")


# --------------------------------------------------------------------------- #
#  시각화: 이미지에 두 방식 쿼드/축 + 다운샘플 점, 옆에 3D 점군+평면
# --------------------------------------------------------------------------- #
def save_viz(img: Image.Image, K, dbg: dict, pose_new: ObjectPose,
             pose_old: ObjectPose, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (좌) 이미지 위 오버레이 -------------------------------------------------- #
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    P, inl = dbg["P"], dbg["inliers"]
    recov, was_m = dbg["recov"], dbg["was_measured"]
    # 5-1) '복원된 사이드'(원래 depth 없던 셀을 평면에 재투영한 점) = 파랑
    uv_r = project(recov[~was_m], K) if (~was_m).any() else np.empty((0, 2))
    for (u, v) in uv_r:
        draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(80, 160, 255))
    # 5-2) 실측 점: 평면 인라이어=초록 / 아웃라이어=주황
    uv = project(P, K)
    for (u, v), good in zip(uv, inl):
        col = (60, 230, 60) if good else (255, 150, 0)
        draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=col)
    if pose_old.valid:                          # 기존 바닥-라인 방식 = 노랑
        draw_quad(draw, pose_old.kpts3d, K, color=(255, 230, 0), w=3, label=False)
        draw_axes(draw, pose_old.R, pose_old.t, K, length=0.18, w=3)
    if pose_new.valid:                          # 새 면-RANSAC(복원) 방식 = 시안
        draw_quad(draw, pose_new.kpts3d, K, color=(0, 230, 230), w=3, label=True)
        draw_axes(draw, pose_new.R, pose_new.t, K, length=0.18, w=5)

    # bbox 크롭(보기 좋게)
    x1, y1, x2, y2 = pose_new.bbox_xyxy if pose_new.valid else pose_old.bbox_xyxy
    pad = 0.4
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - pad * bw)); cy1 = max(0, int(y1 - pad * bh))
    cx2 = min(img.width, int(x2 + pad * bw)); cy2 = min(img.height, int(y2 + pad * bh))
    crop = np.asarray(canvas.crop((cx1, cy1, cx2, cy2)))

    fig = plt.figure(figsize=(15, 7))
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.imshow(crop); ax0.set_axis_off()
    ax0.set_title("overlay  (yellow=bottom-line 2D / cyan=face-RANSAC 3D)\n"
                  "border samples: green=inliers, orange=outliers | blue=recovered side",
                  fontsize=10)

    # (우) 3D 점군 + 피팅 평면 + 복원 코너 ------------------------------------ #
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    Pin, Pout = P[inl], P[~inl]
    Rf = dbg["recov"][~dbg["was_measured"]]            # 복원된 사이드 점
    ax1.scatter(Pin[:, 0], Pin[:, 2], -Pin[:, 1], s=6, c="g", label="measured inliers")
    if len(Pout):
        ax1.scatter(Pout[:, 0], Pout[:, 2], -Pout[:, 1], s=6, c="orange", label="outliers")
    if len(Rf):
        ax1.scatter(Rf[:, 0], Rf[:, 2], -Rf[:, 1], s=6, c="dodgerblue",
                    label="recovered side")
    # 피팅 평면을 코너 범위에 맞춰 메쉬로 그림
    if pose_new.valid:
        k3 = pose_new.kpts3d
        c, n = dbg["c"], dbg["n"]
        span_x = np.linspace(k3[:, 0].min(), k3[:, 0].max(), 8)
        span_y = np.linspace(k3[:, 1].min(), k3[:, 1].max(), 8)
        gx, gy = np.meshgrid(span_x, span_y)
        # 평면식 n·(x-c)=0 에서 z 풀이 (n_z≈0 이면 패스)
        if abs(n[2]) > 1e-3:
            gz = c[2] - (n[0] * (gx - c[0]) + n[1] * (gy - c[1])) / n[2]
            ax1.plot_surface(gx, gz, -gy, alpha=0.25, color="cyan")
        # 복원된 4코너 + 쿼드 외곽
        loop = np.vstack([k3, k3[0]])
        ax1.plot(loop[:, 0], loop[:, 2], -loop[:, 1], "c-", lw=2)
        ax1.scatter(k3[:, 0], k3[:, 2], -k3[:, 1], c="b", s=40)
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z depth (m)"); ax1.set_zlabel("-Y up (m)")
    ax1.set_title("face point cloud + 3D-RANSAC plane + recovered corners", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n시각화 저장: {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/media/otter/otterHD/mando_aug/train",
                    help="images/ labels/ depth/ 를 가진 split 디렉토리")
    ap.add_argument("--ini", default="/media/otter/otterHD/mando_aug/"
                    "CameraParam_Orbbec Femto BoltCL8855300X5_Color1920x1080_Depth640x576.ini")
    ap.add_argument("--intrinsic-section", default="ColorIntrinsic")
    ap.add_argument("--stem", default=None, help="프레임 stem (생략 시 첫 라벨 자동)")
    ap.add_argument("--obj", type=int, default=0, help="프레임 내 몇 번째 오브젝트")
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--num-kpts", type=int, default=4)
    ap.add_argument("--out", default="_face_ransac.png")
    args = ap.parse_args()

    intr = load_orbbec_ini(args.ini, args.intrinsic_section)
    K, img_wh = intr.K, (intr.width, intr.height)

    stem = args.stem
    if stem is None:
        first = sorted(glob(os.path.join(args.root, "labels", "*.txt")))[0]
        stem = os.path.splitext(os.path.basename(first))[0]

    label_path = os.path.join(args.root, "labels", stem + ".txt")
    depth_path = os.path.join(args.root, "depth", stem + ".png")
    image_path = os.path.join(args.root, "images", stem + ".png")
    depth_m = np.asarray(Image.open(depth_path)).astype(np.float64) * args.depth_scale

    lines = [l for l in open(label_path).read().strip().split("\n") if l.strip()]
    cls, bbox, kpts = parse_yolo_kpt_label(lines[args.obj], args.num_kpts)

    # --- 기존 방식: 바닥 변 한 줄 → 2D RANSAC 직선 (yaw만, roll/pitch=0 가정) ---
    pose_old = keypoints_to_pose(cls, bbox, kpts, depth_m, K, img_wh, mode="bottom")
    # --- 새 방식: 면 전체 다운샘플 → 3D RANSAC 평면 → full 6DoF ---------------
    pose_new, dbg = pose_from_face_ransac(cls, bbox, kpts, depth_m, K, img_wh,
                                          return_debug=True)

    print("=" * 72)
    print(f"frame: {stem}  obj#{args.obj}  cls={cls}")
    print("-" * 72)
    print("기존: 바닥 변 한 줄 → 2D RANSAC 직선 (yaw만)")
    print(describe("bottom-line 2D", pose_old))
    print("-" * 72)
    print("신규: 평면 먼저(중앙 depth) → GT 키포인트로 사이드/코너 복원 (full 6DoF)")
    print(describe("face-RANSAC 3D", pose_new))
    print("=" * 72)

    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        save_viz(img, K, dbg, pose_new, pose_old, args.out)


if __name__ == "__main__":
    main()
