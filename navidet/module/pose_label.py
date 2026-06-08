"""
YOLO 4-keypoint + Depth → 6DoF GT 라벨 생성 (RANSAC 강건판).

설계: naviEYE PalletModule/SingleModule(운영 고전 파이프라인)의 강건 기법을 차용.
  - 코너 4점만 보지 않고, 쿼드 내부를 '그리드 다수 점'으로 샘플 (get_lines 방식)
  - depth 마스크(0.1<d<max)로 유효한 점만 3D 언프로젝션 (rgbd2pcd 방식)
  - X–Z(top-down) 평면에서 RANSAC 직선 피팅으로 yaw를 강건 추정
    (get_line_info_via_ransac 방식) → 코너 하나가 depth 구멍이어도 복원됨
  - 알려진 팔레트 깊이(pallet_depth)로 size의 z(두께)를 채움

mode="face" (권장, full 6DoF):
  - 외곽 프레임(테두리 좌·우·하; 상단 제외)을 모양 유지 확장해 띠로 샘플
    → 내부(트레이/구멍)보다 robust → 3D RANSAC '평면'을 깨끗하게 잡음
  - 그 평면에 GT 키포인트 레이를 쏴 코너/사이드를 복원(Femto Bolt 사이드 depth 결손
    대응) → 평면 법선에서 roll/pitch/yaw 전체를 얻음(직립 가정 불필요)

좌표/회전 규약
  - 카메라좌표계(x 우, y 하, z 전방)
  - 오브젝트축: x=가로(전면 폭), y=세로(아래), z=전면 법선
  - (face 외 모드) 팔레트 직립 가정 → roll/pitch≈0, yaw 지배
  - R = camera_R_object (오브젝트축이 카메라좌표계에서 표현된 행렬)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 키포인트 순서 (시계방향: 오른쪽 위부터) TR,BR,BL,TL
TR, BR, BL, TL = 0, 1, 2, 3


@dataclass
class ObjectPose:
    cls: int
    bbox_xyxy: np.ndarray     # [4] 픽셀
    R: np.ndarray             # [3,3] camera_R_object
    t: np.ndarray             # [3] 카메라좌표 전면 중심 (m)
    size: np.ndarray          # [3] (width, height, depth/thickness) (m)
    kpts3d: np.ndarray        # [4,3] 피팅된 전면 4코너 (TR,BR,BL,TL) 카메라좌표 (m)
    fit_rmse: float           # RANSAC 직선 잔차 RMSE (m)
    n_inliers: int            # RANSAC inlier 수 (품질 지표)
    valid: bool


# ----------------------------------------------------------------------------- #
#  샘플링 & 언프로젝션
# ----------------------------------------------------------------------------- #
def sample_face_grid(kpts_px: np.ndarray, num_w: int = 15, num_h: int = 3,
                     offset: float = 0.15, row_lo: float | None = None,
                     row_hi: float | None = None) -> np.ndarray:
    """
    쿼드(TR,BR,BL,TL) 내부를 가로 num_w × 세로 num_h 그리드로 샘플(픽셀).
    offset   : 좌우 가장자리 깊이 번짐 회피용 안쪽 여백 비율.
    row_lo/hi: 세로 샘플 범위(0=top, 1=bottom). cart는 바닥 변 부근만 샘플.
    """
    tr, br, bl, tl = kpts_px
    row_lo = offset if row_lo is None else row_lo
    row_hi = (1 - offset) if row_hi is None else row_hi
    rows = np.linspace(row_lo, row_hi, num_h)            # top→bottom
    cols = np.linspace(offset, 1 - offset, num_w)        # left→right
    pts = []
    for f in rows:
        left = tl * (1 - f) + bl * f                      # 왼쪽 변 보간
        right = tr * (1 - f) + br * f                     # 오른쪽 변 보간
        for g in cols:
            pts.append(left * (1 - g) + right * g)
    return np.array(pts, dtype=np.float64)               # [num_w*num_h, 2]


def sample_face_border(kpts_px: np.ndarray, num_long: int = 24, num_band: int = 3,
                       band: float = 0.12, expand: float = 0.04,
                       include_top: bool = False) -> np.ndarray:
    """
    쿼드(TR,BR,BL,TL) **테두리**를 따라 띠(band)로 샘플(픽셀).

    파렛트/카트의 가장 robust 한 부분은 외곽 프레임(좌·우 기둥 + 바닥 보)이다.
    내부는 트레이/구멍/적재물로 depth 가 들쭉날쭉(평면 아웃라이어)하므로, 테두리
    띠만 뽑으면 3D RANSAC 평면이 훨씬 깨끗하게 잡힌다(인라이어율 ↑, RMSE ↓).
    상단(top)은 적재물 경계/핸들 등으로 불안정해 기본 제외(include_top=False).

    band   : 각 변에서 안쪽으로의 띠 두께(0~1 비율).
    expand : 모양 유지한 채 중심 기준으로 살짝 바깥 확장(테두리 프레임에 정확히 얹기).
    """
    tr, br, bl, tl = kpts_px.astype(np.float64)
    if expand:                                            # 모양 유지 확장(중심 스케일)
        ctr = (tr + br + bl + tl) / 4
        tr, br, bl, tl = (ctr + (p - ctr) * (1 + expand) for p in (tr, br, bl, tl))

    def at(f, g):                                         # f: top→bottom, g: left→right
        left = tl * (1 - f) + bl * f
        right = tr * (1 - f) + br * f
        return left * (1 - g) + right * g

    bands = np.linspace(0.0, band, num_band)
    f_lo = 0.0 if include_top else band                   # 좌/우 띠에서 상단부 제외
    pts = []
    for f in np.linspace(f_lo, 1.0, num_long):            # 좌 변 (g≈0)
        pts += [at(f, g) for g in bands]
    for f in np.linspace(f_lo, 1.0, num_long):            # 우 변 (g≈1)
        pts += [at(f, g) for g in 1 - bands]
    for g in np.linspace(0.0, 1.0, num_long):             # 바닥 변 (f≈1)
        pts += [at(f, g) for f in 1 - bands]
    if include_top:
        for g in np.linspace(0.0, 1.0, num_long):         # 상단 변 (f≈0)
            pts += [at(f, g) for f in bands]
    return np.array(pts, dtype=np.float64)


def rgbd_unproject(px: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                   max_depth: float = 10.0):
    """
    픽셀 다수점 → 유효 depth(0.1<d<max)인 점만 3D로 (rgbd2pcd 방식, 벡터화).
    반환: P[M,3] (유효점만, M≤N).
    """
    H, W = depth_m.shape
    u = np.clip(np.round(px[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(px[:, 1]).astype(int), 0, H - 1)
    z = depth_m[v, u]
    mask = (z > 0.1) & (z < max_depth)
    if not mask.any():
        return np.empty((0, 3))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zz = z[mask]
    xx = (u[mask] - cx) * zz / fx
    yy = (v[mask] - cy) * zz / fy
    return np.stack((xx, yy, zz), axis=1)


# ----------------------------------------------------------------------------- #
#  RANSAC 2D 직선 피팅 (X–Z 평면)
# ----------------------------------------------------------------------------- #
def ransac_line_2d(pts: np.ndarray, iters: int = 150, thresh: float = 0.04,
                   rng: np.random.Generator | None = None):
    """
    pts[N,2] → 강건 직선. 반환: (direction[2] 단위벡터, point[2] 직선상 한 점,
    inlier_mask[N], rmse). 점이 2개 미만이면 None.
    """
    N = len(pts)
    if N < 2:
        return None
    rng = rng or np.random.default_rng(0)
    best_inliers, best_cnt = None, -1
    for _ in range(iters):
        i, j = rng.choice(N, 2, replace=False)
        p0, p1 = pts[i], pts[j]
        d = p1 - p0
        nrm = np.linalg.norm(d)
        if nrm < 1e-9:
            continue
        d /= nrm
        n = np.array([-d[1], d[0]])                       # 법선
        dist = np.abs((pts - p0) @ n)                     # 직선까지 수직거리
        inliers = dist < thresh
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt, best_inliers = cnt, inliers
    if best_inliers is None or best_cnt < 2:
        return None
    # inlier로 TLS(주성분) 재피팅
    P = pts[best_inliers]
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    direction = Vt[0] / np.linalg.norm(Vt[0])
    n = np.array([-direction[1], direction[0]])
    rmse = float(np.sqrt((((P - c) @ n) ** 2).mean()))
    return direction, c, best_inliers, rmse


# ----------------------------------------------------------------------------- #
#  보조: yaw → 회전행렬
# ----------------------------------------------------------------------------- #
def canonical_quad(width: float, height: float) -> np.ndarray:
    """정준 사각형 4코너(오브젝트 중심 원점, x=가로 y=세로 z=0). 순서 TR,BR,BL,TL."""
    w, h = width / 2, height / 2
    return np.array([[+w, -h, 0.0], [+w, +h, 0.0],
                     [-w, +h, 0.0], [-w, -h, 0.0]], dtype=np.float64)


def kabsch(Q: np.ndarray, P: np.ndarray):
    """정준 Q(N×3)를 관측 P(N×3)에 맞추는 강체변환 R,t (R·Q+t≈P), det(R)=+1."""
    cq, cp = Q.mean(0), P.mean(0)
    H = (Q - cq).T @ (P - cp)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cp - R @ cq
    rmse = float(np.sqrt((((R @ Q.T).T + t - P) ** 2).sum(1).mean()))
    return R, t, rmse


def fit_plane(points: np.ndarray):
    """≥3점에 최소제곱 평면 피팅 → (평면상의 점 c, 단위 법선 n)."""
    c = points.mean(0)
    _, _, Vt = np.linalg.svd(points - c)
    n = Vt[2] / (np.linalg.norm(Vt[2]) + 1e-12)            # 최소 분산 방향 = 법선
    return c, n


def ray_plane_intersect(u, v, K, c, n):
    """픽셀(u,v) 카메라 레이와 평면(점 c, 법선 n)의 교점 3D. 평행이면 None."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])      # 레이 방향
    denom = float(n @ d)
    if abs(denom) < 1e-9:
        return None
    s = float(n @ c) / denom
    return s * d


def ransac_plane_full(P: np.ndarray, iters: int = 200, thresh: float = 0.03,
                      rng: np.random.Generator | None = None):
    """
    점군 P(N×3)에 3D RANSAC 평면 피팅.
    반환: (c[3] 평면상 점, n[3] 단위 법선, inlier_mask[N], rmse). 실패 시 None.
    """
    rng = rng or np.random.default_rng(0)
    N = len(P)
    if N < 3:
        return None
    best_mask, best_cnt = None, -1
    for _ in range(iters):
        a, b, cc = P[rng.choice(N, 3, replace=False)]
        n = np.cross(b - a, cc - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:                       # 세 점이 거의 일직선 → 평면 정의 불가
            continue
        n /= nn
        mask = np.abs(P @ n - (n @ a)) < thresh
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_mask = cnt, mask
    if best_mask is None or best_cnt < 3:
        return None
    c, n = fit_plane(P[best_mask])          # 인라이어로 최소제곱 재피팅(정밀화)
    rmse = float(np.sqrt((((P[best_mask] - c) @ n) ** 2).mean()))
    return c, n, best_mask, rmse


def ransac_plane(P: np.ndarray, iters: int = 100, thresh: float = 0.03,
                 rng: np.random.Generator | None = None):
    """점군 P(N×3)에 강건 평면 피팅(RANSAC) → (점 c, 법선 n). 실패 시 None. (간편 래퍼)"""
    res = ransac_plane_full(P, iters=iters, thresh=thresh, rng=rng)
    return None if res is None else (res[0], res[1])


def R_from_xaxis(x_axis_xz: np.ndarray) -> np.ndarray:
    """
    X–Z 평면의 가로방향(직선 direction)으로부터 회전행렬 구성.
    x=가로(수평), y=(0,1,0) 카메라 아래, z=법선=x×y.
    """
    x_axis = np.array([x_axis_xz[0], 0.0, x_axis_xz[1]], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)                     # 재직교화
    return np.stack([x_axis, y_axis, z_axis], axis=1)     # 열 = 오브젝트축


def R_from_plane_corners(n: np.ndarray, P4: np.ndarray) -> np.ndarray:
    """
    평면 법선 n 과 복원된 4코너 P4(TR,BR,BL,TL)로 오브젝트 회전 R 구성.
      z = 면 법선(카메라 쪽), x = 가로(좌→우), y = 세로(아래). 직교화로 강체화.
    Kabsch 처럼 직사각형을 강제하지 않아 z 는 평면에 '정확히' 눕고, 코너(P4)는
    GT 키포인트 레이 위에 그대로 남는다(=재투영 시 코너가 edge 에 정확히 붙음).
    roll/pitch/yaw 전체가 평면 기울기에서 나온다.
    """
    z = n / (np.linalg.norm(n) + 1e-12)
    if z[2] > 0:                                          # 카메라(+z 응시) 쪽을 향하게
        z = -z
    x_dir = (P4[TR] + P4[BR]) / 2 - (P4[TL] + P4[BL]) / 2  # 좌→우
    y_dir = (P4[BL] + P4[BR]) / 2 - (P4[TL] + P4[TR]) / 2  # 상→하(아래)
    x = x_dir - (x_dir @ z) * z                           # 평면에 투영(z 성분 제거)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    if y @ y_dir < 0:                                     # 아래 방향으로 정렬
        y = -y
    z = np.cross(x, y)                                    # 우수계 보장(det=+1)
    return np.stack([x, y, z], axis=1)                    # 열 = 오브젝트축


def line_wb_from_dir(direction: np.ndarray, point: np.ndarray):
    """X–Z 직선 (direction, point) → Z = w*X + b 형태. 수직(가로≈Z축)이면 None."""
    dx, dz = direction
    if abs(dx) < 1e-6:                                    # X 변화 거의 없음 → 발산
        return None
    w = dz / dx
    b = point[1] - w * point[0]
    return float(w), float(b)


def intersect_ray_line(u: float, fx: float, cx: float, w: float, b: float):
    """
    픽셀 u의 카메라 레이(X = m*Z, m=(u-cx)/fx)와 직선 Z=w*X+b 의 교점 X.
    (운영 min_X/max_X 공식과 동치)  레이가 직선과 평행하면 None.
    """
    denom = fx - w * (u - cx)
    if abs(denom) < 1e-6:
        return None
    return float(b * (u - cx) / denom)


def _corner_depth(depth_m, u, v, win=13, min_valid=4):
    """코너 (u,v) 주변 win창에서 유효 depth 중앙값(m). 부족하면 0."""
    H, W = depth_m.shape
    ui, vi, r = int(round(u)), int(round(v)), win // 2
    patch = depth_m[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
    nz = patch[(patch > 0.1) & (patch < 10.0)]
    return float(np.median(nz)) if nz.size >= min_valid else 0.0


def _pose_from_corners(cls, bbox_xyxy, kpts_px, depth_m, K, max_depth, fail,
                       win=13):
    """
    4코너 직접 역투영 → 평면 피팅으로 결측 코너 복원 → Kabsch로 R,t 추정.
    각진/원근 카트도 4점에 정확히 맞는다(재투영오차 ~수 px).
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.full((4, 3), np.nan)
    valid = np.zeros(4, dtype=bool)
    for i, (u, v) in enumerate(kpts_px):
        z = _corner_depth(depth_m, u, v, win)
        if 0 < z < max_depth:
            P[i] = [(u - cx) * z / fx, (v - cy) * z / fy, z]
            valid[i] = True

    if valid.sum() >= 3:
        # 유효 코너로 평면 피팅 (가장 정확, ~2px)
        c, n = fit_plane(P[valid])
        n_in = int(valid.sum())
    else:
        # 코너 depth 부족 → 쿼드 그리드 다수점 RANSAC 평면으로 폴백 (커버리지 보강)
        G = rgbd_unproject(sample_face_grid(kpts_px, 15, 7, offset=0.12),
                           depth_m, K, max_depth)
        plane = ransac_plane(G)
        if plane is None:
            return fail()
        c, n = plane
        n_in = len(G)

    # 모든 코너를 레이∩평면으로 (공면화 + 결측 복원 + 노이즈 평활)
    P4 = np.zeros((4, 3))
    for i, (u, v) in enumerate(kpts_px):
        x = ray_plane_intersect(u, v, K, c, n)
        if x is None:
            return fail()
        P4[i] = x

    width = 0.5 * (np.linalg.norm(P4[TR] - P4[TL]) + np.linalg.norm(P4[BR] - P4[BL]))
    height = 0.5 * (np.linalg.norm(P4[TR] - P4[BR]) + np.linalg.norm(P4[TL] - P4[BL]))
    Q = canonical_quad(width, height)
    R, t, rmse = kabsch(Q, P4)
    size = np.array([width, height, 0.0])
    kpts3d = (Q @ R.T) + t
    return ObjectPose(cls, bbox_xyxy, R, t, size, kpts3d, rmse, n_in, valid=True)


def _pose_from_face(cls, bbox_xyxy, kpts_px, depth_m, K, max_depth, fail,
                    band=0.12, expand=0.04, ransac_thresh=0.03, min_points=8,
                    rigid=False):
    """
    테두리 띠로 평면 먼저 → GT 키포인트 레이로 코너/사이드 복원 → full 6DoF.

    설계 의도
      - 외곽 프레임(좌·우·하; 상단 제외)은 내부(트레이/구멍/적재물)보다 depth 가
        robust → 테두리 띠만 샘플해 3D RANSAC 평면을 깨끗하게 잡는다.
      - Femto Bolt 가 사이드에 depth 를 못 줘도, 4코너는 그 평면 위에서 레이로 복원.
      - 직사각형을 강제(Kabsch)하지 않고 복원 코너를 그대로 kpts3d 로 써서 코너가
        2D 키포인트(=객체 edge)에 정확히 붙는다. rigid=True 면 직사각형 강제(비교용).
      - roll/pitch/yaw 전체가 평면 기울기에서 나온다(직립 가정 불필요).
    """
    grid_px = sample_face_border(kpts_px, band=band, expand=expand)
    P = rgbd_unproject(grid_px, depth_m, K, max_depth)
    if len(P) < min_points:
        return fail()
    plane = ransac_plane_full(P, thresh=ransac_thresh)
    if plane is None:
        return fail()
    c, n, inliers, rmse = plane

    # 4코너 픽셀 레이를 평면에 교차 → 사이드 depth 가 없어도 코너 복원(공면화)
    P4 = np.zeros((4, 3))
    for i, (u, v) in enumerate(kpts_px):
        x = ray_plane_intersect(u, v, K, c, n)
        if x is None:
            return fail()
        P4[i] = x

    R = R_from_plane_corners(n, P4)
    t = P4.mean(0)
    width = 0.5 * (np.linalg.norm(P4[TR] - P4[TL]) + np.linalg.norm(P4[BR] - P4[BL]))
    height = 0.5 * (np.linalg.norm(P4[TR] - P4[BR]) + np.linalg.norm(P4[TL] - P4[BL]))
    size = np.array([width, height, 0.0], dtype=np.float64)
    if rigid:                                  # 직사각형 강제(기존 호환/비교용)
        Q = canonical_quad(width, height)
        R, t, _ = kabsch(Q, P4)
        kpts3d = (Q @ R.T) + t
    else:
        kpts3d = P4                            # 복원 코너 그대로(edge 에 붙음)
    return ObjectPose(cls, bbox_xyxy, R, t, size, kpts3d, rmse,
                      int(inliers.sum()), valid=True)


# ----------------------------------------------------------------------------- #
#  메인 변환 (강건판)
# ----------------------------------------------------------------------------- #
def keypoints_to_pose(cls: int, bbox_norm: np.ndarray, kpts_norm: np.ndarray,
                      depth_m: np.ndarray, K: np.ndarray, img_wh: tuple[int, int],
                      max_depth: float = 10.0, num_w: int = 15, num_h: int = 3,
                      min_points: int = 8, ransac_thresh: float = 0.04,
                      mode: str = "full", border_band: float = 0.12,
                      border_expand: float = 0.04, face_rigid: bool = False,
                      **_legacy) -> ObjectPose:
    """
    단일 오브젝트 (정규화 bbox, 4키포인트) + depth → ObjectPose (RANSAC 강건).

    kpts_norm 순서: TR,BR,BL,TL.
    mode:
      - "face" (권장, full 6DoF): **테두리 띠**(좌·우·하, 상단 제외)를 모양 유지
        확장해 샘플 → 3D RANSAC 평면 → GT 키포인트 레이로 코너/사이드 복원.
        프레임은 내부보다 depth 가 robust → 평면이 깨끗하고, 사이드 depth 가 없어도
        코너가 복원되며 roll/pitch/yaw 전체를 얻는다. (border_band/border_expand 조절,
        face_rigid=True 면 직사각형 강제)
      - "full" (팔레트 전면 등 얇은 평면): 쿼드 전체를 샘플, X–Z 2D 직선(yaw)
      - "bottom" (cart 등 세로로 긴 구조): **바닥 변 부근만** 샘플
        (운영 SingleModule이 cart에서 bottom_line만 쓰는 것과 동일 — top/bottom이
         서로 다른 깊이라 전체를 섞으면 yaw·중심·높이가 틀어지는 문제 방지).
        height는 바닥 깊이에서 2D 세로 길이를 역투영해 추정.
      - "corners" (cart 등 4코너 뚜렷): 4코너 직접 역투영 + 평면복원 + Kabsch
    size = (width, height, 0). z(두께)는 미관측.
    """
    W, H = img_wh
    cx, cy, bw, bh = bbox_norm
    bbox_xyxy = np.array([(cx - bw / 2) * W, (cy - bh / 2) * H,
                          (cx + bw / 2) * W, (cy + bh / 2) * H], dtype=np.float64)
    kpts_px = kpts_norm * np.array([W, H])

    def fail():
        return ObjectPose(cls, bbox_xyxy, np.eye(3), np.zeros(3), np.zeros(3),
                          np.full((4, 3), np.nan), np.inf, 0, valid=False)

    # ── mode="face" (권장): 테두리 띠 평면 → GT 키포인트로 코너/사이드 복원 ──
    if mode == "face":
        return _pose_from_face(cls, bbox_xyxy, kpts_px, depth_m, K, max_depth, fail,
                               band=border_band, expand=border_expand,
                               ransac_thresh=ransac_thresh, min_points=min_points,
                               rigid=face_rigid)

    # ── mode="corners" (cart 등 4코너 뚜렷): 4코너 직접 역투영 + 평면복원 + Kabsch ──
    if mode == "corners":
        return _pose_from_corners(cls, bbox_xyxy, kpts_px, depth_m, K,
                                  max_depth, fail)

    # 1) 샘플 영역: cart는 바닥 변 부근만, 그 외는 쿼드 전체
    if mode == "bottom":
        grid_px = sample_face_grid(kpts_px, num_w, max(num_h, 3),
                                   offset=0.04, row_lo=0.80, row_hi=0.97)
    else:
        grid_px = sample_face_grid(kpts_px, num_w, num_h, offset=0.15)
    P = rgbd_unproject(grid_px, depth_m, K, max_depth)
    if len(P) < min_points:
        return fail()

    # 2) X–Z 평면 RANSAC 직선 피팅 → yaw 방향
    xz = P[:, [0, 2]]
    res = ransac_line_2d(xz, thresh=ransac_thresh)
    if res is None:
        return fail()
    direction, c_pt, inliers, rmse = res
    Pin = P[inliers]
    if len(Pin) < min_points:
        return fail()

    # 3) 좌/우 코너 픽셀 레이를 RANSAC 직선과 교차 → 실제 폭/끝점 복원
    #    (운영 get_line_info_via_ransac 방식: 코너 depth 구멍과 무관하게 정확)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    line = line_wb_from_dir(direction, c_pt)              # Z = w*X + b
    if line is None:
        return fail()
    w_, b_ = line
    order = np.argsort(kpts_px[:, 0])
    left_u = kpts_px[order[:2], 0].mean()                 # 좌측 2코너 평균 u
    right_u = kpts_px[order[2:], 0].mean()                # 우측 2코너 평균 u
    Xl = intersect_ray_line(left_u, fx, cx, w_, b_)
    Xr = intersect_ray_line(right_u, fx, cx, w_, b_)
    if Xl is None or Xr is None:
        return fail()
    left_xz = np.array([Xl, w_ * Xl + b_])
    right_xz = np.array([Xr, w_ * Xr + b_])
    width_m = float(np.linalg.norm(right_xz - left_xz))

    # 기준 중심 t = '면 중심' (모델은 bbox 중심 depth로 t를 예측하므로 일치시켜야 함)
    if mode == "bottom":
        # cart: 4점 중심 픽셀을 바닥 직선 깊이로 역투영 → 면 중심.
        #       (바닥 변만 봤어도 t는 바닥이 아니라 면 중심이어야 쿼드가 안 처짐)
        uc = float(kpts_px[:, 0].mean()); vc = float(kpts_px[:, 1].mean())
        Xc = intersect_ray_line(uc, fx, cx, w_, b_)
        if Xc is None:
            return fail()
        Zc = w_ * Xc + b_
        t = np.array([Xc, (vc - cy) * Zc / fy, Zc])
        mid_top_v = (kpts_px[TR, 1] + kpts_px[TL, 1]) / 2     # 2D 세로길이 → 높이
        mid_bot_v = (kpts_px[BR, 1] + kpts_px[BL, 1]) / 2
        height_m = float(abs(mid_top_v - mid_bot_v) * Zc / fy)
    else:
        mid_xz = (left_xz + right_xz) / 2
        t = np.array([mid_xz[0], float(Pin[:, 1].mean()), mid_xz[1]])
        height_m = float(Pin[:, 1].max() - Pin[:, 1].min())
    size = np.array([width_m, height_m, 0.0], dtype=np.float64)  # z(두께) 미관측

    # 4) 회전: 끝점 방향으로 x축 구성 (yaw 지배)
    R = R_from_xaxis(right_xz - left_xz)

    # 5) 시각화/검증용 전면 4코너 복원
    kpts3d = (canonical_quad(width_m, height_m) @ R.T) + t

    return ObjectPose(cls, bbox_xyxy, R, t, size, kpts3d, rmse,
                      int(inliers.sum()), valid=True)


# ----------------------------------------------------------------------------- #
#  YOLO 라벨 파서
# ----------------------------------------------------------------------------- #
def parse_yolo_kpt_label(line: str, num_kpts: int = 4):
    """'cls cx cy w h kx1 ky1 ...' → (cls, bbox_norm[4], kpts_norm[K,2])."""
    v = list(map(float, line.split()))
    cls = int(v[0])
    bbox = np.array(v[1:5], dtype=np.float64)
    rest = v[5:]
    stride = len(rest) // num_kpts        # 2(xy) 또는 3(xyv)
    kpts = np.array(rest, dtype=np.float64).reshape(num_kpts, stride)[:, :2]
    return cls, bbox, kpts
