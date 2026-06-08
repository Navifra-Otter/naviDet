"""
6DoF 복원에 쓰이는 기하 유틸.

- 6D 회전 표현 ↔ 회전행렬 (Zhou et al., CVPR'19)
- Quaternion ↔ 회전행렬 (옵션)
- Geodesic(측지) 회전 손실
- Depth + Camera Intrinsics → Translation Unprojection
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
#  6D rotation representation
# ----------------------------------------------------------------------------- #
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    6D 표현 → 3x3 회전행렬 (Gram-Schmidt 정규직교화).
    d6: [..., 6]  →  R: [..., 3, 3]
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    # a2에서 b1 성분 제거 후 정규화 → b1과 직교하는 b2
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)         # 행벡터로 stack → [...,3,3]


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """3x3 회전행렬 → 6D 표현 (첫 두 행). R:[...,3,3] → [...,6]"""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


# ----------------------------------------------------------------------------- #
#  Quaternion (옵션)
# ----------------------------------------------------------------------------- #
def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """단위 quaternion(w,x,y,z) → 회전행렬. q:[...,4] → [...,3,3]"""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


# ----------------------------------------------------------------------------- #
#  Geodesic loss
# ----------------------------------------------------------------------------- #
def _rel_cos(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """상대회전의 cos = (trace(R_pred R_gtᵀ) - 1) / 2.  R:[...,3,3] → [...]"""
    R = R_pred @ R_gt.transpose(-2, -1)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return (trace - 1) / 2


def rotation_cosine_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    학습용 안정 회전 손실: 1 - cos(Δangle).  R:[...,3,3] → [...]
    정렬 시 0, 그래디언트가 R에 선형이라 유한(=acos 특이점 없음).
    소각도에서 Δθ²/2 로 거동 → 사실상 각오차의 매끄러운 대리손실.
    """
    return 1.0 - _rel_cos(R_pred, R_gt).clamp(-1.0, 1.0)


def geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    두 회전행렬 간 측지 거리(라디안). 평가/리포트용 (acos는 backward 특이점 有).
    angle = arccos( (trace(R_pred R_gtᵀ) - 1) / 2 )
    """
    cos = _rel_cos(R_pred, R_gt).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos)


# ----------------------------------------------------------------------------- #
#  Translation unprojection
# ----------------------------------------------------------------------------- #
def sample_depth(depth_map: torch.Tensor, uv: torch.Tensor, img_size) -> torch.Tensor:
    """
    depth_map에서 픽셀좌표 uv 위치의 깊이를 bilinear 샘플(미분 가능).

    depth_map: [B, 1, Hd, Wd]  (입력보다 작은 해상도여도 됨)
    uv       : [B, N, 2]       (원본 이미지 픽셀좌표 u,v)
    img_size : (H, W) 원본 입력 크기
    반환     : [B, N] 깊이값
    """
    H, W = img_size
    # grid_sample용 정규좌표 [-1,1]
    grid = torch.empty_like(uv)
    grid[..., 0] = uv[..., 0] / (W - 1) * 2 - 1
    grid[..., 1] = uv[..., 1] / (H - 1) * 2 - 1
    grid = grid.unsqueeze(1)                                   # [B,1,N,2]
    z = F.grid_sample(depth_map, grid, mode="bilinear",
                      align_corners=True)                      # [B,1,1,N]
    return z.view(depth_map.shape[0], -1)                      # [B,N]


def unproject_translation(uv: torch.Tensor, depth: torch.Tensor,
                          K: torch.Tensor) -> torch.Tensor:
    """
    bbox 중심 (u,v) + 깊이 Z + intrinsics K → 카메라좌표 translation (X,Y,Z).

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    uv   : [B, N, 2]
    depth: [B, N]
    K    : [B, 3, 3]
    반환 : [B, N, 3]
    """
    fx = K[:, 0, 0].view(-1, 1)
    fy = K[:, 1, 1].view(-1, 1)
    cx = K[:, 0, 2].view(-1, 1)
    cy = K[:, 1, 2].view(-1, 1)
    Z = depth
    X = (uv[..., 0] - cx) * Z / fx
    Y = (uv[..., 1] - cy) * Z / fy
    return torch.stack((X, Y, Z), dim=-1)                      # [B,N,3]
