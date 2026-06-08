"""
모델 정의 — 한 백본/넥을 공유하는 두 계열 모델을 함께 제공한다.

  · YOLO6DoF : 백본 + PAN 넥 + (Pose6DoFHead + DepthHead)  — 1-stage 6DoF Pose
  · YOLOPose : 백본 + PAN 넥 + MultiTaskHead               — 2D Detect/Segment/Pose

`build_model(task=...)`로 둘 중 하나를 빌드한다 (task ∈ {6dof, detect, segment, pose}).

YOLO6DoF (Direction B) 구조:

    RGB ─ Backbone ─ Neck ─┬─ DepthHead   ─ Dense Depth Map ─┐
                           └─ Pose6DoFHead ─ 2D BBox/Class ──┤
                                            └ Rotation/Size  │
                                                             ▼
            (BBox 중심 Z 샘플 + Camera Intrinsics) Unprojection ─ Translation(X,Y,Z)
            Rotation(6D→R) + Translation + Size ─ Final 6DoF Pose & 3D Size

YOLOPose 구조:

    RGB ─ Backbone ─ Neck ─ MultiTaskHead ┬─ Detect  (box + class)
                                          ├─ Segment (mask coeff + Proto)
                                          └─ Pose    (keypoints)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import PANNeck, YOLOv11Backbone
from ..utils.geometry import (quaternion_to_matrix, rotation_6d_to_matrix,
                       sample_depth, unproject_translation)
from .head import DepthHead, MultiTaskHead, Pose6DoFHead


class YOLO6DoF(nn.Module):
    """
    단일 파이프라인 6DoF Object Pose Estimation 모델.

    forward(x):
        train 모드 → {"det": head raw dict, "depth": dense depth map}
        eval  모드 → {"det": [B, 4+nc+rot+3, A], "depth": [B,1,H/2,W/2]}

    추론 후 `decode_pose(...)`로 Intrinsics를 적용해 최종 R|t를 복원.
    """

    def __init__(self, nc: int = 80, scale: str = "n", reg_max: int = 16,
                 strides: tuple[int, ...] = (8, 16, 32), rot_repr: str = "6d",
                 light_head: bool = True):
        super().__init__()
        self.backbone = YOLOv11Backbone(scale=scale)
        self.neck = PANNeck(self.backbone.out_channels, scale=scale)
        self.head = Pose6DoFHead(
            nc=nc, ch=self.neck.out_channels, reg_max=reg_max,
            strides=strides, rot_repr=rot_repr, light=light_head,
        )
        self.depth_head = DepthHead(self.neck.out_channels[0])  # N3 입력
        self.nc = nc
        self.rot_repr = rot_repr

    def fuse(self):
        """모든 Conv+BN을 융합 (추론 전용, 출력 동일·속도↑). self 반환."""
        from .blocks import Conv
        for mod in self.modules():
            if isinstance(mod, Conv):
                mod.fuse()
        return self

    def forward(self, x: torch.Tensor):
        # x: [B, 3, H, W]
        feats = self.backbone(x)        # (P3,P4,P5)
        n3, n4, n5 = self.neck(feats)   # 융합 특징
        det = self.head((n3, n4, n5))   # train: dict / eval: tensor
        depth = self.depth_head(n3)     # [B,1,H/2,W/2]
        return {"det": det, "depth": depth}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode_pose(self, out: dict, K: torch.Tensor, img_size,
                    boxes_xywh: torch.Tensor, rot_raw: torch.Tensor,
                    size: torch.Tensor, depth_map: torch.Tensor = None) -> dict:
        """
        선택된(예: NMS 통과) 객체들에 대해 최종 6DoF Pose를 복원.

        out        : forward 결과(예측 depth 포함)
        K          : [B,3,3] camera intrinsics
        img_size   : (H, W) 입력 해상도
        boxes_xywh : [B,N,4] (cx,cy,w,h 픽셀) — 선택된 박스
        rot_raw    : [B,N,rot_dim] — 선택된 박스의 회전 raw
        size       : [B,N,3] — 선택된 박스의 3D 크기
        depth_map  : [B,1,Hd,Wd] 센서 depth(m). 주면 translation을 이걸로(우선),
                     없으면 모델 예측 depth(out["depth"])로 폴백.
        반환: {"R":[B,N,3,3], "t":[B,N,3], "size":[B,N,3]}
        """
        centers = boxes_xywh[..., :2]                         # [B,N,2] bbox 중심
        dm = depth_map if depth_map is not None else out["depth"]   # 센서 우선
        Z = sample_depth(dm, centers, img_size)               # [B,N]
        t = unproject_translation(centers, Z, K)              # [B,N,3]
        if self.rot_repr == "6d":
            R = rotation_6d_to_matrix(rot_raw)
        else:
            R = quaternion_to_matrix(rot_raw)
        return {"R": R, "t": t, "size": size}


class YOLOPose(nn.Module):
    """
    2D 멀티태스크 YOLO 모델 (Detection + Segmentation + Pose).

    forward(x):
        train → head raw dict (box/cls/mask_coef/proto/kpt)
        eval  → 디코딩 dict (boxes/scores/mask_coef/proto/kpt)
    """

    def __init__(self, nc: int = 80, scale: str = "n", reg_max: int = 16,
                 strides: tuple[int, ...] = (8, 16, 32),
                 tasks=("detect", "segment", "pose"), nm: int = 32,
                 kpt_shape=(4, 3)):
        super().__init__()
        self.backbone = YOLOv11Backbone(scale=scale)
        self.neck = PANNeck(self.backbone.out_channels, scale=scale)
        self.head = MultiTaskHead(
            nc=nc, ch=self.neck.out_channels, reg_max=reg_max, strides=strides,
            tasks=tasks, nm=nm, kpt_shape=kpt_shape,
        )
        self.nc = nc
        self.tasks = self.head.tasks
        self.kpt_shape = kpt_shape

    def fuse(self):
        """Conv+BN 융합 (추론 가속). self 반환."""
        from .blocks import Conv
        for mod in self.modules():
            if isinstance(mod, Conv):
                mod.fuse()
        return self

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)        # (P3,P4,P5)
        n3, n4, n5 = self.neck(feats)
        return self.head((n3, n4, n5))  # train: dict / eval: 디코딩 dict


# task 1개 → 그 task 전용 독립 모델 (det/seg/pose를 따로따로 학습·추론)
#   seg/pose 모델은 박스가 필요하므로 detection을 기본 포함한다(YOLO 관례).
TASK_PRESET = {
    "detect": ("detect",),
    "segment": ("detect", "segment"),
    "pose": ("detect", "pose"),
}


def build_model(task: str = "6dof", nc: int = 13, scale: str = "n",
                rot_repr: str = "6d", light_head: bool = True,
                nm: int = 32, kpt_shape=(4, 3)):
    """
    통합 빌더. task로 모델 계열을 선택한다.

      task == "6dof"             → YOLO6DoF (Pose6DoFHead + DepthHead)
      task in {detect,segment,pose} → YOLOPose (MultiTaskHead, 해당 task 프리셋)

    예) 6DoF LINEMOD 13객체 → build_model("6dof", nc=13)
        키포인트 pose          → build_model("pose", nc=3, kpt_shape=(4,3))
    """
    if task == "6dof":
        return YOLO6DoF(nc=nc, scale=scale, rot_repr=rot_repr, light_head=light_head)
    assert task in TASK_PRESET, f"task must be '6dof' or one of {list(TASK_PRESET)}"
    return YOLOPose(nc=nc, scale=scale, tasks=TASK_PRESET[task], nm=nm,
                    kpt_shape=kpt_shape)
