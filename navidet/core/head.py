"""
검출 Head 모음 — 한 패키지에서 두 계열의 head를 함께 제공한다.

  · Pose6DoFHead + DepthHead : 6DoF Object Pose Estimation (Direction B)
  · MultiTaskHead   + Proto  : 2D 멀티태스크 (Detection + Segmentation + Pose)

공용 유틸(DFL / make_anchors / dist2bbox)은 두 계열이 공유한다.
아래 설명은 Pose6DoFHead(6DoF) 기준이며, 멀티태스크 head는 MultiTaskHead 참조.

--------------------------------------------------------------------------
6DoF Pose Estimation을 위한 Custom Head (Direction B - Depth + Rotation Regression).

다이어그램 구조에 맞춘 멀티태스크 설계. 멀티스케일 각 위치(anchor point)마다
디커플드(decoupled) 브랜치로 아래를 동시에 예측한다.

    1) 2D BBox      : DFL(Distribution Focal Loss) 분포 (l,t,r,b) → xyxy
    2) Objectness   : 객체 존재 확률 1채널
    3) Class Score  : 클래스별 점수 nc채널
    4) 3D Rotation  : 6D 표현(기본) 또는 Quaternion 직접 회귀
    5) 3D Size      : 객체 metric 크기 (dx, dy, dz)

Translation(X,Y,Z)은 별도 DepthHead가 만든 dense depth map에서 bbox 중심의 Z를
샘플링하고 Camera Intrinsics로 unprojection 하여 복원한다(geometry.py).

설계 포인트
------------
- Anchor-free: 각 grid cell 중심 1점을 anchor point로 사용.
- Decoupled head: box / obj+cls / rot / size 브랜치를 분리해 태스크 간섭 완화.
- Rotation은 연속 표현(6D)을 기본으로 하여 학습 안정성 확보.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv


class DFL(nn.Module):
    """
    Distribution Focal Loss 적분 모듈.

    box의 각 변(l,t,r,b)을 reg_max개 bin에 대한 분포로 예측한 뒤,
    softmax 기대값(expectation)을 취해 실수 거리값 1개로 변환한다.
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False).requires_grad_(False)
        # 가중치를 0..reg_max-1 로 고정 → conv가 곧 기대값 계산
        self.conv.weight.data[:] = nn.Parameter(
            torch.arange(reg_max, dtype=torch.float).view(1, reg_max, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4*reg_max, A]  (A = anchor 총 개수)
        B, _, A = x.shape
        x = x.view(B, 4, self.reg_max, A).transpose(1, 2)        # [B, reg_max, 4, A]
        x = x.softmax(dim=1)                                     # bin 분포
        return self.conv(x).view(B, 4, A)                        # [B, 4, A] (l,t,r,b)


def make_anchors(feats: list[torch.Tensor], strides: tuple[int, ...],
                 grid_cell_offset: float = 0.5):
    """
    각 FPN 레벨의 feature map으로부터 anchor point(셀 중심)와 stride 텐서를 생성.

    반환:
        anchor_points: [A, 2]  (feature-map 좌표계의 셀 중심 x,y)
        stride_tensor: [A, 1]
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))   # [h*w, 2]
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = False):
    """DFL이 뱉은 (l,t,r,b) 거리 → bbox 로 변환. distance: [B,4,A], anchor: [A,2]"""
    lt, rb = distance.chunk(2, dim=1)              # 각 [B,2,A]
    ap = anchor_points.transpose(0, 1)             # [2, A]
    x1y1 = ap - lt                                 # [B,2,A]
    x2y2 = ap + rb
    if xywh:
        c = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c, wh), dim=1)           # [B,4,A] (cx,cy,w,h)
    return torch.cat((x1y1, x2y2), dim=1)          # [B,4,A] (x1,y1,x2,y2)


ROT_DIM = {"6d": 6, "quat": 4}


class Pose6DoFHead(nn.Module):
    """
    멀티스케일 6DoF Head (Direction B).

    Args:
        nc        : 클래스 수
        ch        : 각 입력 스케일(P3,P4,P5)의 채널 튜플
        reg_max   : DFL bin 수
        strides   : 각 스케일의 stride (P3=8, P4=16, P5=32)
        rot_repr  : "6d"(기본) 또는 "quat"
    """

    def __init__(self, nc: int = 80, ch: tuple[int, ...] = (256, 512, 1024),
                 reg_max: int = 16, strides: tuple[int, ...] = (8, 16, 32),
                 rot_repr: str = "6d", light: bool = True):
        super().__init__()
        assert rot_repr in ROT_DIM, f"rot_repr must be one of {list(ROT_DIM)}"
        self.nc = nc
        self.nl = len(ch)                  # 레벨 수 (3)
        self.reg_max = reg_max
        self.rot_repr = rot_repr
        self.rot_dim = ROT_DIM[rot_repr]   # 6 또는 4
        self.size_dim = 3                  # (dx, dy, dz)
        self.no_box = 4 * reg_max          # box 분기 출력 채널
        self.strides = strides
        self.light = light                 # True=경량 head, False=원본 head
        self.pose_dim = self.rot_dim + self.size_dim
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        # True면 eval 모드에서도 loss용 raw dict를 반환 (BN은 eval 통계 사용)
        self.return_raw = False

        c2 = max(16, ch[0] // 4, self.no_box)        # box 브랜치 (정밀도 위해 conv 2개)
        # --- 2D BBox 회귀 (DFL) — 두 head 공통, conv 2개 ---
        self.cv_box = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, self.no_box, 1)) for x in ch
        )
        if light:
            # 경량: cls(conv 1개) + rot/size 통합 pose 브랜치(conv 1개)
            c3 = c4 = max(ch[0] // 2, 48)
            self.cv_cls = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), nn.Conv2d(c3, self.nc + 1, 1)) for x in ch
            )
            self.cv_pose = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), nn.Conv2d(c4, self.pose_dim, 1)) for x in ch
            )
        else:
            # 원본: cls + rot + size 각 분리 브랜치, conv 2개씩
            c3 = max(ch[0], min(nc, 100))
            c4 = max(ch[0] // 2, 32)
            self.cv_cls = nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                              nn.Conv2d(c3, self.nc + 1, 1)) for x in ch
            )
            self.cv_rot = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3),
                              nn.Conv2d(c4, self.rot_dim, 1)) for x in ch
            )
            self.cv_size = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3),
                              nn.Conv2d(c4, self.size_dim, 1)) for x in ch
            )

        self._init_bias()

    def _init_bias(self):
        """objectness/cls bias 초기화 — 초기 학습 안정화 (focal-style prior)."""
        for cls_branch, s in zip(self.cv_cls, self.strides):
            b = cls_branch[-1].bias.data
            b[0] = math.log(5 / (640 / s) ** 2)              # objectness prior
            b[1:] = math.log(0.6 / (self.nc - 0.999999)) if self.nc > 1 else 0.0
        for box_branch in self.cv_box:
            box_branch[-1].bias.data[:] = 1.0                # box 분기 살짝 양수

    # ------------------------------------------------------------------ #
    def forward(self, feats):
        """
        Args:
            feats: (N3,N4,N5) = ([B,c3,H/8,W/8], [B,c4,H/16,W/16], [B,c5,H/32,W/32])

        Returns (train):
            raw per-level 텐서 dict (loss에서 타깃 할당과 함께 사용)
        Returns (eval):
            [B, 4+nc+rot_dim+3, A] 디코딩 예측 (box 픽셀, score, rot raw, size)
        """
        box_out, cls_out, rot_out, size_out = [], [], [], []
        for i, x in enumerate(feats):
            box_out.append(self.cv_box[i](x))    # [B, 4*reg_max, Hi, Wi]
            cls_out.append(self.cv_cls[i](x))    # [B, nc+1,      Hi, Wi]
            if self.light:                       # 통합 pose 브랜치 → rot/size split
                pose = self.cv_pose[i](x)        # [B, rot_dim+3, Hi, Wi]
                rot_out.append(pose[:, :self.rot_dim])
                size_out.append(pose[:, self.rot_dim:])
            else:                                # 원본: 분리 브랜치
                rot_out.append(self.cv_rot[i](x))
                size_out.append(self.cv_size[i](x))

        if self.training or self.return_raw:
            # 학습/검증 손실 계산 시 raw 출력을 그대로 넘겨 loss 쪽에서 디코딩/할당
            return {"box": box_out, "cls": cls_out, "rot": rot_out, "size": size_out,
                    "feats": feats, "strides": self.strides}

        # ---------------- Inference 디코딩 ---------------- #
        # 1) 레벨별 [B,C,Hi,Wi] → [B,C,A]로 flatten 후 concat (A = ΣHi*Wi)
        anchors, strides = make_anchors(feats, self.strides)   # [A,2], [A,1]
        box = torch.cat([b.flatten(2) for b in box_out], 2)    # [B, 4*reg_max, A]
        cls = torch.cat([c.flatten(2) for c in cls_out], 2)    # [B, nc+1,      A]
        rot = torch.cat([r.flatten(2) for r in rot_out], 2)    # [B, rot_dim,   A]
        size = torch.cat([s.flatten(2) for s in size_out], 2)  # [B, 3,         A]

        # 2) 2D box: DFL 기대값 → (cx,cy,w,h)(픽셀)
        dist = self.dfl(box)                                   # [B, 4, A]
        boxes = dist2bbox(dist, anchors, xywh=True)            # [B, 4, A] (셀단위)
        boxes = boxes * strides.transpose(0, 1)                # 픽셀 좌표로 스케일

        # 3) objectness/class
        obj = cls[:, :1].sigmoid()                             # [B, 1, A]
        scores = cls[:, 1:].sigmoid() * obj                    # [B, nc, A]

        # 4) rotation(raw 6D/quat), size(softplus로 양수화)
        size = F.softplus(size)                                # [B, 3, A] > 0

        # 최종: [B, 4+nc+rot_dim+3, A]
        #   (translation은 DepthHead/Intrinsics 필요 → model 레벨에서 결합)
        return torch.cat([boxes, scores, rot, size], dim=1)


class DepthHead(nn.Module):
    """
    Monocular Dense Depth Head.

    Neck의 최고해상도 특징(N3, /8)을 받아 2단계 업샘플로 /2 해상도의 dense depth
    map을 회귀한다. Edge PC를 고려해 가벼운 디코더로 구성.
    출력은 softplus로 양수(metric depth) 보장.
    """

    def __init__(self, in_ch: int, mid_ch: int = 64, max_depth: float = 10.0):
        super().__init__()
        self.max_depth = max_depth
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec = nn.Sequential(
            Conv(in_ch, mid_ch, 3),
            Conv(mid_ch, mid_ch, 3),
        )
        self.dec2 = Conv(mid_ch, mid_ch // 2, 3)
        self.out = nn.Conv2d(mid_ch // 2, 1, 1)

    def forward(self, n3: torch.Tensor) -> torch.Tensor:
        # n3: [B, in_ch, H/8, W/8]
        x = self.dec(n3)            # [B, mid, H/8, W/8]
        x = self.up(x)              # [B, mid, H/4, W/4]
        x = self.dec2(x)            # [B, mid/2, H/4, W/4]
        x = self.up(x)              # [B, mid/2, H/2, W/2]
        # 물리 범위 [0, max_depth]로 bound → 발산(무한대 depth) 원천 차단
        return self.max_depth * torch.sigmoid(self.out(x))   # [B, 1, H/2, W/2]


# ============================================================================ #
#  2D 멀티태스크 계열: Segmentation Proto + Detection/Segment/Pose 통합 Head
# ============================================================================ #
class Proto(nn.Module):
    """
    가장 고해상도 특징(P3)으로 nm장의 프로토타입 마스크를 생성(2배 업샘플).
    최종 인스턴스 마스크 = (per-anchor coeff[nm]) · (proto[nm,H,W]).
    """

    def __init__(self, c1: int, c_: int = 256, nm: int = 32):
        super().__init__()
        self.cv1 = Conv(c1, c_, 3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)   # ×2
        self.cv2 = Conv(c_, c_, 3)
        self.cv3 = Conv(c_, nm, 1)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))   # [B, nm, 2H, 2W]


class MultiTaskHead(nn.Module):
    """
    Detect(+Segment +Pose) 통합 head — 표준 anchor-free 2D 방식.

    멀티스케일 각 anchor point마다 디커플드 브랜치로 예측:
        Detect   : 2D BBox(DFL 분포 l,t,r,b) + Class score(nc)
        Segment  : per-anchor mask coefficient(nm) + 공유 Proto → coeff @ proto
        Pose     : per-anchor keypoint (nk × kpt_dim[x,y,(vis)])

    Args:
        nc       : 클래스 수
        ch       : 입력 스케일 채널 (P3,P4,P5)
        reg_max  : DFL bin 수
        strides  : 각 스케일 stride
        tasks    : 활성 태스크 ("detect" 필수, "segment"/"pose" 선택)
        nm       : 마스크 프로토타입 수 (segment)
        npr      : Proto hidden 채널
        kpt_shape: (nk, kpt_dim) — 키포인트 수, 차원(2=xy / 3=xy+vis)
    """

    def __init__(self, nc=80, ch=(256, 512, 1024), reg_max=16, strides=(8, 16, 32),
                 tasks=("detect", "segment", "pose"), nm=32, npr=256,
                 kpt_shape=(4, 3)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no_box = 4 * reg_max
        self.strides = strides
        self.tasks = tuple(tasks)
        self.segment = "segment" in self.tasks
        self.pose = "pose" in self.tasks
        self.nm = nm
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]      # 총 keypoint 출력 채널
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        self.return_raw = False

        c2 = max(16, ch[0] // 4, self.no_box)      # box 브랜치
        c3 = max(ch[0], min(nc, 100))              # cls 브랜치
        # --- Detect: box(DFL) + cls ---
        self.cv_box = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.no_box, 1))
            for x in ch)
        self.cv_cls = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1))
            for x in ch)
        # --- Segment: per-anchor mask coeff + Proto ---
        if self.segment:
            c4 = max(ch[0] // 4, nm)
            self.proto = Proto(ch[0], npr, nm)
            self.cv_mask = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, nm, 1))
                for x in ch)
        # --- Pose: per-anchor keypoints ---
        if self.pose:
            c5 = max(ch[0] // 4, self.nk)
            self.cv_kpt = nn.ModuleList(
                nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nk, 1))
                for x in ch)

        self._init_bias()

    def _init_bias(self):
        for cls_branch, s in zip(self.cv_cls, self.strides):
            cls_branch[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)
        for box_branch in self.cv_box:
            box_branch[-1].bias.data[:] = 1.0

    # ------------------------------------------------------------------ #
    def forward(self, feats):
        """
        feats: (P3,P4,P5).
        train: raw 텐서 dict / eval: 디코딩된 예측 dict.
        """
        box_out = [self.cv_box[i](x) for i, x in enumerate(feats)]
        cls_out = [self.cv_cls[i](x) for i, x in enumerate(feats)]
        out = {"box": box_out, "cls": cls_out, "feats": feats, "strides": self.strides}

        if self.segment:
            out["mask_coef"] = [self.cv_mask[i](x) for i, x in enumerate(feats)]
            out["proto"] = self.proto(feats[0])           # [B, nm, H/4, W/4]
        if self.pose:
            out["kpt"] = [self.cv_kpt[i](x) for i, x in enumerate(feats)]

        if self.training or self.return_raw:
            return out
        return self._decode(out)

    # ------------------------------------------------------------------ #
    def _decode(self, out):
        feats = out["feats"]
        anchors, strides = make_anchors(feats, self.strides)       # [A,2],[A,1]
        box = torch.cat([b.flatten(2) for b in out["box"]], 2)     # [B,4*rm,A]
        cls = torch.cat([c.flatten(2) for c in out["cls"]], 2)     # [B,nc,A]

        dist = self.dfl(box)
        boxes = dist2bbox(dist, anchors, xywh=True) * strides.transpose(0, 1)  # px
        scores = cls.sigmoid()
        dec = {"boxes": boxes, "scores": scores}                   # [B,4,A],[B,nc,A]

        if self.segment:
            dec["mask_coef"] = torch.cat([m.flatten(2) for m in out["mask_coef"]], 2)  # [B,nm,A]
            dec["proto"] = out["proto"]                            # [B,nm,Hm,Wm]
        if self.pose:
            kpt = torch.cat([k.flatten(2) for k in out["kpt"]], 2)  # [B, nk, A]
            dec["kpt"] = self.decode_keypoints(kpt, anchors, strides)
        return dec

    def decode_keypoints(self, kpt, anchors, strides):
        """raw keypoint offset → 이미지 좌표. kpt:[B, nk_total, A]."""
        B, _, A = kpt.shape
        K, D = self.kpt_shape
        kpt = kpt.view(B, K, D, A)
        ax, ay = anchors[:, 0].view(1, 1, A), anchors[:, 1].view(1, 1, A)
        st = strides.view(1, 1, A)
        kx = (kpt[:, :, 0] * 2.0 + ax) * st
        ky = (kpt[:, :, 1] * 2.0 + ay) * st
        if D == 3:
            vis = kpt[:, :, 2].sigmoid()
            return torch.stack((kx, ky, vis), 2).view(B, K * D, A)
        return torch.stack((kx, ky), 2).view(B, K * D, A)

    def fuse(self):
        for m in self.modules():
            if isinstance(m, Conv):
                m.fuse()
        return self
