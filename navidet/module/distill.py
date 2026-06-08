"""
DINOv3(Teacher) → YOLO6DoF(Student) Knowledge Distillation 구성요소.

4가지 기법:
  1. Task-Specialized Teacher : 미세조정된 DINOv3를 frozen(eval + no_grad)으로 고정
  2. FeatureProjector         : Student neck 특징(저차원) → DINOv3 공간(고차원) 1x1 매핑
  3. Register Token Separation: DINOv3 토큰에서 [CLS]+register 제거, 순수 patch만 정렬
  4. Decoupled Loss Schedule  : Total = α·Distill + β·Task, α/β를 epoch에 따라 동적 조절
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv3는 ImageNet 통계로 정규화된 입력을 기대
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================ #
#  2. FeatureProjector — Student 특징을 Teacher 임베딩 공간으로 매핑
# ============================================================================ #
class FeatureProjector(nn.Module):
    """
    1x1 Conv 기반 어댑터. Student의 낮은 차원 특징맵 [B, C_s, h, w] 을
    DINOv3 임베딩 차원 D 로 매핑 → [B, D, h, w].

    1x1 conv(차원 변환) + BN + (옵션)1x1 conv 로 표현력을 약간 부여한다.
    """

    def __init__(self, in_ch: int, teacher_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or teacher_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, teacher_dim, 1, bias=True),   # 최종 1x1 → Teacher 차원
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)                                 # [B, D, h, w]


# ============================================================================ #
#  3. Register Token Separation
# ============================================================================ #
def separate_register_tokens(tokens: torch.Tensor, num_register: int,
                             num_cls: int = 1):
    """
    DINOv3 토큰 시퀀스 [B, 1(cls)+R(register)+N(patch), D] 를 분리.

    토큰 배치 규약(DINOv2/v3): [CLS, reg_1..reg_R, patch_1..patch_N]
      - cls    : 전역 표현 토큰
      - reg_*  : 배경/노이즈를 흡수하는 register 토큰 (증류에서 버림)
      - patch_*: 순수 공간 특징 (이것만 정렬에 사용)

    반환: (patch_tokens[B,N,D], register_tokens[B,R,D], cls_token[B,num_cls,D])
    """
    cls_token = tokens[:, :num_cls, :]
    register_tokens = tokens[:, num_cls:num_cls + num_register, :]
    patch_tokens = tokens[:, num_cls + num_register:, :]    # ★ register 이후가 순수 patch
    return patch_tokens, register_tokens, cls_token


# ============================================================================ #
#  1. Task-Specialized Teacher (frozen DINOv3 래퍼)
# ============================================================================ #
class DINOv3Teacher(nn.Module):
    """
    미세조정된 DINOv3를 감싸 '순수 patch 토큰의 공간 특징맵'을 추출한다.
    학습 내내 가중치 고정: eval() + requires_grad_(False) + 호출은 no_grad.

    backbone 은 다음 중 하나를 만족해야 한다(둘 다 지원):
      (a) forward_features(x) 가 dict 반환 → 'x_norm_patchtokens' 키 사용
      (b) forward(x) 가 raw 토큰 [B, 1+R+N, D] 반환 → 여기서 직접 슬라이싱
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, patch_size: int = 16,
                 num_register_tokens: int = 4, img_size: int = 640,
                 normalize: bool = True):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_register = num_register_tokens
        self.img_size = img_size
        self.normalize = normalize
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # ★ Teacher 고정: 평가 모드 + 그래디언트 차단
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # Student가 train()으로 바뀌어도 Teacher는 항상 eval 유지
        super().train(mode)
        self.backbone.eval()
        return self

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """입력을 teacher img_size(패치 배수)로 리사이즈 + ImageNet 정규화."""
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        if self.normalize:
            x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B,3,H,W] → 순수 patch 공간 특징맵 [B, D, Hp, Wp].
        (register/cls 토큰은 separate_register_tokens 로 제거)
        """
        x = self._prep(x)
        B = x.shape[0]
        Hp = Wp = self.img_size // self.patch_size

        out = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") \
            else self.backbone(x)

        if isinstance(out, dict):                          # (a) DINOv3 dict 경로
            patch = out["x_norm_patchtokens"]              # 이미 register-free [B,N,D]
        else:                                              # (b) raw 토큰 직접 슬라이싱
            patch, _, _ = separate_register_tokens(out, self.num_register, num_cls=1)

        # [B, N, D] → [B, D, Hp, Wp] (공간 격자로 복원)
        return patch.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)


# ============================================================================ #
#  Distillation Loss (순수 patch 공간 특징끼리만 비교)
# ============================================================================ #
def feature_distillation_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                              kind: str = "cosine", align: str = "teacher") -> torch.Tensor:
    """
    student_feat: [B, D, h, w]  (FeatureProjector 출력)
    teacher_feat: [B, D, Hp, Wp] (순수 patch 공간맵)
    grid 해상도가 다르면 align 기준으로 한쪽을 interpolate 해 맞춘다.
    """
    if align == "teacher":                                 # teacher 격자에 맞춰 student 축소
        student_feat = F.interpolate(student_feat, size=teacher_feat.shape[-2:],
                                     mode="bilinear", align_corners=False)
    else:                                                  # student 격자에 맞춰 teacher 확대
        teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[-2:],
                                     mode="bilinear", align_corners=False)

    if kind == "mse":
        return F.mse_loss(student_feat, teacher_feat)
    # cosine: 채널축 정규화 후 1 - cos (공간 위치별 평균)
    s = F.normalize(student_feat, dim=1)
    t = F.normalize(teacher_feat, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


# ============================================================================ #
#  4. Decoupled Loss Scheduler (α 감소 / β 증가)
# ============================================================================ #
def loss_weights(epoch: int, total_epochs: int, alpha0: float = 1.0,
                 beta0: float = 0.3, beta1: float = 1.0, mode: str = "cosine"):
    """
    Total = α·Distill + β·Task.
    초기: α↑(교사 모방), 후반: β↑(실제 태스크). 반환 (alpha, beta).
    """
    p = epoch / max(total_epochs - 1, 1)                   # 진행도 0→1
    if mode == "cosine":
        alpha = alpha0 * 0.5 * (1 + math.cos(math.pi * p)) # α0 → 0 (부드럽게)
    else:                                                  # linear
        alpha = alpha0 * (1 - p)
    beta = beta0 + (beta1 - beta0) * p                     # β0 → β1
    return alpha, beta


# ============================================================================ #
#  스모크 테스트용 Mock Teacher (DINOv3 미설치 환경에서 파이프라인 검증)
# ============================================================================ #
class MockDINOv3(nn.Module):
    """raw 토큰 [B, 1+R+N, D]를 반환하는 더미 ViT (slicing 로직 검증용)."""

    def __init__(self, embed_dim=384, patch_size=16, num_register=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_register = num_register
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.tokens = nn.Parameter(torch.randn(1, 1 + num_register, embed_dim) * 0.02)

    def forward(self, x):
        p = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        special = self.tokens.expand(x.shape[0], -1, -1)     # [B, 1+R, D]
        return torch.cat([special, p], dim=1)                # [B, 1+R+N, D]
