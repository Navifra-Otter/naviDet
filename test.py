"""
TransPose-style 2D Human Pose Estimation with DINOv3 Backbone
==============================================================

Reference:
    TransPose: Keypoint Localization via Transformer (ICCV 2021)
    - Paper : https://arxiv.org/abs/2012.14214
    - GitHub: https://github.com/yangsenius/TransPose

Architecture (논문 Fig.3 그대로):
    ┌─────────────────────────────────────────────────────────────┐
    │  Image (B, 3, H, W)                                        │
    │     ↓                                                      │
    │  [DINOv3 Backbone]   ← CNN backbone 역할                   │
    │  patch tokens (B, N+1, D)  →  feature map (B, C, h, w)    │
    │     ↓                                                      │
    │  Flatten + 2D Sine Positional Encoding                     │
    │  sequence (B, h*w, C)                                      │
    │     ↓                                                      │
    │  [Transformer Encoder] × L layers                         │
    │  MHSA → Add&Norm → FFN → Add&Norm                         │
    │     ↓                                                      │
    │  Reshape back → (B, C, h, w)                              │
    │     ↓                                                      │
    │  [Prediction Head]   ← 논문: 1×1 conv (optional deconv)   │
    │  heatmaps (B, K, H_out, W_out)                            │
    └─────────────────────────────────────────────────────────────┘

DINOv3 (≈DINOv2) specs:
    dinov2_vits14 : token_dim=384,  patch=14
    dinov2_vitb14 : token_dim=768,  patch=14
    dinov2_vitl14 : token_dim=1024, patch=14
    dinov2_vitg14 : token_dim=1536, patch=14
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


# =============================================================================
# 1. DINOv3 Backbone Wrapper
# =============================================================================

class DINOv3Backbone(nn.Module):
    """
    DINOv3(=DINOv2) backbone wrapper.

    TransPose 논문에서 CNN backbone은 low-level feature를 뽑은 뒤
    Transformer Encoder에 넘겨준다. DINOv3는 patch token 자체가
    이미 rich feature이므로, 그대로 backbone 역할을 수행한다.

    Args:
        model_name : torch.hub에서 로드할 DINOv2/v3 모델 이름
        freeze     : True면 backbone 전체 파라미터 freeze
        freeze_blocks: 앞쪽 N개 block만 freeze (None이면 전체 freeze/unfreeze)
        out_indices : 여러 레이어 출력 사용 시 (현재는 마지막만 사용)
    """

    # token_dim 매핑
    MODEL_DIMS: Dict[str, int] = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
        "dinov2_vits14_reg": 384,
        "dinov2_vitb14_reg": 768,
        "dinov2_vitl14_reg": 1024,
        "dinov2_vitg14_reg": 1536,
    }
    PATCH_SIZE = 14

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze: bool = False,
        freeze_blocks: Optional[int] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        assert model_name in self.MODEL_DIMS, (
            f"Unknown model: {model_name}. Choose from {list(self.MODEL_DIMS)}"
        )
        self.model_name = model_name
        self.token_dim = self.MODEL_DIMS[model_name]
        self.patch_size = self.PATCH_SIZE

        # Load from torch.hub (facebookresearch/dinov2)
        if pretrained:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=True
            )
        else:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=False
            )

        # Freeze 설정
        if freeze:
            if freeze_blocks is None:
                # 전체 freeze
                for p in self.dino.parameters():
                    p.requires_grad_(False)
            else:
                # 앞쪽 N block만 freeze
                for p in self.dino.patch_embed.parameters():
                    p.requires_grad_(False)
                for i, block in enumerate(self.dino.blocks):
                    if i < freeze_blocks:
                        for p in block.parameters():
                            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x : (B, 3, H, W)  — H, W must be divisible by patch_size(14)

        Returns:
            patch_tokens : (B, h_p*w_p, token_dim)  CLS 제거된 patch 토큰
            h_p          : 세로 patch 수  = H // 14
            w_p          : 가로 patch 수  = W // 14
        """
        B, _, H, W = x.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        # DINOv2: forward_features returns dict with 'x_norm_patchtokens'
        out = self.dino.forward_features(x)

        # patch tokens only (CLS already stripped in DINOv2 API)
        patch_tokens = out["x_norm_patchtokens"]  # (B, N, D)

        assert patch_tokens.shape[1] == h_p * w_p, (
            f"Patch count mismatch: {patch_tokens.shape[1]} vs {h_p}×{w_p}="
            f"{h_p*w_p}. Check input resolution divisibility by {self.patch_size}."
        )
        return patch_tokens, h_p, w_p


# =============================================================================
# 2. Feature Projection  (Backbone → TransPose Encoder 사이 브릿지)
# =============================================================================

class FeatureProjection(nn.Module):
    """
    DINOv3 token_dim → TransPose d_model 선형 변환.

    논문에서 CNN backbone 마지막 feature map을 Transformer Encoder에
    바로 넣듯이, 여기서는 token dim을 먼저 맞춰준다.
    """

    def __init__(self, token_dim: int, d_model: int):
        super().__init__()
        if token_dim == d_model:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(token_dim, d_model),
                nn.LayerNorm(d_model),
            )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, token_dim) → (B, N, d_model)"""
        return self.proj(tokens)


# =============================================================================
# 3. 2D Sine Positional Encoding  (논문 Section 3.2)
# =============================================================================

class PositionalEncoding2D(nn.Module):
    """
    논문에서 사용한 2D sine positional encoding.

    "We use the 2D sine positional encoding ...
     which can help generalize better on unseen input resolution."
    (TransPose paper, Section 4.4)

    row / col 각각 d_model/2 차원의 sine 인코딩을 만들어 concat.
    """

    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            pe : (h*w, d_model)
        """
        d_half = self.d_model // 2
        dim_t = torch.arange(d_half, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / d_half)

        pos_y = torch.arange(h, dtype=torch.float32, device=device)  # (h,)
        pos_x = torch.arange(w, dtype=torch.float32, device=device)  # (w,)

        # shape: (h, d_half)
        enc_y = pos_y.unsqueeze(1) / dim_t.unsqueeze(0)
        enc_y[:, 0::2] = torch.sin(enc_y[:, 0::2])
        enc_y[:, 1::2] = torch.cos(enc_y[:, 1::2])

        # shape: (w, d_half)
        enc_x = pos_x.unsqueeze(1) / dim_t.unsqueeze(0)
        enc_x[:, 0::2] = torch.sin(enc_x[:, 0::2])
        enc_x[:, 1::2] = torch.cos(enc_x[:, 1::2])

        # Broadcast to (h, w, d_model)
        enc_y = enc_y.unsqueeze(1).expand(h, w, d_half)  # (h, w, d/2)
        enc_x = enc_x.unsqueeze(0).expand(h, w, d_half)  # (h, w, d/2)
        pe = torch.cat([enc_y, enc_x], dim=-1)            # (h, w, d)
        return pe.view(h * w, self.d_model)                # (N, d)


# =============================================================================
# 4. TransPose Encoder  (논문 Fig.3 중간 블록)
# =============================================================================

class TransPoseEncoderLayer(nn.Module):
    """
    논문의 Transformer Encoder Layer.

    MHSA → Residual+Norm → FFN → Residual+Norm  (Post-LN, 논문 원본과 동일)
    attention map 반환으로 explainability 지원.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",   # 논문: relu (not gelu)
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        act_fn = nn.ReLU() if activation == "relu" else nn.GELU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src : (B, N, d_model)
        Returns:
            out        : (B, N, d_model)
            attn_weight: (B, N, N)   ← 논문 Fig.5, 6의 attention map
        """
        # Multi-Head Self-Attention
        attn_out, attn_weight = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_out))

        # Feed-Forward Network
        src = self.norm2(src + self.ffn(src))
        return src, attn_weight


class TransPoseEncoder(nn.Module):
    """L개 TransPoseEncoderLayer 스택."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransPoseEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            out       : (B, N, d_model)
            attn_maps : List[(B, N, N)] — 레이어별 attention weight
        """
        x = src
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        return x, attn_maps


# =============================================================================
# 5. Prediction Head  (논문: simple 1×1 conv → heatmap)
# =============================================================================

class PredictionHead(nn.Module):
    """
    논문 Fig.3 맨 오른쪽 'Localized heatmap' 출력 head.

    TransPose 원본: 1×1 conv로 바로 K채널 heatmap 출력.
    DINOv3 patch_size=14이므로 해상도가 낮아 deconv upsample 옵션 추가.

    논문 TransPose-H의 경우 HRNet backbone이 stride 4(1/4 해상도)를
    유지하기 때문에 deconv 없이 바로 예측.
    DINOv3는 patch_size=14이므로 stride=14 → 추가 upsample 권장.
    """

    def __init__(
        self,
        d_model: int,
        num_keypoints: int,
        upsample_factor: int = 2,   # 1 = no upsample (논문 원본), 2 or 4 추천
    ):
        super().__init__()

        layers: List[nn.Module] = []

        # Deconv upsample (upsample_factor > 1일 때)
        in_ch = d_model
        factor = upsample_factor
        while factor > 1:
            layers += [
                nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=4,
                                   stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_ch // 2),
                nn.ReLU(inplace=True),
            ]
            in_ch = in_ch // 2
            factor //= 2

        self.upsample = nn.Sequential(*layers) if layers else nn.Identity()

        # 논문과 동일: 1×1 conv for heatmap prediction
        self.heatmap_conv = nn.Conv2d(in_ch, num_keypoints, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.heatmap_conv.weight, std=0.001)
        nn.init.constant_(self.heatmap_conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, d_model, h, w)
        Returns:
            heatmaps : (B, K, h*upsample_factor, w*upsample_factor)
        """
        x = self.upsample(x)
        return self.heatmap_conv(x)


# =============================================================================
# 6. Complete Model: DINOv3 + TransPose
# =============================================================================

class DINOv3TransPose(nn.Module):
    """
    DINOv3를 backbone으로 하는 TransPose 2D Pose Estimation 모델.

    논문 파이프라인 (Fig.3) 완전 구현:
        Image
          ↓  [DINOv3 Backbone]          CNN backbone 역할
          ↓  [FeatureProjection]        token_dim → d_model
          ↓  [2D Sine Pos Encoding]     공간 위치 정보 주입
          ↓  [TransPose Encoder × L]   long-range 관계 캡처
          ↓  Reshape (B,N,C)→(B,C,h,w)
          ↓  [PredictionHead]           1×1 conv → K heatmaps
        Heatmaps (B, K, H_out, W_out)

    Args:
        backbone_name  : DINOv3/DINOv2 모델 이름
        num_keypoints  : 관절 수 (COCO=17, MPII=16, Halpe=26 등)
        d_model        : Transformer 내부 차원 (논문: 256 for TP-R, 96 for TP-H)
        nhead          : Multi-head attention head 수
        num_enc_layers : Encoder 레이어 수 (논문: 3~6)
        dim_feedforward: FFN 은닉 차원 (논문: 1024 for TP-R, 192 for TP-H)
        dropout        : Dropout 비율
        upsample_factor: Prediction head에서 deconv 업샘플 배수
        freeze_backbone: Backbone freeze 여부
        freeze_blocks  : 앞쪽 N block만 freeze (None = 전체)
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        num_keypoints: int = 17,
        d_model: int = 256,
        nhead: int = 8,
        num_enc_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        upsample_factor: int = 2,
        freeze_backbone: bool = False,
        freeze_blocks: Optional[int] = None,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        # ── 1. DINOv3 Backbone ───────────────────────────────────────────────
        self.backbone = DINOv3Backbone(
            model_name=backbone_name,
            freeze=freeze_backbone,
            freeze_blocks=freeze_blocks,
            pretrained=pretrained_backbone,
        )
        token_dim = self.backbone.token_dim

        # ── 2. Feature Projection ────────────────────────────────────────────
        self.proj = FeatureProjection(token_dim, d_model)

        # ── 3. 2D Sine Positional Encoding ───────────────────────────────────
        self.pos_enc = PositionalEncoding2D(d_model)

        # ── 4. TransPose Encoder ─────────────────────────────────────────────
        self.encoder = TransPoseEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_enc_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # ── 5. Prediction Head ───────────────────────────────────────────────
        self.head = PredictionHead(
            d_model=d_model,
            num_keypoints=num_keypoints,
            upsample_factor=upsample_factor,
        )

        # 메타 정보
        self.d_model = d_model
        self.num_keypoints = num_keypoints

    def forward(
        self,
        images: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        Args:
            images     : (B, 3, H, W)
                         H, W must be divisible by 14 (DINOv3 patch size)
                         권장 해상도: 256×192, 384×288 등
            return_attn: True이면 레이어별 attention map 반환

        Returns:
            heatmaps : (B, K, H_out, W_out)
            attn_maps: List[(B, N, N)] — return_attn=True일 때만
        """
        B = images.shape[0]

        # ── Step 1: DINOv3 Backbone ──────────────────────────────────────────
        # CNN backbone처럼 feature 추출
        patch_tokens, h_p, w_p = self.backbone(images)
        # patch_tokens: (B, h_p*w_p, token_dim)

        # ── Step 2: Projection (token_dim → d_model) ─────────────────────────
        x = self.proj(patch_tokens)   # (B, N, d_model)

        # ── Step 3: 2D Sine Positional Encoding 추가 ─────────────────────────
        pe = self.pos_enc(h_p, w_p, device=x.device)   # (N, d_model)
        x = x + pe.unsqueeze(0)                          # (B, N, d_model)

        # ── Step 4: TransPose Encoder ─────────────────────────────────────────
        # 논문 핵심: flattened feature sequence → Transformer → 관절 간 관계 파악
        x, attn_maps = self.encoder(x)   # (B, N, d_model)

        # ── Step 5: Sequence → Spatial Feature Map ────────────────────────────
        x = x.transpose(1, 2).view(B, self.d_model, h_p, w_p)  # (B, C, h, w)

        # ── Step 6: Prediction Head → Heatmaps ───────────────────────────────
        heatmaps = self.head(x)   # (B, K, H_out, W_out)

        if return_attn:
            return heatmaps, attn_maps
        return heatmaps

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        use_dark: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end inference: image → keypoint coordinates + confidence.

        Args:
            images   : (B, 3, H, W)
            use_dark : DARK post-processing 사용 여부

        Returns:
            coords : (B, K, 2)  pixel 좌표 (x, y)
            scores : (B, K)     keypoint 신뢰도
        """
        heatmaps = self.forward(images)
        coords, scores = decode_heatmap(heatmaps, use_dark=use_dark)

        # heatmap 해상도 → 원본 이미지 해상도로 스케일링
        H, W = images.shape[-2:]
        H_map, W_map = heatmaps.shape[-2:]
        coords[..., 0] *= W / W_map
        coords[..., 1] *= H / H_map
        return coords, scores


# =============================================================================
# 7. Loss Functions
# =============================================================================

class JointsMSELoss(nn.Module):
    """
    논문 및 SimpleBaseline과 동일한 weighted MSE loss.

    Args:
        use_target_weight: visibility/weight로 keypoint별 가중치 적용
    """

    def __init__(self, use_target_weight: bool = True):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred          : (B, K, H, W)
            target        : (B, K, H, W) — Gaussian heatmaps
            target_weight : (B, K, 1)    — visibility flags
        """
        K = pred.shape[1]
        total_loss = 0.0
        for k in range(K):
            p = pred[:, k]    # (B, H, W)
            t = target[:, k]
            if self.use_target_weight and target_weight is not None:
                w = target_weight[:, k]  # (B, 1)
                p = p * w.unsqueeze(-1)
                t = t * w.unsqueeze(-1)
            total_loss += self.criterion(p, t)
        return total_loss / K


# =============================================================================
# 8. Heatmap Generation & Post-processing
# =============================================================================

def generate_heatmaps(
    joints: torch.Tensor,
    joints_vis: torch.Tensor,
    heatmap_size: Tuple[int, int],
    sigma: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ground-truth Gaussian heatmap 생성.

    Args:
        joints     : (B, K, 2) — heatmap 좌표계 (x, y)
        joints_vis : (B, K, 1) — visibility (1=visible, 0=invisible)
        heatmap_size: (H, W)
        sigma      : Gaussian 표준편차

    Returns:
        target        : (B, K, H, W)
        target_weight : (B, K, 1)
    """
    B, K, _ = joints.shape
    H, W = heatmap_size
    target = torch.zeros(B, K, H, W, dtype=torch.float32, device=joints.device)
    target_weight = joints_vis.clone().float()

    size = int(6 * sigma + 1)
    ax = torch.arange(size, dtype=torch.float32, device=joints.device)
    gauss = torch.exp(-0.5 * ((ax - size // 2) / sigma) ** 2)
    kernel = gauss.outer(gauss)          # (size, size)

    for b in range(B):
        for k in range(K):
            if joints_vis[b, k, 0] < 1:
                continue
            cx = int(joints[b, k, 0].item())
            cy = int(joints[b, k, 1].item())

            # kernel을 heatmap에 붙여넣을 ROI 계산
            ul = (cx - size // 2, cy - size // 2)
            br = (ul[0] + size, ul[1] + size)

            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                target_weight[b, k, 0] = 0
                continue

            kx0, kx1 = max(0, -ul[0]),       min(size, W - ul[0])
            ky0, ky1 = max(0, -ul[1]),        min(size, H - ul[1])
            ix0, ix1 = max(0, ul[0]),         min(W, br[0])
            iy0, iy1 = max(0, ul[1]),         min(H, br[1])

            target[b, k, iy0:iy1, ix0:ix1] = kernel[ky0:ky1, kx0:kx1]

    return target, target_weight


def decode_heatmap(
    heatmaps: torch.Tensor,
    use_dark: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Heatmap → keypoint 좌표 디코딩.

    Args:
        heatmaps : (B, K, H, W)
        use_dark : DARK post-processing 사용 여부

    Returns:
        coords : (B, K, 2)
        scores : (B, K)
    """
    if use_dark:
        return _dark_decode(heatmaps)
    else:
        return _argmax_decode(heatmaps)


def _argmax_decode(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    scores, idx = flat.max(dim=-1)     # (B, K)
    x = (idx % W).float()
    y = (idx // W).float()
    coords = torch.stack([x, y], dim=-1)   # (B, K, 2)
    return coords, scores


def _dark_decode(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DARK: Distribution-Aware coordinate Representation (CVPR 2020).
    서브픽셀 정밀도를 위해 Gaussian blur 후 Taylor 2차 근사.
    """
    heatmaps = _gaussian_blur2d(heatmaps, kernel_size=11, sigma=2.0)
    B, K, H, W = heatmaps.shape

    flat = heatmaps.view(B, K, -1)
    scores, idx = flat.max(dim=-1)     # (B, K)

    px = (idx % W).clamp(1, W - 2).long()   # (B, K)
    py = (idx // W).clamp(1, H - 2).long()

    # 벡터화된 Taylor 디코딩
    coords = _taylor_decode_vectorized(heatmaps, px, py)   # (B, K, 2)
    return coords, scores


def _gaussian_blur2d(
    x: torch.Tensor, kernel_size: int = 11, sigma: float = 2.0
) -> torch.Tensor:
    B, C, H, W = x.shape
    pad = kernel_size // 2
    ax = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - pad
    kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_2d = kernel_1d.outer(kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(C, 1, -1, -1)
    x_pad = F.pad(x, [pad, pad, pad, pad], mode="reflect")
    return F.conv2d(x_pad, kernel_2d, groups=C)


def _taylor_decode_vectorized(
    heatmaps: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
) -> torch.Tensor:
    """
    Hessian 기반 2차 Taylor 근사로 서브픽셀 좌표 계산 (벡터화 버전).

    Args:
        heatmaps : (B, K, H, W)
        px, py   : (B, K) — argmax 위치 (1 ≤ p ≤ size-2)

    Returns:
        coords   : (B, K, 2)
    """
    B, K, H, W = heatmaps.shape

    def at(dy, dx):
        """heatmaps[b, k, py+dy, px+dx] 를 벡터화해서 가져오기."""
        y_idx = (py + dy).clamp(0, H - 1)
        x_idx = (px + dx).clamp(0, W - 1)
        # gather: (B, K)
        flat_idx = y_idx * W + x_idx
        return heatmaps.view(B, K, -1).gather(2, flat_idx.unsqueeze(-1)).squeeze(-1)

    v00 = at(0,  0)
    vp1 = at(0,  1);  vm1 = at(0, -1)
    vp2 = at(0,  2);  vm2 = at(0, -2)
    vq1 = at(1,  0);  vq2 = at(-1, 0)
    vr2 = at(2,  0);  vs2 = at(-2, 0)
    vpp = at(1,  1);  vpm = at(1, -1)
    vmp = at(-1, 1);  vmm = at(-1, -1)

    dx  = 0.5  * (vp1 - vm1)
    dy  = 0.5  * (vq1 - vq2)
    dxx = 0.25 * (vp2 - 2 * v00 + vm2)
    dyy = 0.25 * (vr2 - 2 * v00 + vs2)
    dxy = 0.25 * (vpp - vpm - vmp + vmm)

    # det(H) = dxx*dyy - dxy^2
    det = dxx * dyy - dxy * dxy
    det = det.abs().clamp(min=1e-8) * det.sign().where(
        det.sign() != 0, torch.ones_like(det)
    )

    # H^-1 · grad
    inv_dxx = dyy  / det
    inv_dyy = dxx  / det
    inv_dxy = -dxy / det

    delta_x = -(inv_dxx * dx + inv_dxy * dy)
    delta_y = -(inv_dxy * dx + inv_dyy * dy)
    delta_x = delta_x.clamp(-1.0, 1.0)
    delta_y = delta_y.clamp(-1.0, 1.0)

    coord_x = px.float() + delta_x   # (B, K)
    coord_y = py.float() + delta_y
    return torch.stack([coord_x, coord_y], dim=-1)   # (B, K, 2)


# =============================================================================
# 9. Factory Functions
# =============================================================================

# 논문 Table 1 설정 (backbone → d, h, heads, layers)
TRANSPOSE_PRESETS = {
    # 이름          : (d_model, dim_ff, nhead, num_layers)
    "TP-R-A3"      : (256, 1024, 8, 3),   # TransPose-R, 3 layers
    "TP-R-A4"      : (256, 1024, 8, 4),   # TransPose-R, 4 layers  (논문 기본)
    "TP-H-S"       : (64,   128, 1, 4),   # TransPose-H-S (HRNet-W32 기준)
    "TP-H-A4"      : (96,   192, 1, 4),   # TransPose-H, 4 layers
    "TP-H-A6"      : (96,   192, 1, 6),   # TransPose-H, 6 layers
}

DINOV3_CONFIGS = {
    "dinov2_vits14" : 384,
    "dinov2_vitb14" : 768,
    "dinov2_vitl14" : 1024,
    "dinov2_vitg14" : 1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}


def build_dinov3_transpose(
    backbone: str = "dinov2_vitb14",
    preset: str = "TP-R-A4",
    num_keypoints: int = 17,
    upsample_factor: int = 2,
    freeze_backbone: bool = False,
    freeze_blocks: Optional[int] = None,
    pretrained_backbone: bool = True,
) -> DINOv3TransPose:
    """
    DINOv3 + TransPose 모델 생성 convenience 함수.

    Example:
        # COCO 17 keypoints, ViT-B backbone
        model = build_dinov3_transpose("dinov2_vitb14", "TP-R-A4", num_keypoints=17)

        # 가벼운 버전
        model = build_dinov3_transpose("dinov2_vits14", "TP-H-S", num_keypoints=17)
    """
    assert backbone in DINOV3_CONFIGS, f"Unknown backbone: {backbone}"
    assert preset in TRANSPOSE_PRESETS, f"Unknown preset: {preset}"

    d_model, dim_ff, nhead, num_layers = TRANSPOSE_PRESETS[preset]

    return DINOv3TransPose(
        backbone_name=backbone,
        num_keypoints=num_keypoints,
        d_model=d_model,
        nhead=nhead,
        num_enc_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=0.1,
        upsample_factor=upsample_factor,
        freeze_backbone=freeze_backbone,
        freeze_blocks=freeze_blocks,
        pretrained_backbone=pretrained_backbone,
    )


# =============================================================================
# 10. Sanity Check (mock backbone으로 torch 없이도 구조 검증 가능)
# =============================================================================

class _MockDINOv3Backbone(nn.Module):
    """torch.hub 없이 구조 검증용 mock backbone."""

    def __init__(self, token_dim: int = 768, patch_size: int = 14):
        super().__init__()
        self.token_dim = token_dim
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        tokens = torch.randn(B, h_p * w_p, self.token_dim, device=x.device)
        return tokens, h_p, w_p


def build_mock_model(
    token_dim: int = 768,
    num_keypoints: int = 17,
    d_model: int = 256,
    nhead: int = 8,
    num_enc_layers: int = 4,
    dim_feedforward: int = 1024,
    upsample_factor: int = 2,
) -> DINOv3TransPose:
    """실제 DINOv3 weight 없이도 전체 구조를 테스트할 수 있는 mock 모델."""
    model = DINOv3TransPose.__new__(DINOv3TransPose)
    nn.Module.__init__(model)

    model.backbone = _MockDINOv3Backbone(token_dim)
    model.proj = FeatureProjection(token_dim, d_model)
    model.pos_enc = PositionalEncoding2D(d_model)
    model.encoder = TransPoseEncoder(d_model, nhead, num_enc_layers, dim_feedforward)
    model.head = PredictionHead(d_model, num_keypoints, upsample_factor)
    model.d_model = d_model
    model.num_keypoints = num_keypoints
    return model


if __name__ == "__main__":
    print("=" * 65)
    print(" DINOv3 + TransPose — Full Model Sanity Check (Mock Backbone)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # 입력 설정
    B        = 2
    IMG_H    = 256   # 256 / 14 = 18 patches
    IMG_W    = 196   # 196 / 14 = 14 patches  (14의 배수)
    NUM_KPT  = 17    # COCO

    fake_images = torch.randn(B, 3, IMG_H, IMG_W).to(device)

    # 모델 생성 (논문 TP-R-A4 설정)
    model = build_mock_model(
        token_dim=768,
        num_keypoints=NUM_KPT,
        d_model=256,
        nhead=8,
        num_enc_layers=4,
        dim_feedforward=1024,
        upsample_factor=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  TransPose head params (excl. backbone): {total_params:,}")

    # ── Forward pass ────────────────────────────────────────────────────────
    heatmaps, attn_maps = model(fake_images, return_attn=True)

    H_p = IMG_H // 14   # 18
    W_p = IMG_W // 14   # 14
    H_out = H_p * 2     # 36  (upsample_factor=2)
    W_out = W_p * 2     # 28

    print(f"\n  Input  images  : {list(fake_images.shape)}")
    print(f"  Patch grid     : {H_p} × {W_p} = {H_p*W_p} tokens")
    print(f"  Output heatmaps: {list(heatmaps.shape)}")
    print(f"  Expected shape : [{B}, {NUM_KPT}, {H_out}, {W_out}]")
    assert heatmaps.shape == (B, NUM_KPT, H_out, W_out), "Shape mismatch!"

    print(f"\n  Attention maps : {len(attn_maps)} layers")
    print(f"  Each attn shape: {list(attn_maps[0].shape)}  (B, N, N)")

    # ── Loss ────────────────────────────────────────────────────────────────
    joints   = torch.rand(B, NUM_KPT, 2) * torch.tensor([W_out - 1, H_out - 1])
    vis      = torch.ones(B, NUM_KPT, 1)
    target, tw = generate_heatmaps(joints.to(device), vis.to(device),
                                   (H_out, W_out), sigma=2.0)
    criterion = JointsMSELoss(use_target_weight=True)
    loss = criterion(heatmaps, target, tw)
    print(f"\n  MSE Loss       : {loss.item():.6f}")

    # ── Decode ──────────────────────────────────────────────────────────────
    coords, scores = decode_heatmap(heatmaps.detach(), use_dark=False)
    print(f"\n  Keypoint coords: {list(coords.shape)}  (B, K, 2) [x, y]")
    print(f"  Keypoint scores: {list(scores.shape)}  (B, K)")

    print("\n  ✓ All checks passed!")
    print("=" * 65)
    print("\n  [Real usage with DINOv3 pretrained weights]")
    print("  >>> from dinov3_transpose_pose import build_dinov3_transpose")
    print("  >>> model = build_dinov3_transpose(")
    print("  ...     backbone='dinov2_vitb14',")
    print("  ...     preset='TP-R-A4',")
    print("  ...     num_keypoints=17,")
    print("  ...     freeze_backbone=True,")
    print("  ... )")
    print("  >>> heatmaps = model(images)  # (B, 17, H_out, W_out)")