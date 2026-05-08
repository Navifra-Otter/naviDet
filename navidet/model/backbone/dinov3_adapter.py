"""DINOv3 backbone in ViTAdapter-compatible form.

Lets ECPose / ECDet / ECSeg use a frozen-or-finetuned DINOv3 backbone in
place of `ViTAdapter`. The output contract (a list of 3 feature maps at
strides [8, 16, 32] with uniform channel count) matches `ViTAdapter` exactly,
so HybridEncoder + DETRTransformer / ECTransformer plug in unchanged.

Two flavors:
  * ConvNext path uses `get_feature_spaces()` (3 stage feats at [/8, /16, /32]).
  * ViT path replicates ViTAdapter's interpolation trick to lift the single
    /16 patch-token map into a [/8, /16, /32] pyramid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from navidet.core.registry import register
from navidet.model.utils import (DINO_CONVNEXT_MODELS, DINO_VIT_MODELS,
                                 convnext_ckps, convnext_sizes, vit_sizes)


@register()
class DinoV3Adapter(nn.Module):
    """Wraps a DINOv3 backbone so it slots in wherever ViTAdapter does.

    Args:
        backbone:        e.g. `dinov3_convnext_small`, `dinov3_vit_base`.
        weights_path:    optional override for the pretrained checkpoint.
        proj_dim:        channel dim of the output feature pyramid (3 levels).
        finetuning:      when True (default), keeps backbone trainable;
                         False freezes it (typical for KD / ablation).
        kind:            'convnext' | 'vit' (auto-inferred from backbone name).
    """

    def __init__(
        self,
        backbone: str = "dinov3_convnext_small",
        weights_path: str | None = None,
        proj_dim: int = 256,
        finetuning: bool = True,
        kind: str | None = None,
        num_levels: int = 3,
    ):
        super().__init__()
        if num_levels != 3:
            raise NotImplementedError("DinoV3Adapter currently emits exactly 3 levels.")

        info = backbone.split("_")
        if len(info) < 3:
            raise ValueError(
                f"backbone={backbone!r} must look like 'dinov3_convnext_small' "
                "or 'dinov3_vit_base'."
            )
        family = info[1]
        size = info[2]
        kind = kind or family
        self.kind = kind
        self.proj_dim = proj_dim

        if kind == "convnext":
            from navidet.model.backbone.dinov3convnext import Dinov3ConvNext
            backbone_id = DINO_CONVNEXT_MODELS[size]
            ckpt = weights_path or convnext_ckps[size]
            self.backbone = Dinov3ConvNext(
                backbone=backbone_id, source="local", weights=ckpt,
            )
            # 3 stage feats; channels = first 3 of `dims`.
            in_channels = convnext_sizes[size]["dims"][:-1]
        elif kind == "vit":
            from navidet.model.backbone.dinov3vit import Dinov3ViT
            backbone_id = DINO_VIT_MODELS[size]
            # ViT path is hub-only; passing weights=None to torch.hub.load on
            # 'local' source would crash deep in dinov3 since it tries to use
            # None as a path. Require a checkpoint file.
            if weights_path is None:
                raise ValueError(
                    f"DinoV3Adapter (kind=vit, size={size!r}) requires "
                    "`weights_path` — set the dinov3 ViT checkpoint path in "
                    "the YAML."
                )
            self.backbone = Dinov3ViT(
                backbone=backbone_id, source="local", weights=weights_path,
            )
            in_channels = [vit_sizes[size]["embed_dim"]] * num_levels
        else:
            raise ValueError(f"Unknown DINOv3 kind {kind!r} (expected 'convnext' or 'vit')")

        if not finetuning:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        # Per-level 1x1 projection to a uniform channel count.
        self.projector = nn.ModuleList([
            nn.Conv2d(c, proj_dim, kernel_size=1, bias=False) for c in in_channels
        ])

    def _convnext_feats(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone.get_feature_spaces(x)
        # Expect feats in NCHW with strides [/8, /16, /32].
        if isinstance(feats, (list, tuple)) and len(feats) >= 3:
            return list(feats[-3:])
        raise RuntimeError(
            f"DinoV3 ConvNext backbone returned {type(feats).__name__} of size "
            f"{getattr(feats, '__len__', lambda: '?')()}, expected list of >=3 feats."
        )

    def _vit_feats(self, x: torch.Tensor) -> list[torch.Tensor]:
        # ViT outputs a single /16 patch map; lift to [/8, /16, /32] with
        # bilinear up/down-sampling, mirroring ViTAdapter.
        feats = self.backbone.get_feature_spaces(x)
        # Dinov3ViT.get_feature_spaces returns NCHW (already reshaped+normalized).
        if feats.dim() == 4 and feats.shape[1] == feats.shape[1]:  # already NCHW
            base = feats
        else:
            base = feats.movedim(-1, 1)  # NHWC → NCHW
        H, W = base.shape[-2:]
        return [
            F.interpolate(base, size=(H * 2, W * 2), mode="bilinear", align_corners=False),
            base,
            F.interpolate(base, size=(H // 2, W // 2), mode="bilinear", align_corners=False),
        ]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self._convnext_feats(x) if self.kind == "convnext" else self._vit_feats(x)
        return [proj(f) for proj, f in zip(self.projector, feats)]
