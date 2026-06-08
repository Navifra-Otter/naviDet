"""
멀티스케일 백본 + FPN 넥 (퍼미시브 라이선스 구성요소).

  · Backbone : timm(Apache-2.0) 백본을 features_only 모드로 사용해 P3/P4/P5
               (stride 8/16/32) 특징을 추출. 사전학습 가중치를 그대로 활용한다.
  · FPNNeck  : torchvision(BSD)의 FeaturePyramidNetwork 로 세 스케일을 융합,
               균일 채널(기본 256)의 (N3, N4, N5) 를 반환.

두 구성요소 모두 외부 퍼미시브 라이브러리에 기반하므로, 검출 헤드(별도 구현)와
함께 쓰면 AGPL 의존 없이 멀티스케일 검출 백본을 구성할 수 있다.
"""

from __future__ import annotations

from collections import OrderedDict

import timm
import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork

# scale(n/s/m/l/x) → timm 백본 이름. Edge PC 실시간 추론을 가정해 경량 위주로 매핑.
# (timm 의 어떤 모델이든 features_only 를 지원하면 교체 가능.)
SCALE_BACKBONE = {
    "n": "mobilenetv3_small_100",
    "s": "mobilenetv3_large_100",
    "m": "efficientnet_b0",
    "l": "convnext_nano",
    "x": "convnext_tiny",
}


def _make_features(name: str, in_ch: int, pretrained: bool):
    """timm features_only 모델 생성. 사전학습 다운로드 실패 시 랜덤 초기화로 폴백."""
    try:
        return timm.create_model(name, features_only=True, pretrained=pretrained,
                                 in_chans=in_ch)
    except Exception as e:                              # 오프라인/다운로드 차단 등
        if pretrained:
            print(f"[backbone] '{name}' 사전학습 로드 실패({type(e).__name__}) "
                  f"→ pretrained=False 로 폴백")
            return timm.create_model(name, features_only=True, pretrained=False,
                                     in_chans=in_ch)
        raise


class Backbone(nn.Module):
    """timm 멀티스케일 백본. forward(img) → (P3, P4, P5).

    stride 8/16/32 에 해당하는 특징 단계를 reduction 으로 자동 선택하므로,
    스테이지 구성이 다른 백본 계열(MobileNet/EfficientNet/ConvNeXt/ResNet ...)에도
    동일하게 동작한다.
    """

    def __init__(self, scale: str = "n", in_ch: int = 3, pretrained: bool = True,
                 name: str | None = None):
        super().__init__()
        name = name or SCALE_BACKBONE.get(scale, SCALE_BACKBONE["s"])
        self.model = _make_features(name, in_ch, pretrained)
        red = list(self.model.feature_info.reduction())
        chs = list(self.model.feature_info.channels())
        # stride 8/16/32 단계의 인덱스를 선택 (P3/P4/P5)
        self.sel = [red.index(r) for r in (8, 16, 32)]
        self.out_channels = tuple(chs[i] for i in self.sel)
        self.name = name

    def forward(self, x: torch.Tensor):
        feats = self.model(x)
        return tuple(feats[i] for i in self.sel)


class FPNNeck(nn.Module):
    """torchvision FeaturePyramidNetwork 기반 넥. (P3,P4,P5) → (N3,N4,N5).

    세 스케일을 top-down 경로로 융합하고 모든 레벨을 균일 채널(out_ch)로 맞춘다.
    """

    def __init__(self, in_channels, out_ch: int = 256):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(list(in_channels), out_ch)
        self.out_channels = (out_ch, out_ch, out_ch)

    def forward(self, feats):
        x = OrderedDict((str(i), f) for i, f in enumerate(feats))
        out = self.fpn(x)
        return tuple(out.values())
