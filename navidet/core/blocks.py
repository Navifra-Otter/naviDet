"""
검출 헤드용 기본 컨볼루션 블록.

백본/넥은 외부 퍼미시브 라이브러리(timm, torchvision)를 사용하므로, 여기에는
헤드가 쓰는 표준 Conv-BatchNorm-SiLU 블록 하나만 둔다. PyTorch 기본 연산만으로
작성했으며 외부 라이브러리 의존성은 없다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Conv → BatchNorm → SiLU.

    in_ch, out_ch : 입출력 채널
    k, s          : 커널/스트라이드
    p             : 패딩(None이면 'same' 출력이 되도록 k//2)
    g             : 그룹 수
    act           : True=SiLU, nn.Module=해당 활성, 그 외=항등
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1,
                 p: int | None = None, g: int = 1, act=True):
        super().__init__()
        pad = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, pad, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if act is True:
            self.act = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        """BatchNorm을 conv 가중치에 흡수(추론 전용, 출력 동일·속도↑). bn→Identity."""
        if not isinstance(self.bn, nn.BatchNorm2d):
            return
        conv, bn = self.conv, self.bn
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std                                   # [out_ch]
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                          conv.stride, conv.padding, dilation=conv.dilation,
                          groups=conv.groups, bias=True).to(conv.weight.device)
        fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        fused.bias.copy_(bn.bias - bn.weight * bn.running_mean / std)
        self.conv = fused.requires_grad_(False)
        self.bn = nn.Identity()
