"""
CSP 기반 컨볼루션 빌딩 블록 모음.

백본/넥을 구성하는 핵심 모듈(Conv, Bottleneck, CSPLayer, SPPF, PSALayer)을
PyTorch 만으로 독립 구현한 것입니다. 외부 라이브러리 의존성은 없습니다.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """'same' 출력 크기를 만드는 padding 계산 (dilation 고려)."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Conv+BN을 단일 Conv(bias 포함)로 융합. 추론 속도용 (출력 동일)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      conv.stride, conv.padding, conv.dilation, conv.groups,
                      bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
    b_conv = (torch.zeros(conv.weight.size(0), device=conv.weight.device)
              if conv.bias is None else conv.bias)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


class Conv(nn.Module):
    """표준 Conv-BN-SiLU 블록 (기본 단위)."""

    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 p: int | None = None, g: int = 1, d: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g,
                              dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuse(self):
        """BN을 conv에 흡수하고 bn을 Identity로 (in-place). 추론 전용."""
        if isinstance(self.bn, nn.BatchNorm2d):
            self.conv = fuse_conv_bn(self.conv, self.bn)
            self.bn = nn.Identity()


class DWConv(Conv):
    """Depthwise(채널별) Conv. 3x3 공간 conv를 저비용으로 — head 경량화에 사용."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, d: int = 1, act: bool = True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    """잔차(residual) bottleneck. shortcut=True면 입력을 더해줌."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 k: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CSPBlock(nn.Module):
    """CSPBlock: 3x3 커널을 쓰는 작은 CSP 블록 (CSPLayer 내부에서 사용)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class CSPLayer(nn.Module):
    """
    메인 CSP 스테이지 블록.

    입력을 2분할(split)하여 한 갈래만 n개의 블록을 통과시키고 다시 concat 하는
    CSP 구조. c3k=True면 내부 블록으로 CSPBlock를, False면 Bottleneck을 사용.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False,
                 e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        block = CSPBlock if c3k else Bottleneck
        self.m = nn.ModuleList(
            block(self.c, self.c, shortcut=shortcut) if c3k
            else block(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cv1으로 2*c 채널 생성 후 절반씩 split → [B, c, H, W] 두 개
        y = list(self.cv1(x).chunk(2, 1))
        # 한 갈래를 블록들에 순차 통과시키며 중간 출력을 모두 누적
        y.extend(m(y[-1]) for m in self.m)
        # 누적된 (2+n)개의 [B, c, H, W]를 concat → cv2로 c2 채널 압축
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast. 다중 수용영역을 저비용으로 결합."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))  # 5,9,13 유효 커널
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """PSALayer 내부의 멀티헤드 셀프 어텐션 (positional-aware)."""

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # depthwise positional encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        qkv = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    """어텐션 + FFN으로 구성된 잔차 블록."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4):
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class PSALayer(nn.Module):
    """
    어텐션 스테이지(보통 백본 마지막, SPPF 뒤).
    CSP 형태로 절반 채널에만 PSA 블록을 적용해 비용을 억제.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, num_heads=max(1, self.c // 64)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
