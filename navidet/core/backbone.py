"""
YOLOv11 스타일 백본 + PAN 넥.

P3/P4/P5 세 스케일의 멀티스케일 특징을 반환하여, 작은~큰 객체를 모두 커버하도록
한다. 6DoF Head는 이 3개 스케일 각각에 대해 예측을 수행한다.

채널 폭(width)과 깊이(depth)는 scale 인자로 조절한다 (n/s/m/l/x 대응).
여기서는 Edge PC를 가정해 'n'(nano) 기본값을 사용한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import C2PSA, C3k2, Conv, SPPF


# (depth_mult, width_mult, max_channels) — ultralytics YOLOv11 scale 표와 동일 의미
YOLO11_SCALES = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


def _round_ch(c: int, width: float, max_c: int, divisor: int = 8) -> int:
    c = min(int(c * width), max_c)
    return max(divisor, int(c + divisor / 2) // divisor * divisor)


def _round_depth(n: int, depth: float) -> int:
    return max(1, round(n * depth))


class YOLOv11Backbone(nn.Module):
    """
    P1~P5 다운샘플 백본. 출력으로 P3(/8), P4(/16), P5(/32) 특징을 반환.
    """

    def __init__(self, scale: str = "n", in_ch: int = 3):
        super().__init__()
        d, w, mc = YOLO11_SCALES[scale]
        ch = [_round_ch(c, w, mc) for c in (64, 128, 256, 512, 1024)]
        self.out_channels = (ch[2], ch[3], ch[4])  # (P3, P4, P5)

        # stem + 다운샘플 스테이지들
        self.stem = Conv(in_ch, ch[0], 3, 2)                       # /2
        self.dark2 = nn.Sequential(
            Conv(ch[0], ch[1], 3, 2),                              # /4
            C3k2(ch[1], ch[1], _round_depth(2, d), c3k=False),
        )
        self.dark3 = nn.Sequential(
            Conv(ch[1], ch[2], 3, 2),                              # /8  -> P3
            C3k2(ch[2], ch[2], _round_depth(2, d), c3k=False),
        )
        self.dark4 = nn.Sequential(
            Conv(ch[2], ch[3], 3, 2),                              # /16 -> P4
            C3k2(ch[3], ch[3], _round_depth(2, d), c3k=True),
        )
        self.dark5 = nn.Sequential(
            Conv(ch[3], ch[4], 3, 2),                              # /32 -> P5
            C3k2(ch[4], ch[4], _round_depth(2, d), c3k=True),
            SPPF(ch[4], ch[4], 5),
            C2PSA(ch[4], ch[4], _round_depth(2, d)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 3, H, W]
        x = self.stem(x)            # [B, ch0, H/2,  W/2]
        x = self.dark2(x)           # [B, ch1, H/4,  W/4]
        p3 = self.dark3(x)          # [B, ch2, H/8,  W/8]
        p4 = self.dark4(p3)         # [B, ch3, H/16, W/16]
        p5 = self.dark5(p4)         # [B, ch4, H/32, W/32]
        return p3, p4, p5


class PANNeck(nn.Module):
    """
    YOLOv11 PAN 넥: top-down(업샘플) + bottom-up(다운샘플) 경로로 멀티스케일
    특징을 융합. 입력/출력 모두 (P3, P4, P5).
    """

    def __init__(self, ch: tuple[int, int, int], scale: str = "n"):
        super().__init__()
        d, _, _ = YOLO11_SCALES[scale]
        c3, c4, c5 = ch
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # top-down (YOLOv8/11 방식: 별도 reduce 없이 concat 후 C3k2가 채널 정리)
        self.td_p4 = C3k2(c5 + c4, c4, _round_depth(2, d), c3k=False)
        self.td_p3 = C3k2(c4 + c3, c3, _round_depth(2, d), c3k=False)

        # bottom-up
        self.down_p3 = Conv(c3, c3, 3, 2)
        self.bu_p4 = C3k2(c3 + c4, c4, _round_depth(2, d), c3k=False)
        self.down_p4 = Conv(c4, c4, 3, 2)
        self.bu_p5 = C3k2(c4 + c5, c5, _round_depth(2, d), c3k=True)

        self.out_channels = (c3, c4, c5)

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        p3, p4, p5 = feats
        # --- top-down ---
        p4_td = self.td_p4(torch.cat([self.up(p5), p4], 1))        # [B, c4, H/16, W/16]
        n3 = self.td_p3(torch.cat([self.up(p4_td), p3], 1))        # [B, c3, H/8,  W/8]  -> N3
        # --- bottom-up ---
        n4 = self.bu_p4(torch.cat([self.down_p3(n3), p4_td], 1))   # [B, c4, H/16, W/16] -> N4
        n5 = self.bu_p5(torch.cat([self.down_p4(n4), p5], 1))      # [B, c5, H/32, W/32] -> N5
        return n3, n4, n5
