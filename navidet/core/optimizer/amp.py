"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch.amp as amp

from navidet.core.registry import register

__all__ = ['GradScaler']


@register()
class GradScaler(amp.GradScaler):
    """`torch.amp.GradScaler` with `device='cuda'` baked in.

    Avoids the `torch.cuda.amp.GradScaler(args...)` deprecation warning while
    keeping the YAML config interface unchanged (no need to specify device).
    """

    def __init__(self, *args, device: str = "cuda", **kwargs):
        super().__init__(device, *args, **kwargs)
