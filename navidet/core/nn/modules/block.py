import torch
import torch.nn as nn
from torch import Tensor
from .mlp import MLP
from .norm import LayerNorm
from typing import List

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ConvBlock(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, c_, k[0], 1)
        self.cv2 = ConvBlock(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = ConvBlock(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, k=1, s=1)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, k=1, s=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.kernel_size = kernel_size
    
    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(self.pool(y[-1]) for _ in range(3))
        return self.conv2(torch.cat(y, dim=1))

    
class PAN(nn.Module):
    """
    Path Aggregation Network (FPN + Bottom-up)
    FIXED: Corrected input channel sizes for Bottom-up path concatenation.
    """
    def __init__(self, in_channels):
        super().__init__()
        c1, c2, c3 = in_channels # [256, 512, 1024]
        self.fpn_conv1 = ConvBlock(c3, c2, k=1)
        self.fpn_conv2 = BottleneckCSP(c2 + c2, c2, n=2, shortcut=False)
        self.fpn_conv3 = ConvBlock(c2, c1, k=1)
        self.fpn_conv4 = BottleneckCSP(c1 + c1, c1, n=2, shortcut=False)
        self.down1 = ConvBlock(c1, c1, k=3, s=2)
        self.pan_conv1 = BottleneckCSP(c1 + c2, c2, n=2, shortcut=False)
        self.down2 = ConvBlock(c2, c2, k=3, s=2)
        self.pan_conv2 = BottleneckCSP(c2 + c3, c3, n=2, shortcut=False)
    
    def forward(self, features):
        c1, c2, c3 = features
        
        # FPN
        c3_up = F.interpolate(self.fpn_conv1(c3), size=c2.shape[2:], mode='bilinear', align_corners=False)
        c2_fused = self.fpn_conv2(torch.cat([c2, c3_up], dim=1))
        
        c2_up = F.interpolate(self.fpn_conv3(c2_fused), size=c1.shape[2:], mode='bilinear', align_corners=False)
        c1_fused = self.fpn_conv4(torch.cat([c1, c2_up], dim=1))
        
        # PAN
        c1_down = self.down1(c1_fused)
        c1_down = F.interpolate(c1_down, size=c2_fused.shape[2:], mode='bilinear', align_corners=False)
        
        # 여기서 256(c1_down) + 512(c2_fused) = 768 채널이 됨
        c2_pan = self.pan_conv1(torch.cat([c1_down, c2_fused], dim=1))
        
        c2_down = self.down2(c2_pan)
        c2_down = F.interpolate(c2_down, size=c3.shape[2:], mode='bilinear', align_corners=False)
        
        # 여기서 512(c2_down) + 1024(c3) = 1536 채널이 됨
        c3_pan = self.pan_conv2(torch.cat([c2_down, c3], dim=1))
        
        return [c1_fused, c2_pan, c3_pan]
    
class AttentionBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., use_gated=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if use_gated:
            from .attention import GatedAttention
            self.attn = GatedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                      attn_drop=attn_drop, proj_drop=drop)
        else:
            from .attention import Attention
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                 attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ConvNext2Block(nn.Module):
    """ConvNeXt V2 Block for 2D data"""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_init_value = layer_scale_init_value
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class FeatureAdaptor(nn.Module):
    def __init__(self, sc:List|int, tc:List=[256, 512, 1024]):
        assert type(sc) == list or type(sc) == int

        super(FeatureAdaptor, self).__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                ConvBlock(sc, c2, k=1),
                ConvBlock(c2, c2, k=3)
            ) for c2 in tc
        ]) if type(sc)==int else nn.ModuleList([
            nn.Sequential(
                ConvBlock(c1, c2, k=1),
                ConvBlock(c2, c2, k=3)
            ) for c1, c2 in zip(sc, tc)
        ])

    def forward(self, features):
        return [adapter(feat) for adapter, feat in zip(self.adapters, features)]
    

class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL)."""

    def __init__(self, c1: int = 16):

        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
