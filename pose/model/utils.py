import torch
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")

def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"

DINOV3REPO = "pose/model/thirdparty/dinov3"

MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_TO_NUM_LAYERS_VIT = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

DINO_VIT_MODELS = {
    'small': MODEL_DINOV3_VITS,
    'small_plus': MODEL_DINOV3_VITSP,   
    'base': MODEL_DINOV3_VITB,
    'large': MODEL_DINOV3_VITL,
    'large_plus': MODEL_DINOV3_VITHP,
    '7b': MODEL_DINOV3_VIT7B
}
MODEL_DINOV3_CONVNEXTT = "dinov3_convnext_tiny"
MODEL_DINOV3_CONVNEXTS = "dinov3_convnext_small"
MODEL_DINOV3_CONVNEXTB = "dinov3_convnext_base"
MODEL_DINOV3_CONVNEXTL = "dinov3_convnext_large"

MODEL_TO_NUM_LAYERS_CONVNEXT = {
    MODEL_DINOV3_CONVNEXTT: 3,
    MODEL_DINOV3_CONVNEXTS: 3,
    MODEL_DINOV3_CONVNEXTB: 3,
    MODEL_DINOV3_CONVNEXTL: 3
}

DINO_CONVNEXT_MODELS = {
    'tiny': MODEL_DINOV3_CONVNEXTT,
    'small': MODEL_DINOV3_CONVNEXTS,
    'base': MODEL_DINOV3_CONVNEXTB,
    'large': MODEL_DINOV3_CONVNEXTL
}

convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}

vit_sizes = {
    "small": dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4.0,
    ),
    "base": dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
    ),
    "large": dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4.0,
    ),

}

convnext_ckps = {
    'tiny': './checkpoints/dinov3/convnext/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth', 
    'small': './checkpoints/dinov3/convnext/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth', 
    'base': './checkpoints/dinov3/convnext/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth', 
    'large': './checkpoints/dinov3/convnext/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'
    }

vit_ckps = {
    'small': './checkpoints/dinov3/vit/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', 
    'base': './checkpoints/dinov3/vit/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', 
    '7b': './checkpoints/dinov3/vit/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth', 
    'large': None
    }

