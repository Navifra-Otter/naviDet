import torch
import torch.nn as nn   
import torch.nn.functional as F
from pose.model.utils import DINOV3REPO, MODEL_TO_NUM_LAYERS_VIT


class Dinov3ViT(nn.Module):
    def __init__(self, backbone, source, weights):
        super().__init__()  
        self.n_layers = range(MODEL_TO_NUM_LAYERS_VIT[backbone])
        self.model = torch.hub.load(DINOV3REPO, backbone, source=source, weights=weights)

    def forward(self, x):
        return self.model(x)

    def get_feature_spaces(self, x):
        feats = self.model.get_intermediate_layers(x, n=self.n_layers, reshape=True, norm=True)[0]
        feats = feats.movedim(-3, -1)  # [1, h, w, D]
        feats = F.normalize(feats, dim=-1, p=2)
        return feats
        

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

