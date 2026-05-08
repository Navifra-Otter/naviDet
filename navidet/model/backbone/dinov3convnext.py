import torch
import torch.nn as nn
import torch.nn.functional as F
from navidet.model.utils import DINOV3REPO, MODEL_TO_NUM_LAYERS_CONVNEXT

class Dinov3ConvNext(nn.Module):
    def __init__(self, backbone, source, weights):
        super().__init__()  
        self.n_layers = range(MODEL_TO_NUM_LAYERS_CONVNEXT[backbone])
        self.model = torch.hub.load(DINOV3REPO, backbone, source=source, weights=weights)

    def forward(self, x):
        return self.model(x)

    def get_feature_spaces(self, x):
        feats = self.model.get_intermediate_layers(x, n=self.n_layers, reshape=True, norm=True)
        # feats = feats.movedim(-3, -1)  # [1, h, w, D]
        # feats = F.normalize(feats, dim=-1, p=2)        
        return feats



