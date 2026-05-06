import math
import torch 
import torch.nn as nn
from pose.model.nn.modules.block import ConvBlock
from pose.model.utils import make_anchors


class PoseHead(nn.Module):
    """YOLOv11-Pose Head with proper output structure."""

    dynamic = False 
    anchors = torch.empty(0)
    strides = torch.empty(0)
    shape = None

    def __init__(self, ncls=1, kpt_shape: tuple = (17, 3), in_ch: tuple=()):
        super().__init__()
        self.nc = ncls
        self.nl = len(in_ch)
        
        self.register_buffer("stride", torch.tensor([8, 16, 32], dtype=torch.float32))    
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c3 = max(in_ch[0], min(self.nc, 100))
        c4 = max(in_ch[0] // 4, self.nk)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(
                    ConvBlock(c1=x, c2=x, k=3, g=math.gcd(x, x)),
                    ConvBlock(x, c3, 1)
                ),
                nn.Sequential(
                    ConvBlock(c3, c3, 3),
                    ConvBlock(c3, c3, 1),
                ),
                nn.Conv2d(c3, ncls, 1)
            ) for x in in_ch
        )
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                ConvBlock(x, c4, 3), 
                ConvBlock(c4, c4, 3), 
                nn.Conv2d(c4, self.nk, 1)
            ) for x in in_ch
        )
        self._init_biases()

    def _init_biases(self):
        """Initialize biases for classification head."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        for cv3_block in self.cv3:
            final_conv = cv3_block[-1] 
            if hasattr(final_conv, 'bias') and final_conv.bias is not None:
                nn.init.constant_(final_conv.bias, bias_value)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Forward pass for pose estimation."""
        for i in range(self.nl):
            cls_feat = self.cv3[i](x[i])
            kpt_feat = self.cv4[i](x[i])
            x[i] = torch.cat([cls_feat, kpt_feat], 1)
        
        if self.training:
            return x
        
        return self._inference_pose(x)

    def _inference_pose(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Pose inference with proper decoding."""
        shape = x[0].shape
        total_ch = self.nc + self.nk
        
        x_cat = torch.cat([xi.view(shape[0], total_ch, -1) for xi in x], 2)
        
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        cls, kpt = x_cat.split((self.nc, self.nk), 1)
        dkpt = self.kpts_decode(kpt)
        
        return torch.cat((cls.sigmoid(), dkpt), 1)
    
    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        y = kpts.clone()  
        
        # x, y 좌표 복원 (Grid 좌표 -> 실제 이미지 좌표)
        y[:, 0::ndim] = (y[:, 0::ndim] - 0.5 + self.anchors[0]) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] - 0.5 + self.anchors[1]) * self.strides
        if ndim == 3:
            y[:, 2::ndim] = y[:, 2::ndim].sigmoid() # Visibility는 Sigmoid
        return y
