import torch
from torch import Tensor
from typing import List, Dict, Optional, Union
from pose.model.utils import convert_path_or_url_to_url, Weights

dinov3_convnext = torch.hub.load(
    repo_or_dir="pose/model/thirdparty/dinov3", 
    model='dinov3_convnext_base', 
    source='local',
)

ConvNext = dinov3_convnext.__class__
dinov3_convnext = None

class CustomDinov3ConvNext(ConvNext):
    def __init__(self, 
                in_chans: int = 3,
                depths: List[int] = [3, 3, 27, 3],
                dims: List[int] = [128, 256, 512, 1024],
                drop_path_rate: float = 0.0,
                layer_scale_init_value: float = 1e-6,
                pretrained: bool = True,
                weights: Union[Weights, str] = Weights.LVD1689M,
                hash: Optional[str] = None,
                **kwargs,
            ):
        
        model_kwargs = dict(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        model_kwargs.update(**kwargs)
        super(CustomDinov3ConvNext, self).__init__(**model_kwargs)
        if pretrained:
            if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
                raise ValueError(f"Unsupported weights for the backbone: {weights}")
            url = convert_path_or_url_to_url(weights)
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        else:
            self.init_weights()

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            x = self.forward_features_list([x], [masks])[-1]
            x_pool = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = torch.flatten(x, 2).transpose(1, 2)

            # concat [CLS] and patch tokens as (N, HW + 1, C), then normalize
            x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1))

            return {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
        else:
            x = self.forward_features_list(x, masks)[-1]
            x_pool = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = torch.flatten(x, 2).transpose(1, 2)
            x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1))

            return {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        output = []
        for x, masks in zip(x_list, masks_list):
            h, w = x.shape[-2:]
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                output.append(x)
        return output

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
        
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

convnext_ckps = {
    'tiny': '/home/otter/study/2D-Human-Pose-Estimation/HPE/checkpoints/dinov3/convnext/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth', 
    'small': '/home/otter/study/2D-Human-Pose-Estimation/HPE/checkpoints/dinov3/convnext/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth', 
    'base': '/home/otter/study/2D-Human-Pose-Estimation/HPE/checkpoints/dinov3/convnext/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth', 
    'large': '/home/otter/study/2D-Human-Pose-Estimation/HPE/checkpoints/dinov3/convnext/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'
    }
