import torch
from torch import Tensor
from typing import Optional, Union, List, Dict
from pose.model.utils import convert_path_or_url_to_url, Weights

dinov3_vit = torch.hub.load(
    "pose/model/thirdparty/dinov3", 
    'dinov3_vits16', 
    source='local', 
)

VisionTransformer = dinov3_vit.__class__
dinov3_vit = None

class Dinov3ViT(VisionTransformer):
    def __init__(
            self,     
            *,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            pos_embed_rope_base: float = 100.0,
            pos_embed_rope_min_period: float | None = None,
            pos_embed_rope_max_period: float | None = None,
            pos_embed_rope_normalize_coords: str = "separate",
            pos_embed_rope_shift_coords: float | None = None,
            pos_embed_rope_jitter_coords: float | None = None,
            pos_embed_rope_rescale_coords: float | None = None,
            pos_embed_rope_dtype: str = "fp32",
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            ffn_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_path_rate: float = 0.0,
            layerscale_init: float | None = None,
            norm_layer: str = "layernorm",
            ffn_layer: str = "mlp",
            ffn_bias: bool = True,
            proj_bias: bool = True,
            n_storage_tokens: int = 0,
            mask_k_bias: bool = False,
            pretrained: bool = False,
            weights: Union[Weights, str] = Weights.LVD1689M,
            hash: Optional[str] = None,
            check_hash: bool = False,
            load_from_url: bool = False,
            **kwargs
        ):
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_min_period=pos_embed_rope_min_period,
            pos_embed_rope_max_period=pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            ffn_bias=ffn_bias,
            proj_bias=proj_bias,
            n_storage_tokens=n_storage_tokens,
            mask_k_bias=mask_k_bias,
        )
        vit_kwargs.update(**kwargs)
        super(Dinov3ViT, self).__init__(**vit_kwargs)
        if pretrained:
            if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
                raise ValueError(f"Unsupported weights for the backbone: {weights}")
            if load_from_url:
                url = convert_path_or_url_to_url(weights)
                state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
            else:
                state_dict = torch.load(weights, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        else:
            self.init_weights()

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0][0]
        else:
            return self.forward_features_list(x, masks)[0]

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
        


