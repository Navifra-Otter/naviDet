import torch
from pose.model.backbone.dinov3vit import Dinov3ViT
from pose.model.backbone.dinov3convnext import Dinov3ConvNext
from pose.model.utils import *

size = 'base'

ckps = vit_ckps[size]
dummy_img = torch.randn(1, 3, 512, 512).cuda()

model = Dinov3ViT(
    backbone=DINO_VIT_MODELS[size],
    source='local',
    weights=ckps
).cuda()

feat_space = model.get_feature_spaces(dummy_img)
feat = model(dummy_img)
for i in range(len(feat_space)):
    print(feat_space[i].shape)
print(feat.shape)

# ckps = convnext_ckps[size]

# model = Dinov3ConvNext(
#     backbone=DINO_CONVNEXT_MODELS[size],
#     source='local',
#     weights=ckps
# ).cuda()

# feat_space = model.get_feature_spaces(dummy_img)
# feat = model(dummy_img)

# for i in range(len(feat_space)):
#     print(feat_space[i].shape)
# print(feat.shape)