"""Data package — adapted from EdgeCrafter, restructured into naviDet layout."""

from ._misc import convert_to_tv_tensor
from .dataloader import *  # noqa: F401,F403  (DataLoader, BatchImageCollateFunction, ...)
from .datasets import *  # noqa: F401,F403
from .transforms import *  # noqa: F401,F403

# CocoEvaluator lives under navidet.eval but EdgeCrafter expects it from `..data`.
# Re-export here for compatibility with the EC trainer code.
from navidet.eval.coco_eval import CocoEvaluator  # noqa: F401
from navidet.eval.coco_utils import get_coco_api_from_dataset  # noqa: F401
