"""Datasets — kept naviDet's existing yolo_pose alongside EdgeCrafter's COCO/VOC."""

from .coco import (CocoDetection, mscoco_category2label,
                   mscoco_category2name, mscoco_label2category)
from .voc import VOCDetection
