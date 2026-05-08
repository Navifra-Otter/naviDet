"""Builder — yacs cfg → modules.

Dispatches by `cfg.task` (`pose` | `detection` | `segmentation`) and supports
both EdgeCrafter `ecvit` backbones (own KD-style backbone) and the legacy
`dinov3` backbones (kept for ablation).

EdgeCrafter components are constructed through the registry (`create()`), the
legacy dinov3 path uses direct imports.
"""

from __future__ import annotations

import os
from typing import Dict, List

import torch.nn as nn
from torch.utils.data import DataLoader

from navidet.utils import DISTRIBUTED


# ---------------------------------------------------------------------------
# Module-import side-effect: ensure all @register() decorated classes are
# loaded before we call create(). Importing the navidet sub-packages triggers
# the @register() decorators on ViTAdapter, HybridEncoder, ECPose, ECDet,
# ECSeg, ECTransformer, DETRTransformer, ECCriterion, DETRPoseCriterion,
# HungarianMatcher, DETRPoseHungarianMatcher, PostProcessor, DETRPosePostProcessor.
# ---------------------------------------------------------------------------
def _eager_register_all() -> None:
    import navidet.model.backbone.ecvit  # noqa: F401
    import navidet.model.backbone.dinov3_adapter  # noqa: F401  (registers DinoV3Adapter)
    import navidet.model.encoder.hybrid_encoder  # noqa: F401
    import navidet.model.detect.decoder  # noqa: F401
    import navidet.model.detect.ecdet  # noqa: F401
    import navidet.model.seg.segmentation_head  # noqa: F401
    import navidet.model.pose.decoder  # noqa: F401
    import navidet.model.pose.postprocess  # noqa: F401
    import navidet.model.pose.ecpose  # noqa: F401
    import navidet.core.loss_fn.detect.criterion  # noqa: F401
    import navidet.core.loss_fn.detect.matcher  # noqa: F401
    import navidet.core.loss_fn.pose.criterion  # noqa: F401
    import navidet.core.loss_fn.pose.matcher  # noqa: F401
    import navidet.core.head.postprocess  # noqa: F401
    import navidet.data.datasets.yolo_pose  # noqa: F401  (registers YoloPoseDataset)
    import navidet.data  # noqa: F401  (registers DataLoader, transforms, COCO, …)
    import navidet.core.optimizer._ec_optim  # noqa: F401 (registers SGD/Adam/AdamW/MultiStepLR/…)
    import navidet.core.optimizer.ema  # noqa: F401  (ModelEMA)
    import navidet.core.optimizer.amp  # noqa: F401  (GradScaler)
    import navidet.core.scheduler.warmup  # noqa: F401  (LinearWarmup)
    import navidet.core.scheduler._ec_lr_scheduler  # noqa: F401  (FlatCosineLRScheduler)
    import navidet.eval  # noqa: F401  (CocoEvaluator/VOCEvaluator)


_eager_register_all()


_ECVIT_BACKBONES = {"ecvit", "ecvit_t", "ecvit_s", "ecvit_b"}
_DINOV3_BACKBONES = {
    "custom_dinov3convnext", "custom_dinov3vit",
    "dinov3convnext", "dinov3vit",
}

_TASK_MODEL = {
    "pose": "ECPose",
    "detection": "ECDet",
    "segmentation": "ECSeg",
}
_TASK_CRITERION = {
    "pose": "DETRPoseCriterion",
    "detection": "ECCriterion",
    "segmentation": "ECCriterion",
}
_TASK_POSTPROCESSOR = {
    "pose": "DETRPosePostProcessor",
    "detection": "PostProcessor",
    "segmentation": "PostProcessor",
}


def _yacs_to_dict(node) -> dict:
    """Recursively convert a yacs CfgNode (or already-dict) to plain dict."""
    if hasattr(node, "items") and not isinstance(node, dict):
        node = dict(node)
    if isinstance(node, dict):
        return {k: _yacs_to_dict(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return type(node)(_yacs_to_dict(v) for v in node)
    return node


class Builder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task = cfg.task.lower()
        if self.task not in _TASK_MODEL:
            raise ValueError(f"cfg.task must be one of {list(_TASK_MODEL)}, got {self.task!r}")

    # ------------------------------------------------------------------ model
    def model(self) -> nn.Module:
        backbone_name = self.cfg.model.backbone_name.lower()

        if backbone_name in _ECVIT_BACKBONES or backbone_name.startswith("ec"):
            return self._build_ec_model()
        if backbone_name in _DINOV3_BACKBONES or backbone_name.startswith("dinov3") \
                or backbone_name.startswith("custom_dinov3"):
            return self._build_dinov3_model()
        raise ValueError(
            f"Unsupported backbone: {backbone_name!r}. "
            f"Expected one of {_ECVIT_BACKBONES | _DINOV3_BACKBONES} or an ec*/dinov3* prefix."
        )

    def _build_ec_model(self) -> nn.Module:
        """Construct ECPose / ECDet / ECSeg via registry."""
        from navidet.core.registry import create

        global_cfg = self._registry_cfg()
        model_cls = _TASK_MODEL[self.task]
        return create(model_cls, global_cfg)

    def _build_dinov3_model(self) -> nn.Module:
        """Legacy dinov3 path — only pose is supported (existing code)."""
        if self.task != "pose":
            raise NotImplementedError(
                f"dinov3 backbone is currently only wired up for cfg.task='pose'. "
                f"Got task={self.task}. Use an ecvit_* backbone instead."
            )
        from navidet.model.pose.dinov3pose import DINOv3Pose
        return DINOv3Pose(
            backbone=self.cfg.model.backbone_name,
            nkpts=self.cfg.model.nkpts,
            ncls=self.cfg.model.ncls,
            backbone_ckps=self.cfg.model.backbone_ckps,
            finetuning=self.cfg.model.finetuning,
        )

    # -------------------------------------------------------------- criterion
    def loss(self, model):
        backbone_name = self.cfg.model.backbone_name.lower()
        if backbone_name in _ECVIT_BACKBONES or backbone_name.startswith("ec"):
            from navidet.core.registry import create
            return create(_TASK_CRITERION[self.task], self._registry_cfg())

        # legacy dinov3 path (pose only)
        from navidet.core.loss_fn.pose._legacy_compute_loss import ComputeLoss
        return ComputeLoss(model, kpt_loss_type='oks')

    def postprocessor(self):
        backbone_name = self.cfg.model.backbone_name.lower()
        if backbone_name in _ECVIT_BACKBONES or backbone_name.startswith("ec"):
            from navidet.core.registry import create
            return create(_TASK_POSTPROCESSOR[self.task], self._registry_cfg())
        return None  # dinov3 path uses head-internal postprocessing

    def metric(self):
        return None

    # -------------------------------------------------- KD: build teacher pair
    def teacher_model(self) -> nn.Module | None:
        """Build a frozen teacher model for distillation.

        Returns None if `cfg.kd.enabled` is false. Teacher backbone defaults to
        `cfg.kd.teacher_backbone` (typically a dinov3 variant).
        """
        kd = getattr(self.cfg, "kd", None)
        if kd is None or not getattr(kd, "enabled", False):
            return None
        teacher_backbone = getattr(kd, "teacher_backbone", "dinov3vit_base")
        if not (teacher_backbone.startswith("dinov3") or teacher_backbone.startswith("custom_dinov3")):
            raise ValueError(
                f"kd.teacher_backbone must be a dinov3* variant (got {teacher_backbone!r})."
            )
        if self.task != "pose":
            raise NotImplementedError(
                "KD with dinov3 teacher is currently only wired for pose. "
                "Add a teacher modeling path for det/seg if needed."
            )
        from navidet.model.pose.dinov3pose import DINOv3Pose
        teacher = DINOv3Pose(
            backbone=teacher_backbone,
            nkpts=self.cfg.model.nkpts,
            ncls=self.cfg.model.ncls,
            backbone_ckps=getattr(kd, "teacher_ckpt", None),
            finetuning=False,
        )
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return teacher

    # ------------------------------------------------------------- optim/sched
    def optimizer(self, params):
        opt_type = getattr(self.cfg.optimizer, 'type', 'adamw').lower()
        lr = getattr(self.cfg.optimizer, 'lr', 1e-4)
        weight_decay = getattr(self.cfg.optimizer, 'weight_decay', 1e-2)
        if opt_type == 'adamw':
            from torch.optim import AdamW
            return AdamW(params, lr=lr, weight_decay=weight_decay)
        if opt_type == 'sgd':
            from torch.optim import SGD
            return SGD(params, lr=lr,
                       momentum=getattr(self.cfg.optimizer, 'momentum', 0.9),
                       weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    def lr_scheduler(self, optimizer):
        sched = getattr(self.cfg.lr_scheduler, 'type', 'cosine').lower()
        if sched == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.trainer.epochs,
                eta_min=getattr(self.cfg.lr_scheduler, 'min_lr', 1e-6),
            )
        if sched == 'flatcosine':
            from navidet.core.scheduler._ec_lr_scheduler import FlatCosineLRScheduler
            return FlatCosineLRScheduler(
                optimizer,
                lr_gamma=getattr(self.cfg.lr_scheduler, 'lr_gamma', 0.1),
                iter_per_epoch=getattr(self.cfg.lr_scheduler, 'iter_per_epoch', 1),
                total_epochs=self.cfg.trainer.epochs,
                no_aug_epochs=getattr(self.cfg.lr_scheduler, 'no_aug_epochs', 0),
                warmup_iter=getattr(self.cfg.lr_scheduler, 'warmup_iter', 0),
            )
        raise ValueError(f"Unsupported LR scheduler type: {sched}")

    # -------------------------------------------------------------- dataset
    def dataset(self, data_path):
        ds_type = self.cfg.dataset.dataset.lower()
        if ds_type == 'yolo_pose':
            from navidet.data.datasets.yolo_pose import YoloPoseDataset
            cache_ram = bool(getattr(self.cfg.dataset, 'cache_ram', False))
            return YoloPoseDataset(
                img_dir=os.path.join(data_path, 'images'),
                label_dir=os.path.join(data_path, 'labels'),
                img_size=self.cfg.dataset.img_size,
                nkpts=self.cfg.model.nkpts[0],
                cache_ram=cache_ram,
            )
        if ds_type == 'coco':
            from navidet.data.datasets.coco import CocoDetection
            return CocoDetection(
                img_folder=os.path.join(data_path, 'images'),
                ann_file=os.path.join(data_path, 'annotations.json'),
                transforms=None,
                return_masks=(self.task == 'segmentation'),
                remap_mscoco_category=getattr(self.cfg.dataset, 'remap_mscoco_category', False),
            )
        raise ValueError(f"Unsupported dataset {ds_type!r} (data_path={data_path!r}).")

    # ----------------------------------------------- DataLoader / DDP wrapping
    def set_device(self, model, trainDS, validDS, device):
        nw = self.cfg.dataloader.num_workers
        prefetch = getattr(self.cfg.dataloader, 'prefetch_factor', 2)
        persistent = bool(getattr(self.cfg.dataloader, 'persistent_workers', True)) and nw > 0
        common = dict(
            num_workers=nw,
            pin_memory=self.cfg.dataloader.pin_memory,
            persistent_workers=persistent,
        )
        if nw > 0:
            common['prefetch_factor'] = prefetch

        collate = getattr(trainDS, 'collate_fn', None)

        if DISTRIBUTED:
            from torch.utils.data.distributed import DistributedSampler
            from torch.nn.parallel import DistributedDataParallel as DDP
            t_sampler = DistributedSampler(trainDS, shuffle=self.cfg.dataloader.shuffle)
            v_sampler = DistributedSampler(validDS, shuffle=False)
            model = DDP(model.to(device), device_ids=[device], output_device=device,
                        find_unused_parameters=False)
            trainloader = DataLoader(trainDS, batch_size=self.cfg.dataloader.batch_size,
                                     sampler=t_sampler, collate_fn=collate,
                                     drop_last=self.cfg.dataloader.drop_last, **common)
            validloader = DataLoader(validDS, batch_size=self.cfg.dataloader.batch_size,
                                     sampler=v_sampler, collate_fn=getattr(validDS, 'collate_fn', None),
                                     drop_last=False, **common)
        else:
            model = model.to(device)
            trainloader = DataLoader(trainDS, batch_size=self.cfg.dataloader.batch_size,
                                     shuffle=self.cfg.dataloader.shuffle, collate_fn=collate,
                                     drop_last=self.cfg.dataloader.drop_last, **common)
            validloader = DataLoader(validDS, batch_size=self.cfg.dataloader.batch_size,
                                     shuffle=False, collate_fn=getattr(validDS, 'collate_fn', None),
                                     drop_last=False, **common)
        return model, trainloader, validloader

    # -------------------------------------------------------------- internals
    def _registry_cfg(self) -> dict:
        """Yacs cfg → dict shape that EdgeCrafter's `create()` expects.

        The registry's `create()` walks GLOBAL_CONFIG entries (which carry the
        per-class `_pymodule` / `_inject` schema). User-provided spec
        (cfg.method) is merged into GLOBAL_CONFIG via `merge_config` so that
        each registered class entry retains its schema while user kwargs
        override the defaults.
        """
        from navidet.core.registry import GLOBAL_CONFIG
        from navidet.core.registry.yaml_utils import merge_config

        method = getattr(self.cfg, "method", None)
        user_cfg = _yacs_to_dict(method) if method is not None else _yacs_to_dict(self.cfg.model)
        # merge_config copies GLOBAL_CONFIG schemas into user_cfg.
        # overwrite=False: GLOBAL_CONFIG fills MISSING keys (schema/defaults), but
        # user-provided values take precedence.
        return merge_config(user_cfg, another_cfg=GLOBAL_CONFIG, inplace=False, overwrite=False)
