"""Knowledge-distillation aware solvers for pose / detection / segmentation.

Subclasses the EdgeCrafter ECSolver / PoseSolver and **only** swaps the
criterion with a wrapper that adds a feature-level distillation loss term.
The fit() / train_one_epoch() flow is unchanged — `loss_dict` simply gains a
`loss_kd` entry that the existing aggregation picks up.

Teacher is built from `cfg.kd.teacher_backbone` and frozen. Feature alignment
is L2 between the student encoder output and the teacher's equivalent feature
pyramid (via `_extract_features` heuristic).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from navidet.core.loss_fn.kd import FeatureKDLoss
from navidet.engine.det_trainer import ECSolver
from navidet.engine.pose_trainer import PoseSolver


def _extract_features(model: nn.Module, samples) -> Any:
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "forward_features"):
        return target.forward_features(samples)
    if hasattr(target, "backbone") and hasattr(target, "encoder"):
        feats = target.backbone(samples)
        return target.encoder(feats)
    if hasattr(target, "backbone"):
        return target.backbone(samples)
    raise RuntimeError(
        f"{type(target).__name__} exposes no usable feature extractor for KD"
    )


class KDCriterionWrapper(nn.Module):
    """Wraps a DETRPoseCriterion / ECCriterion and appends `loss_kd`.

    Captures `samples` via a forward pre-hook on the student model and the
    student encoder output via a forward hook on `student.encoder`. On
    forward(outputs, targets, **metas) the wrapper runs the base criterion,
    re-runs the teacher (no_grad) on the captured samples, and computes a
    feature-level L2 distance.
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        student: nn.Module,
        teacher: nn.Module,
        kd_weight: float = 1.0,
    ):
        super().__init__()
        self.base = base_criterion
        self.teacher = teacher
        self.kd_loss = FeatureKDLoss(weight=kd_weight)

        self._samples: torch.Tensor | None = None
        self._student_feats: Any = None

        student.register_forward_pre_hook(self._capture_samples)
        target = student.module if hasattr(student, "module") else student
        if hasattr(target, "encoder"):
            target.encoder.register_forward_hook(self._capture_student_feats)
        else:
            # Fallback: hook the backbone if no encoder.
            target.backbone.register_forward_hook(self._capture_student_feats)

    # ----------------- hooks -----------------
    def _capture_samples(self, module, args):
        if args:
            self._samples = args[0]

    def _capture_student_feats(self, module, inp, out):
        self._student_feats = out

    # ------------- proxy properties needed by EC train loops --------------
    @property
    def weight_dict(self):
        return getattr(self.base, "weight_dict", {})

    def __getattr__(self, name):
        # Defer unknown attribute lookups to base criterion (e.g. .losses,
        # .matcher) — anything EC's train_one_epoch may touch.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    # ---------------------------- forward ---------------------------------
    def forward(self, outputs, *args, **kwargs):
        loss_dict = self.base(outputs, *args, **kwargs)
        if self._samples is None or self._student_feats is None:
            return loss_dict
        with torch.no_grad():
            teacher_feats = _extract_features(self.teacher, self._samples)
        kd = self.kd_loss(self._student_feats, teacher_feats)
        # Match the dtype/device of an existing entry to be safe.
        if isinstance(loss_dict, dict):
            loss_dict["loss_kd"] = kd
        return loss_dict


def _build_teacher(cfg) -> nn.Module:
    """Construct + freeze teacher model. Currently only DINOv3-based teachers
    are supported (consistent with the legacy ablation backbones)."""
    kd_cfg = cfg.yaml_cfg.get("kd", {}) if hasattr(cfg, "yaml_cfg") else {}
    backbone = kd_cfg.get("teacher_backbone", "custom_dinov3vit_base")
    ckpt = kd_cfg.get("teacher_ckpt", None)
    if not (backbone.startswith("dinov3") or backbone.startswith("custom_dinov3")):
        raise ValueError(f"KD teacher must be a dinov3* backbone, got {backbone!r}")
    from navidet.model.pose.dinov3pose import DINOv3Pose
    teacher = DINOv3Pose(
        backbone=backbone, nkpts=(17, 3), ncls=1,
        backbone_ckps=ckpt, finetuning=False,
    )
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def _wire_kd(solver, kd_weight: float = 1.0):
    """Replace solver.criterion with a KD-aware wrapper after _setup()."""
    teacher = _build_teacher(solver.cfg).to(next(solver.model.parameters()).device)
    solver.criterion = KDCriterionWrapper(
        solver.criterion, solver.model, teacher, kd_weight=kd_weight,
    )
    return solver


class KDECSolver(ECSolver):
    """ECSolver + KD."""

    def _setup(self):
        super()._setup()
        kd_weight = self.cfg.yaml_cfg.get("kd", {}).get("feat_loss_weight", 1.0)
        _wire_kd(self, kd_weight=kd_weight)


class KDPoseSolver(PoseSolver):
    """PoseSolver + KD."""

    def _setup(self):
        super()._setup()
        kd_weight = self.cfg.yaml_cfg.get("kd", {}).get("feat_loss_weight", 1.0)
        _wire_kd(self, kd_weight=kd_weight)
