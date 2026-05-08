"""Feature-level knowledge distillation loss.

Used for ablation studies that compare ecvit student trained alone vs. trained
with a frozen dinov3 teacher providing supervision on intermediate features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureKDLoss(nn.Module):
    """L2 distance between aligned student/teacher feature pyramids.

    Both `student_feats` and `teacher_feats` are sequences of NCHW tensors at
    matched strides. If channel counts disagree, a 1×1 Conv projection is used
    to bring student channels to teacher channels (per level, lazily built on
    first forward).
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.projectors: nn.ModuleList | None = None

    def _ensure_projectors(self, student_feats, teacher_feats) -> None:
        if self.projectors is not None:
            return
        projs = []
        for s, t in zip(student_feats, teacher_feats):
            if s.shape[1] == t.shape[1]:
                projs.append(nn.Identity())
            else:
                projs.append(nn.Conv2d(s.shape[1], t.shape[1], kernel_size=1, bias=False))
        self.projectors = nn.ModuleList(projs).to(student_feats[0].device)

    def forward(self, student_feats, teacher_feats) -> torch.Tensor:
        if not isinstance(student_feats, (list, tuple)):
            student_feats = [student_feats]
        if not isinstance(teacher_feats, (list, tuple)):
            teacher_feats = [teacher_feats]
        assert len(student_feats) == len(teacher_feats), \
            f"feature pyramid mismatch: student={len(student_feats)} teacher={len(teacher_feats)}"
        self._ensure_projectors(student_feats, teacher_feats)

        loss = student_feats[0].new_zeros(())
        for s, t, proj in zip(student_feats, teacher_feats, self.projectors):
            s = proj(s)
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(s, size=t.shape[-2:], mode='bilinear', align_corners=False)
            loss = loss + F.mse_loss(s, t.detach())
        return self.weight * loss / len(student_feats)
