"""yacs-native training loop for naviDet.

Designed for the existing yacs cfg style. The full EdgeCrafter solver
(`engine/det_trainer.py`, `engine/pose_trainer.py`) is also available for users
who want the registry/YAMLConfig-based pipeline; this Trainer keeps the
existing naviDet workflow but adds task-aware loss/eval and optional KD.

Design:
  Trainer(cfg, model, trainloader, validloader, optimizer, lr_scheduler,
          loss_fn, ddp_manager=None, teacher=None, kd_loss=None)
  - teacher / kd_loss are optional; when provided, teacher is run in no_grad
    on each batch and the kd_loss output is added to the supervised loss.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch
import torch.nn as nn

from navidet.utils import (DISTRIBUTED, MASTER_RANK, master_only,
                           printE, printM, printS, printT)


def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(v, device) for v in batch)
    return batch


def _extract_features(model: nn.Module, samples) -> Any:
    """Best-effort feature extraction for KD.

    For ECPose / ECDet / ECSeg the encoder output is the right level to match.
    For dinov3 teacher we run only the backbone+encoder path (no decoder).
    Models without `forward_features` get a backbone+encoder fallback.
    """
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "forward_features"):
        return target.forward_features(samples)
    # ECPose has backbone/encoder/decoder attrs but no forward_features.
    if hasattr(target, "backbone") and hasattr(target, "encoder"):
        feats = target.backbone(samples)
        feats = target.encoder(feats)
        return feats
    raise RuntimeError(
        f"{type(target).__name__} does not expose feature pyramid for KD. "
        "Define forward_features() or expose .backbone/.encoder."
    )


class Trainer:
    def __init__(
        self,
        cfg,
        model: nn.Module,
        trainloader,
        validloader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        loss_fn,
        ddp_manager=None,
        metric=None,
        use_scalar: bool = False,
        teacher: nn.Module | None = None,
        kd_loss: nn.Module | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.ddp = ddp_manager
        self.device = ddp_manager.device if ddp_manager is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.metric = metric
        self.use_amp = bool(getattr(cfg, "use_amp", False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.teacher = teacher
        self.kd_loss = kd_loss
        if self.teacher is not None:
            self.teacher.to(self.device).eval()
        if self.kd_loss is not None:
            self.kd_loss.to(self.device)

        self.save_path = getattr(cfg.trainer, "save_path", "weights")
        os.makedirs(self.save_path, exist_ok=True)

        self.best_val = float("inf")
        self.epochs_no_improve = 0

    # --------------------------------------------------------------- helpers
    def _step(self, samples, targets) -> Dict[str, float]:
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(samples, targets) if self._takes_targets() \
                else self.model(samples)
            sup_loss, sup_log = self._compute_supervised_loss(outputs, targets)
            total = sup_loss

            kd_value = 0.0
            if self.teacher is not None and self.kd_loss is not None:
                with torch.no_grad():
                    teacher_feats = _extract_features(self.teacher, samples)
                student_feats = _extract_features(self.model, samples)
                kd = self.kd_loss(student_feats, teacher_feats)
                total = total + kd
                kd_value = float(kd.detach().item())

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        out = {"loss": float(total.detach().item()), "loss_sup": float(sup_loss.detach().item())}
        if self.teacher is not None:
            out["loss_kd"] = kd_value
        if isinstance(sup_log, dict):
            out.update({k: float(v) for k, v in sup_log.items()})
        return out

    def _takes_targets(self) -> bool:
        # ECPose/ECDet/ECSeg accept (samples, targets); dinov3pose accepts (samples).
        target = self.model.module if hasattr(self.model, "module") else self.model
        return type(target).__name__ in {"ECPose", "ECDet", "ECSeg"}

    def _compute_supervised_loss(self, outputs, targets):
        """Adapter that accepts either DETR-style criterion (returns dict of
        weighted losses) or legacy callable (returns (total, dict))."""
        result = self.loss_fn(outputs, targets)
        if isinstance(result, dict):
            total = sum(v for v in result.values() if isinstance(v, torch.Tensor))
            return total, {k: float(v.detach()) if isinstance(v, torch.Tensor) else v
                           for k, v in result.items()}
        if isinstance(result, tuple) and len(result) == 2:
            total, log = result
            return total, log
        return result, {}

    # ------------------------------------------------------------------ run
    @master_only
    def _log(self, msg):
        printT(msg)

    def train(self):
        epochs = self.cfg.trainer.epochs
        for epoch in range(1, epochs + 1):
            self.model.train()
            if DISTRIBUTED and hasattr(self.trainloader.sampler, "set_epoch"):
                self.trainloader.sampler.set_epoch(epoch)

            running = 0.0
            n = 0
            for batch_idx, batch in enumerate(self.trainloader):
                samples, targets = self._unpack(batch)
                samples = _to_device(samples, self.device)
                targets = _to_device(targets, self.device)
                try:
                    log = self._step(samples, targets)
                except Exception as e:  # don't kill epoch on a single bad batch
                    printE(f"epoch {epoch} batch {batch_idx}: {e}")
                    continue
                running += log["loss"]
                n += 1

            avg = running / max(n, 1)
            self._log(f"[epoch {epoch}/{epochs}] train_loss={avg:.4f}")
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if epoch % getattr(self.cfg.trainer, "valid_term", 1) == 0:
                val = self.validate(epoch)
                self._log(f"[epoch {epoch}/{epochs}] val_loss={val:.4f}")
                if val < self.best_val:
                    self.best_val = val
                    self.epochs_no_improve = 0
                    self._save(epoch, tag="best")
                else:
                    self.epochs_no_improve += 1

            if epoch % getattr(self.cfg.trainer, "save_term", epochs) == 0:
                self._save(epoch, tag=f"e{epoch:04d}")

            es = getattr(self.cfg.trainer, "early_stopping", None)
            if es is not None and getattr(es, "enabled", False) and \
                    self.epochs_no_improve >= getattr(es, "patience", 10):
                self._log(f"early stop @ epoch {epoch}")
                break

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        running, n = 0.0, 0
        for batch in self.validloader:
            samples, targets = self._unpack(batch)
            samples = _to_device(samples, self.device)
            targets = _to_device(targets, self.device)
            outputs = self.model(samples, targets) if self._takes_targets() \
                else self.model(samples)
            sup_loss, _ = self._compute_supervised_loss(outputs, targets)
            running += float(sup_loss.detach().item())
            n += 1
        return running / max(n, 1)

    def _unpack(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch.get("samples", batch.get("images")), batch.get("targets", batch.get("labels"))
        return batch, None

    @master_only
    def _save(self, epoch: int, tag: str):
        target = self.model.module if hasattr(self.model, "module") else self.model
        path = os.path.join(self.save_path, f"{tag}.pth")
        torch.save({
            "epoch": epoch,
            "model": target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }, path)
        printS(f"saved {path}")
