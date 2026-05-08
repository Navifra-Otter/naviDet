"""yacs CfgNode → YAMLConfig adapter.

EdgeCrafter's BaseSolver / ECSolver / PoseSolver consume a `YAMLConfig`-like
object whose lazy properties (model, criterion, optimizer, train_dataloader, …)
build modules through the `create()` registry. We feed those properties from a
yacs cfg by:

  1. Flattening `cfg.method` into `self.yaml_cfg` (a plain dict).
  2. Inheriting `YAMLConfig`'s property machinery — it does the rest.

The yacs side keeps its existing role for non-method knobs (gpus, seed,
trainer.save_path, etc.), but anything the EC solver needs MUST live under
`cfg.method` in the YAML.
"""

from __future__ import annotations

import copy
from typing import Any

from ._config import BaseConfig
from .yaml_config import YAMLConfig
from .yaml_utils import merge_dict


def _yacs_to_dict(node) -> Any:
    """Recursively materialize a yacs CfgNode into a plain dict.

    Lists / tuples are preserved; primitives pass through.
    """
    if hasattr(node, "items") and not isinstance(node, dict):
        node = dict(node)
    if isinstance(node, dict):
        return {k: _yacs_to_dict(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return type(node)(_yacs_to_dict(v) for v in node)
    return node


class YacsYAMLConfig(YAMLConfig):
    """Drop-in YAMLConfig backed by a yacs cfg.

    The constructor skips YAMLConfig's file-loading / `reset_cfg` logic — we
    receive an already-merged yacs cfg and convert its `method` subtree.
    """

    def __init__(self, yacs_cfg, extra: dict | None = None) -> None:
        BaseConfig.__init__(self)

        method = getattr(yacs_cfg, "method", None)
        if method is None:
            raise ValueError(
                "YacsYAMLConfig requires `cfg.method` (configs/method/<task>/<variant>.yaml). "
                "Use the simple Builder/Trainer flow (--simple) for legacy configs."
            )
        cfg_dict = _yacs_to_dict(method)
        if extra:
            cfg_dict = merge_dict(cfg_dict, extra)

        self.yaml_cfg = copy.deepcopy(cfg_dict)
        self._yacs = yacs_cfg

        # Convenience: surface a few yacs top-level keys the solver may want.
        if hasattr(yacs_cfg, "task") and "task" not in self.yaml_cfg:
            self.yaml_cfg["task"] = yacs_cfg.task
        if hasattr(yacs_cfg, "trainer") and "epoches" not in self.yaml_cfg:
            self.yaml_cfg["epoches"] = int(yacs_cfg.trainer.epochs)
        if hasattr(yacs_cfg, "trainer") and "output_dir" not in self.yaml_cfg:
            self.yaml_cfg["output_dir"] = yacs_cfg.trainer.save_path

        # Top-level model / criterion / postprocessor names — required by
        # YAMLConfig's lazy properties. Default from the task when missing.
        _DEFAULTS = {
            "pose":         {"model": "ECPose", "criterion": "DETRPoseCriterion",
                             "postprocessor": "DETRPosePostProcessor"},
            "detection":    {"model": "ECDet",  "criterion": "ECCriterion",
                             "postprocessor": "PostProcessor"},
            "segmentation": {"model": "ECSeg",  "criterion": "ECCriterion",
                             "postprocessor": "PostProcessor"},
        }
        defaults = _DEFAULTS.get(self.yaml_cfg.get("task", ""), {})
        for k, v in defaults.items():
            self.yaml_cfg.setdefault(k, v)

        # Pull EC-style top-level YAML blocks BEFORE applying scalar defaults
        # so user values win over defaults.
        TOP_LEVEL_EC_KEYS = (
            "optimizer", "lr_scheduler", "lr_warmup_scheduler",
            "ema", "scaler", "evaluator",
            "train_dataloader", "val_dataloader",
            "use_amp", "use_ema", "clip_max_norm",
            "print_freq", "checkpoint_freq",
            "sync_bn", "find_unused_parameters",
            "grad_accum_steps", "use_focal_loss",
            "eval_spatial_size",
        )
        for k in TOP_LEVEL_EC_KEYS:
            if k in self.yaml_cfg:
                continue
            if hasattr(yacs_cfg, k):
                v = getattr(yacs_cfg, k)
                self.yaml_cfg[k] = _yacs_to_dict(v) if hasattr(v, "items") else v

        # Scalar defaults the EC solver expects in yaml_cfg. Users override via
        # the method block. Anything left absent gets a sensible value here so
        # that minimal configs still work end-to-end.
        scalar_defaults = {
            "use_amp": False,
            "use_ema": False,
            "clip_max_norm": 0.1,
            "print_freq": 100,
            "checkpoint_freq": 5,
            "sync_bn": False,
            "find_unused_parameters": False,
            "grad_accum_steps": 1,
            "use_focal_loss": True,
            "device": "",
            "writer_output_dir": None,
            "summary_dir": None,
            # FLOPs / param-count profiler input shape; matches the YAML's
            # final Resize size if not overridden.
            "eval_spatial_size": [640, 640],
        }
        for k, v in scalar_defaults.items():
            self.yaml_cfg.setdefault(k, v)

        # Surface KD spec from yacs into yaml_cfg so KD solvers see it.
        if hasattr(yacs_cfg, "kd") and "kd" not in self.yaml_cfg:
            self.yaml_cfg["kd"] = _yacs_to_dict(yacs_cfg.kd)

        # EC's `create()` requires every block it instantiates to carry a
        # `type:` (or be itself a registered class entry). Inject sensible
        # defaults so users don't have to repeat them in every YAML.
        for dl_key in ("train_dataloader", "val_dataloader"):
            block = self.yaml_cfg.get(dl_key)
            if isinstance(block, dict):
                block.setdefault("type", "DataLoader")
                if isinstance(block.get("collate_fn"), dict):
                    block["collate_fn"].setdefault("type", "BatchImageCollateFunction")
                if isinstance(block.get("dataset"), dict):
                    block["dataset"].setdefault("type", "CocoDetection")
                    transforms = block["dataset"].get("transforms")
                    if isinstance(transforms, dict):
                        transforms.setdefault("type", "Compose")
        for top_key, default_type in (
            ("ema", "ModelEMA"),
            ("scaler", "GradScaler"),
            ("evaluator", "CocoEvaluator"),
            ("lr_warmup_scheduler", "LinearWarmup"),
        ):
            block = self.yaml_cfg.get(top_key)
            if isinstance(block, dict):
                block.setdefault("type", default_type)

        # Sync BaseConfig public attrs from the now-fully-populated yaml_cfg
        # (output_dir / use_amp / sync_bn / device / clip_max_norm / etc.). This
        # mirrors YAMLConfig.__init__'s final loop, but runs after our merging
        # so EC-required scalars are visible.
        for k in list(BaseConfig().__dict__):
            if not k.startswith("_") and k in self.yaml_cfg:
                self.__dict__[k] = self.yaml_cfg[k]

        # EC's BaseSolver / pose_trainer reads `cfg.epoches` (D-FINE typo),
        # not `cfg.epochs`. BaseConfig only declares `epochs`, so the sync
        # loop above misses `epoches`. Set it explicitly.
        if "epoches" in self.yaml_cfg:
            self.epoches = self.yaml_cfg["epoches"]
            # mirror to `epochs` too in case other code uses the corrected name
            self.epochs = self.yaml_cfg["epoches"]

    def __getattr__(self, name):
        """Fallback: surface any yaml_cfg key as an attribute.

        Only triggers when normal attribute lookup fails — BaseConfig's
        declared attrs (model / criterion / optimizer / output_dir / use_amp /
        …) keep their existing semantics. Avoids having to enumerate every
        scalar EC's solver code might touch (`grad_accum_steps`,
        `clip_max_norm`, `print_freq`, `sync_bn`, …).
        """
        if name.startswith("_"):
            raise AttributeError(name)
        # Avoid recursion if yaml_cfg itself isn't initialized yet.
        yaml_cfg = self.__dict__.get("yaml_cfg")
        if yaml_cfg is not None and name in yaml_cfg:
            return yaml_cfg[name]
        raise AttributeError(name)

    # YAMLConfig's reset_cfg expects very specific keys — we skip it.
    def reset_cfg(self):  # noqa: D401  (intentionally a no-op)
        return None
