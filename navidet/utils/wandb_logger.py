"""Thin wandb wrapper.

Once `init_wandb(cfg)` runs on the master rank, anywhere else can call
`wandb_log({...}, step=...)` without having to know whether wandb is enabled
or installed. No-op when disabled / not installed / on non-master ranks.
"""

from __future__ import annotations

from typing import Any, Mapping

from navidet.utils import printS, printW
from navidet.utils.dist_utils import is_main_process

_run = None  # populated by init_wandb on master


def _to_dict(node) -> dict:
    """yacs CfgNode → dict (best-effort)."""
    try:
        return {k: _to_dict(v) for k, v in node.items()} if hasattr(node, "items") else node
    except Exception:
        return {}


def init_wandb(cfg) -> None:
    """Initialize wandb run from yacs cfg. No-op if disabled / not master /
    wandb not installed."""
    global _run
    wb_cfg = getattr(cfg, "wandb", None)
    if wb_cfg is None or not getattr(wb_cfg, "enabled", False):
        return
    if not is_main_process():
        return
    try:
        import wandb
    except ImportError:
        printW("wandb not installed — skipping (`pip install wandb` to enable)")
        return

    run_name = wb_cfg.run_name or None
    entity = wb_cfg.entity or None
    group = getattr(wb_cfg, "group", "") or None
    notes = getattr(wb_cfg, "notes", "") or None
    mode = getattr(wb_cfg, "mode", "online")

    _run = wandb.init(
        project=wb_cfg.project,
        entity=entity,
        name=run_name,
        group=group,
        tags=list(wb_cfg.tags) if wb_cfg.tags else None,
        notes=notes,
        mode=mode,
        config=_to_dict(cfg),
    )
    printS(f"wandb initialized — run={_run.name}  project={wb_cfg.project}")


def wandb_log(data: Mapping[str, Any], step: int | None = None) -> None:
    """Log a dict of scalars / images / etc. No-op when wandb isn't running."""
    if _run is None or not is_main_process():
        return
    try:
        import wandb
        wandb.log(dict(data), step=step)
    except Exception:
        pass


def wandb_finish() -> None:
    """Close the run cleanly at end of training."""
    global _run
    if _run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass
    _run = None


def is_active() -> bool:
    return _run is not None
