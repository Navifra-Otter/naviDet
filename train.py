"""naviDet training entrypoint.

All models go through the EdgeCrafter solver pipeline (EMA + AMP + sync_bn +
CocoEvaluator + KD). Both ecvit (own KD-style backbone) and dinov3 (legacy
ablation) are dispatched via the YAML's `method` block.
"""

import os

# Silence TensorFlow stderr noise (cuDNN/cuFFT/cuBLAS factory warnings,
# oneDNN, TF-TRT, etc.). Some indirect dep (e.g. tensorboard, calflops) drags
# TF in; we don't actually use it. Must be set BEFORE any import that could
# transitively import tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTORCH_JIT", "0")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# torch.cuda.amp.* deprecation chatter triggered by EC code paths we keep
# working — already ported to torch.amp where we own the call site, but the
# decorated registry path still walks through wrappers that emit it.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.")

import signal

import torch.multiprocessing as _mp
try:
    _mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from configs import cfg, parse_args, update_config
from navidet.utils import DISTRIBUTED, printE, printM, signal_handler

signal.signal(signal.SIGINT, signal_handler)


def main():
    # Eager-import for registry side effects (model / data / optim / eval).
    import navidet.core.builder  # noqa: F401
    from navidet.core.registry import YacsYAMLConfig
    from navidet.engine import get_solver_cls
    from navidet.utils import dist_utils
    from navidet.utils.wandb_logger import init_wandb, wandb_finish

    dist_utils.setup_distributed(print_rank=0, print_method="builtin",
                                 seed=getattr(cfg, "seed", 0))

    init_wandb(cfg)  # master-only, no-op if cfg.wandb.enabled is False

    yc = YacsYAMLConfig(cfg)
    kd_enabled = bool(getattr(cfg, "kd", None) and cfg.kd.enabled)
    solver_cls = get_solver_cls(cfg.task, kd_enabled=kd_enabled)
    printM(f"[task] {cfg.task}  [backbone] {cfg.model.backbone_name}  "
           f"[solver] {solver_cls.__name__}  [kd] {kd_enabled}")

    try:
        solver = solver_cls(yc)
        if getattr(cfg, "test", False):
            solver.val()
        else:
            solver.fit()
    finally:
        wandb_finish()
    dist_utils.cleanup()


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    try:
        main()
    except Exception as e:
        printE(e)
    finally:
        if DISTRIBUTED:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
