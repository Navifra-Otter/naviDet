"""naviDet training entrypoint.

All models go through the EdgeCrafter solver pipeline (EMA + AMP + sync_bn +
CocoEvaluator + KD). Both ecvit (own KD-style backbone) and dinov3 (legacy
ablation) are dispatched via the YAML's `method` block.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import signal

os.environ["PYTORCH_JIT"] = "0"

from configs import cfg, parse_args, update_config
from navidet.utils import DISTRIBUTED, printE, printM, signal_handler

signal.signal(signal.SIGINT, signal_handler)


def main():
    # Eager-import for registry side effects (model / data / optim / eval).
    import navidet.core.builder  # noqa: F401
    from navidet.core.registry import YacsYAMLConfig
    from navidet.engine import get_solver_cls
    from navidet.utils import dist_utils

    dist_utils.setup_distributed(print_rank=0, print_method="builtin",
                                 seed=getattr(cfg, "seed", 0))

    yc = YacsYAMLConfig(cfg)
    kd_enabled = bool(getattr(cfg, "kd", None) and cfg.kd.enabled)
    solver_cls = get_solver_cls(cfg.task, kd_enabled=kd_enabled)
    printM(f"[task] {cfg.task}  [backbone] {cfg.model.backbone_name}  "
           f"[solver] {solver_cls.__name__}  [kd] {kd_enabled}")

    solver = solver_cls(yc)
    if getattr(cfg, "test", False):
        solver.val()
    else:
        solver.fit()
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
