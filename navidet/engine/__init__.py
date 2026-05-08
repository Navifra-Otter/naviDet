"""naviDet engine — task-aware solver dispatch.

`TASKS[cfg.task]` returns the solver class (or its KD-aware subclass when
`cfg.kd.enabled` is True).
"""

from typing import Dict, Type

from .det_trainer import ECSolver
from .pose_trainer import PoseSolver
from .kd_solver import KDECSolver, KDPoseSolver
from ._solver import BaseSolver


TASKS: Dict[str, Type[BaseSolver]] = {
    "detection": ECSolver,
    "segmentation": ECSolver,
    "pose": PoseSolver,
}

KD_TASKS: Dict[str, Type[BaseSolver]] = {
    "detection": KDECSolver,
    "segmentation": KDECSolver,
    "pose": KDPoseSolver,
}


def get_solver_cls(task: str, kd_enabled: bool = False) -> Type[BaseSolver]:
    table = KD_TASKS if kd_enabled else TASKS
    if task not in table:
        raise ValueError(f"Unsupported task {task!r}. Expected one of {list(TASKS)}.")
    return table[task]
