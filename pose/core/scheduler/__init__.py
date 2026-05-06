from .torch import (
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    LambdaLR
)

__all__ = [
    'CosineAnnealingLR',
    'StepLR',
    'ExponentialLR',
]

SCHEDULERS = {
    'cosineannealinglr': CosineAnnealingLR,
    'steplr': StepLR,
    'exponentiallr': ExponentialLR,
    'reducelronplateau': ReduceLROnPlateau,
    'cycliclr': CyclicLR,
    'onecyclelr': OneCycleLR,
    'lambdalr': LambdaLR,
}