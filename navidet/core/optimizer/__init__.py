from .torch import (
    SGD, 
    Adam, 
    AdamW, 
    RMSprop, 
    Adadelta, 
    Adagrad, 
    Adamax, 
    LBFGS, 
    NAdam, 
    Ftrl, 
    ASGD, 
    Rprop, 
    SparseAdam
)

__all__ = [
    'SGD',
    'Adam',
    'AdamW',
]

OPTIMEZERS = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
    'rmsprop': RMSprop,
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'adamax': Adamax,
    'lbfgs': LBFGS,
    'nadam': NAdam,
    'ftrl': Ftrl,
    'asgd': ASGD,
    'rprop': Rprop,
    'sparseadam': SparseAdam,
}



# EdgeCrafter integrations.
from .ema import ModelEMA  # noqa: F401
from .amp import GradScaler  # noqa: F401
from . import _ec_optim as ec_optim  # noqa: F401  (registers the YAMLConfig 'optimizer' factory)
