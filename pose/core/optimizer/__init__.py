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

