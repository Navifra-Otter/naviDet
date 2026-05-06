import torch.optim as optim

class SGD(optim.SGD):
    """SGD optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0):
        super(SGD, self).__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

class Adam(optim.Adam):
    """Adam optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(Adam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

class AdamW(optim.AdamW):
    """AdamW optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super(AdamW, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

class RMSprop(optim.RMSprop):
    """RMSprop optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0):
        super(RMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)

class Adadelta(optim.Adadelta):
    """Adadelta optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        super(Adadelta, self).__init__(params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

class Adagrad(optim.Adagrad):
    """Adagrad optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
        super(Adagrad, self).__init__(params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)

class Adamax(optim.Adamax):
    """Adamax optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(Adamax, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

class LBFGS(optim.LBFGS):
    """LBFGS optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100, line_search_fn=None):
        super(LBFGS, self).__init__(params, lr=lr, max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size, line_search_fn=line_search_fn)

class NAdam(optim.NAdam):
    """NAdam optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(NAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

class Ftrl(optim.Ftrl):
    """Ftrl optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1e-2, l1=0.0, l2=0.0, lr_power=-0.5, initial_accumulator_value=0.1):
        super(Ftrl, self).__init__(params, lr=lr, l1=l1, l2=l2, lr_power=lr_power, initial_accumulator_value=initial_accumulator_value)

class ASGD(optim.ASGD):
    """ASGD optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        super(ASGD, self).__init__(params, lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)

class Rprop(optim.Rprop):
    """Rprop optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        super(Rprop, self).__init__(params, lr=lr, etas=etas, step_sizes=step_sizes)


class SparseAdam(optim.SparseAdam):
    """SparseAdam optimizer with default parameters suitable for most models."""
    
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        super(SparseAdam, self).__init__(params, lr=lr, betas=betas, eps=eps)


