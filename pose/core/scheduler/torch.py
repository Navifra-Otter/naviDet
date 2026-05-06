import torch.optim.lr_scheduler as scheduler

class CosineAnnealingLR(scheduler.CosineAnnealingLR):
    """Cosine Annealing Learning Rate Scheduler with default parameters."""
    
    def __init__(self, optimizer, T_max=50, eta_min=0, last_epoch=-1):
        super(CosineAnnealingLR, self).__init__(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)

class StepLR(scheduler.StepLR):
    """Step Learning Rate Scheduler with default parameters."""
    
    def __init__(self, optimizer, step_size=30, gamma=0.1, last_epoch=-1):
        super(StepLR, self).__init__(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

class ExponentialLR(scheduler.ExponentialLR):
    """Exponential Learning Rate Scheduler with default parameters."""
    
    def __init__(self, optimizer, gamma=0.9, last_epoch=-1):
        super(ExponentialLR, self).__init__(optimizer, gamma=gamma, last_epoch=last_epoch)

class ReduceLROnPlateau(scheduler.ReduceLROnPlateau):
    """ReduceLROnPlateau Scheduler with default parameters."""
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        super(ReduceLROnPlateau, self).__init__(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)

class CyclicLR(scheduler.CyclicLR):
    """Cyclic Learning Rate Scheduler with default parameters."""
    
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
        super(CyclicLR, self).__init__(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down, mode=mode, gamma=gamma, scale_fn=scale_fn, scale_mode=scale_mode, cycle_momentum=cycle_momentum, base_momentum=base_momentum, max_momentum=max_momentum, last_epoch=last_epoch)

class OneCycleLR(scheduler.OneCycleLR):
    """One Cycle Learning Rate Scheduler with default parameters."""
    
    def __init__(self, optimizer, max_lr, total_steps=None, epochs=10, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e4, last_epoch=-1):
        super(OneCycleLR, self).__init__(optimizer, max_lr=max_lr, total_steps=total_steps, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=pct_start, anneal_strategy=anneal_strategy, cycle_momentum=cycle_momentum, base_momentum=base_momentum, max_momentum=max_momentum, div_factor=div_factor, final_div_factor=final_div_factor, last_epoch=last_epoch)

class LambdaLR(scheduler.LambdaLR):
    """Lambda Learning Rate Scheduler with default parameters."""
    
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(LambdaLR, self).__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

