import torch

from .lr_scheduler import ExponentialDecayLR

__all__ = ["build_optimizer", "build_lr_scheduler", ]


def build_optimizer(cfg, model):
    """
    Build an optimizer from config.
    """
    params = list(model.parameters())
    base_lr = cfg.SOLVER.BASE_LR
    name = cfg.SOLVER.OPTIMIZER_NAME

    if name == "Adam":
        betas = cfg.SOLVER.BETAS
        eps = cfg.SOLVER.EPS
        optimizer = torch.optim.Adam(
            params=params, lr=base_lr, betas=betas, eps=eps)
    else:
        raise NotImplementedError
    
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME

    if name == "ExponentialDecayLR":
        scheduler = ExponentialDecayLR(
            optimizer, 
            decay_steps=cfg.SOLVER.DECAY_STEPS, 
            decay_rate=cfg.SOLVER.DECAY_RATE, 
        )
    else:
        raise NotImplementedError
    
    return scheduler
