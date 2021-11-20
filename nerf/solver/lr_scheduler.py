from typing import List
import torch

__all__ = ["ExponentialDecayLR", ]


class ExponentialDecayLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        decay_steps: int,
        decay_rate: float,
        last_iter: int = -1
    ) -> None:
        """
        Args:
            optimizer, last_iter: See ``torch.optim.lr_scheduler._LRScheduler``.
                ``last_iter`` is the same as ``last_epoch``.
            decay_steps: exponential learning rate decay steps.
            decay_rate: exponential learning rate decay rate.
        """
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> List[float]:
        multiplier = self.decay_rate ** (self.last_epoch / self.decay_steps)
        return [base_lr * multiplier for base_lr in self.base_lrs]
