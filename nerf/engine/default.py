from ..modeling import build_meta_arch
from ..data import build_train_loader
from ..solver import build_optimizer, build_lr_scheduler

__all__ = ["DefaultTrainer", ]


class DefaultTrainer:
    def __init__(self, cfg) -> None:
        self.model = self.build_model(cfg)
        self.data_loader = self.build_train_loader(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        self.iter = self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

    def train(self):
        """
        Run training.
        """
        for self.iter in range(self.start_iter, self.max_iter):
            self.run_step()
    
    def run_step(self):
        data = next(self.data_loader)

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_model(self, cfg):
        """
        Returns:
            torch.nn.Module:
        """
        model = build_meta_arch(cfg)
        print("Model:\n{}".format(model))
        return model
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        """
        return build_train_loader(cfg)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        """
        return build_optimizer(cfg, model)
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        Returns:
            torch.optim.lr_scheduler._LRScheduler:
        """
        return build_lr_scheduler(cfg, optimizer)
