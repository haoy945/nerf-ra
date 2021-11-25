import os
import time
import weakref
import logging
from fvcore.common.checkpoint import Checkpointer

from ..modeling import build_meta_arch
from ..data import build_train_loader, build_test_loader
from ..solver import build_optimizer, build_lr_scheduler
from ..utils import setup_logger, EventWriter
from ..evaluation import inference, DatasetEvaluator

__all__ = ["default_setup", "DefaultTrainer", ]


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the nerf logger
    2. Log basic information about cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    setup_logger(output=output_dir)
    logger = logging.getLogger("nerf")

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                open(args.config_file, "r").read(),
            )
        )

    path = os.path.join(output_dir, "config.yaml")
    logger.info("Running with full config:\n{}".format(cfg.dump(), ".yaml"))
    with open(path, 'w') as f:
        f.write(cfg.dump())
    logger.info("Full config saved to {}".format(path))


class DefaultTrainer:
    def __init__(self, cfg) -> None:
        """
        Args:
            cfg (CfgNode):
        """
        # setup logger
        logger = logging.getLogger("nerf")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger(output=cfg.OUTPUT_DIR)

        self.model = self.build_model(cfg)
        self.data_loader = self.build_train_loader(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        self.writer = self.build_writer(cfg)
        self.checkpointer = Checkpointer(
            self.model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        self.iter = self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def train(self):
        """
        Run training.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        for self.iter in range(self.start_iter, self.max_iter):
            self.run_step()

            if (self.iter + 1) % 20 == 0 or (self.iter + 1) == self.max_iter:
                self.writer.write(self.iter)
            if (self.iter + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 or (self.iter + 1) == self.max_iter:
                save_file_name = "model_{:06d}".format(self.iter)
                self.checkpointer.save(save_file_name)
            if (self.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0 or (self.iter + 1) == self.max_iter:
                self.test(self.cfg, self.model, self.iter)

    def run_step(self):
        start = time.perf_counter()

        # loading data
        data = next(self.data_loader)
        data_time = time.perf_counter() - start

        # running model
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        # optimizing
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        self.scheduler.step()

        total_time = time.perf_counter() - start
        self.writer.store(**{
            "loss": loss_dict, 
            "data_time": data_time,
            "total_time": total_time,
            "lr": self.scheduler._last_lr[0],
        })

    @classmethod
    def build_model(self, cfg):
        """
        Returns:
            torch.nn.Module:
        """
        model = build_meta_arch(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        """
        return build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        """
        Returns:
            iterable
        """
        return build_test_loader(cfg)

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

    @classmethod
    def build_writer(cls, cfg):
        """
        Returns:
            torch.optim.lr_scheduler._LRScheduler:
        """
        max_iter = cfg.SOLVER.MAX_ITER
        return EventWriter(max_iter=max_iter)

    @classmethod
    def test(cls, cfg, model, iteration):
        logger = logging.getLogger(__name__)
        data_loader = cls.build_test_loader(cfg)
        evaluator = DatasetEvaluator(cfg, iteration)
        results = inference(model, data_loader, evaluator)

        logger.info("Evaluation results:")
        logger.info("MSE : {:.4f}".format(results['MSE']))
        logger.info("PSNR: {:.2f}".format(results['PSNR']))
        return results

    def state_dict(self):
        ret = {
            "iteration": self.iter,
            "LRScheduler": self.scheduler.state_dict(),
        }
        return ret

    def load_state_dict(self, state_dict):
        self.iter = state_dict["iteration"]
        self.scheduler.load_state_dict(state_dict["LRScheduler"])
