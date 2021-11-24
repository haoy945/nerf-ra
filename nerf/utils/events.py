import logging
import datetime
from collections import defaultdict

import torch
from fvcore.common.history_buffer import HistoryBuffer

__all__ = ["EventWriter", ]


class EventWriter:
    def __init__(self, max_iter: int = None, window_size: int = 20) -> None:
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter
        self._window_size = window_size
        self._history = defaultdict(HistoryBuffer)

    def store(
        self, *, loss: dict, data_time: float, total_time: float, lr: float, 
        **kwargs,
    ) -> None:
        """
        Add multiple values to the buffer. 
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss.items()}
        if "total_loss" not in metrics_dict:
            metrics_dict['total_loss'] = sum(metrics_dict.values())
        metrics_dict["data_time"] = data_time
        metrics_dict["total_time"] = total_time
        metrics_dict["lr"] = lr

        for k, v in metrics_dict.items():
            self._store(k, v)

    def _store(self, name, value):
        """
        Add the `value` to the `HistoryBuffer` associated with `name`.
        """
        history = self._history[name]
        value = float(value)
        history.update(value)

    def _get_eta(self, iteration):
        if self._max_iter is None:
            return ""
        eta_seconds = self._history["total_time"].median(1000) * (self._max_iter - iteration - 1)
        return str(datetime.timedelta(seconds=int(eta_seconds)))

    def write(self, iteration):
        if iteration >= self._max_iter:
            # reports training progress only
            return

        try:
            data_time = self._history["data_time"].avg(20)
        except KeyError:
            data_time = None
        try:
            iter_time = self._history["total_time"].global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = "{:.6f}".format(self._history["lr"].latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        eta_string = self._get_eta(iteration)

        self.logger.info(
            " {eta}iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.4f}".format(k, v.median(self._window_size))
                        for k, v in self._history.items()
                        if "loss" in k
                    ]
                ),
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )
