import os
import sys
import functools
import logging
from termcolor import colored

__all__ = ["setup_logger", ]


class _ColorfulFormatter(logging.Formatter):
    def formatMessage(self, record):
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(output=None, *, color=True, name="nerf"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
    )

    # create stdout handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    if color:
        ch_formatter = _ColorfulFormatter(
            fmt=colored('[%(asctime)s %(name)s]: ', 'cyan') + '%(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
        )
    else:
        ch_formatter = formatter
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # create file handler
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            filename = output
        else:
            filename = os.path.join(output, 'log.txt')

        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
