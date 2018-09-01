"""
utility functions for handling hyperparams / logging
"""

import logging
from collections import Iterable


def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x


def set_logger(name, log_path):
    """logging 设置使得同时输出到文件和console

    Arguments:
        name {str} -- logger的名字 建议使用__name__
        log_path {str} -- log 文件的地址

    Returns:
        {logging object} -- 设置好的logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # log to log file
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter(
            "%(asctime)s:%(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        # log to console
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        return logger
