"""
utility functions for handling hyperparams / logging
"""

import json
import logging
from collections import Iterable


def flatten(items, ignore_types=(str, bytes)):
    """

    Args:
        items (iterable): 可以迭代展开的对象
        ignore_types (class): Defaults to (str, bytes). 想要忽略的可迭代类

    Yield:
        不可迭代或忽略列表中的元素
    """

    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x


def set_logger(name, log_path):
    """logging 设置使得同时输出到文件和console

    Args:
        name (str): logger的名字 建议使用__name__
        log_path (str): log 文件的地址

    Return:
        (logging object): 设置好的logger
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


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w', encoding='utf-8') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4, ensure_ascii=False)
