"""
utility functions for handling hyperparams / logging
"""

import inspect
import json
import logging
from collections import Iterable

import yaml


def flatten(items, ignore_types=(str, bytes)):
  """展开成一维可迭代的生成器

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


class Params():
  """Class that loads hyperparameters from a yaml file.
  Example:
  ```
  params = Params(yaml_path)
  print(params.learning_rate)
  params.learning_rate = 0.5  # change the value of learning_rate in params
  ```
  """

  def __init__(self, yaml_path):
    self.update(yaml_path)

  def save(self, yaml_path):
    """Saves parameters to yaml file"""
    with open(yaml_path, 'w') as f:
      yaml.dump(self.__dict__, f)

  def update(self, yaml_path):
    """Loads parameters from yaml file"""
    with open(yaml_path) as f:
      params = yaml.load(f)
      self.__dict__.update(params)

  @property
  def dict(self):
    """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
    return self.__dict__


def save_dict_to_yaml(d, yaml_path):
  """Saves dict of floats in yaml file
  Args:
      d: (dict) of float-castable values (np.float, int, float, etc.)
      yaml_path: (string) path to yaml file
  """
  with open(yaml_path, 'w', encoding='utf-8') as f:
    # We need to convert the values to float for yaml (it doesn't accept np.array, np.float, )
    d = {k: float(v) for k, v in d.items()}

    yaml.dump(d, f, encoding='utf-8', indent=4,
              default_flow_style=False, allow_unicode=True)


def set_logger(log_path, file_logging_level="INFO", console_logging_level="INFO",
               name=None):
  """logging 设置使得同时输出到文件和console

  Args:
      log_path (str): log 文件的地址
      file_logging_level (str): "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" 中一个
      console_logging_level (str): "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" 中一个
      name (str): logger的名字 当None时，默认使用module 的__name__

  Return:
      (logging object): 设置好的logger
  """

  frm = inspect.stack()[1]
  mod = inspect.getmodule(frm[0])
  if name is None:
    # use calling module's __name__
    logger = logging.getLogger(mod.__name__)
  else:
    logger = logging.getLogger(name)

  logger.setLevel(getattr(logging, file_logging_level.upper()))

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
    stream_handler.setLevel(
        getattr(logging, console_logging_level.upper()))
    logger.addHandler(stream_handler)
    return logger
