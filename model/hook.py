import time

import tensorflow as tf

from model.helper import set_logger

logger = set_logger('train.log')


class _LoggerHook(tf.train.SessionRunHook):
  """logs loss and steps"""

  def __init__(self, loss, gstep, lr, print_n_step):
    self._loss = loss
    self._print_n_step = print_n_step
    self._gstep = gstep
    self._lr = lr

  def begin(self):
    self._step = -1
    self._start_time = time.time()

  def before_run(self, run_context):
    self._step += 1
    return tf.train.SessionRunArgs([self._loss, self._gstep, self._lr])

  def after_run(self, run_context, run_values):
    if self._step % self._print_n_step == 0:
      current_time = time.time()
      duration = current_time - self._start_time
      self._start_time = current_time

      loss_value = run_values.results[0]
      current_gstep = run_values.results[1]
      lr = run_values.results[2]
      logger.info(
          f"step: {current_gstep:>7}\t loss: {loss_value:.2f}\t lr: {lr:0.7f}\t spent: {duration:.1f} seconds")
