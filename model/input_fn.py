"""
Define the input data pipeline
"""

import os

import tensorflow as tf

CPU_COUNT = os.cpu_count()
_SHUFFLE_BUFFER = 120000


def _set_static_shape(t, shape):
  t.set_shape(shape)
  return t


def _cascade_label_set_shape(dataset, label_flat_length, label_num):
  dataset = dataset.map(lambda line: tf.reshape(line, (-1, label_num)),
                        num_parallel_calls=CPU_COUNT)

  data_shape = [int(label_flat_length / label_num), label_num]
  dataset = dataset.map(lambda line: _set_static_shape(line, data_shape),
                        num_parallel_calls=CPU_COUNT)
  return dataset


def build_dataset(file_path,
                  length,
                  padding=False,
                  cascading_label=False,
                  label_num=4):
  """创建子dataset

  Args:
      file_path (str): 文件名, 若None, 则生成伪标签(for inference)
      length (int): 一行包含的元素数量
      padding (bool, optional): Defaults to False.如果为True, 则当一行中实际
          元素数量 < length时，会用0填充
      cascading_label (bool, optional): Defaults to False. 如果True，则将label堆叠
          成二维
      label_num (int, optional): Defaults to None. 当cascading_label为True时，
          label_num为最里层维度的数量

  Returns:
      Dataset: 返回dataset
  """

  def _label_generator(size):
    while True:
      yield [0] * size

  if file_path is None:
    dataset = tf.data.Dataset.from_generator(
        lambda: _label_generator(length),
        output_types=tf.int32,
        output_shapes=tf.TensorShape([length]))
  else:
    dataset = tf.data.TextLineDataset(file_path)
    dataset = (dataset
               .map(lambda string: tf.string_split(
                   [string], delimiter=",").values,
                   num_parallel_calls=CPU_COUNT)
               .map(lambda strings: tf.string_to_number(strings, tf.int32),
                    num_parallel_calls=CPU_COUNT))

  if padding:  # 填充0至length长度
    dataset = dataset.map(lambda line: tf.pad(
        line, [[0, length - tf.shape(line)[0]]], constant_values=0),
        num_parallel_calls=CPU_COUNT)

  if cascading_label:
    dataset = dataset.map(lambda line: tf.reshape(line, (-1, label_num)),
                          num_parallel_calls=CPU_COUNT)

  # 给dynamic tensor 提供 static shape 以方便后续使用
  data_shape = [length] if not cascading_label else [
      int(length / label_num), label_num]
  dataset = dataset.map(lambda line: _set_static_shape(line, data_shape),
                        num_parallel_calls=CPU_COUNT)
  return dataset


def input_fn(sentences,
             labels,
             batch_size=1,
             is_training=False,
             is_test=False,
             repeat_count=1,
             prefetch=2):
  """得到features & labels

  Args:
      sentences (dataset): tf.data.Dataset 对象
      labels (dataset): tf.data.Dataset 对象
      batch_size (int, optional): Defaults to 1. batch大小
      is_training (bool, optional): Defaults to False. 是否为训练
      is_test (bool, optional): Defaults to False. 是否为推断
      repeat_count (int, optional): Defaults to 1. 若None, 则无限循环
      prefetch (int, optional): Defaults to 2. 预备到pipeline的数量

  Returns:
     features & labels
  """

  dataset = tf.data.Dataset.zip((sentences, labels))

  if is_training:
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = (dataset
             .repeat(count=repeat_count)
             .batch(batch_size, drop_remainder=True)
             .prefetch(prefetch))

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels
