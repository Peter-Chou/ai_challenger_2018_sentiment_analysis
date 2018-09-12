"""
where you define the input data pipeline
"""

import tensorflow as tf


def build_dataset(file_path, length, padding=False):
    """创建子dataset

    Args:
        file_path (str): 文件名
        length (int): 一行包含的元素数量
        padding (bool, optional): Defaults to False.如果为True, 则当一行中实际
            元素数量 < length时，会用0填充

    Returns:
        Dataset: 返回dataset
    """

    def _set_static_shape(t, shape):
        t.set_shape(shape)
        return t

    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda string: tf.string_split(
        [string], delimiter=",").values)
    dataset = dataset.map(
        lambda strings: tf.string_to_number(strings, tf.int32))

    if padding:  # 填充0至length长度
        dataset = dataset.map(lambda line: tf.pad(
            line, [[0, length - tf.shape(line)[0]]], constant_values=0))

    # 给dynamic tensor 提供 static shape 以方便后续使用
    dataset = dataset.map(lambda line: _set_static_shape(line, [length]))
    return dataset
