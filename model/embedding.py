import numpy as np
import tensorflow as tf


def word_embedding(inputs, vector_path, scale=False, scope="word_embedding"):
  """word index -> 词向量转换

  Args:
      inputs (2d tensor): batch中每个句子的word index
      vector_path (str): 预训练中文词向量的npy文件地址
      scale (bool, optional): Defaults to False. 是否对词向量进行缩放
      scope (str, optional): Defaults to "word_embedding". scope名称

  Returns:
      3d tensor: (n, sentence_length, embed_size)
  """

  pretrained_embs = np.load(vector_path)
  embed_size = pretrained_embs.shape[1]

  with tf.variable_scope(scope):
    pretrained_embs = tf.get_variable(
        name="embs_pretrained",
        initializer=tf.constant_initializer(
            np.asarray(pretrained_embs), dtype=tf.float32),
        shape=pretrained_embs.shape, trainable=False)

    num_vector = tf.get_variable(
        name="NUM",
        shape=[1, embed_size],
        initializer=tf.random_uniform_initializer(-0.04, 0.04),
        trainable=True)

    lookup_table = tf.concat(
        [pretrained_embs, num_vector], axis=0)

    outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    if scale:
      outputs = outputs * (embed_size ** 0.5)

  return outputs


def position_embedding(inputs,
                       num_units,
                       mask_pad=False,
                       scale=True,
                       scope="position_embedding",
                       reuse=None):
  '''Sinusoidal Positional_Encoding.
  Args:
    inputs: A 3d Tensor with shape of (batch, N, T).
    num_units: Output dimensionality
    mask_pad: Boolean. If True, <pad>:0 will be ignored (replaced by zero vectors)
    scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
  Returns:
      A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
  '''

  # N 为 batch size; T 为 max sentence length
  N, T = inputs.get_shape().as_list()
  with tf.variable_scope(scope, reuse=reuse):
    position_ind = tf.tile(tf.expand_dims(
        tf.range(1, T + 1), 0), [N, 1])  # (N, T)

    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
        for pos in range(T)])  # (T, num_units)

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    # Convert to a tensor
    lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

    lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                              lookup_table), 0)

    # * 如果是pad（id为0）则不给予位置信息 (added)
    if mask_pad:
      pad_mask = tf.cast(tf.not_equal(inputs, 0), tf.int32)
      position_ind = position_ind * pad_mask

    outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

    if scale:
      outputs = outputs * num_units**0.5

    return outputs
