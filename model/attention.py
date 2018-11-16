# -*- coding: utf-8 -*-

"""
创建模型所需的函数
transformer part 代码主要借鉴自：
https://github.com/Kyubyong/transformer/blob/master/modules.py
"""

import numpy as np
import tensorflow as tf


# ###################################################
# Self Attention Part
def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
  '''Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  '''
  with tf.variable_scope(scope, reuse=reuse):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.get_variable("beta",
                           shape=params_shape,
                           dtype=tf.float32,
                           initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma",
                            shape=params_shape,
                            dtype=tf.float32,
                            initializer=tf.ones_initializer())
    # 原实现代码
    # beta = tf.Variable(tf.zeros(params_shape))
    # gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

  return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        kernel_initializer=None,
                        kernel_regularizer=None,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
  '''Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    num_units: A scalar. Attention size.
    dropout_rate: A floating point number.
    kernel_initializer: weight initializer
    kernel_regularizer: weight regularizer
    is_training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # Linear projections
    # ? remove activation?
    # 使得不同的multi-head attention 都回应不同的query，以及句子的不同子理解
    Q = tf.layers.dense(queries,
                        num_units,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activation=tf.nn.relu)  # (N, T_q, C)
    K = tf.layers.dense(keys,
                        num_units,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activation=tf.nn.relu)  # (N, T_k, C)
    V = tf.layers.dense(keys,
                        num_units,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activation=tf.nn.relu)  # (N, T_k, C)

    # Split and concat
    # 将Q, K, V 分成 num_heads 个子集合
    # Q_ = {Q1, Q2,...,Qh}
    # K_ = {K1, K2, ..., Kh}
    # V_  = {V1, V2, ..., Vh}
    # {Qi, Ki, Vi} 对应 head_i 的 scale dot-product attention 的三个输入
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2),
                   axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, C/h)

    # Multiplication
    # 每个 head的query 与 keys矩阵相乘并拼接成一个大矩阵
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

    # Scale
    # 防止 数值过大落入到softmax的饱和区域
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # Key Masking
    # 使得与该query无关的keys不要参与回答问题，(将其MatMul得到的值设为极小，
    # 则Softmax后得到的权重近似 0)
    # 对于key里某向量全为0时，则将其key-value 对应的权重设为近似0的极小数
    # 源代码实现中： <PAD>  为 zeros vector
    # tf.sign(x) = -1 if x < 0
    # tf.sign(x) = 0 if x == 0 or tf.is_nan(x)
    # tf.sign(x) = 1 if x > 0
    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [
                        1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2**32 + 1)

    # 如果是 0 （即<pad>）则返回极小值，使得softmax得出的权重接近0
    # 这样使得对sentence的query，<pad>不具有发言权
    outputs = tf.where(tf.equal(key_masks, 0), paddings,
                       outputs)  # (h*N, T_q, T_k)

    # Causality = Future blinding
    # tf.linalg.LinearOperatorLowerTriangular:
    # [[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 10],[10, 11, 12, 13]]的tril:
    #  ################################################################
    # [[ 1.  0.  0.  0.]
    #  [ 4.  5.  0.  0.]
    #  [ 7.  8.  9.  0.]
    #  [10. 11. 12. 13.]]
    if causality:
      diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
      tril = tf.linalg.LinearOperatorLowerTriangular(
          diag_vals).to_dense()  # (T_q, T_k)
      masks = tf.tile(tf.expand_dims(tril, 0), [
                      tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

      paddings = tf.ones_like(masks) * (-2**32 + 1)
      outputs = tf.where(tf.equal(masks, 0), paddings,
                         outputs)  # (h*N, T_q, T_k)

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    # 将零含义的quey得到的attention设为0向量（去噪）
    # 原理与key masking相同
    query_masks = tf.sign(
        tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(
        query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (N, T_q, T_k)

    # Dropouts
    # 此时的outputs仍是 key-value 对应的权重,对该权重集合进行dropout
    # 使得与value相乘得到的attention更加robust（即不仅仅依赖极少数的value）
    outputs = tf.layers.dropout(
        outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Weighted sum
    # 此时的outputs是 attention = w1*v1 + ... + w_Tk*v_Tk
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    # 将 head_i 都concat起来, 因为结果维度与inputs(即query)相同,所以不需要全连接调整维度
    outputs = tf.concat(tf.split(outputs, num_heads,
                                 axis=0), axis=2)  # (N, T_q, C)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = normalize(outputs, scope="ln_1")  # (N, T_q, C)
    # outputs = normalize(outputs)  # (N, T_q, C)

  return outputs


def feedforward(inputs,
                kernel_initializer=None,
                kernel_regularizer=None,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
  '''Point-wise feed forward net.

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    kernel_initializer: weight initializer
    kernel_regularizer: weight regularizer
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Inner layer
    # kernel_size = 1 即 element-wise (T中每个元素从C维变换到num_units[0]维)
    params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
              "activation": tf.nn.relu, "use_bias": True,
              "kernel_initializer": kernel_initializer,
              "kernel_regularizer": kernel_regularizer}
    outputs = tf.layers.conv1d(**params)

    # Readout layer
    params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
              "activation": None, "use_bias": True,
              "kernel_initializer": kernel_initializer,
              "kernel_regularizer": kernel_regularizer}
    outputs = tf.layers.conv1d(**params)

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = normalize(outputs, scope="ln_2")
    # outputs = normalize(outputs)

  return outputs


# ###################################################
# CNN Part
def conv_maxpool(inputs,
                 filter_size,
                 num_filters,
                 hidden_size,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 scope="conv_maxpool",
                 reuse=None):
  """conv + maxpool

  以attention为单位，按filter_size单位的attentions为窗口卷积params.num_filters个feature map,
  对每个feature map求整体最大值，作为这个feature map的特征值

  Args:
      inputs (4d tensor): (N, T, C, d), d一般是1
      filter_size (int): filter的高度（宽度默认与attention的维度相同,这里可以理解为
          选用filter_size个attention作为filter的窗口, 每次按一个attention滑动
      num_filters (int): 一个filter生成的feature map数量
      hidden_size (int): attention的维度
    kernel_initializer: weight initializer
      kernel_regularizer: weight regularizer
      scope (str, optional):  Defaults to "conv_maxpool". scope名称
      reuse (Bool, optional): Defaults to None. 是否重复是否该命名域的变量

  Returns:
      4d tensor: (N, 1, 1, params.num_filters)
  """

  with tf.variable_scope(scope, reuse=reuse):
    inputs_height = inputs.get_shape().as_list()[1]

    # conv 为 (n, new_height, 1, params.num_filters)
    # new_height = (inputs_height - filter_size / stride_height) + 1
    conv = tf.layers.conv2d(
        inputs,
        filters=num_filters,
        kernel_size=(filter_size, hidden_size),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        activation=tf.nn.relu)

    # pool 为 (n, 1, 1, params.num_filters)
    # 将一个feature map 池化为一个特征值
    pool = tf.layers.max_pooling2d(
        conv,
        pool_size=(inputs_height - filter_size + 1, 1),
        strides=(1, 1))
    return pool


def inception(inputs,
              filter_size_list,
              num_filters,
              hidden_size,
              kernel_initializer=None,
              kernel_regularizer=None,
              scope="inception",
              reuse=None):
  """将不同filter_size得到的不同特征组合在一起并返回

  Args:
      inputs (4d tensor): (N, T, C, d), d一般是1
      filter_size_list (A list of int): 含有多个filter_size的list
      num_filters (int): 一个filter生成的feature map数量
      hidden_size (int): attention的维度
    kernel_initializer: weight initializer
      kernel_regularizer: weight regularizer
      scope (str, optional):  Defaults to "conv_maxpool". scope名称
      reuse (Bool, optional): Defaults to None. 是否重复是否该命名域的变量

  Returns:
      4d tensor: (N, 1, 1, len(params.filter_size_list) * params.num_filters)
  """

  with tf.variable_scope(scope, reuse=reuse):
    pooled_outputs = []
    for filter_size in filter_size_list:
      feature = conv_maxpool(inputs,
                             filter_size=filter_size,
                             num_filters=num_filters,
                             hidden_size=hidden_size,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer,
                             scope=f"conv_maxpool_{filter_size}_filter",
                             reuse=reuse)
      pooled_outputs.append(feature)

  # (N, 1, 1, len(params.filter_sizes) * params.num_filters)
  return tf.concat(pooled_outputs, -1)


def dense_logits(inputs,
                 label_num,
                 kernel_regularizer,
                 kernel_initializer=None,
                 scope="dense_logits",
                 inner_dense_outshape=None,
                 inner_dense_activation=tf.nn.relu,
                 use_bias=True,
                 reuse=None
                 ):
  """经过（多层）dense输出category各个label class的logits

  Args:
      inputs (2d tensor): 特征向量：(n, feature_num)
      label_num (int): label class的数量
      kernel_regularizer (regularizer): 矩阵w的约束器
    kernel_initializer: weight initializer
      scope (str, optional): Defaults to "dense_logits". socpe namespace
      inner_dense_outshape (list, optional): Defaults to None.
          若为None / []，则没有中间的dense层
      inner_dense_activation (operation, optional): Defaults to tf.nn.relu
      use_bias (bool, optional): Defaults to True. 是否在所有层中使用偏置
      reuse (Bool, optional): Defaults to None. 是否重复是否该命名域的变量

  Returns:
      2d tensor: (n, label_num)
  """

  out = inputs
  inner_dense_sizes = [] if inner_dense_outshape is None else inner_dense_outshape

  with tf.variable_scope(scope, reuse=reuse):
    for out_size in inner_dense_sizes:
      out = tf.layers.dense(
          out,
          out_size,
          activation=inner_dense_activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer
      )

    out = tf.layers.dense(
        out,
        label_num,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )

  return out


def label_smoothing(inputs, epsilon=0.1):
  '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

  Args:
    inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

  For example,

  ```
  import tensorflow as tf
  inputs = tf.convert_to_tensor([[[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]],
    [[1, 0, 0],
     [1, 0, 0],
     [0, 1, 0]]], tf.float32)

  outputs = label_smoothing(inputs)

  with tf.Session() as sess:
      print(sess.run([outputs]))

  >>
  [array([[[ 0.03333334,  0.03333334,  0.93333334],
      [ 0.03333334,  0.93333334,  0.03333334],
      [ 0.93333334,  0.03333334,  0.03333334]],
     [[ 0.93333334,  0.03333334,  0.03333334],
      [ 0.93333334,  0.03333334,  0.03333334],
      [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
  ```
  '''
  K = inputs.get_shape().as_list()[-1]  # number of channels
  return ((1 - epsilon) * inputs) + (epsilon / K)
