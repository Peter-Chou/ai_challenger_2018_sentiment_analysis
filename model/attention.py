# -*- coding: utf-8 -*-

"""
creates the deep learning model
from https://github.com/Kyubyong/transformer/blob/master/modules.py
"""

import numpy as np
import tensorflow as tf


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
    # TODO : change tf.Variable to tf.get_variable (may cause restore problem)
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        # beta = tf.get_variable("beta",
        #                        shape=params_shape,
        #                        dtype=tf.float64,
        #                        initializer=tf.zeros_initializer())
        # gamma = tf.get_variable("gamma",
        #                         shape=params_shape,
        #                         dtype=tf.float64,
        #                         initializer=tf.ones_initializer())

        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embed_lookup(inputs, vector_path, scale=False, scope="embedding"):

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


def positional_encoding(inputs,
                        num_units,
                        mask_pad=False,
                        scale=True,
                        scope="positional_encoding",
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
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
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

# # 原始的代码实现
# def positional_encoding(inputs,
#                         num_units,
#                         zero_pad=False,
#                         scale=True,
#                         scope="positional_encoding",
#                         reuse=None):
#     '''Sinusoidal Positional_Encoding.
#     Args:
#       inputs: A 3d Tensor with shape of (batch, N, T).
#       num_units: Output dimensionality
#       zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
#         and <pad>:0 will have zero vectors as positional embedding
#       scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
#       scope: Optional scope for `variable_scope`.
#       reuse: Boolean, whether to reuse the weights of a previous layer
#         by the same name.
#     Returns:
#         A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
#     '''

#     N, T = inputs.get_shape().as_list()
#     with tf.variable_scope(scope, reuse=reuse):
#         position_ind = tf.tile(tf.expand_dims(
#             tf.range(T), 0), [N, 1])  # (N, T)

#         # First part of the PE function: sin and cos argument
#         position_enc = np.array([
#             [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
#             for pos in range(T)])

#         # Second part, apply the cosine to even columns and sin to odds.
#         position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
#         position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

#         # Convert to a tensor
#         lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

#         if zero_pad:
#             lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
#                                       lookup_table[1:, :]), 0)

#         outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

#         if scale:
#             outputs = outputs * num_units**0.5

#         return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
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

        # Q = tf.layers.dense(queries, num_units, use_bias=False)  # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, use_bias=False)  # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, use_bias=False)  # (N, T_k, C)

        # Linear projections
        # ? remove activation?
        # 使得不同的multi-head attention 都回应不同的query，以及句子的不同子理解
        Q = tf.layers.dense(queries, num_units,
                            activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(
            keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(
            keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

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

        paddings = tf.ones_like(outputs)*(-2**32+1)

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

            paddings = tf.ones_like(masks)*(-2**32+1)
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
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
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
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


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
    return ((1-epsilon) * inputs) + (epsilon / K)
