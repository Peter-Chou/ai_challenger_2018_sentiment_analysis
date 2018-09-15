"""Define the model."""

import tensorflow as tf

from model.attention import feedforward, multihead_attention
from model.embedding import position_embedding, word_embedding


def build_model(mode, vector_path, inputs, params, reuse=False):
    is_training = (mode == "train")

    x = inputs["sentence"]
    label = inputs["label"]
    # iterator_init_op = inputs["iterator_init_op"]

    # build embedding vectors
    vector = word_embedding(x, vector_path, scale=False)

    # ! reduce the fiexed word dimensions to appropriate dimension
    if params.hidden_size != vector.get_shape().as_list()[-1]:
        # 原论文中使用全连接降维
        # vector = tf.layers.dense(
        #     vector,
        #     params.hidden_size,
        #     activation=None,
        #     use_bias=False,
        #     kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))

        # ? (尝试) use conv1d to reduce the dimension (with shared weights)
        vector = tf.layers.conv1d(
            vector,
            filters=params.hidden_size,
            kernel_size=1,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))

    # scale the word embedding
    vector = vector * (params.hidden_size ** 0.5)

    # 给词向量 增加位置信息
    vector += position_embedding(x,
                                 num_units=params.hidden_size,
                                 mask_pad=True,
                                 #   zero_pad=False,
                                 scale=False)

    # # * add dropout mask vector may be not a good idea
    vector = tf.layers.dropout(vector, rate=params.dropout_rate,
                               training=tf.convert_to_tensor(is_training))

    # # transformer attention stacks
    attns = []
    for i in range(params.num_attention_stacks):
        with tf.variable_scope(f"num_attention_stacks_{i + 1}"):
            # multi-head attention
            vector = multihead_attention(queries=vector,
                                         keys=vector,
                                         num_units=params.hidden_size,
                                         num_heads=params.num_heads,
                                         dropout_rate=params.dropout_rate,
                                         is_training=is_training,
                                         causality=False)

    #         # feed forward
            vector = feedforward(vector,
                                 num_units=[4*params.hidden_size, params.hidden_size])
            attns.append(vector)
    # concat all attentions (like DenseNet)
    features = tf.concat(attns, 1)  # (N, attention_stacks*T, C)

    # 最里增加一维，以模拟一维黑白通道
    features = tf.expand_dims(features, -1)  # (N, attention_stacks*T, C, 1)
# ************************************************************
# complete attention part, now CNN capture part
# ************************************************************

    # return vector
    return features


if __name__ == "__main__":
    main()
