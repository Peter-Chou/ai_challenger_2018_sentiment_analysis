"""Define the model."""

import tensorflow as tf

from model.attention import (dense_logits, feedforward, inception,
                             label_smoothing, multihead_attention)
from model.embedding import position_embedding, word_embedding


def build_model(mode, vector_path, inputs, params, reuse=False):
    is_training = (mode == "train")

    x = inputs["sentence"]
    labels = inputs["label"]

    if params.label_smooth:
        labels = tf.cast(labels, tf.float32)
        labels = label_smoothing(labels)

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
    attentions = tf.concat(attns, 1)  # (N, attention_stacks*T, C)

    # 最里增加一维，以模拟一维黑白通道
    # (N, attention_stacks*T, C, 1)
    attentions = tf.expand_dims(attentions, -1)

# ************************************************************
# complete attention part, now CNN capture part
# ************************************************************
    logits = []
    # 每个category对应一个inception_maxpool classifier
    for topic in range(params.multi_categories):

        features = inception(attentions,
                             filter_size_list=params.filter_size_list,
                             num_filters=params.num_filters,
                             hidden_size=params.hidden_size,
                             scope=f"category_{topic+1}_inception")  # (n, 1, 1, total_filter_num)

        total_feature_num = len(params.filter_size_list) * params.num_filters

        # features: (n, total_filter_num)
        features = tf.reshape(features, (-1, total_feature_num))

        # logit: (n, num_sentiment)
        # logit = tf.layers.dense(features,
        #                         params.num_sentiment,
        #                         activation=None,
        #                         use_bias=True,
        #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))

        # category_logit: (n, num_sentiment)
        category_logits = dense_logits(
            features,
            params.num_sentiment,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                1.0),
            scope=f"category_{topic+1}_logits",
            inner_dense_outshape=params.inner_dense_outshape,
            inner_dense_activation=tf.tanh,
            use_bias=True)

        # 将该category的logit加入列表
        logits.append(category_logits)

    # logits: (n, multi_categories, num_sentiment)
    logits = tf.stack(logits, axis=1)

    # loss: (n, multi_categories)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits)
    # loss: scala tensor. return total batch loss
    loss = tf.reduce_sum(loss)

    if mode == "infer":
        prediction = tf.squeeze(logits)
        prediction = tf.subtract(tf.argmax(prediction, axis=1), 2)

    return loss


if __name__ == "__main__":
    main()
