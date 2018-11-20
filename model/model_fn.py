"""Define the model."""

import tensorflow as tf

from model.attention import (dense_logits, feedforward, inception,
                             label_smoothing, multihead_attention)
from model.embedding import position_embedding, word_embedding
from model.hook import _LoggerHook
from model.metrics import average_macro_f1


def model_fn(
        features,
        labels,
        mode,
        params):
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  x = features
  unchanged_labels = labels  # non-smoothing labels for f1 metrics

  if params.label_smooth and labels is not None:
    labels = tf.cast(labels, tf.float32)
    labels = label_smoothing(labels, epsilon=params.epsilon)

  # build embedding vectors
  vector = word_embedding(x, params.vector_path, scale=False)

  # ! reduce the fiexed word dimensions to appropriate dimension
  if params.hidden_size != vector.get_shape().as_list()[-1]:
    # 原论文中使用全连接降维
    with tf.variable_scope("dimension_reduction"):
      vector = tf.layers.dense(
          vector,
          params.hidden_size,
          activation=None,
          use_bias=False,
          kernel_initializer=tf.contrib.layers.xavier_initializer(),
          kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))

  # scale the word embedding
  vector = vector * (params.hidden_size ** 0.5)

  # 给词向量 增加位置信息
  vector += position_embedding(x,
                               num_units=params.hidden_size,
                               scale=False)

  # # * add dropout mask vector may be not a good idea
  vector = tf.layers.dropout(vector, rate=params.dropout_rate,
                             training=tf.convert_to_tensor(is_training))

  # # transformer attention stacks
  for i in range(params.num_attention_stacks):
    with tf.variable_scope(f"num_attention_stacks_{i + 1}"):
      # multi-head attention
      vector = multihead_attention(queries=vector,
                                   keys=vector,
                                   num_units=params.hidden_size,
                                   num_heads=params.num_heads,
                                   dropout_rate=params.dropout_rate,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       1.0),
                                   is_training=is_training,
                                   causality=False)

      # feed forward
      vector = feedforward(vector,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           num_units=[2 * params.hidden_size, params.hidden_size])
  attentions = vector

  # 最里增加一维，以模拟一维黑白通道
  # (N, attention_stacks*T, C, 1)
  attentions = tf.expand_dims(attentions, -1)

  # ************************************************************
  # complete attention part, now CNN capture part
  # ************************************************************
  logits = []
  # 每个category对应一个inception_maxpool classifier
  for topic in range(params.multi_categories):
    cnn_features = inception(attentions,
                             filter_size_list=params.filter_size_list,
                             num_filters=params.num_filters,
                             hidden_size=params.hidden_size,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                 1.0),
                             scope=f"category_{topic+1}_inception")  # (n, 1, 1, total_filter_num)

    total_feature_num = len(params.filter_size_list) * params.num_filters
    # cnn_features: (n, total_filter_num)
    cnn_features = tf.reshape(cnn_features, (-1, total_feature_num))

    # category_logit: (n, num_sentiment)
    category_logits = dense_logits(
        cnn_features,
        params.num_sentiment,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(
            1.0),
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        scope=f"category_{topic+1}_logits",
        inner_dense_outshape=params.inner_dense_outshape,
        inner_dense_activation=tf.tanh,
        use_bias=True)

    # 将该category的logit加入列表
    logits.append(category_logits)

  # logits: (n, multi_categories, num_sentiment)
  logits = tf.stack(logits, axis=1)

  # * train & eval common part
  if (mode == tf.estimator.ModeKeys.TRAIN or
          mode == tf.estimator.ModeKeys.EVAL):

    gstep = tf.train.get_or_create_global_step()

    # loss: (n, multi_categories)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits)
    loss = tf.reduce_sum(loss, axis=1)  # (n,)
    loss = tf.reduce_mean(loss, axis=0)  # scala

    if params.use_regularizer:
      loss_reg = sum(tf.get_collection(
          tf.GraphKeys.REGULARIZATION_LOSSES))
      loss += params.reg_const * loss_reg
    loss = tf.identity(loss, name="loss")

    # predictions = tf.nn.softmax(logits)
    predictions = tf.cast(tf.equal(tf.reduce_max(
        logits, axis=-1, keepdims=True), logits), tf.float32)

    avg_macro_f1, avg_macro_f1_update_op = average_macro_f1(
        labels=tf.cast(unchanged_labels, tf.float32),
        predictions=predictions)

    eval_metric_ops = {
        'avg_macro_f1': (avg_macro_f1, avg_macro_f1_update_op)}

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("f1", avg_macro_f1)

    summary_hook = tf.train.SummarySaverHook(save_steps=params.print_n_step,
                                             output_dir="./summary",
                                             summary_op=tf.summary.merge_all())

  else:
    loss = None
    eval_metric_ops = None

  # * train specific part
  if (mode == tf.estimator.ModeKeys.TRAIN):
    learning_rate = tf.train.cosine_decay_restarts(
        learning_rate=params.learning_rate,
        global_step=gstep,
        first_decay_steps=params.first_decay_steps,
        t_mul=params.t_mul,
        m_mul=params.m_mul,
        alpha=params.alpha,
        name="learning_rate")
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params.momentum)

    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, params.max_norm)
    train_op = optimizer.apply_gradients(zip(gradients, variables),
                                         global_step=gstep)

    # add custom training logger
    custom_logger = _LoggerHook(
        loss, gstep, learning_rate, params.print_n_step)
  else:
    train_op = None

  # * predict part
  if mode == tf.estimator.ModeKeys.PREDICT:
    # 在预测时， logits：(multi_categories, num_sentiment)
    # pred: (multi_categories,)
    pred = tf.subtract(tf.argmax(logits, axis=-1), 2)
    predictions = {
        "classes": pred,
    }
    export_outputs = {
        "classify": tf.estimator.export.PredictOutput(predictions)
    }
  else:
    predictions = None
    export_outputs = None

  training_hooks = [custom_logger, summary_hook] if is_training else None

  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions=predictions,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops=eval_metric_ops,
                                    training_hooks=training_hooks)
