import tensorflow as tf


EPSILON = 1e-6


def average_macro_f1(labels, predictions):
  """计算 average F1

  Args:
    labels (tensor): (batch, num_categories, num_sentiments)
    predictions (tensor): (batch, num_categories, num_sentiments)

  Returns:
    (average_f1_scalar, update_op_group)
  """

  batch_size = predictions.get_shape().as_list()[0]
  categories = predictions.get_shape().as_list()[1]
  sentiments = predictions.get_shape().as_list()[2]

  update_op_list = []
  f1_list = []

  with tf.variable_scope("macro_f1"):
    for category in range(categories):
      for sentiment in range(sentiments):
        precision, precision_update_op = tf.metrics.precision(
            labels=labels[:, category, sentiment],
            predictions=predictions[:, category, sentiment],
            name=f"p_{category}_{sentiment}")

        recall, recall_update_op = tf.metrics.recall(
            labels=labels[:, category, sentiment],
            predictions=predictions[:, category, sentiment],
            name=f"r_{category}_{sentiment}")

        f1 = 2 * (precision * recall) / (precision + recall + EPSILON)

        f1_list.append(f1)
        update_op_list.extend([precision_update_op, recall_update_op])

  f1_list = tf.stack(f1_list)

  return tf.reduce_mean(f1_list), tf.group(*update_op_list)
