import tensorflow as tf


EPSILON = 1e-6


def average_macro_f1(labels, predictions):
  batch_size = predictions.get_shape().as_list()[0]
  total_sentiments = (predictions.get_shape().as_list()[1]
                      * predictions.get_shape().as_list()[2])
  labels = tf.reshape(labels, [batch_size, total_sentiments])
  predictions = tf.reshape(predictions, [batch_size, total_sentiments])

  update_op_list = []
  f1_list = []
  with tf.variable_scope("macro_f1"):
    for sentiment in range(total_sentiments):
      precision, precision_update_op = tf.metrics.precision(
          labels=labels[:, sentiment],
          predictions=predictions[:, sentiment],
          name=f"p_{sentiment}")

      recall, recall_update_op = tf.metrics.recall(
          labels=labels[:, sentiment],
          predictions=predictions[:, sentiment],
          name=f"r_{sentiment}")

      f1 = 2 * (precision * recall) / (precision + recall + EPSILON)

      f1_list.append(f1)
      update_op_list.extend([precision_update_op, recall_update_op])

    #   tp, tp_update_op = tf.metrics.true_positives(
    #       predictions=predictions[:, sentiment],
    #       labels=labels[:, sentiment],
    #       name=f"tp_{sentiment}")

    #   fp, fp_update_op = tf.metrics.false_positives(
    #       predictions=predictions[:, sentiment],
    #       labels=labels[:, sentiment],
    #       name=f"fp_{sentiment}")

    #   fn, fn_update_op = tf.metrics.false_negatives(
    #       predictions=predictions[:, sentiment],
    #       labels=labels[:, sentiment],
    #       name=f"fn_{sentiment}")

    #   tp = tf.cast(tp, dtype=tf.float32)
    #   fp = tf.cast(fp, dtype=tf.float32)
    #   fn = tf.cast(fn, dtype=tf.float32)

    #   precision = tp / (tp + fp + EPSILON)
    #   recall = tp / (tp + fn + EPSILON)

    # f1 = tf.cond(tf.reduce_any([tf.equal(precision, 0.), tf.equal(recall, 0.)]),
    #              true_fn=lambda: 0.,
    #              false_fn=lambda: (2. * (precision * recall) / (precision + recall)))
      # f1_list.append(f1)
      # update_op_list.extend([tp_update_op, fp_update_op, fn_update_op])

      # f1, update_op = tf.contrib.metrics.f1_score(
      #     labels=labels[:, sentiment],
      #     predictions=predictions[:, sentiment],
      #     name=f"f1_{sentiment}")

  f1_list = tf.stack(f1_list)

  return tf.reduce_mean(f1_list), tf.group(*update_op_list)
