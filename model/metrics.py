import tensorflow as tf


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
      tp, tp_update_op = tf.metrics.true_positives(
          predictions=predictions[:, sentiment],
          labels=labels[:, sentiment],
          name=f"tp_{sentiment}")

      fp, fp_update_op = tf.metrics.false_positives(
          predictions=predictions[:, sentiment],
          labels=labels[:, sentiment],
          name=f"fp_{sentiment}")

      fn, fn_update_op = tf.metrics.false_negatives(
          predictions=predictions[:, sentiment],
          labels=labels[:, sentiment],
          name=f"fn_{sentiment}")

      tp = tf.cast(tp, dtype=tf.float32)
      fp = tf.cast(fp, dtype=tf.float32)
      fn = tf.cast(fn, dtype=tf.float32)

      precision = tp / (tp + fp)
      recall = tp / (tp + fn)

      f1 = tf.cond(tf.reduce_any([tf.equal(precision, 0.), tf.equal(recall, 0.)]),
                   true_fn=lambda: 0.,
                   false_fn=lambda: (2. * (precision * recall) / (precision + recall)))
      f1_list.append(f1)
      update_op_list.extend([tp_update_op, fp_update_op, fn_update_op])

  return tf.reduce_mean(f1_list), tf.group(*update_op_list)
