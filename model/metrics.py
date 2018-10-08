import tensorflow as tf


def average_macro_f1(labels, predictions):
    batch_size = predictions.get_shape().as_list()[0]
    total_sentiments = (predictions.get_shape().as_list()[1]
                        * predictions.get_shape().as_list()[2])
    labels = tf.reshape(labels, [batch_size, total_sentiments])
    predictions = tf.reshape(predictions, [batch_size, total_sentiments])

    update_op_list = []
    f1_list = []
    for sentiment in range(total_sentiments):
        f1, update_op = tf.contrib.metrics.f1_score(
            labels=labels[:, sentiment],
            predictions=predictions[:, sentiment],
            name=f"f1_{sentiment}"
        )
        update_op_list.append(update_op)
        f1_list.append(f1)

    return tf.reduce_mean(f1_list), tf.group(*update_op_list)


# def average_macro_f1(labels, predictions):
#     batch_size = predictions.get_shape().as_list()[0]
#     total_sentiments = (predictions.get_shape().as_list()[1]
#                         * predictions.get_shape().as_list()[2])
#     labels = tf.reshape(labels, [batch_size, total_sentiments])
#     predictions = tf.reshape(predictions, [batch_size, total_sentiments])

#     update_op_list = []
#     f1_list = []
#     for sentiment in range(total_sentiments):
#         tp, tf_update_op = tf.contrib.metrics.streaming_true_positives(
#             predictions=predictions[:, sentiment],
#             labels=labels[:, sentiment],
#             name=f"tp_{sentiment}")

#         fp, fp_update_op = tf.contrib.metrics.streaming_false_positives(
#             predictions=predictions[:, sentiment],
#             labels=labels[:, sentiment],
#             name=f"fp_{sentiment}")

#         fn, fn_update_op =

    # update_op_list.append(update_op)
    # f1_list.append(f1)

    return tf.reduce_mean(f1_list), tf.group(*update_op_list)

    # category_num = predictions.get_shape().as_list()[1]
    # sentiment_num = predictions.get_shape().as_list()[2]
    # update_op_list = []
    # macro_f1_list = []
    # for category in range(category_num):
    #     category_f1_list = []
    #     for sentiment in range(sentiment_num):
    #         f1, update_op = tf.contrib.metrics.f1_score(
    #             labels=labels[:, category, sentiment],
    #             predictions=predictions[:, category, sentiment],
    #             name=f"f1_{category}_{sentiment}"
    #         )
    #         update_op_list.append(update_op)
    #         category_f1_list.append(f1)
    #     macro_f1 = tf.reduce_mean
