"""Define the model."""

import tensorflow as tf

from model.attention import (embed_lookup, feedforward, multihead_attention,
                             positional_encoding)


def build_model(mode, vector_path, inputs, params, reuse=False):
    is_training = (mode == "train")

    x = inputs["sentence"]
    label = inputs["label"]
    # iterator_init_op = inputs["iterator_init_op"]

    # build embedding vectors
    vector = embed_lookup(x, vector_path)
    vector += positional_encoding(x,
                                  num_units=params.embed_size,
                                  zero_pad=False,
                                  scale=False)

    # * add dropout mask vector may be not a good idea
    vector = tf.layers.dropout(vector, rate=params.dropout_rate,
                               training=tf.convert_to_tensor(is_training))

    # transformer attention stacks
    for i in range(params.num_attention_stacks):
        with tf.variable_scope(f"num_attention_stacks_{i + 1}"):
            # multi-head attention
            vector = multihead_attention(queries=vector,
                                         keys=vector,
                                         num_units=params.embed_size,
                                         num_heads=params.num_heads,
                                         dropout_rate=params.dropout_rate,
                                         is_training=is_training,
                                         causality=False)

            # feed forward
            vector = feedforward(vector,
                                 num_units=[4*params.embed_size, params.embed_size])

# ************************************************************
# complete attention part, now CNN capture part
# ************************************************************

    return vector


if __name__ == "__main__":
    main()
