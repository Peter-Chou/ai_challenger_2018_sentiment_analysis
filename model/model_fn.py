"""Define the model."""

import tensorflow as tf

from model.attention import embed_lookup, positional_encoding


def build_model(mode, vector_path, inputs, params, reuse=False):

    sentence = inputs["sentence"]
    label = inputs["label"]
    # iterator_init_op = inputs["iterator_init_op"]

    # build embedding vectors
    sentence = embed_lookup(sentence, vector_path)
    sentence += positional_encoding(tf.ones_like(inputs["sentence"]),
                                    num_units=300,
                                    zero_pad=False,
                                    scale=False)
    return sentence


if __name__ == "__main__":
    main()
