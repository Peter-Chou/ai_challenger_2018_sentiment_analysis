import argparse

import numpy as np
import tensorflow as tf

from model.helper import Params
from model.input_fn import build_dataset, input_fn
from model.model_fn import model_fn

_MIN_EVAL_FREQUENCY = 100
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=None,
                    help="Directory containing the model graph & metrics")

parser.add_argument('--mode', default=None,
                    help="train or eval")
parser.add_argument(
    "-t", "--test", help="predict", action="store_true")


def main(unused):
    params = Params("params.yml")
    args = parser.parse_args()
    if args.model_dir is None:
        raise Exception("must give a folder for model")

    # load training data
    train_feature = build_dataset(
        "./data/train/sentences_idx.csv",
        length=params.max_len,
        padding=True)
    train_label = build_dataset(
        "./data/train/labels.csv",
        length=params.multi_categories * params.num_sentiment,
        padding=False,
        cascading_label=True,
        label_num=params.num_sentiment)

    # load eval data
    eval_feature = build_dataset(
        "./data/val/sentences_idx.csv",
        length=params.max_len,
        padding=True)
    eval_label = build_dataset(
        "./data/val/labels.csv",
        length=params.multi_categories * params.num_sentiment,
        padding=False,
        cascading_label=True,
        label_num=params.num_sentiment)

    # load test data
    test_feature = build_dataset(
        "./data/test/a/sentences_idx.csv",
        length=params.max_len,
        padding=True)
    test_label = build_dataset(  # pseudo labels
        None,
        length=params.multi_categories * params.num_sentiment,
        padding=False,
        cascading_label=True,
        label_num=params.num_sentiment)

    # define train, eval, test's input_fn
    def train_input_fn():
        return input_fn(train_feature,
                        train_label,
                        batch_size=params.batch_size,
                        is_training=True,
                        repeat_count=None,
                        prefetch=params.prefetch
                        )

    def eval_input_fn():
        return input_fn(eval_feature,
                        eval_label,
                        batch_size=params.batch_size,
                        is_training=False,
                        repeat_count=1,
                        prefetch=params.prefetch)

    def test_input_fn():
        return input_fn(test_feature,
                        test_label,
                        batch_size=1,
                        is_training=False,
                        repeat_count=1)

        # define strategy
        # NUM_GPUS = 2
        # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)

        # define config
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        keep_checkpoint_max=10,
        save_checkpoints_steps=100,
        # train_distribute=strategy
    )

    # define estimator
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params
    )

    # define spec
    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=params.train_steps)
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=100,
        # exporters=[exporters],
        name="eval")

    # train and evaluate
    if not args.test:
        tf.estimator.train_and_evaluate(
            nn,
            train_spec,
            eval_spec)

    else:  # 'pred'
        predict_results = nn.predict(input_fn=test_input_fn)
        print(predict_results)


if __name__ == "__main__":
    tf.app.run(main)
