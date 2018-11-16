import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from model.helper import Params
from model.input_fn import build_dataset, input_fn
from model.model_fn import model_fn

_MIN_EVAL_FREQUENCY = 100
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=None,
                    help="Directory containing the model graph & metrics")

parser.add_argument('--test_dir', default="data/test/a",
                    help="Directory containing the test data")

parser.add_argument(
    "-e", "--eval", help="evaluate", action="store_true")
parser.add_argument(
    "-t", "--test", help="predict", action="store_true")


def save_or_update_predict(predicts,
                           dirname,
                           predict_save_name):
  """将推断得到的predicts按要求格式保存到本地

  Args:
      predicts (numpy array): 模型预测得到的结果集合
      dirname (str): 测试集所在的文件夹地址
      predict_save_name (str): 保存到本地的文件名称
  """

  for filename in os.listdir(dirname):
    if "sentiment_analysis" in filename:
      original_test_data = os.path.join(dirname, filename)
      predict_save_file = os.path.join(dirname, predict_save_name)

  if os.path.isfile(predict_save_file):
    os.remove(predict_save_file)

  test_data = pd.read_csv(original_test_data)
  test_data.iloc[:, 1] = ""  # erase contents
  test_data.iloc[:, 2:] = predicts  # replace in place
  test_data.to_csv(predict_save_file, index=False)


def main(unused):
  params = Params("params.yaml")
  args = parser.parse_args()
  if args.model_dir is None:
    raise Exception("You must give a folder to save / retore the model")

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
      os.path.join(args.test_dir, "sentences_idx.csv"),
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
                    is_test=True,
                    repeat_count=1)

  # define strategy
  # NUM_GPUS = 2
  # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)

  # define config
  config = tf.estimator.RunConfig(
      model_dir=args.model_dir,
      tf_random_seed=params.random_seed,
      keep_checkpoint_max=params.keep_checkpoint_max,
      save_checkpoints_steps=params.save_n_step,
      # train_distribute=strategy
  )

  # define estimator
  nn = tf.estimator.Estimator(
      model_fn=model_fn,
      config=config,
      params=params
  )

  # define train spec
  train_spec = tf.estimator.TrainSpec(
      train_input_fn,
      max_steps=params.train_steps)

  # define eval spec
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=500,
      throttle_secs=0,
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
    results = []
    for result in predict_results:  # result is dict object
      results.append(result["classes"])
    results = np.asarray(results)
    save_or_update_predict(results,
                           args.test_dir,
                           "ai_competition_submission_predict_label_data.csv")


if __name__ == "__main__":
  # Enable logging for tf.estimator
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
