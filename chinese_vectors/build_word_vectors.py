"""生成word转idx字典，以及vectors向量集"""

import argparse
import bz2
import csv
import json
import os

import numpy as np

VOCAB_FILE = "./sgns.weibo.bigram-char.bz2"


def build_word_idx_and_vectors(file_path):
  """从word—vectors文件中导出 word_idx_map, vectors matrix (对vector进行
  Frobenius Norm标准化)

  Args:
      file_path (str): word-vector文件地址

  """

  word_idx_dict = dict()
  with bz2.open(file_path, 'rt', newline='', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    next(reader)
    vectors = []
    word_idx_dict["<PAD>"] = 0  # add <PAD>'s id is zero
    vectors.append([0.0] * 300)
    for idx, (word, *vector) in enumerate(reader, start=1):
      word_idx_dict[word] = idx
      vector = np.asarray([float(x) for x in vector if x != ''])
      # // vector = vector / np.linalg.norm(vector)  # Frobenius norm
      vectors.append(vector)
    vectors = np.asarray(vectors)

  dict_save_path = os.path.join(os.path.dirname(
      file_path), "word_idx_table.json")
  vectors_save_path = os.path.join(os.path.dirname(
      file_path), "vectors.npy")

  np.save(vectors_save_path, vectors)
  with open(dict_save_path, 'w', encoding='utf-8') as f:
    json.dump(word_idx_dict, f, indent=4, ensure_ascii=False)


def main():
  build_word_idx_and_vectors(VOCAB_FILE)


if __name__ == '__main__':
  main()
