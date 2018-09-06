# -*- coding: utf-8 -*-

"""
creates or transforms the dataset e.g. sentences.txt, labels.txt
"""

import argparse
import csv
import json
import os
import re

import jieba
import numpy as np

from model.helper import flatten

CHINESE_WORD_INT_SWITCH = "./chinese_vectors/word_idx_table.json"
# STOPWORDS_PATH = './chinese_vectors/chinese_stopwords.txt'

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--test", help="clean test format data", action="store_true")
parser.add_argument('--data_dir', default=None,
                    help="Directory containing the raw data to split into sentence & label txt")
parser.add_argument('--max_len', default=None,
                    help="Directory containing the raw data to split into sentence & label txt")


def _add_sub_or_unk_word(word, vocab):
    res = []
    tmp = jieba.lcut(word, cut_all=True)
    for i in (0, -1):
        if tmp[i] in vocab:
            res.append(tmp[i])
    return res if len(res) > 0 else "<UNK>"  # 将vocab里未出现的word替换为<UNK>


def _add_num_token(word):
    word = int(word)
    if word >= 10:
        return "<num>"  # 将数字 -> <num>
    else:
        return str(word)  # 0-9 保留


def tokenize_sentence(line, vocab):
    rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('', line)

    sentence = []
    for word in jieba.cut(line, cut_all=False):
        if word in vocab:
            try:
                sentence.append(_add_num_token(word))
            except ValueError:
                sentence.append(word)
        else:
            sentence.append(_add_sub_or_unk_word(word, vocab))
    return list(flatten(sentence))


def _write_rows_to_csv(lists, saved_csv_name):
    with open(saved_csv_name, 'w', newline='', encoding='utf-8', errors='ignore') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(lists)


def sentence_label_save(file_path, w2i_dict, test=False):
    lengths = []
    sentences_path = os.path.join(os.path.dirname(file_path), "sentences.csv")
    labels = []
    with open(sentences_path, 'w', newline='', encoding='utf-8', errors='ignore') as save_f:
        writer = csv.writer(save_f, delimiter=',')
        with open(file_path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for idx, sentence, *label in reader:
                sentence = tokenize_sentence(sentence, w2i_dict)
                lengths.append([int(len(sentence))])  # for test
                if not test:
                    label = [int(x) for x in label]
                    labels.append(label)
                writer.writerow(sentence)

    labels_path = os.path.join(os.path.dirname(file_path), "labels.csv")
    if not test:
        _write_rows_to_csv(labels, labels_path)
    # collect data
    lengths_path = os.path.join(os.path.dirname(
        file_path), "lengths_stopwords.csv")
    _write_rows_to_csv(lengths, lengths_path)

    # with open(path_csv) as in_file:
    #     next(in_file)
    #     for idx, line in enumerate(in_file):
    #         if idx < 3:--
    #             print(line)
    #         else:
    # break
    # with open(path_csv, newline='', encoding='utf-8', errors='ignore') as in_file:
    #     reader = csv.reader(in_file, delimiter=',')
    #     _, _, *label_headers = next(reader)
    #     for idx, sentence, *labels in reader:
    # if int(idx) < 20:
    #     print(_remove_punctuation(sentence))
    # else:
    #     break

    # load_dataset("./data/train/sentiment_analysis_trainingset.csv")


def load_word_int_table(file_path):
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def main():
    args = parser.parse_args()
    if args.data_dir is None:
        raise Exception("must give a dataset folder")
    word_int_dict = load_word_int_table(CHINESE_WORD_INT_SWITCH)
    dataset_path = os.path.join(
        args.data_dir, os.path.basename(args.data_dir) + "_sc.csv")
    max_Length = sentence_label_save(
        dataset_path, word_int_dict, test=args.test)


if __name__ == "__main__":
    main()
