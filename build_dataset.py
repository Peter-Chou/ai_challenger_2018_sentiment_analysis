# encoding: utf-8
"""
creates or transforms the dataset e.g. sentences.txt, labels.txt
"""

import argparse
import csv
import re

import jieba

from model.helper import flatten

# import unicodedata


VOCAB_PATH = "./chinese_vectors/sgns.weibo.bigram-char"

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/small',
                    help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")


test_line = "🤔 又是一家菜单让人眼花缭乱的店！蔡塘的蝴碟轩不在广场美食区内，而是在万佳酒店的大厅旁边，里面曲曲折折的分成很多用餐区，但其实桌数不少，用餐区隔得比较有私密感，适合小聚，装潢跟它的菜色一样走混搭风！\n\n蝴碟轩也是包山包海，口味就一般以上，达人未满，如果交通方便在附近可以考虑，但不必要特地来吃，菜的份量很刚好，2人也可以点出花样来。\n\n🐓法国鹅肝炒饭：鹅肝味道很淡份量不多，所以应该把重点放在饭炒得有多香，一般般像妈妈菜，钟意牛排的好吃很多。\n\n🐙熟悉的墨鱼仔：泰式凉拌海鲜口味，ok!\n\n🐗生菜包养的肉碎：肉末炒四季豆，比较咸一点，包在生菜中刚好，有辣度！\n\n🌙月亮虾薄饼：NG! 虾在哪呢？月亮上吗？ 请更名炸薄饼，但也炸得不好，油哈味很重。这45大洋我觉得太贵！讓"

# 判断中文范围
# ! https://blog.csdn.net/JohinieLi/article/details/76152549


def get_vocab(file_path):
    word_set = set()
    with open(file_path, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        next(reader)    # 跳过第一行
        for word, *_ in reader:
            word_set.add(word)
    return word_set


def _full_cut(word, vocab):
    res = []
    tmp = jieba.lcut(word, cut_all=True)
    for i in (0, -1):
        if tmp[i] in vocab:
            res.append(tmp[i])
    return res if len(res) > 0 else "<UNK>"  # 将vocab里未出现的word替换为<UNK>


def tokenize_word(line, vocab):
    rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('', line)

    sentence = []
    for word in jieba.cut(line, cut_all=False):
        if word in vocab:
            try:
                int(word)
                sentence.append("<NUM>")  # 将数字 -> <NUM>
            except ValueError:
                sentence.append(word)
        else:
            sentence.append(_full_cut(word, vocab))
    return list(flatten(sentence))


def load_dataset(path_csv, test=False):
    # with open(path_csv) as in_file:
    #     next(in_file)
    #     for idx, line in enumerate(in_file):
    #         if idx < 3:
    #             print(line)
    #         else:
                # break
    with open(path_csv, newline='', encoding='utf-8', errors='ignore') as in_file:
        reader = csv.reader(in_file, delimiter=',')
        _, _, *label_headers = next(reader)
        for idx, sentence, *labels in reader:
            if int(idx) < 20:
                print(_remove_punctuation(sentence))
            else:
                break


# load_dataset("./data/train/sentiment_analysis_trainingset.csv")


def main():
    args = parser.parse_args()
    vocab = get_vocab(VOCAB_PATH)
    print(tokenize_word(test_line, vocab))

    # print(len(vocab))


if __name__ == "__main__":
    main()
