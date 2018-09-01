"""
creates or transforms the dataset e.g. sentences.txt, labels.txt
"""

import argparse
import csv
import re
import unicodedata

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/small',
                    help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")


# test_line = "又是一家菜單讓人眼花繚亂的店！2.34蔡塘的蝴碟軒不在廣場美食區內，而是在萬佳酒店的大廳旁邊，裡面曲曲折折的分成很多用餐區，但其實桌數不少，用餐區隔得比較有私密感，適合小聚，裝潢跟它的菜色一樣走混搭風！\n\n蝴碟軒也是包山包海，口味就一般以上，達人未滿，如果交通方便在附近可以考慮，但不必要特地來吃，菜的份量很剛好，2人也可以點出花樣來。\n\n法國鵝肝炒飯：鵝肝味道很淡份量不多，所以應該把重點放在飯炒得有多香，一般般像媽媽菜，鍾意牛排的好吃很多。\n\n熟悉的墨魚仔：泰式涼拌海鮮口味，ok!\n\n生菜包養的肉碎：肉末炒四季豆，比較鹹一點，包在生菜中剛好，有辣度！\n\n月亮蝦薄餅：NG! 蝦在哪呢？月亮上嗎？ 請更名炸薄餅，但也炸得不好，油哈味很重。這45大洋我覺得太貴！"

# 判断中文范围
# ! https://blog.csdn.net/JohinieLi/article/details/76152549


# table = {ord(f):ord(t) for f,t in zip(
#      u'，。！？【】（）％＃＠＆１２３４５６７８９０',
#      u',.!?[]()%#@&1234567890')}


def _remove_punctuation(line):
    # rule = re.compile(ur"[^a-zA-Z0-9\u4e00-\u9fa5]")
    # rule = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z^0-9]")
    line = unicodedata.normalize("NFKC", line)
    rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5,.!?%#@&]")
    line = rule.sub('', line)
    return line


# print(_remove_punctuation(test_line.decode('utf-8')))


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


if __name__ == "__main__":
    main()
