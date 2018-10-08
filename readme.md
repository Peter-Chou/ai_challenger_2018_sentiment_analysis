# 细粒度用户评论情感分析 （全球AI挑战赛 2018）

## 项目简介

根据客户的评论，对20个方面进行情感分析（-2：未提及、-1：负面、0：中性、1：正面）  
解决问题思路：将这20个多分类任务问题看作一个 **多任务学习** 问题来搭建模型  
**解决方案模型**： SemEval-2018中论文 ([Attention-based Convolutional Neural Networks for Multi-label Emotion Classification](http://aclweb.org/anthology/S18-1019)) 的实现及应用。

## 预处理

### 繁体转简体

使用opencc 将文件中的繁体转换成简体

```sh
opencc -i data/train/sentiment_analysis_trainingset.csv -o data/train/train_sc.csv -c t2s.json
opencc -i data/val/sentiment_analysis_validationset.csv -o data/val/val_sc.csv -c t2s.json
opencc -i data/test/a/sentiment_analysis_testa.csv -o data/test/a/a_sc.csv -c t2s.json
```

### 中文词向量

简体中文的词向量[chinese word vectors](https://github.com/Embedding/Chinese-Word-Vectors) 里的Word2vec / Skip-Gram with Negative Sampling，内容选择微博 （Word + Character + Ngram)  
**中文停用词**使用此[微博中文停用词库](
https://github.com/chdd/weibo/blob/master/stopwords/%E4%B8%AD%E6%96%87%E5%81%9C%E7%94%A8%E8%AF%8D%E5%BA%93.txt) (其中去除0-9)

### 分词

分词使用的是[jieba](https://github.com/fxsjy/jieba)包, 主要先按词组拆分，如果词组不在词库(已去除停用词）中出现，再将该词组按字拆分,  
因为考虑到项目为辨析情绪非翻译，考虑弱化语言结构，所以这里对未在词库中出现的新词不进行保留。

## 模型

### 模型结构

模型由参数共享的语句理解层和参数独立的情感辨别层：

- 特征共享层：由1词向量层 + 1位置向量层(提供位置信息) + 3个Transformer自注意力模块组成
- 情感辨别层：由1卷积层 + 1最大池化层 + 1全连接层组成

![attn_conv picture](/pic/attnconv_all_in_one.png)

该模型的思路是模仿人处理该问题的行为：第一步理解语句（自注意力模块），第二步辨别情感（卷积+最大池化）

### Transformer: 自注意力模块

Transformer是由谷歌团队在[Attention Is All You Need]( https://arxiv.org/pdf/1706.03762.pdf)首次提出，这里使用的是Encoder中的自注意力Transformer  
自注意力Transformer对输入进行线性变换得到每个位置的query和(key, value)键值对,  
通过对query和key求点积来寻找与query最相关的key并对其结果使用softmax得到该键值对的权重。  
这个query的回答就是：sum(value $\times$ 对应权重)  
最后对这个query的回答进行维度缩放（使用position-wise feed forword，即一维卷积，stride=1, 激活函数为relu）  
这样若有N个位置，得到N个query及其对应的回答

### CNN情感辨别模块

这里借鉴的是Yoon Kim在[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)提出的架构。其中：  
卷积层kernel的宽度为Transformer提取的Attention的维度大小，kernel的高度取10（即对临近的10个Attention进行卷积操作）。kernel的数量取64  
最大池化的作用范围为整个feature map，即每个Kernel得到的feature map在经过最大池化后被提炼为一个值

## 效果

Average Macro F1 = 0.6, 比赛排名为111名（总共256名）