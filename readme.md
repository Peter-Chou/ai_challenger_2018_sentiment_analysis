
# 文本处理

## 繁体转简体

使用opencc 将文件中的繁体转换成简体  

```sh
opencc -i sentiment_analysis_trainingset.csv -o train_sc.csv -c t2s.json
opencc -i sentiment_analysis_validationset.csv -o val_sc.csv -c t2s.json
opencc -i sentiment_analysis_testa.csv -o a_sc.csv -c t2s.json
```

## 中文停用词

使用此[微博中文停用词库](
https://github.com/chdd/weibo/blob/master/stopwords/%E4%B8%AD%E6%96%87%E5%81%9C%E7%94%A8%E8%AF%8D%E5%BA%93.txt) (其中去除0-9)  

# label vector生成