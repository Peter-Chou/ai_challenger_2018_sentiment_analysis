
# 文本处理

## 繁体转简体

使用opencc 将文件中的繁体转换成简体  

```sh
opencc -i sentiment_analysis_trainingset.csv -o trainset_sc.csv -c t2s.json
opencc -i sentiment_analysis_validationset.csv -o validationset_sc.csv -c t2s.json
opencc -i sentiment_analysis_testa.csv -o test_sc.csv -c t2s.json
```