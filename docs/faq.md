# 怎么设计词汇表

1. 如果是纯中文数据，可以把`model_type`设置为`char`，然后`build_vocab_size`设置为`null`，这样会使用数据集出现的全部字符。
2. 如果是英文数据，也可以把`model_type`设置为`word`，然后`build_vocab_size`设置为`null`，这样会使用数据集出现的全部单词，但是不建议这么做，因为这样的词汇表会变成的非常大，不利于训练，例如Librispeech数据集就会有9万多个单词，所以建议使用把`model_type`设置为`unigram`，然后`build_vocab_size`设置为`5000`，也可以跟更大一些，根据数据集量设置。
3. 如果是中混合数据，可以把`model_type`设置为`unigram`，然后`build_vocab_size`设置为`10000`左右。
4. 如果是其他语言，可以直接使用`model_type`=`unigram`，然后`build_vocab_size`设置为`10000`左右，如果报错太大了，可以更加提示设置小一些。


