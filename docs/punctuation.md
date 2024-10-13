# 前言

在语音识别中，模型输出的结果只是单纯的文本结果，并没有根据语法添加标点符号，本教程就是针对这种情况，在语音识别文本中根据语法情况加入标点符号，使得语音识别系统能够输出在标点符号的最终结果。

# 使用

使用主要分为三4步：

1. 首先是[下载标点的模型](https://pan.baidu.com/s/1GgPU753eJd4pZ1LxDjeyow?pwd=7wof)，放在`models/`目录下。
2. 在使用时，将`use_punc`参数设置为True，输出的结果就自动加上了标点符号，如下。

```
消耗时间：101, 识别结果: 近几年，不但我用书给女儿儿压岁，也劝说亲朋，不要给女儿压岁钱而改送压岁书。
```

# 单独使用标点符号模型

如果只是使用标点符号模型的话，可以参考一下代码。
```python
from masr.infer_utils.punc_predictor import PunctuationPredictor

pun_predictor = PunctuationPredictor(model_dir='models/punc_models')
result = pun_predictor('近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书')[0]
print(result)
```

输出结果：
```
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\test\AppData\Local\Temp\jieba.cache
Loading model cost 0.502 seconds.
Prefix dict has been built successfully.
近几年，不但我用书给女儿儿压岁，也劝说亲朋，不要给女儿压岁钱而改送压岁书。
```
