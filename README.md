![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/MASR)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/MASR)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/MASR)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

# MASR流式与非流式语音识别项目 (此分支未完成，请勿使用)

MASR是一款基于Pytorch实现的自动语音识别框架，MASR全称是神奇的自动语音识别框架（Magical Automatic Speech Recognition），当前为V2版本，如果想使用V1版本，请在这个分支[r1.x](https://github.com/yeyupiaoling/MASR/tree/r1.x)。MASR致力于简单，实用的语音识别项目。可部署在服务器，Nvidia Jetson设备，未来还计划支持Android等移动设备。

**如果熟悉PaddlePaddle，请优先使用：[PPASR](https://github.com/yeyupiaoling/PPASR)**

**欢迎大家扫码入QQ群讨论**，或者直接搜索QQ群号`1169600237`，问题答案为博主Github的ID`yeyupiaoling`。

<div align="center">
  <img src="docs/images/qq.png"/>
</div>


本项目使用的环境：
 - Anaconda 3
 - Python 3.8
 - Pytorch 1.12.1
 - Windows 10 or Ubuntu 18.04


## 项目快速了解

 1. 本项目支持流式识别模型`deepspeech2`、`conformer`，每个模型又分online(在线)和offline(离线)，对应的是流式识别和非流式识别。
 2. 本项目支持两种解码器，分别是集束搜索解码器`ctc_beam_search`和贪心解码器`ctc_greedy`，集束搜索解码器`ctc_beam_search`准确率更高，但不支持Windows。
 3. 下面提供了一系列预训练模型的下载，下载预训练模型之后，需要把全部文件复制到项目根目录，并执行导出模型才可以使用语音识别。


## 更新记录

 - 2022.11: 正式发布最终级的V2版本。


## 视频讲解

这个是PPSAR的视频教程，项目是通用的，可以参考使用。

 - [知识点讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Rr4y1D7iZ)
 - [流式识别的使用讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Te4y1h7KK)


## 模型下载


1. `conformer`预训练模型列表：

|       使用模型        |                                  数据集                                  | 预处理方式 | 语言  | 测试集字错率（词错率） | 下载地址 |
|:-----------------:|:---------------------------------------------------------------------:|:-----:|:---:|:-----------:|:----:|
| conformer_online  |            [WenetSpeech](./docs/wenetspeech.md) (10000小时)             | fbank | 中文  |             |      |
| conformer_online  |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank | 中文  |             |      |
| conformer_offline |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank | 中文  |             |      |
| conformer_online  | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank | 英文  |             |      | 
| conformer_offline | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank | 英文  |             |      | 


2. `deepspeech2`预训练模型列表：

|        使用模型         |                                  数据集                                  | 预处理方式 | 语言  | 测试集字错率（词错率） | 下载地址 |
|:-------------------:|:---------------------------------------------------------------------:|:-----:|:---:|:-----------:|:----:|
| deepspeech2_online  |            [WenetSpeech](./docs/wenetspeech.md) (10000小时)             | fbank | 中文  |             |      |
| deepspeech2_online  |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank | 中文  |             |      |
| deepspeech2_offline |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank | 中文  |             |      |
| deepspeech2_online  | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank | 英文  |             |      | 
| deepspeech2_offline | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank | 英文  |             |      | 


**说明：** 
1. 这里字错率或者词错率是使用`eval.py`程序并使用集束搜索解码`ctc_beam_search`方法计算得到的。
2. 没有提供预测模型，需要把全部文件复制到项目的根目录下，执行`export_model.py`导出预测模型。

>有问题欢迎提 [issue](https://github.com/yeyupiaoling/MASR/issues) 交流

# 语言模型

|                                          语言模型                                          |                                                      训练数据                                                       |  数据量  |  文件大小   |                 说明                  |
|:--------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:-----:|:-------:|:-----------------------------------:|
|         [自定义中文语言模型](https://pan.baidu.com/s/1vdQsqnoKHO9jdFU_1If49g?pwd=ea09)          |                       [自定义中文语料](https://download.csdn.net/download/qq_33200967/87002687)                        | 约2千万  | 572 MB  |           训练参数`-o 5`，无剪枝            |
|  [英文语言模型](https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm)  | [CommonCrawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz) | 18.5亿 | 8.3 GB  | 训练参数`-o 5`，剪枝参数`'--prune 0 1 1 1 1` |
| [中文语言模型（剪枝）](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm) |                                                     百度内部语料库                                                     | 1.3亿  | 2.8 GB  | 训练参数`-o 5`，剪枝参数`'--prune 0 1 1 1 1` |                                     |
|            [中文语言模型](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm)            |                                                     百度内部语料库                                                     |  37亿  | 70.4 GB |           训练参数`-o 5`，无剪枝            |                                     


## 文档教程

- [快速安装](./docs/install.md)
- [快速使用](./docs/GETTING_STARTED.md)
- [数据准备](./docs/dataset.md)
- [WenetSpeech数据集](./docs/wenetspeech.md)
- [合成语音数据](./docs/generate_audio.md)
- [数据增强](./docs/augment.md)
- [训练模型](./docs/train.md)
- [集束搜索解码](./docs/beam_search.md)
- [执行评估](./docs/eval.md)
- [导出模型](./docs/export_model.md)
- [使用标点符号模型](./docs/punctuation.md)
- [使用语音活动检测（VAD）](./docs/vad.md)
- 预测
   - [本地预测](./docs/infer.md)
   - [长语音预测](./docs/infer.md)
   - [Web部署模型](./docs/infer.md)
   - [GUI界面预测](./docs/infer.md)


## 快速预测

 - 下载作者提供的模型或者训练模型，然后执行[导出模型](./docs/export_model.md)，使用`infer_path.py`预测音频，通过参数`--wav_path`指定需要预测的音频路径，完成语音识别，详情请查看[模型部署](./docs/infer.md)。
```shell script
python infer_path.py --wav_path=./dataset/test.wav
```

输出结果：
```
消耗时间：132, 识别结果: 近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书, 得分: 94
```


 - 长语音预测

```shell script
python infer_path.py --wav_path=./dataset/test_vad.wav --is_long_audio=True
```


 - Web部署

![录音测试页面](./docs/images/infer_server.jpg)


 - GUI界面部署

![GUI界面](./docs/images/infer_gui.jpg)


## 相关项目
 - 基于Pytorch实现的声纹识别：[VoiceprintRecognition-Pytorch](https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch)
 - 基于Pytorch实现的分类：[AudioClassification-Pytorch](https://github.com/yeyupiaoling/AudioClassification-Pytorch)
 - 基于PaddlePaddle实现的语音识别：[PPASR](https://github.com/yeyupiaoling/PPASR)


## 参考资料
 - https://github.com/yeyupiaoling/PPASR
 - https://github.com/jiwidi/DeepSpeech-pytorch
 - https://github.com/wenet-e2e/WenetSpeech
 - https://github.com/SeanNaren/deepspeech.pytorch
