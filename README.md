![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/MASR)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/MASR)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/MASR)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

# MASR流式与非流式语音识别项目

MASR是一款基于Pytorch实现的自动语音识别框架，MASR全称是神奇的自动语音识别框架（Magical Automatic Speech Recognition），MASR致力于简单，实用的语音识别项目。可部署在服务器，Nvidia Jetson设备，未来还计划支持Android等移动设备。


**欢迎大家扫码入QQ群讨论**，或者直接搜索QQ群号`1169600237`，问题答案为博主Github的ID`yeyupiaoling`。

<div align="center">
  <img src="docs/images/qq.png"/>
</div>


本项目使用的环境：
 - Anaconda 3
 - Python 3.8
 - Pytorch 1.12.1
 - Windows 10 or Ubuntu 18.04

<!--
## 在线使用

 - [在线使用Dome](https://masr.yeyupiaoling.cn)
-->

## 项目快速了解

 1. 本项目支持流式识别模型`deepspeech2`、`deepspeech2_big`，非流式模型`deepspeech2_no_stream`、`deepspeech2_big_no_stream`。
 2. 本项目支持两种解码器，分别是集束搜索解码器`ctc_beam_search`和贪心解码器`ctc_greedy`，集束搜索解码器`ctc_beam_search`准确率更高，但不支持Windows。


## 更新记录

 - 2022.09.18: 支持使用WebSocket调用流式识别。
 - 2022.08.27: 修改使用kaldi实现`fbank`和`mfcc`预处理方法。
 - 2022.08.22: 增加非流式模型`deepspeech2_no_stream`和`deepspeech2_big_no_stream`。
 - 2022.08.04: 发布1.0版本，优化实时识别流程。
 - 2022.07.12: 完成GUI界面的录音实时识别。
 - 2022.06.14: 支持`deepspeech2_big`模型，适合WenetSpeech大数据集训练模型。
 - 2022.01.16: 支持多种预处理方法。
 - 2022.01.15: 支持英文语音识别。
 - 2022.01.13: 支持给识别结果加标点符号
 - 2021.12.26: 支持pip方式安装。
 - 2021.12.25: 初步完成基本程序。


## 视频讲解

这个是PPSAR的视频教程，项目是通用的，可以参考使用。

 - [知识点讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Rr4y1D7iZ)
 - [流式识别的使用讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Te4y1h7KK)


## 模型下载

本项目支持流式识别模型`deepspeech2`、`deepspeech2_big`，非流式模型`deepspeech2_no_stream`、`deepspeech2_big_no_stream`。

|           使用模型            |                                  数据集                                  | 预处理方式  | 参数大小（M）`*` | 语言  |     测试集字错率（词错率）      |                                    下载地址                                    |
|:-------------------------:|:---------------------------------------------------------------------:|:------:|:-------:|:---:|:--------------------:|:--------------------------------------------------------------------------:|
|      deepspeech2_big      |            [WenetSpeech](./docs/wenetspeech.md) (10000小时)             | fbank  |   167   | 中文  | 0.08944(AIShell的测试集) |    [点击下载](链接：https://pan.baidu.com/s/1tGlHCBHF7vIWfU2N_7FE7A?pwd=j8hi)     |
|        deepspeech2        |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank  |   35    | 中文  |       0.07321        |      [点击下载](https://pan.baidu.com/s/1TuN6AmTk2EzEvwdf7cMZdg?pwd=quez)      |
|      deepspeech2_big      |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank  |   167   | 中文  |      0.04879`*`      |      [点击下载](https://pan.baidu.com/s/1TuN6AmTk2EzEvwdf7cMZdg?pwd=quez)      |
|   deepspeech2_no_stream   |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank  |   98    | 中文  |       0.06518        |      [点击下载](https://pan.baidu.com/s/1TuN6AmTk2EzEvwdf7cMZdg?pwd=quez)      |
| deepspeech2_big_no_stream |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank  |   473   | 中文  |                      |      [点击下载](https://pan.baidu.com/s/1TuN6AmTk2EzEvwdf7cMZdg?pwd=quez)      |
|        deepspeech2        |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | linear |   35    | 中文  |       0.07991        |      [点击下载](https://pan.baidu.com/s/1TuN6AmTk2EzEvwdf7cMZdg?pwd=quez)      |
|      deepspeech2_big      |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | linear |   167   | 中文  |       0.09148        |      [点击下载](https://pan.baidu.com/s/1TuN6AmTk2EzEvwdf7cMZdg?pwd=quez)      |
|   deepspeech2_no_stream   |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | linear |   98    | 中文  |       0.06865        |      [点击下载](https://pan.baidu.com/s/1TuN6AmTk2EzEvwdf7cMZdg?pwd=quez)      |
| deepspeech2_big_no_stream |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | linear |   473   | 中文  |       0.07588        |      [点击下载](https://pan.baidu.com/s/1TuN6AmTk2EzEvwdf7cMZdg?pwd=quez)      |
|        deepspeech2        | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank  |   35    | 英文  |       0.17229        |      [点击下载](https://pan.baidu.com/s/1c57J718blFgUAGqDO-dbJA?pwd=lcjw)      | 
|      deepspeech2_big      | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank  |   167   | 英文  |                      |      [点击下载](https://pan.baidu.com/s/1c57J718blFgUAGqDO-dbJA?pwd=lcjw)      | 
|   deepspeech2_no_stream   | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank  |   98    | 英文  |                      |      [点击下载](https://pan.baidu.com/s/1c57J718blFgUAGqDO-dbJA?pwd=lcjw)      | 
| deepspeech2_big_no_stream | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank  |   473   | 英文  |                      |      [点击下载](https://pan.baidu.com/s/1c57J718blFgUAGqDO-dbJA?pwd=lcjw)      | 
|        deepspeech2        |                   超大数据集(1600多小时真实数据)+(1300多小时合成数据)                    | linear |   35    | 中文  |       0.06215        | [点击下载](https://download.csdn.net/download/qq_33200967/75138230)(需要重新导出模型)  |
|      deepspeech2_big      |                   超大数据集(1600多小时真实数据)+(1300多小时合成数据)                    | linear |   167   | 中文  |       0.05517        | 先`star`项目再[点击下载](https://pan.baidu.com/s/1IW7HJP16IxRHeqSfMfNK5g?pwd=0w36) |


**说明：** 
1. 这里字错率是使用`eval.py`程序并使用集束搜索解码`ctc_beam_search`方法计算得到的。
2. 中文解码参数为：`alpha=2.2，beta=4.3，beam_size=300，cutoff_prob=0.99，cutoff_top_n=40`。
3. 英文解码参数为：`alpha=1.9，beta=0.3，beam_size=500，cutoff_prob=1.0，cutoff_top_n=40`。
4. 除了aishell数据集按照数据集本身划分的训练数据和测试数据，其他的都是按照项目设置的固定比例划分训练数据和测试数据。
5. 下载的压缩文件已经包含了`mean_std.npz`和`vocabulary.txt`，需要把解压得到的全部文件复制到项目根目录下。
6. 模型名称包含`no_stream`为非流式模型，不能用于流式识别。
7. 带有`*`的使用了`WenetSpeech`作为预训练模型。
8. 由于算力不足，大部分的模型都没有训练足够轮数，有算力的同学，欢迎提供模型。
9. 由于音频的长度不一，所以参数大小也有所变化，以上参数大小为同一音频长度下的结果，仅供对比使用。

>有问题欢迎提 [issue](https://github.com/yeyupiaoling/MASR/issues) 交流


## 文档教程

- [快速安装](./docs/install.md)
- [数据准备](./docs/dataset.md)
- [WenetSpeech数据集](./docs/wenetspeech.md)
- [合成语音数据](./docs/generate_audio.md)
- [数据增强](./docs/augment.md)
- [训练模型](./docs/train.md)
- [集束搜索解码](./docs/beam_search.md)
- [执行评估](./docs/eval.md)
- [导出模型](./docs/export_model.md)
- [使用标点符号模型](./docs/punctuation.md)
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
-----------  Configuration Arguments -----------
alpha: 1.2
beam_size: 10
beta: 0.35
cutoff_prob: 1.0
cutoff_top_n: 40
decoding_method: ctc_greedy
enable_mkldnn: False
is_long_audio: False
lang_model_path: ./lm/zh_giga.no_cna_cmn.prune01244.klm
mean_std_path: ./dataset/mean_std.npz
model_dir: ./models/infer/
to_an: True
use_gpu: True
use_tensorrt: False
vocab_path: ./dataset/zh_vocab.txt
wav_path: ./dataset/test.wav
------------------------------------------------
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
