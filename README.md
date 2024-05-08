![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/MASR)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/MASR)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/MASR)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

# MASR流式与非流式语音识别项目

MASR是一款基于Pytorch实现的自动语音识别框架，MASR全称是神奇的自动语音识别框架（Magical Automatic Speech Recognition），当前为V2版本，如果想使用V1版本，请在这个分支[r1.x](https://github.com/yeyupiaoling/MASR/tree/r1.x)。MASR致力于简单，实用的语音识别项目。可部署在服务器，Nvidia Jetson设备，未来还计划支持Android等移动设备。


**欢迎大家扫码入知识星球或者QQ群讨论，知识星球里面提供项目的模型文件和博主其他相关项目的模型文件，也包括其他一些资源。**

<div align="center">
  <img src="https://yeyupiaoling.cn/zsxq.png" alt="知识星球" width="400">
  <img src="https://yeyupiaoling.cn/qq.png" alt="QQ群" width="400">
</div>


本项目使用的环境：
 - Anaconda 3
 - Python 3.11
 - Pytorch 2.0.1
 - Windows 10 or Ubuntu 18.04


## 项目快速了解

 1. 本项目支持流式识别模型`deepspeech2`、`conformer`、`squeezeformer`，`efficient_conformer`，每个模型都支持流式识别和非流式识别，在配置文件中`streaming`参数设置。
 2. 本项目支持两种解码器，分别是集束搜索解码器`ctc_beam_search`和贪心解码器`ctc_greedy`，集束搜索解码器`ctc_beam_search`准确率更高。
 3. 下面提供了一系列预训练模型的下载，下载预训练模型之后，需要把全部文件复制到项目根目录，并执行导出模型才可以使用语音识别。


## 更新记录

 - 2023.01.28: 调整配置文件结构，支持efficient_conformer模型。
 - 2022.11: 正式发布最终级的V2版本。


## 视频讲解

这个是PPSAR的视频教程，项目是通用的，可以参考使用。

 - [知识点讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Rr4y1D7iZ)
 - [流式识别的使用讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Te4y1h7KK)

## 在线使用

**- [在线使用Dome](https://www.doiduoyi.com/?app=SPEECHRECOG)**

# 快速使用

这里介绍如何使用MASR快速进行语音识别，前提是要安装MASR，文档请看[快速安装](./docs/install.md)。执行过程不需要手动下载模型，全部自动完成。

1. 短语音识别
```python
from masr.predict import MASRPredictor

predictor = MASRPredictor(model_tag='conformer_streaming_fbank_aishell')

wav_path = 'dataset/test.wav'
result = predictor.predict(audio_data=wav_path, use_pun=False)
score, text = result['score'], result['text']
print(f"识别结果: {text}, 得分: {int(score)}")
```

2. 长语音识别
```python
from masr.predict import MASRPredictor

predictor = MASRPredictor(model_tag='conformer_streaming_fbank_aishell')

wav_path = 'dataset/test_long.wav'
result = predictor.predict_long(audio_data=wav_path, use_pun=False)
score, text = result['score'], result['text']
print(f"识别结果: {text}, 得分: {score}")
```

3. 模拟流式识别
```python
import time
import wave

from masr.predict import MASRPredictor

predictor = MASRPredictor(model_tag='conformer_streaming_fbank_aishell')

# 识别间隔时间
interval_time = 0.5
CHUNK = int(16000 * interval_time)
# 读取数据
wav_path = 'dataset/test.wav'
wf = wave.open(wav_path, 'rb')
data = wf.readframes(CHUNK)
# 播放
while data != b'':
    start = time.time()
    d = wf.readframes(CHUNK)
    result = predictor.predict_stream(audio_data=data, use_pun=False, is_end=d == b'')
    data = d
    if result is None: continue
    score, text = result['score'], result['text']
    print(f"【实时结果】：消耗时间：{int((time.time() - start) * 1000)}ms, 识别结果: {text}, 得分: {int(score)}")
# 重置流式识别
predictor.reset_stream()
```


## 模型下载


1. [WenetSpeech](./docs/wenetspeech.md) (10000小时) 的预训练模型列表：

|   使用模型    | 是否为流式 | 预处理方式 | 语言  | 测试集字错率 | 下载地址 |
|:---------:|:-----:|:-----:|:---:|:------:|:----:|
| conformer | True  | fbank | 普通话 |        |      |


2.  [WenetSpeech](./docs/wenetspeech.md) (10000小时)+[中文语音数据集](https://download.csdn.net/download/qq_33200967/87003964) (3000+小时) 的预训练模型列表：

|    使用模型    | 是否为流式 | 预处理方式 | 语言  |                               测试集字错率                                |   下载地址   |
|:----------:|:-----:|:-----:|:---:|:-------------------------------------------------------------------:|:--------:|
| conformere | True  | fbank | 普通话 | 0.03179(aishell_test)<br>0.16722(test_net)<br>0.20317(test_meeting) | 加入知识星球获取 |


3. [AIShell](https://openslr.magicdatatech.com/resources/33) (179小时) 的预训练模型列表：

|        使用模型         | 是否为流式 | 预处理方式 | 语言  | 测试集字错率  |   下载地址   |
|:-------------------:|:-----:|:-----:|:---:|:-------:|:--------:|
|    squeezeformer    | True  | fbank | 普通话 | 0.04137 | 加入知识星球获取 |
|      conformer      | True  | fbank | 普通话 | 0.04491 | 加入知识星球获取 |
| efficient_conformer | True  | fbank | 普通话 | 0.04073 | 加入知识星球获取 |
|     deepspeech2     | True  | fbank | 普通话 | 0.06907 | 加入知识星球获取 |


4. [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) 的预训练模型列表：

|        使用模型         | 是否为流式 | 预处理方式 | 语言 | 测试集词错率  |   下载地址   |
|:-------------------:|:-----:|:-----:|:--:|:-------:|:--------:|
|    squeezeformer    | True  | fbank | 英文 | 0.09715 | 加入知识星球获取 | 
|      conformer      | True  | fbank | 英文 | 0.09265 | 加入知识星球获取 | 
| efficient_conformer | True  | fbank | 英文 |         | 加入知识星球获取 | 
|     deepspeech2     | True  | fbank | 英文 | 0.19423 | 加入知识星球获取 | 


**说明：** 
1. 这里字错率或者词错率是使用`eval.py`程序并使用集束搜索解码`ctc_beam_search`方法计算得到的。
2. 没有提供预测模型，需要把全部文件复制到项目的根目录下，执行`export_model.py`导出预测模型。
3. 由于算力不足，这里只提供了流式模型，但全部模型都支持流式和非流式的，在配置文件中`streaming`参数设置。

>有问题欢迎提 [issue](https://github.com/yeyupiaoling/MASR/issues) 交流


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


## 相关项目
 - 基于Pytorch实现的声纹识别：[VoiceprintRecognition-Pytorch](https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch)
 - 基于Pytorch实现的分类：[AudioClassification-Pytorch](https://github.com/yeyupiaoling/AudioClassification-Pytorch)
 - 基于PaddlePaddle实现的语音识别：[PPASR](https://github.com/yeyupiaoling/PPASR)


## 打赏作者

<br/>
<div align="center">
<p>打赏一块钱支持一下作者</p>
<img src="https://yeyupiaoling.cn/reward.png" alt="打赏作者" width="400">
</div>


## 参考资料
 - https://github.com/yeyupiaoling/PPASR
 - https://github.com/jiwidi/DeepSpeech-pytorch
 - https://github.com/wenet-e2e/WenetSpeech
 - https://github.com/SeanNaren/deepspeech.pytorch
