![python version](https://img.shields.io/badge/python-3.11+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/MASR)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/MASR)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/MASR)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

# MASR流式与非流式语音识别项目

MASR是一款基于Pytorch实现的自动语音识别框架，MASR全称是神奇的自动语音识别框架（Magical Automatic Speech Recognition），当前为V3版本，与V2版本不兼容，如果想使用V2版本，请在这个分支[V2](https://github.com/yeyupiaoling/MASR/tree/release/2.3.x)。MASR致力于简单，实用的语音识别项目。可部署在服务器，Nvidia Jetson设备，未来还计划支持Android等移动设备。


**欢迎大家扫码入知识星球或者QQ群讨论，知识星球里面提供项目的模型文件和博主其他相关项目的模型文件，也包括其他一些资源。**

<div align="center">
  <img src="https://yeyupiaoling.cn/zsxq.png" alt="知识星球" width="400">
  <img src="https://yeyupiaoling.cn/qq.png" alt="QQ群" width="400">
</div>


本项目使用的环境：
 - Anaconda 3
 - Python 3.11
 - Pytorch 2.5.1
 - Windows 11 or Ubuntu 22.04


# 在线试用

[在线试用地址](https://tools.yeyupiaoling.cn/speech/masr)


## 项目特点

1. 支持多个语音识别模型，包含`deepspeech2`、`conformer`、`squeezeformer`、`efficient_conformer`等，每个模型都支持流式识别和非流式识别，在配置文件中`streaming`参数设置。
2. 支持多种解码器，包含`ctc_greedy_search`、`ctc_prefix_beam_search`、`attention_rescoring`、`ctc_beam_search`等。
3. 支持多种预处理方法，包含`fbank`、`mfcc`等。
4. 支持多种数据增强方法，包含噪声增强、混响增强、语速增强、音量增强、重采样增强、位移增强、SpecAugmentor、SpecSubAugmentor等。
5. 支持多种推理方法，包含短音频推理、长音频推理、流式推理、说话人分离推理等。
6. 更多特点等待你发现。


## 与V2版本的区别

1. 项目结构的优化，大幅度降低的使用难度。
2. 更换预处理的库，改用kaldi_native_fbank，在提高数据预处理的速度，同时也支持多平台。
3. 修改token的方法，使用sentencepiece制作token，这个框架极大的降低了多种语言的处理难度，同时还使中英文混合训练成为可能。


## 更新记录

 - 2025.3: 正式发布最终级的V3版本。

## 模型下载


1. [WenetSpeech](./docs/wenetspeech.md) (10000小时，普通话) 的预训练模型列表，错误率类型为字错率（CER）：

|    使用模型     | 是否为流式 | 预处理方式 |          解码方式          | test_net | test_meeting | aishell_test |   下载地址   |
|:-----------:|:-----:|:-----:|:----------------------:|:--------:|:------------:|:------------:|:--------:|
|  Conformer  | True  | fbank |   ctc_greedy_search    | 0.14391  |   0.18665    |   0.06751    | 加入知识星球获取 |
|  Conformer  | True  | fbank | ctc_prefix_beam_search | 0.14326  |   0.18488    |   0.06763    | 加入知识星球获取 |
|  Conformer  | True  | fbank |  attention_rescoring   | 0.13523  |   0.18069    |   0.06079    | 加入知识星球获取 |
|  Conformer  | True  | fbank |    ctc_beam_search     | 0.18227  |   0.21586    |   0.04981    | 加入知识星球获取 |
| DeepSpeech2 | True  | fbank |   ctc_greedy_search    |          |              |              | 加入知识星球获取 |
| DeepSpeech2 | True  | fbank | ctc_prefix_beam_search |          |              |              | 加入知识星球获取 |
| DeepSpeech2 | True  | fbank |    ctc_beam_search     |          |              |              | 加入知识星球获取 |

2. [AIShell](https://openslr.magicdatatech.com/resources/33) (179小时，普通话) 的预训练模型列表，错误率类型为字错率（CER）：

|    使用模型     | 是否为流式 | 预处理方式 |          解码方式          | 自带的测试集  |   下载地址   |
|:-----------:|:-----:|:-----:|:----------------------:|:-------:|:--------:|
|  Conformer  | True  | fbank |   ctc_greedy_search    | 0.06134 | 加入知识星球获取 |
|  Conformer  | True  | fbank | ctc_prefix_beam_search | 0.06132 | 加入知识星球获取 |
|  Conformer  | True  | fbank |  attention_rescoring   | 0.05366 | 加入知识星球获取 |
|  Conformer  | True  | fbank |    ctc_beam_search     | 0.04409 | 加入知识星球获取 |
| DeepSpeech2 | True  | fbank |   ctc_greedy_search    | 0.12000 | 加入知识星球获取 |
| DeepSpeech2 | True  | fbank | ctc_prefix_beam_search | 0.12016 | 加入知识星球获取 |
| DeepSpeech2 | True  | fbank |    ctc_beam_search     | 0.08748 | 加入知识星球获取 |


3. [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时，英语) 的预训练模型列表，错误率类型为词错率（WER）：

|    使用模型     | 是否为流式 | 预处理方式 |          解码方式          | 自带的测试集  |   下载地址   |
|:-----------:|:-----:|:-----:|:----------------------:|:-------:|:--------:|
|  Conformer  | True  | fbank |   ctc_greedy_search    | 0.07432 | 加入知识星球获取 |
|  Conformer  | True  | fbank | ctc_prefix_beam_search | 0.07418 | 加入知识星球获取 |
|  Conformer  | True  | fbank |  attention_rescoring   | 0.06549 | 加入知识星球获取 |
|  Conformer  | True  | fbank |    ctc_beam_search     |    /    | 加入知识星球获取 |
| DeepSpeech2 | True  | fbank |   ctc_greedy_search    | 0.15491 | 加入知识星球获取 |
| DeepSpeech2 | True  | fbank | ctc_prefix_beam_search | 0.15307 | 加入知识星球获取 |
| DeepSpeech2 | True  | fbank |    ctc_beam_search     |    /    | 加入知识星球获取 |


4. 其他数据集的预训练模型列表，错误率类型，如果是中文就是字错率（CER），英文则是词错率（WER），中英混合为混合错误率（MER）：

|   使用模型    | 是否为流式 | 预处理方式 |       数据集       | 语言  |          解码方式          |                                                        测试数据                                                         |   下载地址   |
|:---------:|:-----:|:-----:|:---------------:|:---:|:----------------------:|:-------------------------------------------------------------------------------------------------------------------:|:--------:|
| Conformer | True  | fbank |      粤语数据集      | 粤语  |   ctc_greedy_search    |                                                       0.05596                                                       | 加入知识星球获取 |
| Conformer | True  | fbank |      粤语数据集      | 粤语  | ctc_prefix_beam_search |                                                       0.05595                                                       | 加入知识星球获取 |
| Conformer | True  | fbank |      粤语数据集      | 粤语  |  attention_rescoring   |                                                       0.04846                                                       | 加入知识星球获取 |
| Conformer | True  | fbank |      粤语数据集      | 粤语  |    ctc_beam_search     |                                                       0.05280                                                       | 加入知识星球获取 |
| Conformer | True  | fbank |     中英混合数据集     | 中英文 |   ctc_greedy_search    |                                                       0.09582                                                       | 加入知识星球获取 |
| Conformer | True  | fbank |     中英混合数据集     | 中英文 | ctc_prefix_beam_search |                                                       0.09523                                                       | 加入知识星球获取 |
| Conformer | True  | fbank |     中英混合数据集     | 中英文 |  attention_rescoring   |                                                       0.08470                                                       | 加入知识星球获取 |
| Conformer | True  | fbank |     中英混合数据集     | 中英文 |    ctc_beam_search     |                                                          /                                                          | 加入知识星球获取 |
| Conformer | True  | fbank | 更大数据集（16000+小时） | 中英文 |   ctc_greedy_search    |              test_net: 0.17378<br>test_meeting: 0.20505<br>Librispeech-Test: 0.20888<br>中英混合: 0.14189               | 加入知识星球获取 |
| Conformer | True  | fbank | 更大数据集（16000+小时） | 中英文 | ctc_prefix_beam_search |              test_net: 0.17311<br>test_meeting: 0.20408<br>Librispeech-Test: 0.20508<br>中英混合: 0.14009               | 加入知识星球获取 |
| Conformer | True  | fbank | 更大数据集（16000+小时） | 中英文 |  attention_rescoring   |              test_net: 0.15607<br>test_meeting: 0.19188<br>Librispeech-Test: 0.17477<br>中英混合: 0.12389               | 加入知识星球获取 |


**说明：** 
1. 这里字错率或者词错率是使用`eval.py`。
2. 分别给出了使用三个解码器的错误率，其中`ctc_prefix_beam_search`、`attention_rescoring`的解码搜索大小为10。
3. 训练时使用了噪声增强和混响增强，以及其他增强方法，具体请看配置参数`configs/augmentation.yml`。
4. 这里只提供了流式模型，但全部模型都支持流式和非流式的，在配置文件中`streaming`参数设置。
5. `更大数据集`准确率比其他的低最主要的是应为训练的epoch太少，但是足以作为其他微调任务的预训练模型。

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
- 预测
   - [本地预测](./docs/infer.md)
   - [说话人日志语音识别](./docs/infer.md)
   - [Web部署模型](./docs/infer.md)
   - [GUI界面预测](./docs/infer.md)
- [常见问题解答](./docs/faq.md)


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
 - https://github.com/wenet-e2e/wenet
 - https://github.com/SeanNaren/deepspeech.pytorch
