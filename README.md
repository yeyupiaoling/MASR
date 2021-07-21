# MASR 中文语音识别

**MASR**是一个基于**端到端的深度神经网络**的**中文普通话语音识别**项目，本项目是基于[masr](https://github.com/nobody132/masr) 进行开发的。
## 模型原理

MASR使用的是门控卷积神经网络（Gated Convolutional Network），网络结构类似于Facebook在2016年提出的Wav2letter，只使用卷积神经网络（CNN）实现的语音识别。但是使用的激活函数不是`ReLU`或者是`HardTanh`，而是`GLU`（门控线性单元）。因此称作门控卷积网络。根据实验结显示，使用`GLU`的收敛速度比`HardTanh`要快。

**以下用字错误率CER来衡量模型的表现，CER = 编辑距离 / 句子长度，越低越好，大致可以理解为 1 - CER 就是识别准确率。**

## 安装环境

1. 执行`requirements.txt`安装依赖环境，在安装过程中出现Pyaudio安装错误，可以先执行`sudo apt-get install portaudio19-dev`这个安装，再重新执行。
```shell script
pip install -r requirements.txt
```

2. 安装ctcdecode依赖，该库笔者只在Ubuntu执行成功过，Windows无法编译。
```shell script
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode/third_party
```

由于网络问题，在安装过程中可能无法正常下载以下这两个文件，你需要自行下载这两个文件，并把它们解压到`third_party`目录下。
```shell script
https://sites.google.com/site/openfst/home/openfst-down/openfst-1.6.7.tar.gz
https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz
```

然后回到该源码的根目录，编辑`ctcdecode/setup.py`，注释以下4行代码。
```python
# Download/Extract openfst, boost
download_extract('https://sites.google.com/site/openfst/home/openfst-down/openfst-1.6.7.tar.gz',
                 'third_party/openfst-1.6.7.tar.gz')
download_extract('https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz',
                 'third_party/boost_1_67_0.tar.gz')
```

在ctcdecode根目录下执行以下命令开始安装ctcdecode。
```shell script
pip install .
```

3. 安装warp-CTC，目前warp-CTC只支持CUDA10.1。如果安装过程中出现`c10/cuda/CUDAGuard.h: 没有那个文件或目录`错误，将`pytorch_binding/src/binding.cpp`将`#include <c10/cuda/CUDAGuard.h>`修改成`#include "ATen/cuda/CUDAGuard.h"`。
```shell script
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build
cd build
cmake ..
make -j4
cd ../pytorch_binding
python setup.py install
```
## 准备语言模型和数据集

### 语言模型
下载语言模型并放在lm目录下，下面下载的小语言模型，如何有足够大性能的机器，可以下载70G的超大语言模型，点击下载[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) ，这个模型会大超多。
```shell script
git clone https://github.com/yeyupiaoling/MASR.git
cd MASR/
mkdir lm
cd lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
```

### 语音数据集
1. 在`data`目录下是公开数据集的下载和制作训练数据列表和字典的，本项目提供了下载公开的中文普通话语音数据集，分别是Aishell，Free ST-Chinese-Mandarin-Corpus，THCHS-30 这三个数据集，总大小超过28G。下载这三个数据只需要执行一下代码即可，当然如何想快速训练，也可以只下载其中一个。
```shell script
cd data/
python aishell.py
python free_st_chinese_mandarin_corpus.py
python thchs_30.py
```

如果开发者有自己的数据集，可以使用自己的数据集进行训练，当然也可以跟上面下载的数据集一起训练。自定义的语音数据需要符合一下格式：
1. 语音文件需要放在`dataset/audio/`目录下，例如我们有个`wav`的文件夹，里面都是语音文件，我们就把这个文件存放在`dataset/audio/`。
2. 然后把数据列表文件存在`dataset/annotation/`目录下，程序会遍历这个文件下的所有数据列表文件。例如这个文件下存放一个`my_audio.txt`，它的内容格式如下。每一行数据包含该语音文件的相对路径和该语音文件对应的中文文本，要注意的是该中文文本只能包含纯中文，不能包含标点符号、阿拉伯数字以及英文字母。
```shell script
dataset/audio/wav/0175/H0175A0171.wav 我需要把空调温度调到二十度
dataset/audio/wav/0175/H0175A0377.wav 出彩中国人
dataset/audio/wav/0175/H0175A0470.wav 据克而瑞研究中心监测
dataset/audio/wav/0175/H0175A0180.wav 把温度加大到十八
```

2. 生成训练的数据列表和数据字典。
```shell script
python create_manifest.py
python build_vocab.py
```

如果你的音频数据集的采样率不一致，在执行`create_manifest.py`的时候可以取消以下这个代码的注释，把所有的音频采样率全部转换成16000Hz。
```python
f = wave.open(audio_path, "rb")
str_data = f.readframes(f.getnframes())
f.close()
file = wave.open(audio_path, 'wb')
file.setnchannels(1)
file.setsampwidth(4)
file.setframerate(16000)
file.writeframes(str_data)
file.close()
```

## 训练模型

执行`train.py`代码开始训练。
```shell script
python train.py
```

 - `train_manifest_path`为训练数据列表路径。
 - `dev_manifest_path`每一轮评估的数据列表路径。
 - `vocab_path`数据字典路径。
 - `save_model_path`保存模型的路径。
 - `epochs`训练轮数。
 - `batch_size`batch size大小，最好使用默认的。
 
训练输出结果如下：
```
-----------  Configuration Arguments -----------
batch_size: 32
dev_manifest_path: dataset/manifest.dev
epochs: 1000
save_model_path: save_model/
train_manifest_path: dataset/manifest.train
vocab_path: dataset/zh_vocab.json
------------------------------------------------
[2/200][26600/48334]	Loss = 32.5724	Remain time: 45 days, 18:03:00
[2/200][26700/48334]	Loss = 21.3798	Remain time: 32 days, 22:29:04
[2/200][26800/48334]	Loss = 26.1648	Remain time: 35 days, 2:18:24
[2/200][26900/48334]	Loss = 19.1140	Remain time: 31 days, 11:30:28
[2/200][27000/48334]	Loss = 19.2719	Remain time: 28 days, 17:53:47
[2/200][27100/48334]	Loss = 23.5359	Remain time: 33 days, 18:43:57
[2/200][27200/48334]	Loss = 22.5717	Remain time: 35 days, 18:35:44
[2/200][27300/48334]	Loss = 19.6255	Remain time: 28 days, 17:17:51
```

## 预测
本项目提供了三种预测方式，分别是通过音频路径识别`infer_path.py`，实时录音识别`infer_record.py`和提供HTTP接口识别`infer_server.py`，他们的公共参数`model_path`训练保存的模型路径，`lm_path`为语言模型路径，根据你的电脑性能，使用超大语言模型还是小的语言模型。

 - `infer_path.py`的参数`wav_path`为语音识别的的音频路径。
 - `infer_record.py`的参数`record_time`为录音时间。
 - `infer_server.py`的参数`host`为服务的访问地址，当为localhost时，本地访问页面，可以在浏览器chrome上在线录音，其他的地址可以使用选择音频文件上传获取预测结果。
 
 
## 模型下载
| 训练数据 | 下载链接 |
| :---: | :---: |
| 三个公开的数据集 | [点击下载](https://download.csdn.net/download/qq_33200967/14028460) |
| 超大数据集（超过1300小时） | [点击下载](https://download.csdn.net/download/qq_33200967/16200011) |

## 参考资料
1. https://github.com/nobody132/masr
