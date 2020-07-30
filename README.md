# MASR 中文语音识别

**MASR**是一个基于**端到端的深度神经网络**的**中文普通话语音识别**项目，本项目是基于[https://github.com/nobody132/masr](https://github.com/nobody132/masr)进行开发的，。

## 模型原理

MASR使用的是门控卷积神经网络（Gated Convolutional Network），网络结构类似于Facebook在2016年提出的Wav2letter，只使用卷积神经网络（CNN）实现的语音识别。但是使用的激活函数不是`ReLU`或者是`HardTanh`，而是`GLU`（门控线性单元）。因此称作门控卷积网络。根据我的实验，使用`GLU`的收敛速度比`HardTanh`要快。

**以下用字错误率CER来衡量模型的表现，CER = 编辑距离 / 句子长度，越低越好，大致可以理解为 1 - CER 就是识别准确率。**

## 安装环境

1. 执行`requirements.txt`安装依赖环境。
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

然后回到该源码的根目录，编辑`ctcdecode/build.py`，注释以下4行代码。
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
3. 安装warp-CTC。
```shell script
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build
cd build
cmake ..
make
cd ../pytorch_binding
python setup.py install
```
## 准备语言模型和数据集

### 语言模型
下载语言模型并放在lm目录下，以下是下载的是70G的超大语言模型，如果不想使用这么大的，可以下载[Mandarin LM Small](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm) ，这个模型会小很多。
```shell script
git clone https://github.com/yeyupiaoling/MASR.git
cd MASR/
mkdir lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm
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
./dataset/audio/wav/0175/H0175A0171.wav 我需要把空调温度调到二十度
./dataset/audio/wav/0175/H0175A0377.wav 出彩中国人
./dataset/audio/wav/0175/H0175A0470.wav 据克而瑞研究中心监测
./dataset/audio/wav/0175/H0175A0180.wav 把温度加大到十八
```

2. 生成训练的数据列表和数据字典。
```shell script
python create_manifest.py
python build_vocab.py
```

## 训练模型

执行`train.py`代码开始训练。
```shell script
python train.py
```

 - `train_manifest_path`
 - `dev_manifest_path`
 - `vocab_path`
 - `save_model_path`
 - `epochs`
 - `batch_size`