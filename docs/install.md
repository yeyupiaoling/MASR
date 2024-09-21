# 安装MASR环境

本人用的就是本地环境和使用Anaconda，并创建了Python3.11的虚拟环境，出现安装问题，随时提[issue](https://github.com/yeyupiaoling/MASR/issues)。

 - 首先安装的是Pytorch 2.4.0的GPU版本，如果已经安装过了，请跳过。
```shell
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
```

 - 安装MASR库。

使用pip安装，命令如下：
```shell
python -m pip install masr -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**建议源码安装**，源码安装能保证使用最新代码。
```shell
git clone https://github.com/yeyupiaoling/MASR.git
cd MASR
pip install .
```

**常见安装问题：** 

1. Linux 报错 OSError: sndfile library not found

```shell
sudo apt-get install libsndfile1
```

2. 如果提示缺少`it`依赖库，请安装。
```shell
python -m pip install WeTextProcessing>=1.0.4.1
```

3. 安装pynini出错，可以执行下面命令安装。
```shell
conda install -c conda-forge pynini
```