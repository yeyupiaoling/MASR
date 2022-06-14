import math

import torch
from torch import nn
from masr.model_utils.deepspeech2.model import DeepSpeech2Model

__all__ = ['Normalizer', 'DeepSpeech2ModelExport']


# 对数据归一化模型
class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = 1e-20

    def forward(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x


# 导出使用的DeepSpeech2Model模型
class DeepSpeech2ModelExport(torch.nn.Module):
    def __init__(self, model:DeepSpeech2Model, feature_mean, feature_std):
        super(DeepSpeech2ModelExport, self).__init__()
        self.normalizer = Normalizer(feature_mean, feature_std)
        self.model = model
        # 在输出层加上Softmax
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, audio, audio_len, init_state_h_box, init_state_c_box):
        x = self.normalizer(audio)
        logits, _, final_chunk_state_h_box, final_chunk_state_c_box = self.model(x, audio_len, init_state_h_box, init_state_c_box)
        output = self.softmax(logits)
        return output, final_chunk_state_h_box, final_chunk_state_c_box


class LinearSpecgram(nn.Module):
    def __init__(self, stride_ms=10.0, window_ms=20.0, audio_rate=16000, use_db_normalization=True, target_db=-20):
        super().__init__()
        self._stride_ms = stride_ms
        self._window_ms = window_ms
        self._audio_rate = audio_rate
        self._use_dB_normalization = use_db_normalization
        self._target_dB = target_db
        self._eps = 1e-14
        self._stride_size = int(0.001 * self._audio_rate * self._stride_ms)
        self._window_size = int(0.001 * self._audio_rate * self._window_ms)

    def forward(self, audio):
        audio = audio.astype(torch.float64)
        audio = self.normalize(audio)
        truncate_size = (audio.shape[-1] - self._window_size) % self._stride_size
        audio = audio[:audio.shape[-1] - truncate_size]
        windows = self.as_strided(audio, kernel_size=self._window_size, strides=self._stride_size)
        weighting = self.hanning(self._window_size)[:, None]
        # 快速傅里叶变换
        fft = torch.fft.rfft(windows * weighting, n=None, dim=0)
        fft = torch.abs(fft)
        fft = fft ** 2
        scale = torch.sum(weighting ** 2) * self._audio_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[0, :] /= scale
        fft[-1, :] /= scale
        return torch.log(fft + self._eps)

    def normalize(self, audio, max_gain_db=300.0):
        """将音频归一化，使其具有所需的有效值(以分贝为单位)"""
        gain = self._target_dB - self.rms_db(audio)
        return self.gain_db(audio, min(max_gain_db, gain))

    @staticmethod
    def gain_db(audio, gain):
        """对音频施加分贝增益"""
        audio *= 10. ** (gain / 20.)
        return audio

    @staticmethod
    def rms_db(audio):
        """返回以分贝为单位的音频均方根能量"""
        mean_square = torch.mean(audio ** 2)
        return 10 * torch.log10(mean_square)

    @staticmethod
    def as_strided(x, kernel_size, strides):  # x.shape[L] kernel_size=320, strides=160
        x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x = torch.nn.functional.unfold(x, kernel_sizes=[1, kernel_size], stride=strides)
        x = x.squeeze(0)
        return x

    @staticmethod
    def hanning(M):  # M=320
        pi = torch.from_numpy(math.pi)
        n = torch.arange(1 - M, M, 2)
        k = pi * n / torch.from_numpy((M - 1))
        return 0.5 + 0.5 * torch.cos(k)
