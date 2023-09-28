import numpy as np
import torch
from torchaudio.compliance.kaldi import mfcc, fbank

from masr.data_utils.audio import AudioSegment


class AudioFeaturizer(object):
    """音频特征器

    :param sample_rate: 用于训练的音频的采样率
    :type sample_rate: int
    :param use_dB_normalization: 是否对音频进行音量归一化
    :type use_dB_normalization: bool
    :param target_dB: 对音频进行音量归一化的音量分贝值
    :type target_dB: float
    :param train: 是否训练使用
    :type train: bool
    """

    def __init__(self,
                 feature_method='fbank',
                 n_mels=80,
                 n_mfcc=40,
                 sample_rate=16000,
                 cmvn_file: str = None,
                 window: str = 'hamming',
                 frame_length: int = 25,
                 frame_shift: int = 10,
                 lfr_m: int = 1,
                 lfr_n: int = 1,
                 dither: float = 1.0,
                 snip_edges: bool = True,
                 upsacle_samples: bool = True,
                 use_dB_normalization=True,
                 target_dB=-20,
                 train=False):
        self.feature_method = feature_method
        self.target_sample_rate = sample_rate
        self.upsacle_samples = upsacle_samples
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.cmvn_file = cmvn_file
        self.window = window
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.snip_edges = snip_edges
        self.use_dB_normalization = use_dB_normalization
        self.target_dB = target_dB
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.dither = dither
        self.train = train
        self.cmvn = None if self.cmvn_file is None else self.load_cmvn(self.cmvn_file)

    @staticmethod
    def load_cmvn(cmvn_file):
        with open(cmvn_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if line_item[0] == '<AddShift>':
                line_item = lines[i + 1].split()
                if line_item[0] == '<LearnRateCoef>':
                    add_shift_line = line_item[3:(len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            elif line_item[0] == '<Rescale>':
                line_item = lines[i + 1].split()
                if line_item[0] == '<LearnRateCoef>':
                    rescale_line = line_item[3:(len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue
        means = np.array(means_list).astype(np.float32)
        vars = np.array(vars_list).astype(np.float32)
        cmvn = np.array([means, vars])
        cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
        return cmvn

    def featurize(self, audio_segment):
        """从AudioSegment中提取音频特征

        :param audio_segment: Audio segment to extract features from.
        :type audio_segment: AudioSegment
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        # upsampling or downsampling
        if audio_segment.sample_rate != self.target_sample_rate:
            audio_segment.resample(self.target_sample_rate)
        # decibel normalization
        if self.use_dB_normalization:
            audio_segment.normalize(target_db=self.target_dB)
        samples = audio_segment.to('int16')
        waveform = torch.from_numpy(samples).float()
        if self.upsacle_samples:
            waveform = waveform * (1 << 15)
        waveform = waveform.unsqueeze(0)
        dither = self.dither if self.train else 0.0
        # feature method
        if self.feature_method == 'mfcc':
            # 计算MFCC
            feat = mfcc(waveform,
                        num_mel_bins=self.n_mels,
                        num_ceps=self.n_mfcc,
                        frame_length=self.frame_length,
                        frame_shift=self.frame_shift,
                        dither=dither,
                        sample_frequency=audio_segment.sample_rate)
        elif self.feature_method == 'fbank':
            feat = fbank(waveform,
                         num_mel_bins=self.n_mels,
                         frame_length=self.frame_length,
                         frame_shift=self.frame_shift,
                         dither=dither,
                         energy_floor=0.0,
                         window_type=self.window,
                         sample_frequency=audio_segment.sample_rate,
                         snip_edges=self.snip_edges)
        else:
            raise Exception('没有{}预处理方法'.format(self.feature_method))

        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)
        if self.cmvn is not None:
            feat = self.apply_cmvn(feat, self.cmvn)
        feat = feat.numpy()  # (T, 40)
        return feat

    @staticmethod
    def apply_lfr(inputs, lfr_m, lfr_n):
        LFR_inputs = []
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
        inputs = torch.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n:i * lfr_n + lfr_m]).view(1, -1))
            else:
                # process last LFR frame
                num_padding = lfr_m - (T - i * lfr_n)
                frame = (inputs[i * lfr_n:]).view(-1)
                for _ in range(num_padding):
                    frame = torch.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)
        LFR_outputs = torch.vstack(LFR_inputs)
        return LFR_outputs.type(torch.float32)

    @staticmethod
    def apply_cmvn(inputs, cmvn):
        """
        Apply CMVN with mvn data
        """

        device = inputs.device
        frame, dim = inputs.shape

        means = cmvn[0:1, :dim]
        vars = cmvn[1:2, :dim]
        inputs += means.to(device)
        inputs *= vars.to(device)

        return inputs.type(torch.float32)

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self.feature_method == 'mfcc':
            return self.n_mfcc * self.lfr_m
        elif self.feature_method == 'fbank':
            return self.n_mels * self.lfr_m
        else:
            raise Exception('没有{}预处理方法'.format(self.feature_method))
