import json

import numpy as np
from torch.utils.data import Dataset

from masr.data_utils.audio import AudioSegment
from masr.data_utils.augmentor.augmentation import AugmentationPipeline
from masr.data_utils.binary import DatasetReader
from masr.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from masr.data_utils.featurizer.text_featurizer import TextFeaturizer
from masr.utils.logger import setup_logger

logger = setup_logger(__name__)


# 音频数据加载器
class MASRDataset(Dataset):
    def __init__(self,
                 preprocess_configs,
                 data_manifest,
                 vocab_filepath,
                 min_duration=0,
                 max_duration=20,
                 augmentation_config='{}',
                 manifest_type='txt',
                 train=False):
        super(MASRDataset, self).__init__()
        self._augmentation_pipeline = AugmentationPipeline(augmentation_config=augmentation_config)
        self._audio_featurizer = AudioFeaturizer(train=train, **preprocess_configs)
        self._text_featurizer = TextFeaturizer(vocab_filepath)
        self.manifest_type = manifest_type
        if self.manifest_type == 'txt':
            # 获取文本格式数据列表
            with open(data_manifest, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.data_list = []
            for line in lines:
                line = json.loads(line)
                # 跳过超出长度限制的音频
                if line["duration"] < min_duration:
                    continue
                if max_duration != -1 and line["duration"] > max_duration:
                    continue
                self.data_list.append(dict(line))
        else:
            # 获取二进制的数据列表
            self.dataset_reader = DatasetReader(data_path=data_manifest,
                                                min_duration=min_duration,
                                                max_duration=max_duration)
            self.data_list = self.dataset_reader.get_keys()

    def __getitem__(self, idx):
        data_list = self.get_one_list(idx)
        # 分割音频路径和标签
        audio_file, transcript = data_list["audio_filepath"], data_list["text"]
        # 如果后缀名为.npy的文件，那么直接读取
        if audio_file.endswith('.npy'):
            start_frame, end_frame = data_list["start_frame"], data_list["end_frame"]
            feature = np.load(audio_file)
            feature = feature[start_frame:end_frame, :]
        else:
            if 'start_time' not in data_list.keys():
                # 读取音频
                audio_segment = AudioSegment.from_file(audio_file)
            else:
                start_time, end_time = data_list["start_time"], data_list["end_time"]
                # 分割读取音频
                audio_segment = AudioSegment.slice_from_file(audio_file, start=start_time, end=end_time)
            # 音频增强
            self._augmentation_pipeline.transform_audio(audio_segment)
            # 预处理，提取特征
            feature = self._audio_featurizer.featurize(audio_segment)
        transcript = self._text_featurizer.featurize(transcript)
        # 特征增强
        feature = self._augmentation_pipeline.transform_feature(feature)
        transcript = np.array(transcript, dtype=np.int32)
        return feature.astype(np.float32), transcript

    def __len__(self):
        return len(self.data_list)

    def get_one_list(self, idx):
        # 获取数据列表
        if self.manifest_type == 'txt':
            data_list = self.data_list[idx]
        elif self.manifest_type == 'binary':
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            raise Exception(f'没有该类型：{self.manifest_type}')
        return data_list

    @property
    def feature_dim(self):
        """返回音频特征大小

        :return: 词汇表大小
        :rtype: int
        """
        return self._audio_featurizer.feature_dim

    @property
    def vocab_size(self):
        """返回词汇表大小

        :return: 词汇表大小
        :rtype: int
        """
        return self._text_featurizer.vocab_size

    @property
    def vocab_list(self):
        """返回词汇表列表

        :return: 词汇表列表
        :rtype: list
        """
        return self._text_featurizer.vocab_list
