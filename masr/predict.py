import json
import os
import sys
from io import BufferedReader
from typing import Union, List

import numpy as np
import torch
import yaml
from loguru import logger
from yeaudio.audio import AudioSegment
from yeaudio.vad_model import VadOnlineModel

from masr.data_utils.audio_featurizer import AudioFeaturizer
from masr.data_utils.tokenizer import MASRTokenizer
from masr.decoders.ctc_prefix_beam_search import ctc_prefix_beam_search
from masr.decoders.attention_rescoring import attention_rescoring
from masr.decoders.ctc_greedy_search import ctc_greedy_search
from masr.infer_utils.inference_predictor import InferencePredictor
from masr.utils.utils import dict_to_object, print_arguments


class MASRPredictor:
    def __init__(self,
                 model_dir: str = 'models/ConformerModel_fbank/inference_model/',
                 decoder: str = "ctc_greedy_search",
                 decoder_configs: Union[str, dict] = None,
                 punc_model_dir: str = None,
                 punc_online_model_dir: str = None,
                 punc_device_id: Union[str, int] = "-1",
                 use_gpu: bool = True,
                 log_level: str = "info"):
        """MASR语音识别识别工具类

        :param model_dir: 导出的预测模型文件夹路径
        :param decoder: 解码器，支持 ctc_greedy_search、ctc_prefix_beam_search、attention_rescoring、ctc_beam_search
        :param decoder_configs: 解码器配置参数文件路径，支持yaml格式
        :param punc_model_dir: 离线标点符号的模型文件夹路径
        :param punc_online_model_dir: 在线标点符号的模型文件夹路径
        :param punc_device_id: 标点符号预测设备ID，-1表示使用CPU预测，否则使用指定GPU预测
        :param use_gpu: 是否使用GPU预测
        :param log_level: 打印的日志等级，可选值有："debug", "info", "warning", "error"
        """
        model_path = os.path.join(model_dir, 'inference.pth')
        model_info_path = os.path.join(model_dir, 'inference.json')
        assert os.path.exists(model_path), f'模型文件[{model_path}]不存在，请检查该文件是否存在！'
        assert os.path.exists(model_info_path), f'模型配置文件[{model_info_path}]不存在，请检查该文件是否存在！'
        self.log_level = log_level.upper()
        logger.remove()
        logger.add(sink=sys.stdout, level=self.log_level)
        with open(model_info_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
            print_arguments(configs=configs, title="模型参数配置")
        self.model_info = dict_to_object(configs)
        self.decoder = decoder
        if self.model_info.model_name == "DeepSpeech2Model":
            assert self.decoder != "attention_rescoring", f'DeepSpeech2Model不支持使用{decoder}解码器！'
        # 读取解码器配置文件
        if isinstance(decoder_configs, str):
            # 获取默认解码配置文件路径
            decoder_configs_path = os.path.join(os.path.dirname(__file__), f"configs/{decoder_configs}.yml")
            decoder_configs = decoder_configs_path if os.path.exists(decoder_configs_path) else decoder_configs
            if os.path.exists(decoder_configs):
                with open(decoder_configs, 'r', encoding='utf-8') as f:
                    decoder_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
                print_arguments(configs=decoder_configs, title='解码器参数配置')
            print_arguments(configs=decoder_configs, title='解码器参数配置')
        self.decoder_configs = decoder_configs if decoder_configs is not None else {}
        self.running = False
        self.use_gpu = use_gpu
        self.inv_normalizer = None
        self.punc_predictor = None
        self.pun_online_predictor = None
        self.sd_predictor = None
        vocab_model_dir = os.path.join(model_dir, 'vocab_model/')
        assert os.path.exists(vocab_model_dir), f'词表模型文件夹[{vocab_model_dir}]不存在，请检查该文件是否存在！'
        self._tokenizer = MASRTokenizer(vocab_model_dir=vocab_model_dir)
        self._audio_featurizer = AudioFeaturizer(**self.model_info.preprocess_conf)
        # 流式解码参数
        self.remained_wav = None
        self.cached_feat = None
        self.last_chunk_text = ""
        self.punc_param_dict = {"cache": []}
        self.vad_param_dict = {"in_cache": []}
        self.last_punc_text_len = 0
        self.last_chunk_time = 0
        self.reset_state_time = 30
        self.beam_search_decoder = None
        if self.decoder == "ctc_beam_search":
            from masr.decoders.beam_search_decoder import BeamSearchDecoder
            decoder_args = self.decoder_configs.get('ctc_beam_search_args', {})
            self.beam_search_decoder = BeamSearchDecoder(vocab_list=self._tokenizer.vocab_list,
                                                         blank_id=self._tokenizer.blank_id,
                                                         **decoder_args)
        # 创建模型
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在，请检查{}是否存在！".format(model_path))
        # 加载标点符号模型
        if punc_model_dir:
            from masr.infer_utils.punc_predictor import PunctuationPredictor
            self.punc_predictor = PunctuationPredictor(model_dir=punc_model_dir, device_id=punc_device_id)
            logger.info("标点符号模型已加载完成")
        # 加载在线标点符号模型，仅在流式模式下使用
        if punc_online_model_dir and self.model_info.streaming:
            from masr.infer_utils.punc_predictor import PunctuationOnlinePredictor
            self.pun_online_predictor = PunctuationOnlinePredictor(model_dir=punc_online_model_dir,
                                                                   device_id=punc_device_id)
            logger.info("在线标点符号模型已加载完成")
        # 获取预测器
        self.predictor = InferencePredictor(model_name=self.model_info.model_name,
                                            streaming=self.model_info.streaming,
                                            model_path=model_path,
                                            use_gpu=self.use_gpu)
        # 加载流式VAD模型
        if self.model_info.streaming:
            self.vad_online_model = VadOnlineModel(device_id=punc_device_id)
            logger.info("流式VAD模型已加载完成")
        # 预热
        logger.info("开始预热预测器...")
        warmup_audio = np.random.uniform(low=-2.0, high=2.0, size=(130000,))
        self.predict(audio_data=warmup_audio, is_itn=False)
        if self.model_info.streaming:
            for i in range(0, warmup_audio.shape[0], 1600):
                chunk_waveform = warmup_audio[i:i + 1600]
                self.predict_stream(audio_data=chunk_waveform, is_itn=False)
            self.reset_predictor()
            self.reset_stream_state()
        logger.info("预测器已准备完成！")

    # 解码模型输出结果
    def decode(self, encoder_outs, ctc_probs, ctc_lens, use_punc, is_itn):
        """解码模型输出结果

        :param encoder_outs: 编码器输出
        :param ctc_probs: 模型输出的CTC概率
        :param ctc_lens: 模型输出的CTC长度
        :param use_punc: 是否使用加标点符号
        :param is_itn: 是否对文本进行反标准化
        :return:
        """
        # 执行解码
        if self.decoder == "ctc_greedy_search":
            result = ctc_greedy_search(ctc_probs=ctc_probs, ctc_lens=ctc_lens, blank_id=self._tokenizer.blank_id)
            text = self._tokenizer.ids2text(result[0])
        elif self.decoder == "ctc_prefix_beam_search":
            decoder_args = self.decoder_configs.get('ctc_prefix_beam_search_args', {})
            result, _ = ctc_prefix_beam_search(ctc_probs=ctc_probs, ctc_lens=ctc_lens,
                                               blank_id=self._tokenizer.blank_id, **decoder_args)
            text = self._tokenizer.ids2text(result[0])
        elif self.decoder == "attention_rescoring":
            decoder_args = self.decoder_configs.get('attention_rescoring_args', {})
            result = attention_rescoring(model=self.predictor.model,
                                         ctc_probs=ctc_probs,
                                         ctc_lens=ctc_lens,
                                         blank_id=self._tokenizer.blank_id,
                                         encoder_outs=encoder_outs,
                                         encoder_lens=ctc_lens,
                                         **decoder_args)
            text = self._tokenizer.ids2text(result[0])
        elif self.decoder == "ctc_beam_search":
            text = self.beam_search_decoder.ctc_beam_search_decoder(ctc_probs=ctc_probs)
        else:
            raise ValueError(f"不支持该解码器：{self.decoder}")

        # 加标点符号
        if use_punc and len(text) > 0:
            if self.punc_predictor is not None:
                text = self.punc_predictor(text)[0]
            else:
                logger.warning('标点符号模型没有初始化！')
        # 是否对文本进行反标准化
        if is_itn:
            text = self.inverse_text_normalization(text)
        return text

    @staticmethod
    def _load_audio(audio_data, sample_rate=16000):
        """加载音频
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        return audio_segment

    # 预测音频
    def _infer(self,
               audio_data,
               use_punc=False,
               is_itn=False,
               sample_rate=16000):
        """
        预测函数，只预测完整的一句话。
        :param audio_data: 音频数据
        :type audio_data: np.ndarray
        :param use_punc: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 提取音频特征
        audio_feature = self._audio_featurizer.featurize(waveform=audio_data, sample_rate=sample_rate)
        audio_feature = torch.tensor(audio_feature, dtype=torch.float32).unsqueeze(0)
        audio_len = torch.tensor([audio_feature.size(1)], dtype=torch.int64)

        # 运行predictor
        encoder_outs, ctc_probs, ctc_lens = self.predictor.predict(audio_feature, audio_len)

        # 解码
        text = self.decode(encoder_outs=encoder_outs, ctc_probs=ctc_probs, ctc_lens=ctc_lens,
                           use_punc=use_punc, is_itn=is_itn)
        return text

    # 语音预测
    def predict(self,
                audio_data,
                use_punc=False,
                is_itn=False,
                sample_rate=16000,
                allow_use_vad=True):
        """
        预测函数，只预测完整的一句话。
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param use_punc: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :param allow_use_vad: 当音频长度大于30秒，是否允许使用语音活动检测分割音频进行识别
        :return: 识别的文本结果和解码的得分数
        """
        if isinstance(audio_data, np.ndarray):
            assert isinstance(sample_rate, int), '当传入的是numpy数据时，需要指定采样率'
        # 加载音频文件，并进行预处理
        audio_segment = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        # 重采样，方便进行语音活动检测
        if audio_segment.sample_rate != self.model_info.sample_rate:
            audio_segment.resample(self.model_info.sample_rate)
        if audio_segment.duration <= 30 or not allow_use_vad:
            text = self._infer(audio_data=audio_segment.samples, use_punc=use_punc, is_itn=is_itn,
                               sample_rate=audio_segment.sample_rate)
            result = {'text': text,
                      'sentences': [{'text': text, 'start': 0, 'end': audio_segment.duration}]}
            return result
        elif allow_use_vad and audio_segment.duration > 30:
            last_audio_ndarray = None
            # 获取语音活动区域
            speech_timestamps = audio_segment.vad()
            texts, sentences = '', []
            for t in speech_timestamps:
                audio_ndarray = audio_segment.samples[int(t['start']): int(t['end'])]
                # 如果语音片段小于0.5秒，则跳过推理，下次合并使用
                if (t['end'] - t['start']) * audio_segment.sample_rate < 0.5 and last_audio_ndarray is None:
                    continue
                if last_audio_ndarray is not None:
                    audio_ndarray = np.concatenate((last_audio_ndarray, audio_ndarray))
                    last_audio_ndarray = None
                # 执行识别
                text = self._infer(audio_data=audio_ndarray, use_punc=False, is_itn=is_itn,
                                   sample_rate=audio_segment.sample_rate)
                if text != '':
                    texts = texts + text if use_punc else texts + '，' + text
                sentences.append({'text': text,
                                  'start': round(t['start'] / audio_segment.sample_rate, 3),
                                  'end': round(t['end'] / audio_segment.sample_rate, 3)})
                logger.info(f'长语音识别片段结果：{text}')
            if texts[0] == '，': texts = texts[1:]
            # 加标点符号
            if use_punc and len(texts) > 0:
                if self.punc_predictor is not None:
                    texts = self.punc_predictor(texts)
                else:
                    logger.warning('标点符号模型没有初始化！')
            result = {'text': texts, 'sentences': sentences}
            return result

    def predict_sd_asr(self,
                       audio_data,
                       vector_configs,
                       vector_model_path,
                       vector_threshold=0.6,
                       audio_db_path=None,
                       speaker_num=None,
                       search_audio_db=None,
                       use_punc=False,
                       is_itn=False,
                       sample_rate=16000):
        """
        预测函数，只预测完整的一句话。
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param vector_configs: 配置参数
        :param vector_model_path: 声纹模型文件夹路径
        :param vector_threshold: 判断是否为同一个人的阈值
        :param audio_db_path: 声纹库路径
        :param speaker_num: 说话人数量，提供说话人数量可以提高准确率
        :param search_audio_db: 是否在音频库中搜索对应的说话人
        :param use_punc: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        if self.sd_predictor is None:
            from mvector.predict import MVectorPredictor
            # 获取识别器
            self.sd_predictor = MVectorPredictor(configs=vector_configs,
                                                 model_path=vector_model_path,
                                                 threshold=vector_threshold,
                                                 audio_db_path=audio_db_path,
                                                 use_gpu=self.use_gpu)
        # 进行说话人日志识别
        sd_results = self.sd_predictor.speaker_diarization(audio_data,
                                                           speaker_num=speaker_num,
                                                           search_audio_db=search_audio_db)
        # 合并相近的说话人
        sd_results = self._merge_nearby_speakers(sd_results)
        # 加载音频文件
        audio_segment = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        results = []
        for result in sd_results:
            speaker = int(result['speaker'])
            start_time = result['start']
            end_time = result['end']
            # 如果音频时长大于30秒，需要循环分段识别
            if end_time - start_time > 30:
                text = ""
                for i in range(int((end_time - start_time) / 30) + 1):
                    start = int((start_time + i * 30) * audio_segment.sample_rate)
                    end = int((start_time + (i + 1) * 30) * audio_segment.sample_rate)
                    audio_ndarray = audio_segment.samples[start: end]
                    text += self._infer(audio_data=audio_ndarray, use_punc=use_punc, is_itn=is_itn,
                                        sample_rate=audio_segment.sample_rate)
            else:
                start = int(start_time * audio_segment.sample_rate)
                end = int(end_time * audio_segment.sample_rate)
                audio_ndarray = audio_segment.samples[start: end]
                text = self._infer(audio_data=audio_ndarray, use_punc=use_punc, is_itn=is_itn,
                                   sample_rate=audio_segment.sample_rate)
            result = {'speaker': speaker, 'text': text, 'start': result['start'], 'end': result['end']}
            logger.info(f'说话人识别结果：{result}')
            results.append(result)
        return results

    # 把相邻的说话人合并
    @staticmethod
    def _merge_nearby_speakers(sd_results: List[dict]):
        results = [sd_results[0]]
        for i in range(1, len(sd_results)):
            sd_result = sd_results[i]
            if results[-1]['speaker'] == sd_result['speaker'] and sd_result['end'] - results[-1]['start'] < 20:
                results[-1]['end'] = sd_result['end']
            else:
                results.append(sd_result)
        return results

    # 预测音频
    def predict_stream(self,
                       audio_data,
                       is_final=False,
                       use_punc=False,
                       is_itn=False,
                       channels=1,
                       samp_width=2,
                       sample_rate=16000):
        """
        预测函数，流式预测，通过一直输入音频数据，实现实时识别。
        :param audio_data: 需要预测的音频wave读取的字节流或者未预处理的numpy值
        :param is_final: 是否结束语音识别
        :param use_punc: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :param channels: 如果传入的是pcm字节流数据，需要指定通道数
        :param samp_width: 如果传入的是pcm字节流数据，需要指定音频宽度
        :param sample_rate: 如果传入的是numpy或者pcm字节流数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        assert self.model_info.streaming, f'不支持改该模型流式识别，当前模型：{self.model_info.model_name}'
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, np.ndarray):
            audio_data = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_data = AudioSegment.from_pcm_bytes(audio_data, channels=channels,
                                                     samp_width=samp_width, sample_rate=sample_rate)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        # 统计时间
        self.last_chunk_time += audio_data.duration
        if self.remained_wav is None:
            self.remained_wav = audio_data
        else:
            self.remained_wav = AudioSegment(np.concatenate([self.remained_wav.samples, audio_data.samples]),
                                             audio_data.sample_rate)

        # 预处理语音块
        x_chunk = self._audio_featurizer.featurize(waveform=self.remained_wav.samples,
                                                   sample_rate=self.remained_wav.sample_rate)
        x_chunk = x_chunk.astype(np.float32)[np.newaxis, :]
        if self.cached_feat is None:
            self.cached_feat = x_chunk
        else:
            self.cached_feat = np.concatenate([self.cached_feat, x_chunk], axis=1)
        self.remained_wav._samples = self.remained_wav.samples[160 * x_chunk.shape[1]:]

        # 识别的数据块大小
        decoding_chunk_size = 16
        context = 7
        subsampling = 4

        cached_feature_num = context - subsampling
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        stride = subsampling * decoding_chunk_size

        # 保证每帧数据长度都有效
        num_frames = self.cached_feat.shape[1]
        if num_frames < decoding_window and not is_final: return None
        if num_frames < context: return None

        # 如果识别结果，要使用最后一帧
        left_frames = context if is_final else decoding_window

        score, text, end = None, None, None
        for cur in range(0, num_frames - left_frames + 1, stride):
            end = min(cur + decoding_window, num_frames)
            # 获取数据块
            x_chunk = self.cached_feat[:, cur:end, :]

            # 执行识别
            if self.model_info.model_name == 'DeepSpeech2Model':
                output_chunk_probs, output_lens = self.predictor.predict_chunk_deepspeech(x_chunk=x_chunk)
            elif self.model_info.model_name == 'ConformerModel' or self.model_info.model_name == 'SqueezeformerModel' or \
                    self.model_info.model_name == 'EfficientConformerModel':
                num_decoding_left_chunks = -1
                required_cache_size = decoding_chunk_size * num_decoding_left_chunks
                output_chunk_probs = self.predictor.predict_chunk_conformer(x_chunk=x_chunk,
                                                                            required_cache_size=required_cache_size)
                output_lens = torch.tensor([output_chunk_probs.size(1)], dtype=torch.int32,
                                           device=output_chunk_probs.device)
            else:
                raise Exception(f'当前模型不支持该方法，当前模型为：{self.model_info.model_name}')
            # 执行解码
            chunk_result = ctc_greedy_search(ctc_probs=output_chunk_probs, ctc_lens=output_lens,
                                             blank_id=self._tokenizer.blank_id)[0]
            chunk_text = self._tokenizer.ids2text(chunk_result)
            self.last_chunk_text = self.last_chunk_text + chunk_text
        # 更新特征缓存
        self.cached_feat = self.cached_feat[:, end - cached_feature_num:, :]

        # 实时检测VAD
        vad_state = []
        if self.vad_online_model is not None:
            self.vad_param_dict["is_final"] = is_final
            vad_state = self.vad_online_model(audio_data.samples, param_dict=self.vad_param_dict)

        # 如果是静音并且推理音频时间足够长，重置流式识别状态，以免显存不足
        if (len(vad_state) > 0 and vad_state[-1][0] != -1 and self.last_chunk_time > self.reset_state_time) or is_final:
            self.reset_predictor()
            self.last_chunk_time = 0
            # 加标点符号
            if use_punc and len(self.last_chunk_text) > 0:
                if self.pun_online_predictor is not None:
                    result_text = self.pun_online_predictor(text=self.last_chunk_text[self.last_punc_text_len:],
                                                            param_dict=self.punc_param_dict)[0]
                    self.last_chunk_text += result_text
                    self.last_punc_text_len = len(self.last_chunk_text)
                else:
                    logger.warning('标点符号模型没有初始化！')
            # 是否对文本进行逆标准化
            if is_itn and len(self.last_chunk_text) > 0:
                self.last_chunk_text = self.inverse_text_normalization(self.last_chunk_text)

        result = {'text': self.last_chunk_text}
        return result

    # 重置预测器
    def reset_predictor(self):
        self.predictor.reset_stream()
        logger.info('重置预测器')

    # 重置流式识别状态
    def reset_stream_state(self):
        self.last_chunk_time = 0
        self.remained_wav = None
        self.cached_feat = None
        self.last_chunk_text = ''
        self.last_punc_text_len = 0
        self.punc_param_dict = {"cache": []}
        self.vad_param_dict = {"in_cache": []}

    # 对文本进行反标准化
    def inverse_text_normalization(self, text):
        if self.inv_normalizer is None:
            # 需要安装WeTextProcessing>=1.0.4.1
            from itn.chinese.inverse_normalizer import InverseNormalizer
            user_dir = os.path.expanduser('~')
            cache_dir = os.path.join(user_dir, '.cache/itn_v1.0.4.1')
            exists = os.path.exists(os.path.join(cache_dir, 'zh_itn_tagger.fst')) and \
                     os.path.exists(os.path.join(cache_dir, 'zh_itn_verbalizer.fst'))
            self.inv_normalizer = InverseNormalizer(cache_dir=cache_dir, enable_0_to_9=False,
                                                    overwrite_cache=not exists)
        result_text = self.inv_normalizer.normalize(text)
        return result_text
