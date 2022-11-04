import os

import numpy as np
import torch

from masr.utils.logger import setup_logger

logger = setup_logger(__name__)


class InferencePredictor:
    def __init__(self,
                 configs,
                 use_model,
                 model_dir='models/conformer_online_fbank/infer/',
                 use_gpu=True):
        """
        语音识别预测工具
        :param configs: 配置参数
        :param use_model: 使用模型的名称
        :param model_dir: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        :param gpu_mem: 预先分配的GPU显存大小
        :param num_threads: 只用CPU预测的线程数量
        """
        self.configs = configs
        self.use_gpu = use_gpu
        self.use_model = use_model
        # 流式参数
        self.output_state_h = None
        self.output_state_c = None
        self.cnn_cache = np.zeros([0, 0, 0, 0], dtype=np.float32)
        self.att_cache = np.zeros([0, 0, 0, 0], dtype=np.float32)
        self.offset = np.array([0], dtype=np.int32)
        # 创建 config
        model_path = os.path.join(model_dir, 'model.pdmodel')
        params_path = os.path.join(model_dir, 'model.pdiparams')
        if not os.path.exists(model_path) or not os.path.exists(params_path):
            raise Exception("模型文件不存在，请检查%s和%s是否存在！" % (model_path, params_path))

        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
            self.predictor = torch.load(model_path)
            self.predictor.to(self.device)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
            self.predictor = torch.load(model_path, map_location='cpu')
        self.predictor.eval()
        logger.info(f'已加载模型：{model_path}')

    # 预测音频
    def predict(self, speech, speech_lengths):
        """
        预测函数，只预测完整的一句话。
        :param speech: 经过处理的音频数据
        :param speech_lengths: 音频长度
        :return: 识别的文本结果和解码的得分数
        """
        audio_data = torch.from_numpy(speech).float()
        audio_len = torch.from_numpy(speech_lengths)

        if self.use_gpu:
            audio_data = audio_data.cuda()

        # 非流式模型的输入
        if 'offline' in self.use_model:
            output_data = self.predictor(audio_data, audio_len)
        # 对流式deepspeech2模型
        if self.use_model == 'deepspeech2_online':
            init_state_h_box = None
            init_state_c_box = None
            output_data = self.predictor(audio_data, audio_len, init_state_h_box, init_state_c_box)
        # 对流式conformer模型全
        if self.use_model == 'conformer_online':
            self.reset_stream()
            required_cache_size = np.array([-1], dtype=np.int32)
            output_data = self.predictor(audio_data, self.offset, required_cache_size, self.cnn_cache, self.att_cache)

        return output_data

    def predict_chunk_deepspeech(self, x_chunk):
        if self.use_model != 'deepspeech2_online':
            raise Exception(f'当前模型不支持该方法，当前模型为：{self.use_model}')
        # 设置输入
        x_chunk_lens = np.array([x_chunk.shape[1]])
        self.speech_data_handle.reshape([x_chunk.shape[0], x_chunk.shape[1], x_chunk.shape[2]])
        self.speech_lengths_handle.reshape([x_chunk.shape[0]])
        self.speech_data_handle.copy_from_cpu(x_chunk.astype(np.float32))
        self.speech_lengths_handle.copy_from_cpu(x_chunk_lens.astype(np.int64))

        if self.output_state_h is None:
            # 全零初始化
            self.output_state_h = np.zeros(shape=(self.configs.encoder_conf.num_rnn_layers,
                                                  x_chunk.shape[0],
                                                  self.configs.encoder_conf.rnn_size), dtype=np.float32)
            self.output_state_c = np.zeros(shape=(self.configs.encoder_conf.num_rnn_layers,
                                                  x_chunk.shape[0],
                                                  self.configs.encoder_conf.rnn_size), dtype=np.float32)
        self.init_state_h_box_handle.reshape(self.output_state_h.shape)
        self.init_state_h_box_handle.copy_from_cpu(self.output_state_h)
        self.init_state_c_box_handle.reshape(self.output_state_c.shape)
        self.init_state_c_box_handle.copy_from_cpu(self.output_state_c)

        # 运行predictor
        self.predictor.run()

        # 获取输出
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        output_chunk_probs = output_handle.copy_to_cpu()
        output_lens_handle = self.predictor.get_output_handle(self.output_names[1])
        output_lens = output_lens_handle.copy_to_cpu()
        output_state_h_handle = self.predictor.get_output_handle(self.output_names[2])
        self.output_state_h = output_state_h_handle.copy_to_cpu()
        output_state_c_handle = self.predictor.get_output_handle(self.output_names[3])
        self.output_state_c = output_state_c_handle.copy_to_cpu()
        return output_chunk_probs, output_lens

    def predict_chunk_conformer(self, x_chunk, required_cache_size):
        if self.use_model != 'conformer_online':
            raise Exception(f'当前模型不支持该方法，当前模型为：{self.use_model}')
        # 设置输入
        self.speech_data_handle.reshape([x_chunk.shape[0], x_chunk.shape[1], x_chunk.shape[2]])
        self.speech_data_handle.copy_from_cpu(x_chunk.astype(np.float32))

        self.offset_handle.reshape(self.offset.shape)
        self.offset_handle.copy_from_cpu(self.offset)
        required_cache_size = np.array([required_cache_size], dtype=np.int32)
        self.required_cache_size_handle.reshape(required_cache_size.shape)
        self.required_cache_size_handle.copy_from_cpu(required_cache_size)
        self.cnn_cache_handle.reshape(self.cnn_cache.shape)
        self.cnn_cache_handle.copy_from_cpu(self.cnn_cache)
        self.att_cache_handle.reshape(self.att_cache.shape)
        self.att_cache_handle.copy_from_cpu(self.att_cache)

        # 运行predictor
        self.predictor.run()

        # 获取输出
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        output_chunk_probs = output_handle.copy_to_cpu()
        att_cache_handle = self.predictor.get_output_handle(self.output_names[1])
        self.att_cache = att_cache_handle.copy_to_cpu()
        cnn_cache_handle = self.predictor.get_output_handle(self.output_names[2])
        self.cnn_cache = cnn_cache_handle.copy_to_cpu()
        self.offset += output_chunk_probs.shape[1]
        return output_chunk_probs

    # 重置流式识别，每次流式识别完成之后都要执行
    def reset_stream(self):
        self.output_state_h = None
        self.output_state_c = None
        self.att_cache = np.zeros([0, 0, 0, 0], dtype=np.float32)
        self.cnn_cache = np.zeros([0, 0, 0, 0], dtype=np.float32)
        self.offset = np.array([0], dtype=np.int32)
