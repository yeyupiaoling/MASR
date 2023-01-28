import os
import torch

from masr.utils.logger import setup_logger

logger = setup_logger(__name__)


class InferencePredictor:
    def __init__(self,
                 configs,
                 use_model,
                 streaming=True,
                 model_path='models/conformer_streaming_fbank/inference.pt',
                 use_gpu=True):
        """
        语音识别预测工具
        :param configs: 配置参数
        :param use_model: 使用模型的名称
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        self.configs = configs
        self.use_gpu = use_gpu
        self.use_model = use_model
        self.streaming = streaming
        # 创建模型
        if not os.path.exists(model_path):
            raise Exception(f"模型文件不存在，请检查{model_path}是否存在！")

        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
            self.predictor = torch.jit.load(model_path)
            self.predictor.to(self.device)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
            self.predictor = torch.jit.load(model_path, map_location='cpu')
            self.predictor.to(self.device)
        self.predictor.eval()
        logger.info(f'已加载模型：{model_path}')

        # 流式参数
        self.output_state_h = torch.zeros([0, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.output_state_c = torch.zeros([0, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.cnn_cache = torch.zeros([0, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.att_cache = torch.zeros([0, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.offset = torch.tensor([0], dtype=torch.int32, device=self.device)

    # 预测音频
    def predict(self, speech, speech_lengths):
        """
        预测函数，只预测完整的一句话。
        :param speech: 经过处理的音频数据
        :param speech_lengths: 音频长度
        :return: 识别的文本结果和解码的得分数
        """
        audio_data = torch.tensor(speech, dtype=torch.float32, device=self.device)
        audio_len = torch.tensor(speech_lengths, dtype=torch.int64, device=self.device)

        # 非流式模型的输入
        output_data = self.predictor.get_encoder_out(speech=audio_data, speech_lengths=audio_len)
        return output_data.cpu().detach().numpy()

    def predict_chunk_deepspeech(self, x_chunk):
        if not (self.use_model == 'deepspeech2' and self.streaming):
            raise Exception(f'当前模型不支持该方法，当前模型为：{self.use_model}，参数streaming为：{self.streaming}')

        x_chunk = torch.tensor(x_chunk, dtype=torch.float32, device=self.device)
        audio_len = torch.tensor([x_chunk.shape[1]], dtype=torch.int64, device=self.device)

        output_chunk_probs, output_lens, self.output_state_h, self.output_state_c = \
            self.predictor.get_encoder_out_chunk(speech=x_chunk,
                                                 speech_lengths=audio_len,
                                                 init_state_h=self.output_state_h,
                                                 init_state_c=self.output_state_c)
        return output_chunk_probs.cpu().detach().numpy(), output_lens.cpu().detach().numpy()

    def predict_chunk_conformer(self, x_chunk, required_cache_size):
        if not ('former' in self.use_model and self.streaming):
            raise Exception(f'当前模型不支持该方法，当前模型为：{self.use_model}，参数streaming为：{self.streaming}')
        x_chunk = torch.tensor(x_chunk, dtype=torch.float32, device=self.device)
        required_cache_size = torch.tensor([required_cache_size], dtype=torch.int32, device=self.device)

        output_chunk_probs, self.att_cache, self.cnn_cache = \
            self.predictor.get_encoder_out_chunk(speech=x_chunk,
                                                 offset=self.offset,
                                                 required_cache_size=required_cache_size,
                                                 att_cache=self.att_cache,
                                                 cnn_cache=self.cnn_cache)

        self.offset += output_chunk_probs.shape[1]
        return output_chunk_probs.cpu().detach().numpy()

    # 重置流式识别，每次流式识别完成之后都要执行
    def reset_stream(self):
        self.output_state_h = torch.zeros([0, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.output_state_c = torch.zeros([0, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.cnn_cache = torch.zeros([0, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.att_cache = torch.zeros([0, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.offset = torch.tensor([0], dtype=torch.int32, device=self.device)
