import importlib

from loguru import logger

from masr.model_utils.conformer.model import ConformerModel
from masr.model_utils.deepspeech2.model import DeepSpeech2Model

__all__ = ['build_model']


def build_model(input_size, vocab_size, mean_istd_path, encoder_conf, decoder_conf, configs):
    use_model = configs.model_conf.get('model', 'ConformerModel')
    model_args = configs.model_conf.get('model_args', {})
    mod = importlib.import_module(__name__)
    model = getattr(mod, use_model)(input_size=input_size, vocab_size=vocab_size, mean_istd_path=mean_istd_path,
                                    encoder_conf=encoder_conf, decoder_conf=decoder_conf, **model_args)
    logger.info(f'成功创建模型：{use_model}，参数为：{model_args}')
    return model
