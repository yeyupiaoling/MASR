# @Time    : 2024-09-05
# @Author  : yeyupiaoling
import importlib

from loguru import logger

from masr.data_utils.tokenizer.char_tokenizer import CharTokenizer
from masr.data_utils.tokenizer.word_tokenizer import WordTokenizer

__all__ = ['build_tokenizer']


def build_tokenizer(tokenizer_configs):
    use_tokenizer = tokenizer_configs.get('tokenizer', 'CharTokenizer')
    tokenizer_args = tokenizer_configs.get('tokenizer_args', {})
    token = importlib.import_module(__name__)
    tokenizer = getattr(token, use_tokenizer)(**tokenizer_args)
    logger.info(f'成功创建：{use_tokenizer}')
    return tokenizer
