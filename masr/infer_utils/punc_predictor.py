# This implementation is adapted from https://github.com/modelscope/FunASR
import json
import os.path
from pathlib import Path
from typing import Union, Tuple

import numpy as np
from loguru import logger

from masr.infer_utils.utils import ONNXRuntimeError, OrtInferSession, read_yaml
from masr.infer_utils.utils import (
    TokenIDConverter,
    split_to_mini_sentence,
    code_mix_split_words,
    code_mix_split_words_jieba,
)

__all__ = ['PunctuationPredictor', 'PunctuationOnlinePredictor']


class PunctuationPredictor:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    CT-Transformer: Controllable time-delay transformer for real-time punctuation prediction and disfluency detection
    https://arxiv.org/pdf/2003.01309.pdf
    """

    def __init__(
            self,
            model_dir: Union[str, Path],
            device_id: Union[str, int] = "-1",
            quantize: bool = False,
            intra_op_num_threads: int = 4,
    ):
        """离线标点符合模型

        Args:
            model_dir (str): 模型路径文件夹
            device_id (Union[str, int], optional): 设备ID，用于指定模型运行的设备，默认为"-1"表示使用CPU。如果指定为GPU，则为GPU的ID。
            quantize (bool, optional): 是否使用量化模型，默认为False。
            intra_op_num_threads (int, optional): ONNX Runtime的线程数，默认为4。
        """
        model_file = os.path.join(model_dir, "model.onnx")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        config_file = os.path.join(model_dir, "config.yaml")
        config = read_yaml(config_file)
        token_list = os.path.join(model_dir, "tokens.json")
        with open(token_list, "r", encoding="utf-8") as f:
            token_list = json.load(f)

        self.converter = TokenIDConverter(token_list)
        self.ort_infer = OrtInferSession(model_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.batch_size = 1
        self.punc_list = config["model_conf"]["punc_list"]
        self.period = 0
        for i in range(len(self.punc_list)):
            if self.punc_list[i] == ",":
                self.punc_list[i] = "，"
            elif self.punc_list[i] == "?":
                self.punc_list[i] = "？"
            elif self.punc_list[i] == "。":
                self.period = i
        self.jieba_usr_dict_path = os.path.join(model_dir, "jieba_usr_dict")
        if os.path.exists(self.jieba_usr_dict_path):
            self.seg_jieba = True
            self.code_mix_split_words_jieba = code_mix_split_words_jieba(self.jieba_usr_dict_path)
        else:
            self.seg_jieba = False

    def __call__(self, text: str, split_size=20):
        """
        对输入的文本进行分句和标点预测
        Args:
            text (str): 输入的文本
            split_size (int, optional): 分句的大小，默认为20。表示每个小句子包含的最大token数
        Returns:
            Tuple[str, List[int]]:
                str: 处理后的文本，包含预测的标点
                List[int]: 处理后的文本对应的标点id列表
        """
        if self.seg_jieba:
            split_text = self.code_mix_split_words_jieba(text)
        else:
            split_text = code_mix_split_words(text)
        split_text_id = self.converter.tokens2ids(split_text)
        mini_sentences = split_to_mini_sentence(split_text, split_size)
        mini_sentences_id = split_to_mini_sentence(split_text_id, split_size)
        assert len(mini_sentences) == len(mini_sentences_id)
        cache_sent = []
        cache_sent_id = []
        new_mini_sentence = ""
        new_mini_sentence_punc = []
        cache_pop_trigger_limit = 200
        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = np.array(cache_sent_id + mini_sentence_id, dtype="int32")
            data = {
                "text": mini_sentence_id[None, :],
                "text_lengths": np.array([len(mini_sentence_id)], dtype="int32"),
            }
            try:
                outputs = self.infer(data["text"], data["text_lengths"])
                y = outputs[0]
                punctuations = np.argmax(y, axis=-1)[0]
                assert punctuations.size == len(mini_sentence)
            except ONNXRuntimeError:
                logger.exception("error")

            # Search for the last Period/QuestionMark as cache
            if mini_sentence_i < len(mini_sentences) - 1:
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if self.punc_list[punctuations[i]] == "。" or self.punc_list[punctuations[i]] == "？":
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if sentenceEnd < 0 <= last_comma_index and len(mini_sentence) > cache_pop_trigger_limit:
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.period
                cache_sent = mini_sentence[sentenceEnd + 1:]
                cache_sent_id = mini_sentence_id[sentenceEnd + 1:].tolist()
                mini_sentence = mini_sentence[0: sentenceEnd + 1]
                punctuations = punctuations[0: sentenceEnd + 1]

            new_mini_sentence_punc += [int(x) for x in punctuations]
            words_with_punc = []
            for i in range(len(mini_sentence)):
                if i > 0:
                    if (
                            len(mini_sentence[i][0].encode()) == 1
                            and len(mini_sentence[i - 1][0].encode()) == 1
                    ):
                        mini_sentence[i] = " " + mini_sentence[i]
                words_with_punc.append(mini_sentence[i])
                if self.punc_list[punctuations[i]] != "_":
                    words_with_punc.append(self.punc_list[punctuations[i]])
            new_mini_sentence += "".join(words_with_punc)
            # Add Period for the end of the sentence
            new_mini_sentence_out = new_mini_sentence
            new_mini_sentence_punc_out = new_mini_sentence_punc
            if mini_sentence_i == len(mini_sentences) - 1:
                if new_mini_sentence[-1] == "，" or new_mini_sentence[-1] == "、":
                    new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
                elif new_mini_sentence[-1] != "。" and new_mini_sentence[-1] != "？":
                    new_mini_sentence_out = new_mini_sentence + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
        return new_mini_sentence_out, new_mini_sentence_punc_out

    def infer(self, feats: np.ndarray, feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer([feats, feats_len])
        return outputs


class PunctuationOnlinePredictor(PunctuationPredictor):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    CT-Transformer: Controllable time-delay transformer for real-time punctuation prediction and disfluency detection
    https://arxiv.org/pdf/2003.01309.pdf
    """

    def __init__(self, *args, **kwargs):
        """在线标点符合模型

        Args:
            model_dir (str): 模型路径文件夹
            device_id (Union[str, int], optional): 设备ID，用于指定模型运行的设备，默认为"-1"表示使用CPU。如果指定为GPU，则为GPU的ID。
            quantize (bool, optional): 是否使用量化模型，默认为False。
            intra_op_num_threads (int, optional): ONNX Runtime的线程数，默认为4。
        """
        super().__init__(*args, **kwargs)

    def __call__(self, text: str, param_dict: dict, split_size=20):
        """对输入的文本进行分句和标点预测

        Args:
            text (str): 输入的文本
            param_dict (dict): 参数字典，包含"cache"键值对，其中"cache"是当前句子的缓存，用于在下一次调用时继续处理
            split_size (int, optional): 分句的大小，默认为20。表示每个小句子包含的最大token数
        Returns:
            Tuple[str, List[int]]:
                str: 处理后的文本，包含预测的标点
                List[int]: 处理后的文本对应的标点id列表
        """
        cache_key = "cache"
        assert cache_key in param_dict
        cache = param_dict[cache_key]
        if cache is not None and len(cache) > 0:
            precache = "".join(cache)
        else:
            precache = ""
            cache = []
        full_text = precache + " " + text
        split_text = code_mix_split_words(full_text)
        split_text_id = self.converter.tokens2ids(split_text)
        mini_sentences = split_to_mini_sentence(split_text, split_size)
        mini_sentences_id = split_to_mini_sentence(split_text_id, split_size)
        new_mini_sentence_punc = []
        assert len(mini_sentences) == len(mini_sentences_id)

        cache_sent = []
        cache_sent_id = np.array([], dtype="int32")
        sentence_punc_list = []
        sentence_words_list = []
        cache_pop_trigger_limit = 200
        skip_num = 0
        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = np.concatenate((cache_sent_id, mini_sentence_id), axis=0, dtype=np.int32)
            text_length = len(mini_sentence_id)
            vad_mask = self.vad_mask(text_length, len(cache))[None, None, :, :].astype(np.float32)
            data = {
                "input": mini_sentence_id[None, :],
                "text_lengths": np.array([text_length], dtype="int32"),
                "vad_mask": vad_mask,
                "sub_masks": vad_mask,
            }
            try:
                outputs = self.infer(
                    data["input"], data["text_lengths"], data["vad_mask"], data["sub_masks"]
                )
                y = outputs[0]
                punctuations = np.argmax(y, axis=-1)[0]
                assert punctuations.size == len(mini_sentence)
            except ONNXRuntimeError:
                logger.exception("error")

            # Search for the last Period/QuestionMark as cache
            if mini_sentence_i < len(mini_sentences) - 1:
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if self.punc_list[punctuations[i]] == "。" or self.punc_list[punctuations[i]] == "？":
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if sentenceEnd < 0 <= last_comma_index and len(mini_sentence) > cache_pop_trigger_limit:
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.period
                cache_sent = mini_sentence[sentenceEnd + 1:]
                cache_sent_id = mini_sentence_id[sentenceEnd + 1:]
                mini_sentence = mini_sentence[0: sentenceEnd + 1]
                punctuations = punctuations[0: sentenceEnd + 1]

            punctuations_np = [int(x) for x in punctuations]
            new_mini_sentence_punc += punctuations_np
            sentence_punc_list += [self.punc_list[int(x)] for x in punctuations_np]
            sentence_words_list += mini_sentence

        assert len(sentence_punc_list) == len(sentence_words_list)
        words_with_punc = []
        sentence_punc_list_out = []
        for i in range(0, len(sentence_words_list)):
            if i > 0:
                if (
                        len(sentence_words_list[i][0].encode()) == 1
                        and len(sentence_words_list[i - 1][-1].encode()) == 1
                ):
                    sentence_words_list[i] = " " + sentence_words_list[i]
            if skip_num < len(cache):
                skip_num += 1
            else:
                words_with_punc.append(sentence_words_list[i])
            if skip_num >= len(cache):
                sentence_punc_list_out.append(sentence_punc_list[i])
                if sentence_punc_list[i] != "_":
                    words_with_punc.append(sentence_punc_list[i])
        sentence_out = "".join(words_with_punc)

        sentenceEnd = -1
        for i in range(len(sentence_punc_list) - 2, 1, -1):
            if sentence_punc_list[i] == "。" or sentence_punc_list[i] == "？":
                sentenceEnd = i
                break
        cache_out = sentence_words_list[sentenceEnd + 1:]
        if sentence_out[-1] in self.punc_list:
            sentence_out = sentence_out[:-1]
            sentence_punc_list_out[-1] = "_"
        param_dict[cache_key] = cache_out
        return sentence_out, sentence_punc_list_out, cache_out

    @staticmethod
    def vad_mask(size, vad_pos, dtype=bool):
        """Create mask for decoder self-attention.

        :param int size: size of mask
        :param int vad_pos: index of vad index
        :param torch.dtype dtype: result dtype
        :rtype: torch.Tensor (B, Lmax, Lmax)
        """
        ret = np.ones((size, size), dtype=dtype)
        if vad_pos <= 0 or vad_pos >= size:
            return ret
        sub_corner = np.zeros((vad_pos - 1, size - vad_pos), dtype=dtype)
        ret[0: vad_pos - 1, vad_pos:] = sub_corner
        return ret

    def infer(
            self, feats: np.ndarray, feats_len: np.ndarray, vad_mask: np.ndarray, sub_masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer([feats, feats_len, vad_mask, sub_masks])
        return outputs
