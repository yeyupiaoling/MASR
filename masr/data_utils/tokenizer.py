import os
import tempfile
from abc import ABC
from typing import List, Union

import sentencepiece as spm

from masr.data_utils.utils import read_manifest


class MASRTokenizer(ABC):
    def __init__(self,
                 vocab_model_dir: str,
                 model_type: str = "char",
                 build_vocab_size: int = None,
                 non_linguistic_symbols: List[str] = None,
                 remove_non_linguistic_symbols: bool = False,
                 is_build_vocab: bool = False):
        self.vocab_model_dir = vocab_model_dir
        self.build_vocab_size = build_vocab_size
        self.non_linguistic_symbols = non_linguistic_symbols
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols
        os.makedirs(self.vocab_model_dir, exist_ok=True)
        self.model_prefix = os.path.join(self.vocab_model_dir, "model")
        if not is_build_vocab:
            model_path = self.model_prefix + ".model"
            assert os.path.exists(model_path), f"模型文件不存在: {model_path}"
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_path)
            # 获取词汇表内容
            vocab_path = self.model_prefix + ".vocab"
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.token_list = [line.strip().split("\t")[0] for line in f.readlines()]
            assert len(self.token_list) == self.sp.vocab_size(), "词汇表大小不一致"
        else:
            self.smp_args = dict(model_type=model_type,
                                 model_prefix=self.model_prefix,
                                 pad_id=0,
                                 unk_id=1,
                                 bos_id=2,
                                 eos_id=3,
                                 pad_piece="<blank>",
                                 unk_piece="<unk>",
                                 bos_piece="<bos>",
                                 eos_piece="</eos>",
                                 input_sentence_size=1e8,
                                 character_coverage=0.9995,
                                 minloglevel=4)
            if self.build_vocab_size is not None:
                self.smp_args["vocab_size"] = self.build_vocab_size
            if model_type == "unigram":
                assert self.build_vocab_size is not None, "构建unigram模型需要指定词汇表大小"
            else:
                self.smp_args["use_all_vocab"] = True

    def build_vocab(self, manifest_paths: List[str]):
        fp = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="utf-8")
        for manifest_path in manifest_paths:
            manifest_data = read_manifest(manifest_path)
            for line in manifest_data:
                text = line["text"]
                # 移除非语言符号
                if self.remove_non_linguistic_symbols:
                    for symbol in self.non_linguistic_symbols:
                        text = text.replace(symbol, "")
                fp.write(text + "\n")
        fp.close()
        spm.SentencePieceTrainer.Train(input=fp.name, **self.smp_args)
        os.unlink(fp.name)

    def text2tokens(self, text: str) -> List[str]:
        return self.sp.EncodeAsPieces(text)

    def text2ids(self, text: str) -> List[int]:
        return self.sp.EncodeAsIds(text)

    def ids2text(self, ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        return self.sp.DecodeIds(ids)

    @property
    def blank_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    @property
    def vocab_list(self) -> List[str]:
        return self.token_list
