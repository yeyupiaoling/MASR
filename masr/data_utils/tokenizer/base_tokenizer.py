from abc import ABC, abstractmethod
from typing import List, Union, Dict


class BaseTokenizer(ABC):
    def __init__(self,
                 token_list: Union[str, List[str]],
                 unk_symbol: str = "<unk>",
                 delimiter: str = " ",
                 non_linguistic_symbols: List[str] = None,
                 remove_non_linguistic_symbols: bool = False):
        self.delimiter = delimiter,
        self.non_linguistic_symbols = None
        self.token_list: List[str] = []
        if isinstance(token_list, str):
            with open(token_list, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip().split("\t")[0]
                    self.token_list.append(line)
        elif isinstance(token_list, list):
            self.token_list = token_list
        else:
            raise RuntimeError(f"输入的 token_list 格式错误，请输入文件路径或列表")

        self.token2id: Dict[str, int] = {}
        for i, t in enumerate(self.token_list):
            if t in self.token2id:
                raise RuntimeError(f'词汇表中存在重复的字符：{t}')
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(f"未找到 <unk> 符号，请检查 token_list 中是否有该字符")
        self.unk_id = self.token2id[self.unk_symbol]
        # 获取非语言符号
        if remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            raise Exception("移除非语言符号需要提供non_linguistic_symbols")
        if remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return (f'{self.__class__.__name__}(delimiter={self.delimiter}, '
                f'unk_symbol={self.unk_symbol}, '
                f'non_linguistic_symbols={self.non_linguistic_symbols}, '
                f'remove_non_linguistic_symbols={self.remove_non_linguistic_symbols},'
                f'vocab_size={self.vocab_size})')

    def encode(self, text: str) -> List[int]:
        tokens = self.text2tokens(text)
        text_ids = self.tokens2ids(tokens)
        return text_ids

    def decode(self, text_ids: List[int]) -> str:
        token = self.ids2tokens(text_ids)
        text = self.tokens2text(token)
        return text

    @abstractmethod
    def text2tokens(self, text: str) -> List[str]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def tokens2text(self, tokens: List[str]) -> str:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def tokens2ids(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def ids2tokens(self, ids: List[int]) -> List[str]:
        raise NotImplementedError("abstract method")

    @property
    def vocab_size(self) -> int:
        return len(self.token_list)

    @property
    def vocab_list(self) -> List[str]:
        return self.token_list
