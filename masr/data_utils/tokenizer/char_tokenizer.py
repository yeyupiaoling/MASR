import re
from typing import Iterable
from typing import List
from typing import Union

from masr.data_utils.tokenizer.base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def __init__(
            self,
            token_list: Union[str, List[str]],
            unk_symbol: str = "<unk>",
            delimiter: str = " ",
            space_symbol: str = "<space>",
            non_linguistic_symbols: List[str] = None,
            remove_non_linguistic_symbols: bool = False):
        super().__init__(token_list=token_list, unk_symbol=unk_symbol, delimiter=delimiter,
                         non_linguistic_symbols=non_linguistic_symbols,
                         remove_non_linguistic_symbols=remove_non_linguistic_symbols)
        self.space_symbol = space_symbol
        self.pattern = re.compile(r"^[A-Za-z'-]+$")

    def text2tokens(self, text: str) -> List[str]:
        tokens = []
        for split_text in text.split(self.delimiter):
            # 判断split_text是否为英文单词
            if self.pattern.match(split_text) is not None:
                tokens.append(split_text)
                continue
            while len(split_text) != 0:
                for w in self.non_linguistic_symbols:
                    if split_text.startswith(w):
                        if not self.remove_non_linguistic_symbols:
                            tokens.append(split_text[: len(w)])
                        split_text = split_text[len(w):]
                        break
                else:
                    t = split_text[0]
                    split_text = split_text[1:]
                    tokens.append(t)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)

    def ids2tokens(self, ids: List[int]) -> List[str]:
        return [self.token_list[i] for i in ids]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]
