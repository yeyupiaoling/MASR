from typing import Iterable
from typing import List
from typing import Union

from masr.data_utils.tokenizer.base_tokenizer import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    def __init__(self,
                 token_list: Union[str, List[str]],
                 unk_symbol: str = "<unk>",
                 delimiter: str = " ",
                 non_linguistic_symbols: List[str] = None,
                 remove_non_linguistic_symbols: bool = False):
        super().__init__(token_list=token_list, unk_symbol=unk_symbol, delimiter=delimiter,
                         non_linguistic_symbols=non_linguistic_symbols,
                         remove_non_linguistic_symbols=remove_non_linguistic_symbols)

    def __repr__(self):
        return (f'{self.__class__.__name__}(delimiter={self.delimiter}, '
                f'unk_symbol={self.unk_symbol}, '
                f'non_linguistic_symbols={self.non_linguistic_symbols}, '
                f'remove_non_linguistic_symbols={self.remove_non_linguistic_symbols})')

    def text2tokens(self, text: str) -> List[str]:
        tokens = []
        for t in text.split(self.delimiter):
            if self.remove_non_linguistic_symbols and t in self.non_linguistic_symbols:
                continue
            tokens.append(t)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.delimiter.join(tokens)

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]

    def ids2tokens(self, ids: List[int]) -> List[str]:
        return [self.token_list[i] for i in ids]
