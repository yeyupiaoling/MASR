import math
from typing import Tuple

import torch
from torch import nn

__all__ = ["PositionalEncoding", "RelPositionalEncoding"]


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Positional encoding.
            PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
            PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
        Args:
            d_model (int): embedding dim.
            dropout_rate (float): dropout rate.
            max_len (int, optional): maximum input length. Defaults to 5000.
            reverse (bool, optional): Not used. Defaults to False.
        """
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros([1, self.max_len, self.d_model])  # [B=1,T,D]
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, :, 0::2] = torch.sin(position * div_term)  # TODO
        self.pe[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding, (batch=1, time, ...)
        """
        assert offset + x.shape[
            1] < self.max_len, "offset: {} + x.shape[1]: {} is larger than the max_len: {}".format(
            offset, x.shape[1], self.max_len)
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset:offset + x.shape[1]]  # TODO
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion
        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.
        Args:
            offset (int): start offset
            size (int): requried size of position encoding
        Returns:
            torch.Tensor: Corresponding position encoding, #[1, T, D].
        """
        assert offset + size < self.max_len
        return self.dropout(self.pe[:, offset:offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """
        Args:
            d_model (int): Embedding dimension.
            dropout_rate (float): Dropout rate.
            max_len (int, optional): [Maximum input length.]. Defaults to 5000.
        """
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        assert offset + x.shape[
            1] < self.max_len, "offset: {} + x.shape[1]: {} is larger than the max_len: {}".format(
            offset, x.shape[1], self.max_len)
        x = x * self.xscale
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset:offset + x.shape[1]]
        return self.dropout(x), self.dropout(pos_emb)


class NoPositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        super().__init__()

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return torch.zeros(1, size, self.d_model)
