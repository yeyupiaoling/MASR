from typing import Tuple
import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=1024,
                 bidirectional=False):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=bidirectional)

    def forward(self, x: torch.nn.utils.rnn.PackedSequence, init_state: Tuple[torch.Tensor, torch.Tensor]):
        init_state_h, init_state_c = init_state
        x, final_state_h = self.rnn(x, init_state_h)  # [B, T, D]
        final_state_c = final_state_h
        return x, (final_state_h, final_state_c)
