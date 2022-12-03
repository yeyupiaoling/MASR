from typing import Tuple, List

import torch
from torch import nn

from masr.model_utils.deepspeech2.conv import Conv2dSubsampling4Pure


class RNN(nn.Module):
    def __init__(self,
                 rnn_input_size,
                 layernorm_size,
                 rnn_size=1024,
                 bidirectional=False,
                 use_gru=False):
        super().__init__()
        self.rnn_size = rnn_size
        if bidirectional:
            self.num_state = 2
        else:
            self.num_state = 1
        # if not use_gru:
        #     self.rnn = nn.LSTM(input_size=rnn_input_size,
        #                        hidden_size=rnn_size,
        #                        num_layers=1,
        #                        batch_first=True,
        #                        bidirectional=bidirectional)
        # else:
        self.rnn = nn.GRU(input_size=rnn_input_size,
                          hidden_size=rnn_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=bidirectional)
        self.layer_norm = nn.LayerNorm(layernorm_size)

    def forward(self, x, x_lens, init_state: torch.Tensor):
        if init_state.size(2) == 0:
            init_state = torch.zeros([self.num_state, x.size(0), self.rnn_size], device=x.device, dtype=x.dtype)
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens.cpu(), batch_first=True)
        x, final_state = self.rnn(x, init_state)  # [B, T, D]
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.layer_norm(x)
        return x, final_state


class CRNNEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 vocab_size,
                 global_cmvn=None,
                 num_rnn_layers=5,
                 rnn_size=1024,
                 rnn_direction='forward',
                 use_gru=False):
        super().__init__()
        self.rnn_size = rnn_size
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.num_rnn_layers = num_rnn_layers
        self.rnn_direction = rnn_direction
        self.use_gru = use_gru
        self.conv = Conv2dSubsampling4Pure(input_dim, 32)

        self.global_cmvn = global_cmvn
        self.output_dim = self.conv.output_dim

        i_size = self.conv.output_dim
        self.rnns = nn.ModuleList()
        self.layernorm_list = nn.ModuleList()
        if rnn_direction == 'bidirect' or rnn_direction == 'bidirectional':
            bidirectional = True
            layernorm_size = 2 * rnn_size
        elif rnn_direction == 'forward':
            bidirectional = False
            layernorm_size = rnn_size
        else:
            raise Exception("Wrong rnn direction")
        for i in range(0, num_rnn_layers):
            if i == 0:
                rnn_input_size = i_size
            else:
                rnn_input_size = layernorm_size
            self.rnns.append(RNN(rnn_input_size=rnn_input_size,
                                 layernorm_size=layernorm_size,
                                 rnn_size=rnn_size,
                                 bidirectional=bidirectional,
                                 use_gru=use_gru))
            self.output_dim = layernorm_size

    @property
    def output_size(self):
        return self.output_dim

    def forward(self, x, x_lens, init_state: torch.Tensor = torch.zeros([0, 0, 0, 0])):
        """Compute Encoder outputs

        Args:
            x (Tensor): [B, T, D]
            x_lens (Tensor): [B]
            init_state(Tensor): init_states h for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
        Return:
            x (Tensor): encoder outputs, [B, T, D]
            x_lens (Tensor): encoder length, [B]
            final_state(Tensor): final_states h for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
        """
        if init_state.size(0) == 0:
            init_state = torch.zeros([self.num_rnn_layers, 0, 0, 0], device=x.device, dtype=x.dtype)

        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        x, x_lens = self.conv(x, x_lens)
        final_chunk_state_list: List[torch.Tensor] = []
        for i, rnn in enumerate(self.rnns):
            x, final_state = rnn(x, x_lens, init_state[i])  # [B, T, D]
            final_chunk_state_list.append(final_state.unsqueeze(0))
        final_chunk_state = torch.cat(final_chunk_state_list, dim=0)

        return x, x_lens, final_chunk_state
