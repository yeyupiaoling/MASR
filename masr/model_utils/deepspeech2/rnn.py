import torch
from torch import nn

__all__ = ['RNNStack']


class RNNForward(nn.Module):
    def __init__(self, rnn_input_size, h_size):
        super().__init__()
        self.rnn = nn.GRU(input_size=rnn_input_size,
                          hidden_size=h_size,
                          bidirectional=False,
                          batch_first=True)
        self.norm = nn.LayerNorm(h_size)

    def forward(self, x, x_lens, init_state):
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens.cpu(), batch_first=True)
        x, final_state = self.rnn(x, init_state)  # [B, T, D]
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.norm(x)
        return x, final_state


class RNNStack(nn.Module):
    """堆叠单向GRU层

    :param i_size: GRU层的输入大小
    :type i_size: int
    :param h_size: GRU层的隐层大小
    :type h_size: int
    :param num_rnn_layers: rnn层数
    :type num_rnn_layers: int

    :return: RNN组的输出层
    :rtype: nn.Layer
    """

    def __init__(self, i_size: int, h_size: int, num_rnn_layers: int):
        super().__init__()
        self.rnns = nn.ModuleList()
        self.output_dim = h_size
        self.num_rnn_layers = num_rnn_layers
        self.rnns.append(RNNForward(rnn_input_size=i_size, h_size=h_size))
        for i in range(0, self.num_rnn_layers - 1):
            self.rnns.append(RNNForward(rnn_input_size=h_size, h_size=h_size))

    def forward(self, x, x_lens, init_state_h_box=None):
        if init_state_h_box is not None:
            init_state_list = torch.split(init_state_h_box, 1, dim=0)
        else:
            init_state_list = [None] * self.num_rnn_layers
        final_chunk_state_list = []
        for rnn, init_state in zip(self.rnns, init_state_list):
            x, final_state = rnn(x, x_lens, init_state)
            final_chunk_state_list.append(final_state)

        final_chunk_state_h_box = torch.concat(final_chunk_state_list, dim=0)
        return x, final_chunk_state_h_box
