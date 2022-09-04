from torch import nn

__all__ = ['RNNStack']


class BiRNNWithBN(nn.Module):
    """具有顺序批标准化的双向RNN层。批标准化只对输入状态权值执行。
    :param i_size: GRUCell的输入大小
    :type i_size: int
    :param h_size: GRUCell的隐藏大小
    :type h_size: string
    :return: 双向RNN层
    :rtype: nn.Layer
    """

    def __init__(self, i_size: int, h_size: int, use_gru:bool):
        super().__init__()
        hidden_size = h_size * 3

        self.fc = nn.Linear(i_size, hidden_size)
        self.bn = nn.LayerNorm(hidden_size)
        if use_gru:
            self.rnn = nn.GRU(input_size=hidden_size, hidden_size=h_size, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=h_size, bidirectional=True)

    def forward(self, x, x_len):
        # x, shape [B, T, D]
        x = self.fc(x)
        x = self.bn(x)
        x = nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(), batch_first=True)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x


class RNNStack(nn.Module):
    """RNN组与堆叠双向简单RNN或GRU层
    :param i_size: GRU层的输入大小
    :type i_size: int
    :param h_size: GRU层的隐层大小
    :type h_size: int
    :param num_stacks: 堆叠的rnn层数
    :type num_stacks: int
    :return: RNN组的输出层
    :rtype: nn.Layer
    """

    def __init__(self, i_size: int, h_size: int, num_stacks: int, use_gru:bool):
        super().__init__()
        rnn_stacks = []
        for i in range(num_stacks):
            rnn_stacks.append(BiRNNWithBN(i_size=i_size, h_size=h_size, use_gru=use_gru))
            i_size = h_size * 2

        self.rnn_stacks = nn.ModuleList(rnn_stacks)

    def forward(self, x, x_len):
        """
        x: shape [B, T, D]
        x_len: shpae [B]
        """
        for i, rnn in enumerate(self.rnn_stacks):
            x = rnn(x, x_len)
        return x