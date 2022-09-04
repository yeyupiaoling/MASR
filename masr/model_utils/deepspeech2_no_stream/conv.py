import torch
from torch import nn

__all__ = ['ConvStack']


class ConvBn(nn.Module):
    """带BN层的卷积
    :param num_channels_in: 输入通道的大小
    :type num_channels_in: int
    :param num_channels_out: 输出通道的大小
    :type num_channels_out: int
    :param kernel_size: 卷积核的大小
    :type kernel_size: int|tuple|list
    :param stride: 卷积核滑动的步数
    :type stride: int|tuple|list
    :param padding: 填充的大小
    :type padding: int|tuple|list
    :return: 带BN层的卷积
    :rtype: nn.Layer
    """

    def __init__(self, num_channels_in, num_channels_out, kernel_size, stride, padding):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(num_channels_in,
                              num_channels_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

        self.bn = nn.BatchNorm2d(num_channels_out)
        self.act = nn.Hardtanh(min_val=0.0, max_val=24.0)

    def forward(self, x, x_len):
        """
        x(Tensor): audio, shape [B, C, D, T]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        mask = torch.BoolTensor(x.size()).fill_(0)
        if x.is_cuda:
            mask = mask.cuda()
        for i, length in enumerate(x_len):
            length = length.item()
            if (mask[i].size(2) - length) > 0:
                mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
        x = x.masked_fill(mask, 0)

        x_len = torch.div(x_len - self.kernel_size[1] + 2 * self.padding[1], self.stride[1], rounding_mode='trunc') + 1

        return x, x_len


class ConvStack(nn.Module):
    """具有堆叠卷积层的卷积组
    :param feat_size: 输入音频的特征大小
    :type feat_size: int
    :param num_stacks: 堆叠卷积层的数量
    :type num_stacks: int
    """

    def __init__(self, feat_size, num_stacks):
        super().__init__()
        self.feat_size = feat_size  # D
        self.num_stacks = num_stacks
        out_channel = 32

        self.conv_in = ConvBn(num_channels_in=1,
                              num_channels_out=32,
                              kernel_size=(41, 11),  # [D, T]
                              stride=(2, 3),
                              padding=(20, 5))

        conv_stacks = []
        for _ in range(self.num_stacks - 1):
            conv_stacks.append(ConvBn(num_channels_in=32,
                                      num_channels_out=out_channel,
                                      kernel_size=(21, 11),
                                      stride=(2, 1),
                                      padding=(10, 5)))
        self.conv_stack = nn.ModuleList(conv_stacks)

        # 卷积层输出的特征大小
        output_height = torch.div(self.feat_size - 1, 2, rounding_mode='trunc') + 1
        for i in range(self.num_stacks - 1):
            output_height = torch.div(output_height - 1, 2, rounding_mode='trunc') + 1
        self.output_height = out_channel * output_height

    def forward(self, x, x_len):
        """
        x: shape [B, C, D, T]
        x_len : shape [B]
        """
        x, x_len = self.conv_in(x, x_len)
        for i, conv in enumerate(self.conv_stack):
            x, x_len = conv(x, x_len)
        return x, x_len