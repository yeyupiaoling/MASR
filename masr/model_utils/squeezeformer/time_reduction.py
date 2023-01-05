from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from masr.model_utils.squeezeformer.conv2d import Conv2dValid


class TimeReductionLayer1D(nn.Module):
    """
    Modified NeMo,
    Squeezeformer Time Reduction procedure.
    Downsamples the audio by `stride` in the time dimension.
    Args:
        channel (int): input dimension of
                       MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for
                           depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self, channel: int, out_dim: int, kernel_size: int = 5, stride: int = 2):
        super(TimeReductionLayer1D, self).__init__()

        self.channel = channel
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = max(0, self.kernel_size - self.stride)

        self.dw_conv = nn.Conv1d(in_channels=channel,
                                 out_channels=channel,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=self.padding,
                                 groups=channel)

        self.pw_conv = nn.Conv1d(in_channels=channel, out_channels=out_dim,
                                 kernel_size=1, stride=1, padding=0, groups=1)

        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.channel ** -0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    def forward(self, xs, xs_lens: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)):
        xs = xs.transpose(1, 2)  # [B, C, T]
        xs = xs.masked_fill(mask_pad.eq(0), 0.0)

        xs = self.dw_conv(xs)
        xs = self.pw_conv(xs)

        xs = xs.transpose(1, 2)  # [B, T, C]

        B, T, D = xs.size()
        mask = mask[:, ::self.stride, ::self.stride]
        mask_pad = mask_pad[:, :, ::self.stride]
        L = mask_pad.size(-1)
        # For JIT exporting, we remove F.pad operator.
        if L - T < 0:
            xs = xs[:, :L - T, :].contiguous()
        else:
            dummy_pad = torch.zeros(B, L - T, D, device=xs.device)
            xs = torch.cat([xs, dummy_pad], dim=1)

        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        return xs, xs_lens, mask, mask_pad


class TimeReductionLayer2D(nn.Module):
    def __init__(
            self, kernel_size: int = 5, stride: int = 2, encoder_dim: int = 256):
        super(TimeReductionLayer2D, self).__init__()
        self.encoder_dim = encoder_dim
        self.kernel_size = kernel_size
        self.dw_conv = Conv2dValid(in_channels=encoder_dim,
                                   out_channels=encoder_dim,
                                   kernel_size=(kernel_size, 1),
                                   stride=stride,
                                   valid_trigy=True)
        self.pw_conv = Conv2dValid(in_channels=encoder_dim,
                                   out_channels=encoder_dim,
                                   kernel_size=1,
                                   stride=1,
                                   valid_trigx=False,
                                   valid_trigy=False)

        self.kernel_size = kernel_size
        self.stride = stride
        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.encoder_dim ** -0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    def forward(
            self, xs: torch.Tensor, xs_lens: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xs = xs.masked_fill(mask_pad.transpose(1, 2).eq(0), 0.0)
        xs = xs.unsqueeze(2)
        padding1 = self.kernel_size - self.stride
        xs = F.pad(xs, (0, 0, 0, 0, 0, padding1, 0, 0), mode='constant', value=0.)
        xs = self.dw_conv(xs.permute(0, 3, 1, 2))
        xs = self.pw_conv(xs).permute(0, 3, 2, 1).squeeze(1).contiguous()
        tmp_length = xs.size(1)
        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        padding2 = max(0, (xs_lens.max() - tmp_length).data.item())
        batch_size, hidden = xs.size(0), xs.size(-1)
        dummy_pad = torch.zeros(batch_size, padding2, hidden, device=xs.device)
        xs = torch.cat([xs, dummy_pad], dim=1)
        mask = mask[:, ::2, ::2]
        mask_pad = mask_pad[:, :, ::2]
        return xs, xs_lens, mask, mask_pad


class TimeReductionLayerStream(nn.Module):
    """
    Squeezeformer Time Reduction procedure.
    Downsamples the audio by `stride` in the time dimension.
    Args:
        channel (int): input dimension of
            MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for
            depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self, channel: int, out_dim: int,
                 kernel_size: int = 1, stride: int = 2):
        super(TimeReductionLayerStream, self).__init__()

        self.channel = channel
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride

        self.dw_conv = nn.Conv1d(in_channels=channel,
                                 out_channels=channel,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=0,
                                 groups=channel)

        self.pw_conv = nn.Conv1d(in_channels=channel, out_channels=out_dim,
                                 kernel_size=1, stride=1, padding=0, groups=1
                                 )

        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.channel ** -0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    def forward(self, xs, xs_lens: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)):
        xs = xs.transpose(1, 2)  # [B, C, T]
        xs = xs.masked_fill(mask_pad.eq(0), 0.0)

        xs = self.dw_conv(xs)
        xs = self.pw_conv(xs)

        xs = xs.transpose(1, 2)  # [B, T, C]

        B, T, D = xs.size()
        mask = mask[:, ::self.stride, ::self.stride]
        mask_pad = mask_pad[:, :, ::self.stride]
        L = mask_pad.size(-1)
        # For JIT exporting, we remove F.pad operator.
        if L - T < 0:
            xs = xs[:, :L - T, :].contiguous()
        else:
            dummy_pad = torch.zeros(B, L - T, D, device=xs.device)
            xs = torch.cat([xs, dummy_pad], dim=1)

        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        return xs, xs_lens, mask, mask_pad
