import torch
from torch import nn


class Conv2dSubsampling4Pure(nn.Module):
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(), )
        self.subsampling_rate = 4
        self.output_dim = ((idim - 1) // 2 - 1) // 2 * odim

    def forward(self, x: torch.Tensor, x_len: torch.Tensor):
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3)  # [B, T, C, D]
        x = x.reshape([x.shape[0], x.shape[1], -1])  # [B, T, C*D]
        x_len = torch.div(torch.div((x_len - 1), 2, rounding_mode='trunc') - 1, 2, rounding_mode='trunc')
        return x, x_len
