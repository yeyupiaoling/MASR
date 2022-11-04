import torch
import torch.nn.functional as F
from torch import nn
from typeguard import check_argument_types


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.
    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.
        e.g.
        smoothing=0.1
        [0,1,2] ->
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]

    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Label-smoothing loss.

        Args:
            size (int): the number of class
            padding_idx (int): padding class id which will be ignored for loss
            smoothing (float): smoothing rate (0.0 means the conventional CE)
            normalize_length (bool):
                True, normalize loss by sequence length;
                False, normalize loss by batch size.
                Defaults to False.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.
        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


class CTCLoss(torch.nn.Module):
    """CTC module"""

    def __init__(
            self,
            odim: int,
            encoder_output_size: int,
            dropout_rate: float = 0.0,
            reduce: bool = True,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.shape[0]
        return loss

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
