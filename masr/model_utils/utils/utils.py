from torch import nn

from masr.model_utils.conformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention, \
    MultiHeadedCrossAttention
from masr.model_utils.conformer.embedding import PositionalEncoding, RelPositionalEncoding, NoPositionalEncoding
from masr.model_utils.conformer.positionwise import PositionwiseFeedForward
from masr.model_utils.conformer.subsampling import LinearNoSubsampling, \
    Conv2dSubsampling4, Conv2dSubsampling6, Conv2dSubsampling8
from masr.model_utils.utils.common import Swish

activation_classes = {
    "hardtanh": nn.Hardtanh,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": getattr(nn, "SiLU", Swish),
    "gelu": nn.GELU,
}

rnn_classes = {
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}

subsample_classes = {
    "linear": LinearNoSubsampling,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
}

emb_classes = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
}

attention_classes = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
    "crossattn": MultiHeadedCrossAttention,
}

mlp_classes = {
    'position_wise_feed_forward': PositionwiseFeedForward,
}

norm_classes = {
    'layer_norm': nn.LayerNorm,
    'batch_norm': nn.BatchNorm1d,
    'rms_norm': nn.RMSNorm
}