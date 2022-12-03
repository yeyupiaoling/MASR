import torch
from torch import nn

from masr.data_utils.normalizer import FeatureNormalizer
from masr.model_utils.deepspeech2.encoder import CRNNEncoder
from masr.model_utils.loss.ctc import CTCLoss
from masr.model_utils.utils.cmvn import GlobalCMVN


class DeepSpeech2Model(nn.Module):
    """The DeepSpeech2 network structure.

    :param input_dim: feature size for audio.
    :type input_dim: int
    :param vocab_size: Dictionary size for tokenized transcription.
    :type vocab_size: int
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput
    """

    def __init__(self,
                 configs,
                 input_dim: int,
                 vocab_size: int,
                 rnn_direction='forward'):
        super().__init__()
        feature_normalizer = FeatureNormalizer(mean_istd_filepath=configs.dataset_conf.mean_istd_path)
        global_cmvn = GlobalCMVN(torch.from_numpy(feature_normalizer.mean).float(),
                                 torch.from_numpy(feature_normalizer.istd).float())
        self.encoder = CRNNEncoder(input_dim=input_dim,
                                   vocab_size=vocab_size,
                                   global_cmvn=global_cmvn,
                                   rnn_direction=rnn_direction,
                                   **configs.encoder_conf)
        self.decoder = CTCLoss(vocab_size, self.encoder.output_size, **configs.decoder_conf)

    def forward(self, speech, speech_lengths, text, text_lengths):
        """Compute Model loss

        Args:
            speech (Tensor): [B, T, D]
            speech_lengths (Tensor): [B]
            text (Tensor): [B, U]
            text_lengths (Tensor): [B]

        Returns:
            loss (Tensor): [1]
        """
        eouts, eouts_len, final_state = self.encoder(speech, speech_lengths)
        loss = self.decoder(eouts, eouts_len, text, text_lengths)
        return {'loss': loss}

    @torch.jit.export
    def get_encoder_out(self, speech, speech_lengths):
        eouts, _, _ = self.encoder(speech, speech_lengths)
        ctc_probs = self.decoder.softmax(eouts)
        return ctc_probs

    @torch.jit.export
    def get_encoder_out_chunk(self, speech, speech_lengths, init_state: torch.Tensor = torch.zeros([0, 0, 0])):
        eouts, eouts_len, final_chunk_state = self.encoder(speech, speech_lengths, init_state)
        ctc_probs = self.decoder.softmax(eouts)
        return ctc_probs, eouts_len, final_chunk_state

    def export(self):
        static_model = torch.jit.script(self)
        return static_model


def DeepSpeech2ModelOnline(configs,
                           input_dim: int,
                           vocab_size: int):
    model = DeepSpeech2Model(configs=configs,
                             input_dim=input_dim,
                             vocab_size=vocab_size,
                             rnn_direction='forward')
    return model


def DeepSpeech2ModelOffline(configs,
                            input_dim: int,
                            vocab_size: int):
    model = DeepSpeech2Model(configs=configs,
                             input_dim=input_dim,
                             vocab_size=vocab_size,
                             rnn_direction='bidirect')
    return model
