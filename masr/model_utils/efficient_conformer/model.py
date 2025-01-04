import importlib
from typing import Tuple

import torch

from masr.data_utils.normalizer import FeatureNormalizer
# noinspection PyUnresolvedReferences
from masr.model_utils.transformer.decoder import *
# noinspection PyUnresolvedReferences
from masr.model_utils.efficient_conformer.encoder import *
from masr.model_utils.loss.ctc import CTCLoss
from masr.model_utils.loss.label_smoothing_loss import LabelSmoothingLoss
from masr.model_utils.utils.cmvn import GlobalCMVN
from masr.model_utils.utils.common import (IGNORE_ID, add_sos_eos, th_accuracy, reverse_pad_list)

from masr.utils.utils import DictObject

__all__ = ["EfficientConformerModel"]


class EfficientConformerModel(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            vocab_size: int,
            mean_istd_path: str,
            eos_id: int,
            streaming: bool = True,
            encoder_conf: DictObject = None,
            decoder_conf: DictObject = None,
            ctc_weight: float = 0.5,
            ignore_id: int = IGNORE_ID,
            reverse_weight: float = 0.0,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        super().__init__()
        self.input_size = input_size
        # 设置是否为流式模型
        self.streaming = streaming
        use_dynamic_chunk = False
        causal = False
        if self.streaming:
            use_dynamic_chunk = True
            causal = True
        feature_normalizer = FeatureNormalizer(mean_istd_filepath=mean_istd_path)
        global_cmvn = GlobalCMVN(torch.from_numpy(feature_normalizer.mean).float(),
                                 torch.from_numpy(feature_normalizer.istd).float())
        # 创建编码器和解码器
        mod = importlib.import_module(__name__)
        self.encoder = getattr(mod, encoder_conf.encoder_name)
        self.encoder = self.encoder(input_size=input_size,
                                    global_cmvn=global_cmvn,
                                    use_dynamic_chunk=use_dynamic_chunk,
                                    causal=causal,
                                    **encoder_conf.encoder_args if encoder_conf.encoder_args is not None else {})
        self.decoder = getattr(mod, decoder_conf.decoder_name)
        self.decoder = self.decoder(vocab_size=vocab_size,
                                    encoder_output_size=self.encoder.output_size(),
                                    **decoder_conf.decoder_args if decoder_conf.decoder_args is not None else {})

        self.ctc = CTCLoss(vocab_size, self.encoder.output_size())
        # sos 和 eos 使用相同的ID
        self.sos = eos_id
        self.eos = eos_id
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        Returns:
            total_loss, attention_loss, ctc_loss
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)  # [B, 1, T] -> [B]

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths,
                                                    self.reverse_weight)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        return {"loss": loss, "loss_att": loss_att, "loss_ctc": loss_ctc}

    def _calc_att_loss(self,
                       encoder_out: torch.Tensor,
                       encoder_mask: torch.Tensor,
                       ys_pad: torch.Tensor,
                       ys_pad_lens: torch.Tensor,
                       reverse_weight: float) -> Tuple[torch.Tensor, float]:
        """Calc attention loss.

        Args:
            encoder_out (torch.Tensor): [B, Tmax, D]
            encoder_mask (torch.Tensor): [B, 1, Tmax]
            ys_pad (torch.Tensor): [B, Umax]
            ys_pad_lens (torch.Tensor): [B]
            reverse_weight (float): reverse decoder weight.

        Returns:
            Tuple[torch.Tensor, float]: attention_loss, accuracy rate
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos, self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad, self.reverse_weight)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size),
                              ys_out_pad,
                              ignore_label=self.ignore_id)
        return loss_att, acc_att

    @torch.jit.export
    def subsampling_rate(self) -> int:
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        return self.encoder.embed.right_context

    @torch.jit.export
    def ignore_symbol(self) -> int:
        return self.ignore_id

    @torch.jit.export
    def sos_symbol(self) -> int:
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        return self.eos

    @torch.jit.export
    def get_encoder_out(self, speech: torch.Tensor, speech_lengths: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get encoder output

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_lengths (torch.Tensor): (batch, )
        Returns:
            Tensor: ctc softmax output
        """
        encoder_outs, encoder_mask = self.encoder(speech,
                                                  speech_lengths,
                                                  decoding_chunk_size=-1,
                                                  num_decoding_left_chunks=-1)  # (B, maxlen, encoder_dim)
        ctc_probs = self.ctc.log_softmax(encoder_outs)
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        return encoder_outs, ctc_probs, encoder_lens

    @torch.jit.export
    def get_encoder_out_chunk(self,
                              speech: torch.Tensor,
                              offset: int,
                              required_cache_size: int,
                              att_cache: torch.Tensor = torch.zeros([0, 0, 0, 0]),
                              cnn_cache: torch.Tensor = torch.zeros([0, 0, 0, 0])) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get encoder output

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
        Returns:
            Tensor: ctc softmax output
        """
        xs, att_cache, cnn_cache = self.encoder.forward_chunk(xs=speech,
                                                              offset=offset,
                                                              required_cache_size=required_cache_size,
                                                              att_cache=att_cache,
                                                              cnn_cache=cnn_cache)
        ctc_probs = self.ctc.log_softmax(xs)
        return ctc_probs, att_cache, cnn_cache

    @torch.jit.export
    def get_decoder_out(
            self,
            hyps: torch.Tensor,
            hyps_lens: torch.Tensor,
            encoder_out: torch.Tensor,
            reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)
        r_hyps_lens = hyps_lens - 1
        r_hyps = hyps[:, 1:]
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        index = index * seq_mask
        r_hyps = torch.gather(r_hyps, 1, index)
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
                                                     reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out

    @torch.no_grad()
    def export(self):
        static_model = torch.jit.script(self.eval())
        return static_model
