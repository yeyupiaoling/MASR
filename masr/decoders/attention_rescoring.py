from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from masr.decoders.ctc_prefix_beam_search import ctc_prefix_beam_search
from masr.model_utils.utils.common import add_sos_eos


def attention_rescoring(
        model,
        ctc_probs: torch.Tensor,
        ctc_lens: torch.Tensor,
        encoder_outs: torch.Tensor,
        encoder_lens: torch.Tensor,
        beam_size: int = 5,
        blank_id: int = 0,
        ctc_weight: float = 0.3,
        reverse_weight: float = 0.5,
) -> List:
    """Attention rescoring

    param model: 模型
    param ctc_probs: (B, maxlen, vocab_size) 模型编码器输出的概率分布
    param ctc_lens: (B, ) 每个样本的实际长度
    param encoder_outs: (B, maxlen, encoder_dim) 编码器输出
    param encoder_lens: (B, ) 每个样本的实际长度
    param beam_size: 解码搜索大小
    param blank_id: 空白标签的id
    param ctc_weight: CTC解码器权重
    param reverse_weight: 反向解码器权重
    return: 解码结果，和所有解码结果，用于attention_rescoring解码器使用
    """
    device = encoder_outs.device
    batch_size = encoder_outs.shape[0]
    sos, eos, ignore_id = model.sos_symbol(), model.eos_symbol(), model.ignore_symbol()
    # len(hyps) = beam_size, encoder_out: (1, maxlen, encoder_dim)
    _, hyps_list = ctc_prefix_beam_search(ctc_probs=ctc_probs, ctc_lens=ctc_lens, blank_id=blank_id)
    assert len(hyps_list[0]) == beam_size

    results = []
    for b in range(batch_size):
        hyps = hyps_list[b]
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)

        hyp_list = []
        for hyp in hyps:
            hyp_content = hyp[0]
            # Prevent the hyp is empty
            if len(hyp_content) == 0:
                hyp_content = (blank_id,)
            hyp_content = torch.tensor(hyp_content, device=device, dtype=torch.int64)
            hyp_list.append(hyp_content)
        hyps_pad = pad_sequence(hyp_list, True, ignore_id)
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps], device=device, dtype=torch.int64)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at beginning

        # ctc score in ln domain
        # (beam_size, max_hyps_len, vocab_size)
        decoder_out, r_decoder_out = model.forward_attention_decoder(hyps_pad, hyps_lens, encoder_out, reverse_weight)

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        # hyps is List[(Text=List[int], Score=float)], len(hyps)=beam_size
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            # last decoder output token is `eos`, for laste decoder input token.
            score += decoder_out[i][len(hyp[0])][eos]
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score (which in ln domain)
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        results.append(list(hyps[best_index][0]))
    return results
