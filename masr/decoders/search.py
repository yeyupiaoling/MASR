import math
from collections import defaultdict
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from masr.decoders.context_graph import ContextGraph
from masr.decoders.decod_status import DecodeResult, PrefixScore
from masr.decoders.utils import log_add
from masr.decoders.utils import make_pad_mask
from masr.decoders.utils import remove_duplicates_and_blank
from masr.model_utils.utils.common import add_sos_eos

__all__ = ["ctc_greedy_search", "ctc_prefix_beam_search", "attention_rescoring"]


def ctc_greedy_search(ctc_probs: torch.Tensor,
                      ctc_lens: torch.Tensor,
                      blank_id: int = 0) -> List[DecodeResult]:
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores, _ = topk_prob.max(1)
    scores = scores.cpu().detach().numpy().tolist()
    results = []
    for hyp, score in zip(hyps, scores):
        r = DecodeResult(tokens=remove_duplicates_and_blank(hyp, blank_id), score=score[0])
        results.append(r)
    return results


def ctc_prefix_beam_search(
        ctc_probs: torch.Tensor,
        ctc_lens: torch.Tensor,
        beam_size: int = 5,
        context_graph: ContextGraph = None,
        blank_id: int = 0,
) -> List[DecodeResult]:
    """
        Returns:
            List[List[List[int]]]: nbest result for each utterance
    """
    print("ctc_lens:", ctc_lens)
    batch_size = ctc_probs.shape[0]
    results = []
    # CTC prefix beam search can not be paralleled, so search one by one
    for i in range(batch_size):
        ctc_prob = ctc_probs[i]
        num_t = ctc_lens[i]
        cur_hyps = [(tuple(), PrefixScore(s=0.0,
                                          ns=-float('inf'),
                                          v_s=0.0,
                                          v_ns=0.0,
                                          context_state=None if context_graph is None
                                          else context_graph.root,
                                          context_score=0.0))]
        # 2. CTC beam search step by step
        for t in range(0, num_t):
            logp = ctc_prob[t]  # (vocab_size,)
            # key: prefix, value: PrefixScore
            next_hyps = defaultdict(lambda: PrefixScore())
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for u in top_k_index:
                u = u.item()
                prob = logp[u].item()
                for prefix, prefix_score in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if u == blank_id:  # blank
                        next_score = next_hyps[prefix]
                        next_score.s = log_add(next_score.s, prefix_score.score() + prob)
                        next_score.v_s = prefix_score.viterbi_score() + prob
                        next_score.times_s = prefix_score.times().copy()
                        # perfix not changed, copy the context from prefix
                        if context_graph and not next_score.has_context:
                            next_score.copy_context(prefix_score)
                            next_score.has_context = True
                    elif u == last:
                        #  Update *uu -> *u;
                        next_score1 = next_hyps[prefix]
                        next_score1.ns = log_add(next_score1.ns, prefix_score.ns + prob)
                        if next_score1.v_ns < prefix_score.v_ns + prob:
                            next_score1.v_ns = prefix_score.v_ns + prob
                            if next_score1.cur_token_prob < prob:
                                next_score1.cur_token_prob = prob
                                next_score1.times_ns = prefix_score.times_ns.copy()
                                next_score1.times_ns[-1] = t
                        if context_graph and not next_score1.has_context:
                            next_score1.copy_context(prefix_score)
                            next_score1.has_context = True

                        # Update *u-u -> *uu, - is for blank
                        n_prefix = prefix + (u,)
                        next_score2 = next_hyps[n_prefix]
                        next_score2.ns = log_add(next_score2.ns, prefix_score.s + prob)
                        if next_score2.v_ns < prefix_score.v_s + prob:
                            next_score2.v_ns = prefix_score.v_s + prob
                            next_score2.cur_token_prob = prob
                            next_score2.times_ns = prefix_score.times_s.copy()
                            next_score2.times_ns.append(t)
                        if context_graph and not next_score2.has_context:
                            next_score2.update_context(context_graph, prefix_score, u)
                            next_score2.has_context = True
                    else:
                        n_prefix = prefix + (u,)
                        next_score = next_hyps[n_prefix]
                        next_score.ns = log_add(next_score.ns, prefix_score.score() + prob)
                        if next_score.v_ns < prefix_score.viterbi_score() + prob:
                            next_score.v_ns = prefix_score.viterbi_score() + prob
                            next_score.cur_token_prob = prob
                            next_score.times_ns = prefix_score.times().copy()
                            next_score.times_ns.append(t)
                        if context_graph and not next_score.has_context:
                            next_score.update_context(context_graph, prefix_score, u)
                            next_score.has_context = True

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(), key=lambda x: x[1].total_score(), reverse=True)
            cur_hyps = next_hyps[:beam_size]

        # We should backoff the context score/state when the context is
        # not fully matched at the last time.
        if context_graph is not None:
            for i, hyp in enumerate(cur_hyps):
                context_score, new_context_state = context_graph.finalize(hyp[1].context_state)
                cur_hyps[i][1].context_score = context_score
                cur_hyps[i][1].context_state = new_context_state

        nbest = [list(y[0]) for y in cur_hyps]
        nbest_scores = [y[1].total_score() for y in cur_hyps]
        nbest_times = [y[1].times() for y in cur_hyps]
        best = nbest[0]
        best_score = nbest_scores[0]
        best_time = nbest_times[0]
        results.append(DecodeResult(tokens=best,
                                    score=best_score,
                                    times=best_time,
                                    nbest=nbest,
                                    nbest_scores=nbest_scores,
                                    nbest_times=nbest_times))
    return results


def attention_rescoring(
        model,
        ctc_prefix_results: List[DecodeResult],
        encoder_outs: torch.Tensor,
        encoder_lens: torch.Tensor,
        ctc_weight: float = 0.3,
        reverse_weight: float = 0.5,
) -> List[DecodeResult]:
    """
        Args:
            ctc_prefix_results(List[DecodeResult]): ctc prefix beam search results
    """
    sos, eos = model.sos_symbol(), model.eos_symbol()
    device = encoder_outs.device
    assert encoder_outs.shape[0] == len(ctc_prefix_results)
    batch_size = encoder_outs.shape[0]
    results = []
    for b in range(batch_size):
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = ctc_prefix_results[b].nbest
        ctc_scores = ctc_prefix_results[b].nbest_scores
        hyps_pad = pad_sequence([
            torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps
        ], True, model.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps], device=device, dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        prefix_len = 1
        decoder_out, r_decoder_out = model.forward_attention_decoder(hyps_pad, hyps_lens, encoder_out, reverse_weight)
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        confidences = []
        tokens_confidences = []
        for i, hyp in enumerate(hyps):
            score = 0.0
            tc = []  # tokens confidences
            for j, w in enumerate(hyp):
                s = decoder_out[i][j + (prefix_len - 1)][w]
                score += s
                tc.append(math.exp(s))
            score += decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
            # add right to left decoder score
            if reverse_weight > 0 and r_decoder_out.dim() > 0:
                r_score = 0.0
                for j, w in enumerate(hyp):
                    s = r_decoder_out[i][len(hyp) - j - 1 + (prefix_len - 1)][w]
                    r_score += s
                    tc[j] = (tc[j] + math.exp(s)) / 2
                r_score += r_decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            confidences.append(math.exp(score / (len(hyp) + 1)))
            # add ctc score
            score += ctc_scores[i] * ctc_weight
            if score > best_score:
                best_score = score.item()
                best_index = i
            tokens_confidences.append(tc)
        results.append(DecodeResult(hyps[best_index],
                                    best_score,
                                    confidence=confidences[best_index],
                                    times=ctc_prefix_results[b].nbest_times[best_index],
                                    tokens_confidence=tokens_confidences[best_index]))
    return results
