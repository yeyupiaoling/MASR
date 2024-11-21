import multiprocessing
import os
import platform
from math import log

import kenlm
import numpy as np
from loguru import logger

from masr.utils.utils import download


class BeamSearchDecoder:
    def __init__(self,
                 vocab_list,
                 alpha=1.2,
                 beta=0.35,
                 beam_size=300,
                 cutoff_prob=0.99,
                 cutoff_top_n=40,
                 language_model_path=None,
                 num_processes=10,
                 blank_id=0):
        self.vocab_list = vocab_list
        self.alpha = alpha
        self.beta = beta
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.vocab_list = vocab_list
        self.num_processes = num_processes
        self.language_model_path = language_model_path
        self.blank_id = blank_id
        if (not os.path.exists(self.language_model_path) and
                self.language_model_path == 'lm/zh_giga.no_cna_cmn.prune01244.klm'):
            logger.info('=' * 70)
            language_model_url = 'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm'
            logger.info("语言模型不存在，正在下载，下载地址： %s ..." % language_model_url)
            os.makedirs(os.path.dirname(self.language_model_path), exist_ok=True)
            download(url=language_model_url, download_target=self.language_model_path)
            logger.info('=' * 70)
        logger.info('=' * 70)
        logger.info("初始化解码器...")
        assert os.path.exists(self.language_model_path), f'语言模型不存在：{self.language_model_path}'
        self._ext_scorer = Scorer(self.alpha, self.beta, self.language_model_path)
        logger.info("初始化解码器完成!")
        logger.info('=' * 70)

    def reset_params(self, alpha, beta):
        self._ext_scorer.reset_params(alpha, beta)

    def ctc_beam_search_decoder(self, ctc_probs):
        if not isinstance(ctc_probs, list):
            ctc_probs = ctc_probs.tolist()
        ctc_probs = np.array(ctc_probs)
        # 必须要计算softmax，否则会导致结果不正确
        ctc_probs = self.softmax(ctc_probs)
        # 初始化前缀集合，以制表符作为前缀，概率为1.0
        prefix_set_prev = {'\t': 1.0}
        # 初始化前缀为制表符时的空白概率和非空白概率
        probs_b_prev, probs_nb_prev = {'\t': 1.0}, {'\t': 0.0}

        # 在循环中扩展前缀
        for time_step in range(len(ctc_probs)):
            # 初始化下一个时间步的前缀集合和对应的概率字典
            prefix_set_next, probs_b_cur, probs_nb_cur = {}, {}, {}

            # 获取当前时间步的概率索引
            prob_idx = list(enumerate(ctc_probs[time_step]))
            cutoff_len = len(prob_idx)

            # 如果启用了剪枝
            if self.cutoff_prob < 1.0 or self.cutoff_top_n < cutoff_len:
                # 按概率从高到低排序
                prob_idx = sorted(prob_idx, key=lambda asd: asd[1], reverse=True)
                cutoff_len, cum_prob = 0, 0.0
                # 计算累积概率并确定剪枝长度
                for i in range(len(prob_idx)):
                    cum_prob += prob_idx[i][1]
                    cutoff_len += 1
                    if cum_prob >= self.cutoff_prob:
                        break
                # 确定最终的剪枝长度
                cutoff_len = min(cutoff_len, self.cutoff_top_n)
                # 取前cutoff_len个概率最高的索引
                prob_idx = prob_idx[0:cutoff_len]

            # 遍历前一个时间步的前缀集合
            for l in prefix_set_prev:
                # 如果当前前缀不在下一个时间步的前缀集合中，则初始化其空白概率和非空白概率为0.0
                if l not in prefix_set_next:
                    probs_b_cur[l], probs_nb_cur[l] = 0.0, 0.0

                # 遍历当前时间步的概率索引，扩展前缀
                for index in range(cutoff_len):
                    c, prob_c = prob_idx[index][0], prob_idx[index][1]

                    # 如果当前字符为空白字符
                    if c == self.blank_id:
                        # 更新空白概率
                        probs_b_cur[l] += prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                    else:
                        # 获取当前字符和前一个字符
                        last_char = l[-1]
                        new_char = self.vocab_list[c]
                        l_plus = l + new_char
                        # 如果新前缀不在下一个时间步的前缀集合中，则初始化其空白概率和非空白概率为0.0
                        if l_plus not in prefix_set_next:
                            probs_b_cur[l_plus], probs_nb_cur[l_plus] = 0.0, 0.0

                        # 根据当前字符和前一个字符的关系更新非空白概率
                        if new_char == last_char:
                            probs_nb_cur[l_plus] += prob_c * probs_b_prev[l]
                            probs_nb_cur[l] += prob_c * probs_nb_prev[l]
                        elif new_char == ' ':
                            # 处理空格字符的特殊情况
                            if len(l) == 1:
                                score = 1.0
                            else:
                                prefix = l[1:]
                                score = self._ext_scorer(prefix)
                            probs_nb_cur[l_plus] += score * prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                        else:
                            probs_nb_cur[l_plus] += prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                        # 将新前缀添加到下一个时间步的前缀集合中
                        # add l_plus into prefix_set_next
                        prefix_set_next[l_plus] = probs_nb_cur[l_plus] + probs_b_cur[l_plus]
                # 将当前前缀添加到下一个时间步的前缀集合中
                prefix_set_next[l] = probs_b_cur[l] + probs_nb_cur[l]
            # 更新概率字典
            probs_b_prev, probs_nb_prev = probs_b_cur, probs_nb_cur

            # 存储前beam_size个概率最高的前缀
            prefix_set_prev = sorted(prefix_set_next.items(), key=lambda asd: asd[1], reverse=True)
            if self.beam_size < len(prefix_set_prev):
                prefix_set_prev = prefix_set_prev[:self.beam_size]
            prefix_set_prev = dict(prefix_set_prev)

        # 初始化beam结果列表
        beam_result = []
        # 遍历最终时间步的前缀集合，计算每个前缀的概率和对应的序列
        for seq, prob in prefix_set_prev.items():
            if prob > 0.0 and len(seq) > 1:
                result = seq[1:]
                # 使用外部评分器对最后一个单词进行评分
                if result[-1] != ' ':
                    prob = prob * self._ext_scorer(result)
                # 计算对数概率
                log_prob = log(prob)
                # 将对数概率和序列添加到beam结果列表中
                beam_result.append((log_prob, result))
            else:
                # 对于不符合条件的前缀，添加负无穷大的对数概率和空字符串
                beam_result.append((float('-inf'), ''))

        # 输出前beam_size个解码结果
        beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)
        result = beam_result[0][1].replace('▁', ' ').strip()
        # 返回概率最高的解码结果
        return result

    def ctc_beam_search_decoder_batch(self, ctc_probs, ctc_lens):
        assert len(ctc_probs) == len(ctc_lens)
        # Windows系统不支持多进程
        beam_search_results = []
        if platform.system() == 'Windows' or self.num_processes <= 1:
            for i, ctc_prob in enumerate(ctc_probs):
                ctc_prob = ctc_prob[:ctc_lens[i]]
                beam_search_results.append(self.ctc_beam_search_decoder(ctc_prob))
            return beam_search_results
        # 使用多进程进行并行解码
        if not isinstance(ctc_probs, list):
            ctc_probs = ctc_probs.tolist()
            ctc_lens = ctc_lens.tolist()
        pool = multiprocessing.Pool(processes=self.num_processes)
        results = []
        for i, ctc_prob in enumerate(ctc_probs):
            ctc_prob = ctc_prob[:ctc_lens[i]]
            results.append(pool.apply_async(self.ctc_beam_search_decoder, args=(ctc_prob,)))
        pool.close()
        pool.join()
        beam_search_results = [result.get() for result in results]
        return beam_search_results

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)


class Scorer(object):
    """外部评分器评估波束搜索解码中的前缀或整个句子，包括n-gram语言模型的评分和单词计数

    :param alpha: 与语言模型相关的参数。当alpha=0时，不使用语言模型。
    :type alpha: float
    :param beta: 与单词计数相关的参数。当beta=0时，不使用单词计数。
    :type beta: float
    :model_path: 语言模型路径
    :type model_path: str
    """

    def __init__(self, alpha, beta, model_path):
        self._alpha = alpha
        self._beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invaid language model path: %s" % model_path)
        self._language_model = kenlm.LanguageModel(model_path)

    # n-gram语言模型评分
    def _language_model_score(self, sentence):
        # log10 prob of last word
        log_cond_prob = list(self._language_model.full_scores(sentence, eos=False))[-1][0]
        return np.power(10, log_cond_prob)

    @staticmethod
    def _word_count(sentence):
        words = sentence.strip().split(' ')
        return len(words)

    def reset_params(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    def __call__(self, sentence, use_log=False):
        """评估函数，收集所有不同的得分并返回最终得分

        :param sentence: 用于计算的输入句子
        :type sentence: str
        :param use_log: 是否使用对数
        :type use_log: bool
        :return: 评价分数，用小数或对数表示
        :rtype: float
        """
        lm = self._language_model_score(sentence)
        word_cnt = self._word_count(sentence)
        if not use_log:
            score = np.power(lm, self._alpha) * np.power(word_cnt, self._beta)
        else:
            score = self._alpha * np.log(lm) + self._beta * np.log(word_cnt)
        return score
