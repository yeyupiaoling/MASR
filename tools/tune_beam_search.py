import argparse
import functools
import time
from datetime import timedelta

import numpy as np
from loguru import logger
from tqdm import tqdm

from masr.decoders.beam_search_decoder import BeamSearchDecoder
from masr.trainer import MASRTrainer
from masr.utils.metrics import wer, cer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("use_gpu",          bool,  True,                        "是否使用GPU评估模型")
add_arg('configs',          str,   'configs/conformer.yml',     "配置文件")
add_arg('decoder_configs',  str,   'configs/decoder.yml',       "解码器配置参数文件路径")
add_arg('resume_model',     str,   'models/ConformerModel_fbank/best_model/',  "模型的路径")
add_arg('metrics_type',     str,   'cer',                       "评估指标类型，中文用cer，英文用wer，中英混合用mer")
add_arg('num_alphas',       int,    30,    "用于调优的alpha候选项")
add_arg('num_betas',        int,    20,    "用于调优的beta候选项")
add_arg('alpha_from',       float,  1.0,   "alpha调优开始大小")
add_arg('alpha_to',         float,  3.2,   "alpha调优结速大小")
add_arg('beta_from',        float,  0.1,   "beta调优开始大小")
add_arg('beta_to',          float,  4.5,   "beta调优结速大小")
args = parser.parse_args()
print_arguments(args=args)


def tune():
    # 逐步调整alphas参数和betas参数
    assert args.num_alphas >= 0, "num_alphas must be non-negative!"
    assert args.num_betas >= 0, "num_betas must be non-negative!"

    # 创建用于搜索的alphas参数和betas参数
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(round(alpha, 2), round(beta, 2)) for alpha in cand_alphas for beta in cand_betas]
    logger.info(f'解码alpha和beta的组合数量：{len(params_grid)}，排列：{params_grid}')

    # 获取训练器
    trainer = MASRTrainer(configs=args.configs,
                          use_gpu=args.use_gpu,
                          decoder_configs=args.decoder_configs)

    # 开始评估
    start = time.time()
    all_ctc_probs, all_ctc_lens, all_label = trainer.evaluate(resume_model=args.resume_model, only_ctc_probs=True)
    logger.info(f'获取模型输出消耗时间：{int(time.time() - start)}s')

    # 获取解码器
    decoder_args = trainer.decoder_configs.get('ctc_beam_search_args', {})
    beam_search_decoder = BeamSearchDecoder(vocab_list=trainer.tokenizer.vocab_list,
                                            blank_id=trainer.tokenizer.blank_id,
                                            **decoder_args)

    logger.info('开始使用识别结果解码...')
    # 搜索alphas参数和betas参数
    best_alpha, best_beta, best_result = 0, 0, 1
    for i, (alpha, beta) in enumerate(params_grid):
        error_results = []
        start_time = time.time()
        for j in tqdm(range(len(all_ctc_probs))):
            ctc_probs, ctc_lens, label = all_ctc_probs[j], all_ctc_lens[j], all_label[j]
            beam_search_decoder.reset_params(alpha, beta)
            text = beam_search_decoder.ctc_beam_search_decoder_batch(ctc_probs=ctc_probs, ctc_lens=ctc_lens)
            for l, t in zip(label, text):
                # 计算字错率或者词错率
                if args.metrics_type == 'wer':
                    error_rate = wer(l, t)
                else:
                    error_rate = cer(l, t)
                error_results.append(error_rate)
        error_result = np.mean(error_results)
        if error_result < best_result:
            best_alpha = alpha
            best_beta = beta
            best_result = error_result
        eta_sec = start_time * (len(params_grid) - i - 1) / 1000
        eta_str = str(timedelta(seconds=int(eta_sec)))
        logger.info(
            f'[{i + 1}/{len(params_grid)}] 当alpha为：{alpha}, beta为：{beta}，{args.metrics_type}：{error_result:.5f}, '
            f'【目前最优】当alpha为：{best_alpha}, beta为：{best_beta}，{args.metrics_type}：{best_result:.5f}, '
            f'预计剩余时间：{eta_str}')
    logger.info(f'【最终最优】当alpha为：%f, {best_alpha}, beta为：{best_beta}，{args.metrics_type}：{best_result:.5f}')


if __name__ == '__main__':
    tune()
