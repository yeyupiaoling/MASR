import argparse
import functools
import time

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/conformer.yml',       "配置文件")
add_arg("use_gpu",          bool,  True,                          "是否使用GPU评估模型")
add_arg('metrics_type',     str,   'cer',                         "评估指标类型，中文用cer，英文用wer")
add_arg('decoder',          str,   'ctc_beam_search',             "解码器，支持ctc_greedy、ctc_beam_search")
add_arg('decoder_configs',  str,   'configs/chinese_decoder.yml', "解码器配置参数文件路径")
add_arg('resume_model',     str,   'models/ConformerModel_fbank/best_model/',  "模型的路径")
args = parser.parse_args()
print_arguments(args=args)


# 获取训练器
trainer = MASRTrainer(configs=args.configs,
                      use_gpu=args.use_gpu,
                      metrics_type=args.metrics_type,
                      decoder=args.decoder,
                      decoder_configs=args.decoder_configs)

# 开始评估
start = time.time()
loss, error_result = trainer.evaluate(resume_model=args.resume_model,
                                      display_result=True)
end = time.time()
print('评估消耗时间：{}s，错误率：{:.5f}'.format(int(end - start), error_result))
