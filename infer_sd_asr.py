import argparse
import functools
import time

from yeaudio.audio import AudioSegment
from masr.predict import MASRPredictor
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_dir',        str,    'models/ConformerModel_fbank/inference_model/', "导出的预测模型文件夹路径")
add_arg('audio_path',       str,    'dataset/test_long.wav',       "预测音频的路径")
add_arg('use_gpu',          bool,   True,                          "是否使用GPU预测")
add_arg('use_punc',         bool,   False,                         "是否给识别结果加标点符号")
add_arg('is_itn',           bool,   False,                         "是否对文本进行反标准化")
add_arg('decoder',          str,    'ctc_greedy_search',           "解码器，支持 ctc_greedy_search、ctc_prefix_beam_search、attention_rescoring")
add_arg('decoder_configs',  str,    'configs/decoder.yml',         "解码器配置参数文件路径")
add_arg('punc_device_id',   str,    '-1',                          "标点符合模型使用的设备，-1表示使用CPU预测，否则使用指定GPU预测")
add_arg('punc_model_dir',   str,    None,                          "标点符号的模型文件夹路径")
add_arg('vector_configs',   str,    'models/CAMPPlus_Fbank/cam++.yml',   "说话人日志配置文件")
add_arg('vector_model_path',str,    'models/CAMPPlus_Fbank/best_model/', "说话人日志模型文件路径")
add_arg('audio_db_path',    str,    'audio_db/',                   "音频库的路径")
add_arg('speaker_num',      int,    None,                          "说话人数量，提供说话人数量可以提高准确率")
add_arg('search_audio_db',  bool,   False,                         "是否在音频库中搜索对应的说话人")
add_arg('vector_threshold', float,  0.6,                           "判断是否为同一个人的阈值")
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MASRPredictor(model_dir=args.model_dir,
                          use_gpu=args.use_gpu,
                          decoder=args.decoder,
                          decoder_configs=args.decoder_configs,
                          punc_device_id=args.punc_device_id,
                          punc_model_dir=args.punc_model_dir)


# 短语音识别
def predict_audio():
    start = time.time()
    result = predictor.predict_sd_asr(audio_data=args.audio_path,
                                      vector_configs=args.vector_configs,
                                      vector_model_path=args.vector_model_path,
                                      vector_threshold=args.vector_threshold,
                                      audio_db_path=args.audio_db_path,
                                      speaker_num=args.speaker_num,
                                      search_audio_db=args.search_audio_db,
                                      use_punc=args.use_punc,
                                      is_itn=args.is_itn)
    print(f"消耗时间：{int(round((time.time() - start) * 1000))}ms, 识别结果: {result}")


if __name__ == "__main__":
    predict_audio()
