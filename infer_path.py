import argparse
import functools
import time

from yeaudio.audio import AudioSegment
from masr.predict import MASRPredictor
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_dir',        str,    'models/ConformerModel_fbank/inference_model/', "导出的预测模型文件夹路径")
add_arg('audio_path',       str,    'dataset/test.wav',            "预测音频的路径")
add_arg('real_time_demo',   bool,   False,                         "是否使用实时语音识别演示")
add_arg('use_gpu',          bool,   True,                          "是否使用GPU预测")
add_arg('use_punc',         bool,   False,                         "是否给识别结果加标点符号")
add_arg('is_itn',           bool,   False,                         "是否对文本进行反标准化")
add_arg('allow_use_vad',    bool,   True,                          "当音频长度大于30秒，是否允许使用语音活动检测分割音频进行识别")
add_arg('decoder',          str,    'ctc_greedy_search',           "解码器，支持 ctc_greedy_search、ctc_prefix_beam_search、attention_rescoring")
add_arg('decoder_configs',  str,    'configs/decoder.yml',         "解码器配置参数文件路径")
add_arg('punc_device_id',   str,    '-1',                          "标点符合模型使用的设备，-1表示使用CPU预测，否则使用指定GPU预测")
add_arg('punc_model_dir',   str,    None,                          "标点符号的模型文件夹路径")
add_arg('punc_online_model_dir',    str,   None,                   "流式标点符号的模型文件夹路径")
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MASRPredictor(model_dir=args.model_dir,
                          use_gpu=args.use_gpu,
                          decoder=args.decoder,
                          decoder_configs=args.decoder_configs,
                          punc_device_id=args.punc_device_id,
                          punc_model_dir=args.punc_model_dir,
                          punc_online_model_dir=args.punc_online_model_dir)


# 短语音识别
def predict_audio():
    start = time.time()
    result = predictor.predict(audio_data=args.audio_path,
                               use_punc=args.use_punc,
                               is_itn=args.is_itn,
                               allow_use_vad=args.allow_use_vad)
    print(f"消耗时间：{int(round((time.time() - start) * 1000))}ms, 识别结果: {result}")


# 实时识别模拟
def real_time_predict_demo():
    # 识别间隔时间
    interval_time = 0.5
    CHUNK = int(16000 * interval_time)
    # 读取数据
    audio_segment = AudioSegment.from_file(args.audio_path)
    audio_bytes = audio_segment.to_bytes(dtype='int16')
    sample_rate = audio_segment.sample_rate
    index = 0
    # 流式识别
    while index < len(audio_bytes):
        start = time.time()
        data = audio_bytes[index:index + CHUNK]
        result = predictor.predict_stream(audio_data=data, use_punc=args.use_punc, is_itn=args.is_itn,
                                          is_final=len(data) < CHUNK, sample_rate=sample_rate)
        index += CHUNK
        if result is None: continue
        text = result['text']
        print(f"【实时结果】：消耗时间：{int((time.time() - start) * 1000)}ms, 识别结果: {text}")
    # 重置流式识别
    predictor.reset_predictor()
    predictor.reset_stream_state()


if __name__ == "__main__":
    if args.real_time_demo:
        real_time_predict_demo()
    else:
        predict_audio()
