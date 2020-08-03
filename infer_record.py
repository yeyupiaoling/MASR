import time
import torch
import functools
import argparse
import pyaudio
import wave
import torch.nn.functional as F
from utils import feature
from ctcdecode import CTCBeamDecoder
from data.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--model_path",
                    default="save_model/model.pth",
                    type=str,
                    help="trained model path. (default: %(default)s)")
parser.add_argument("--lm_path",
                    default="lm/zh_giga.no_cna_cmn.prune01244.klm",
                    type=str,
                    help="language model path. (default: %(default)s)")
parser.add_argument("--record_time",
                    default=5,
                    type=int,
                    help="record time for second. (default: %(default)s)")
args = parser.parse_args()

alpha = 0.8
beta = 0.3
cutoff_top_n = 40
cutoff_prob = 1.0
beam_width = 32
num_processes = 4
blank_index = 0

model = torch.load(args.model_path)
model = model.cuda()
model.eval()

decoder = CTCBeamDecoder(model.vocabulary,
                         args.lm_path,
                         alpha,
                         beta,
                         cutoff_top_n,
                         cutoff_prob,
                         beam_width,
                         num_processes,
                         blank_index)


def translate(vocab, out, out_len):
    return "".join([vocab[x] for x in out[0:out_len]])


def predict(wav_path):
    wav = feature.load_audio(wav_path)
    spec = feature.spectrogram(wav)
    spec.unsqueeze_(0)
    with torch.no_grad():
        spec = spec.cuda()
        y = model.cnn(spec)
        y = F.softmax(y, 1)
    y_len = torch.tensor([y.size(-1)])
    y = y.permute(0, 2, 1)  # B * T * V
    print("decoding...")
    out, score, offset, out_len = decoder.decode(y, y_len)
    return translate(model.vocabulary, out[0][0], out_len[0][0])


def save_wave_file(filename, data):
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPWIDTH)
    wf.setframerate(RATE)
    wf.writeframes(b"".join(data))
    wf.close()


def record(wav_path, time=5):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    my_buf = []
    print("录音中(%ds)" % time)
    for i in range(0, int(RATE / CHUNK * time)):
        data = stream.read(CHUNK)
        my_buf.append(data)
        print(".", end="", flush=True)

    save_wave_file(wav_path, my_buf)
    stream.close()


if __name__ == '__main__':
    print_arguments(args)
    # 录音格式
    RATE = 16000
    CHUNK = 1024
    CHANNELS = 1
    SAMPWIDTH = 2
    # 临时保存路径
    save_path = 'dataset/record.wav'
    while True:
        _ = input("按下回车键开机录音，录音%s秒中：" % args.record_time)
        record(save_path, time=args.record_time)
        start = time.time()
        result_text = predict(save_path)
        end = time.time()
        print("识别时间：%dms，识别结果：%s" % (round((end - start) * 1000), result_text))
