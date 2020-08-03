import argparse
import functools
import time
import torch
import torch.nn.functional as F
from ctcdecode import CTCBeamDecoder
from data.utility import add_arguments, print_arguments
from utils import feature

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--model_path",
                    default="save_model/model.pth",
                    type=str,
                    help="trained model path. (default: %(default)s)")
parser.add_argument("--lm_path",
                    default="lm/zhidao_giga.klm",
                    type=str,
                    help="language model path. (default: %(default)s)")
parser.add_argument("--wav_path",
                    default="dataset/test.wav",
                    type=str,
                    help="infer audio file path. (default: %(default)s)")
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


if __name__ == '__main__':
    print_arguments(args)
    start = time.time()
    result_text = predict(args.wav_path)
    end = time.time()
    print("识别时间：%dms，识别结果：%s" % (round((end - start) * 1000), result_text))
