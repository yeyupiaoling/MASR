import argparse
import functools
import torch
import torch.nn.functional as F
from utils import feature
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm
from data.utility import add_arguments, print_arguments
from utils import data
from utils.decoder import GreedyDecoder

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
parser.add_argument("--dev_manifest_path",
                    default="dataset/manifest.dev",
                    type=str,
                    help="train manifest file path. (default: %(default)s)")
parser.add_argument("--vocab_path",
                    default="dataset/zh_vocab.json",
                    type=str,
                    help="vocab file path. (default: %(default)s)")
parser.add_argument("--batch_size",
                    default=64,
                    type=int,
                    help="number for batch size. (default: %(default)s)")
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
    y = y.permute(0, 2, 1)
    out, score, offset, out_len = decoder.decode(y, y_len)
    return translate(model.vocabulary, out[0][0], out_len[0][0])


def evaluate(dataloader, dev_manifest_path):
    cer = 0
    decoder1 = GreedyDecoder(dataloader.dataset.labels_str)
    with open(dev_manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with torch.no_grad():
        for line in tqdm(lines):
            path, label = line.replace('\n', '').split(',')
            out_strings = predict(path)
            cer += decoder1.cer(out_strings, label) / float(len(label))
        cer /= len(lines)
    return cer


def main():
    print_arguments(args)
    dev_dataset = data.MASRDataset(args.dev_manifest_path, args.vocab_path)
    dev_dataloader = data.MASRDataLoader(dev_dataset, batch_size=args.batch_size, num_workers=8)
    cer = evaluate(dev_dataloader, args.dev_manifest_path)
    print("CER=%f" % cer)


if __name__ == '__main__':
    main()
