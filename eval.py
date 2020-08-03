import argparse
import functools
import torch
import torch.nn.functional as F
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


def evaluate(model, dataloader):
    decoder1 = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
    print("decoding...")
    with torch.no_grad():
        for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
            x = x.cuda()
            outs, out_lens = model(x, x_lens)
            outs = F.softmax(outs, 1)
            outs = outs.transpose(1, 2)
            ys = []
            offset = 0
            for y_len in y_lens:
                ys.append(y[offset: offset + y_len])
                offset += y_len
            out_strings, out_offsets = decoder.decode(outs, out_lens)
            y_strings = decoder1.convert_to_strings(ys)
            for pred, truth in zip(out_strings, y_strings):
                trans, ref = pred[0], truth[0]
                cer += decoder1.cer(trans, ref) / float(len(ref))
        cer /= len(dataloader.dataset)
    return cer


def main():
    print_arguments(args)
    dev_dataset = data.MASRDataset(args.dev_manifest_path, args.vocab_path)
    dev_dataloader = data.MASRDataLoader(dev_dataset, batch_size=args.batch_size, num_workers=8)
    cer = evaluate(model, dev_dataloader)
    print("CER=%f" % cer)


if __name__ == '__main__':
    main()
