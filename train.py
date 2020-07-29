import torch
import torch.nn as nn
import data
from models.conv import GatedConv
from tqdm import tqdm
from decoder import GreedyDecoder
from warpctc_pytorch import CTCLoss
import tensorboardX as tensorboard
import torch.nn.functional as F
import json
import functools
import argparse
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--train_index_path",
                    default="dataset/manifest.train",
                    type=str,
                    help="train manifest file path. (default: %(default)s)")
parser.add_argument("--dev_index_path",
                    default="dataset/manifest.dev",
                    type=str,
                    help="train manifest file path. (default: %(default)s)")
parser.add_argument("--vocab_path",
                    default="dataset/zh_vocab.json",
                    type=str,
                    help="vocab file path. (default: %(default)s)")
parser.add_argument("--epochs",
                    default=1000,
                    type=int,
                    help="train number. (default: %(default)s)")
parser.add_argument("--batch_size",
                    default=64,
                    type=int,
                    help="number for batch size. (default: %(default)s)")
args = parser.parse_args()


def train(model,
          train_manifest_path,
          dev_manifest_path,
          vocab_path,
          epochs,
          batch_size,
          learning_rate=0.6,
          momentum=0.8,
          max_grad_norm=0.2,
          weight_decay=0):
    train_dataset = data.MASRDataset(train_manifest_path, vocab_path)
    batchs = (len(train_dataset) + batch_size - 1) // batch_size
    dev_dataset = data.MASRDataset(dev_manifest_path, vocab_path)
    train_dataloader = data.MASRDataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    train_dataloader_shuffle = data.MASRDataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    dev_dataloader = data.MASRDataLoader(dev_dataset, batch_size=batch_size, num_workers=8)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters,
                                lr=learning_rate,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=weight_decay)
    ctcloss = CTCLoss(size_average=True)
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)
    writer = tensorboard.SummaryWriter()
    gstep = 0
    for epoch in range(epochs):
        epoch_loss = 0
        if epoch > 0:
            train_dataloader = train_dataloader_shuffle
        # lr_sched.step()
        lr = get_lr(optimizer)
        writer.add_scalar("lr/epoch", lr, epoch)
        for i, (x, y, x_lens, y_lens) in enumerate(train_dataloader):
            x = x.to("cuda")
            out, out_lens = model(x, x_lens)
            out = out.transpose(0, 1).transpose(0, 2)
            loss = ctcloss(out, y, out_lens, y_lens)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar("loss/step", loss.item(), gstep)
            gstep += 1
            print(
                "[{}/{}][{}/{}]\tLoss = {}".format(epoch + 1, epochs, i, int(batchs), loss.item()))
        epoch_loss = epoch_loss / batchs
        cer = eval(model, dev_dataloader)
        writer.add_scalar("loss/epoch", epoch_loss, epoch)
        writer.add_scalar("cer/epoch", cer, epoch)
        print("Epoch {}: Loss= {}, CER = {}".format(epoch, epoch_loss, cer))
        torch.save(model, "pretrained/model_{}.pth".format(epoch))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def eval(model, dataloader):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
    print("decoding")
    with torch.no_grad():
        for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
            x = x.to("cuda")
            outs, out_lens = model(x, x_lens)
            outs = F.softmax(outs, 1)
            outs = outs.transpose(1, 2)
            ys = []
            offset = 0
            for y_len in y_lens:
                ys.append(y[offset: offset + y_len])
                offset += y_len
            out_strings, out_offsets = decoder.decode(outs, out_lens)
            y_strings = decoder.convert_to_strings(ys)
            for pred, truth in zip(out_strings, y_strings):
                trans, ref = pred[0], truth[0]
                cer += decoder.cer(trans, ref) / float(len(ref))
        cer /= len(dataloader.dataset)
    model.train()
    return cer


def main():
    print_arguments(args)
    with open("data_aishell/labels.json") as f:
        vocabulary = json.load(f)
        vocabulary = "".join(vocabulary)
    model = GatedConv(vocabulary)
    model.to("cuda")
    train(model=model,
          train_manifest_path=args.train_manifest_path,
          dev_manifest_path=args.dev_manifest_path,
          vocab_path=args.vocab_path,
          epochs=args.epochs,
          batch_size=args.batch_size)


if __name__ == "__main__":
    main()
