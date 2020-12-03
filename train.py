import datetime
import os
import time
import torch
import torch.nn as nn
from models.conv import GatedConv
from tqdm import tqdm
from utils.decoder import GreedyDecoder
from warpctc_pytorch import CTCLoss
from utils import data
import tensorboardX as tensorboard
import torch.nn.functional as F
import functools
import argparse
from data.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--train_manifest_path",
                    default="dataset/manifest.train",
                    type=str,
                    help="train manifest file path. (default: %(default)s)")
parser.add_argument("--dev_manifest_path",
                    default="dataset/manifest.dev",
                    type=str,
                    help="train manifest file path. (default: %(default)s)")
parser.add_argument("--vocab_path",
                    default="dataset/zh_vocab.json",
                    type=str,
                    help="vocab file path. (default: %(default)s)")
parser.add_argument("--save_model_path",
                    default="save_model/",
                    type=str,
                    help="vocab file path. (default: %(default)s)")
parser.add_argument("--restore_model",
                    default=None,
                    type=str,
                    help="restore model path. (default: %(default)s)")
parser.add_argument("--epochs",
                    default=200,
                    type=int,
                    help="train number. (default: %(default)s)")
parser.add_argument("--batch_size",
                    default=64,
                    type=int,
                    help="number for batch size. (default: %(default)s)")
parser.add_argument("--learning_rate",
                    default=0.6,
                    type=int,
                    help="number for batch size. (default: %(default)s)")
args = parser.parse_args()


def train(model,
          train_manifest_path,
          dev_manifest_path,
          vocab_path,
          epochs,
          batch_size,
          learning_rate,
          momentum=0.8,
          max_grad_norm=0.2,
          weight_decay=0):
    train_dataset = data.MASRDataset(train_manifest_path, vocab_path)
    dev_dataset = data.MASRDataset(dev_manifest_path, vocab_path)
    # 获取总的batch数量
    batchs = (len(train_dataset) + batch_size - 1) // batch_size
    # 获取一个打乱的数据和未打乱的
    train_dataloader = data.MASRDataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    train_dataloader_shuffle = data.MASRDataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    # 获取测试数据
    dev_dataloader = data.MASRDataLoader(dev_dataset, batch_size=batch_size, num_workers=8)
    # 定义优化方法
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters,
                                lr=learning_rate,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=weight_decay)
    # 学习率衰减
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8, last_epoch=epochs // 2)
    # 创建CTC损失函数，该函是DeepSpeech实现的Pytorch版本
    ctcloss = CTCLoss(size_average=True)
    # 记录日志信息
    writer = tensorboard.SummaryWriter()
    gstep = 0
    for epoch in range(epochs):
        # scheduler.step()
        epoch_loss = 0
        # 第一次训练使用不打乱的数据
        if epoch > 0:
            train_dataloader = train_dataloader_shuffle
        # 获取实时的学习率，记录到日志中
        lr = get_lr(optimizer)
        writer.add_scalar("lr/epoch", lr, epoch)
        # 开始一轮训练
        for i, (x, y, x_lens, y_lens) in enumerate(train_dataloader):
            start_time = time.time()
            x = x.cuda()
            out, out_lens = model(x, x_lens)
            out = out.transpose(0, 1).transpose(0, 2)
            loss = ctcloss(out, y, out_lens, y_lens)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            if i % 100 == 0:
                writer.add_scalar("loss/step", loss.item(), gstep)
                gstep += 1
                remain_step = (epochs - epoch) * batchs - i
                remain_time = remain_step * (time.time() - start_time)
                remain_time = str(datetime.timedelta(seconds=int(remain_time)))
                print("[{}/{}][{}/{}]\tLoss = {:.4f}\tRemain time: {}".format(epoch + 1, epochs, i, int(batchs),
                                                                              loss.item(), remain_time))
        epoch_loss = epoch_loss / batchs
        cer = evaluate(model, dev_dataloader)
        writer.add_scalar("loss/epoch", epoch_loss, epoch)
        writer.add_scalar("cer/epoch", cer, epoch)
        print("Epoch {}: Loss= {}, CER = {}".format(epoch + 1, epoch_loss, cer))
        torch.save(model, os.path.join(args.save_model_path, "model_{}.pth".format(epoch + 1)))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def evaluate(model, dataloader):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
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
            y_strings = decoder.convert_to_strings(ys)
            for pred, truth in zip(out_strings, y_strings):
                trans, ref = pred[0], truth[0]
                cer += decoder.cer(trans, ref) / float(len(ref))
        cer /= len(dataloader.dataset)
    model.train()
    return cer


def main():
    print_arguments(args)
    # 创建保存模型的文件夹
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    # 加载数据字典
    with open(args.vocab_path, 'r', encoding='utf-8') as f:
        vocabulary = eval(f.read())
        vocabulary = "".join(vocabulary)
    # 获取模型
    model = GatedConv(vocabulary)
    # 加载预训练模型
    if args.restore_model:
        model = torch.load(args.restore_model)
    model = model.cuda()
    train(model=model,
          train_manifest_path=args.train_manifest_path,
          dev_manifest_path=args.dev_manifest_path,
          vocab_path=args.vocab_path,
          epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
