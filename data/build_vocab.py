import argparse
import functools
import codecs
from collections import Counter
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('count_threshold',  int,    0,  "Truncation threshold for char counts.")
add_arg('vocab_path',       str,    '../dataset/zh_vocab.json', "Filepath to write the vocabulary.")
add_arg('manifest_path',   str,    '../dataset/manifest.train', "manifest path")
# yapf: disable
args = parser.parse_args()


def count_manifest(counter, manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            for char in line.strip(',')[1].replace('\n', ''):
                counter.update(char)


def main():
    print_arguments(args)

    counter = Counter()
    count_manifest(counter, args.manifest_path)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with codecs.open(args.vocab_path, 'w', 'utf-8') as fout:
        labels = ['?']
        for char, count in count_sorted:
            if count < args.count_threshold: break
            labels.append(char)
        fout.write(str(labels))


if __name__ == '__main__':
    main()
