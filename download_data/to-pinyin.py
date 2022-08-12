import os
import functools
import argparse

from tqdm import tqdm
from pypinyin import lazy_pinyin, Style
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--annotation_text",
                    default="../dataset/annotation/",
                    type=str,
                    help="存放音频标注文件的目录 (默认: %(default)s)")
args = parser.parse_args()

def convert_annotation_pinyin(annotation_path):
    for annotation_text in os.listdir(annotation_path):
        annotation_text_path = os.path.join(annotation_path, annotation_text)
        with open(annotation_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        newlines = []
        for line in tqdm(lines):
            sections = line.split('\t')
            if len(sections)>= 2:
                audio_path = sections[0]
                text = sections[1].replace('\n', '').replace('\r', '').lower()
                if len(text)>0:
                    py = lazy_pinyin(text, style=Style.TONE3)
                    newline = [audio_path, ' '.join(py)]
                    newlines.append('\t'.join(newline)+os.linesep)
                
        with open(annotation_text_path, 'w', encoding='utf-8') as f:
            f.writelines(newlines)

def main():
    print_arguments(args)
    convert_annotation_pinyin(args.annotation_text)

if __name__ == '__main__':
    main()
