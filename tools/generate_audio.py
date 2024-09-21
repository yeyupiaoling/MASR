import argparse
import os
from pathlib import Path

import numpy as np
import requests
import torch
import torchaudio
from tqdm import tqdm


def generate(args):
    # construct dataset for generate
    sentences = []
    with open(args.text, 'rt', encoding='utf-8') as f:
        for line in f:
            utt_id, sentence = line.strip().split()
            sentences.append((utt_id, sentence))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(os.path.dirname(args.annotation_path))
    annotation_path.mkdir(parents=True, exist_ok=True)
    start_num = 0
    if os.path.exists(args.annotation_path):
        with open(args.annotation_path, 'r', encoding='utf-8') as f_ann:
            start_num = len(f_ann.readlines())
    f_ann = open(args.annotation_path, 'a', encoding='utf-8')

    url = f"http://{args.host}:{args.port}/inference_sft"
    # 开始生成音频
    for i in tqdm(range(start_num, len(sentences))):
        utt_id, sentence = sentences[i]
        save_audio_path = str(output_dir / (utt_id + ".wav"))
        # 调用接口合成语音
        payload = {
            'tts_text': sentence,
            'spk_id': args.spk_id
        }
        response = requests.request("GET", url, data=payload, stream=True)
        tts_audio = b''
        for r in response.iter_content(chunk_size=16000):
            tts_audio += r
        tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
        save_audio_path = save_audio_path[3:].replace('\\', '/')
        torchaudio.save(save_audio_path, tts_speech, 22050)
        sentence = sentence.replace('。', '').replace('，', '').replace('！', '').replace('？', '')
        f_ann.write(f'{save_audio_path}\t{sentence}\n')
        f_ann.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='localhost')
    parser.add_argument('--port',
                        type=int,
                        default='50000')
    parser.add_argument('--spk_id',
                        type=str,
                        default='中文女')
    parser.add_argument("--text",
                        type=str,
                        default='generate_audio/corpus.txt',
                        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='../dataset/audio/generate',
                        help="output audio dir.")
    parser.add_argument("--annotation_path",
                        type=str,
                        default='../dataset/annotation/generate.txt',
                        help="audio annotation path.")
    parser.add_argument("--device",
                        type=str,
                        default="gpu",
                        help="device type to use.")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
