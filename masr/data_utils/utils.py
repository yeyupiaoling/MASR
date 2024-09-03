import json
import os
import time
import wave

import numpy as np
import soundfile
from loguru import logger
from tqdm import tqdm
from yeaudio.audio import AudioSegment
from zhconv import convert

from masr.data_utils.binary import DatasetWriter


def read_manifest(manifest_path, max_duration=float('inf'), min_duration=0.5):
    """解析数据列表
    持续时间在[min_duration, max_duration]之外的实例将被过滤。

    :param manifest_path: 数据列表的路径
    :type manifest_path: str
    :param max_duration: 过滤的最长音频长度
    :type max_duration: float
    :param min_duration: 过滤的最短音频长度
    :type min_duration: float
    :return: 数据列表，JSON格式
    :rtype: list
    :raises IOError: If failed to parse the manifest.
    """
    manifest = []
    for json_line in open(manifest_path, 'r', encoding='utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        if max_duration >= json_data["duration"] >= min_duration:
            manifest.append(json_data)
    return manifest


# 创建数据列表
def create_manifest(annotation_path, train_manifest_path, test_manifest_path, is_change_frame_rate=True,
                    only_keep_zh_en=True, max_test_manifest=10000, target_sr=16000):
    data_list = []
    test_list = []
    durations = []
    for annotation_text in os.listdir(annotation_path):
        annotation_text_path = os.path.join(annotation_path, annotation_text)
        if os.path.splitext(annotation_text_path)[-1] == '.json':
            with open(annotation_text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                try:
                    d = json.loads(line)
                except Exception as e:
                    logger.warning(f'{line} 错误，已跳过，错误信息：{e}')
                    continue
                audio_path, text = d["audio_filepath"], d["text"]
                start_time, end_time, duration = d["start_time"], d["end_time"], d["duration"]
                # 重新调整音频格式并保存
                if is_change_frame_rate:
                    change_rate(audio_path, target_sr=target_sr)
                # 获取音频长度
                durations.append(duration)
                text = text.lower().strip()
                if only_keep_zh_en:
                    # 过滤非法的字符
                    text = is_ustr(text)
                if len(text) == 0: continue
                # 保证全部都是简体
                text = convert(text, 'zh-cn')
                # 加入数据列表中
                line = dict(audio_filepath=audio_path.replace('\\', '/'),
                            text=text,
                            duration=duration,
                            start_time=start_time,
                            end_time=end_time)
                if annotation_text == 'test.json':
                    test_list.append(line)
                else:
                    data_list.append(line)
        else:
            with open(annotation_text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                try:
                    audio_path, text = line.replace('\n', '').replace('\r', '').split('\t')
                except Exception as e:
                    logger.warning(f'{line} 错误，已跳过，错误信息：{e}')
                    continue
                # 重新调整音频格式并保存
                if is_change_frame_rate:
                    change_rate(audio_path, target_sr=target_sr)
                # 获取音频长度
                audio_segment = AudioSegment.from_file(audio_path)
                duration = audio_segment.duration
                durations.append(duration)
                text = text.lower().strip()
                if only_keep_zh_en:
                    # 过滤非法的字符
                    text = is_ustr(text)
                if len(text) == 0 or text == ' ': continue
                # 保证全部都是简体
                text = convert(text, 'zh-cn')
                # 加入数据列表中
                line = dict(audio_filepath=audio_path.replace('\\', '/'),
                            text=text,
                            duration=duration)
                if annotation_text == 'test.txt':
                    test_list.append(line)
                else:
                    data_list.append(line)

    # 按照音频长度降序
    data_list.sort(key=lambda x: x["duration"], reverse=False)
    if len(test_list) > 0:
        test_list.sort(key=lambda x: x["duration"], reverse=False)
    # 数据写入到文件中
    f_train = open(train_manifest_path, 'w', encoding='utf-8')
    f_test = open(test_manifest_path, 'w', encoding='utf-8')
    for line in test_list:
        line = json.dumps(line, ensure_ascii=False)
        f_test.write('{}\n'.format(line))
    interval = 500
    if len(data_list) / 500 > max_test_manifest:
        interval = len(data_list) // max_test_manifest
    for i, line in enumerate(data_list):
        line = json.dumps(line, ensure_ascii=False)
        if i % interval == 0:
            if len(test_list) == 0:
                f_test.write('{}\n'.format(line))
            else:
                f_train.write('{}\n'.format(line))
        else:
            f_train.write('{}\n'.format(line))
    f_train.close()
    f_test.close()
    logger.info("完成生成数据列表，数据集总长度为{:.2f}小时！".format(sum(durations) / 3600.))


# 将多段短音频合并为长音频，减少文件数量
def merge_audio(annotation_path, save_audio_path, max_duration=600, target_sr=16000):
    # 合并数据列表
    train_list_path = os.path.join(annotation_path, 'merge_audio.json')
    if os.path.exists(train_list_path):
        f_ann = open(train_list_path, 'a', encoding='utf-8')
    else:
        f_ann = open(train_list_path, 'w', encoding='utf-8')
    wav, duration_sum, list_data = [], [], []
    for annotation_text in os.listdir(annotation_path):
        annotation_text_path = os.path.join(annotation_path, annotation_text)
        if os.path.splitext(annotation_text_path)[-1] != '.txt': continue
        if os.path.splitext(annotation_text_path)[-1] == 'test.txt': continue
        with open(annotation_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            audio_path, text = line.replace('\n', '').replace('\r', '').split('\t')
            if not os.path.exists(audio_path): continue
            audio_segment = AudioSegment.from_file(audio_path)
            # 重采样
            if audio_segment.sample_rate != target_sr:
                audio_segment.resample(target_sample_rate=target_sr)
            # 合并数据
            duration_sum.append(audio_segment.duration)
            wav.append(audio_segment.samples)
            # 列表数据
            list_d = dict(text=text,
                          duration=round(audio_segment.duration, 5),
                          start_time=round(sum(duration_sum) - audio_segment.duration, 5),
                          end_time=round(sum(duration_sum), 5))
            list_data.append(list_d)
            # 删除已处理的音频文件
            os.remove(audio_path)
            # 保存合并音频文件
            if sum(duration_sum) >= max_duration:
                # 保存路径
                dir_num = len(os.listdir(save_audio_path)) - 1 if os.path.exists(save_audio_path) else 0
                save_dir = os.path.join(save_audio_path, str(dir_num))
                os.makedirs(save_dir, exist_ok=True)
                if len(os.listdir(save_dir)) >= 1000:
                    save_dir = os.path.join(save_audio_path, str(dir_num + 1))
                    os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{int(time.time() * 1000)}.wav').replace('\\', '/')
                data = np.concatenate(wav)
                soundfile.write(save_path, data=data, samplerate=target_sr, format='WAV')
                # 写入到列表文件
                for list_d in list_data:
                    list_d['audio_filepath'] = save_path
                    f_ann.write('{}\n'.format(json.dumps(list_d)))
                f_ann.flush()
                wav, duration_sum, list_data = [], [], []
        # 删除已处理的标注文件
        os.remove(annotation_text_path)
    f_ann.close()


# 改变音频采样率
def change_rate(audio_path, target_sr=16000):
    is_change = False
    audio_segment = AudioSegment.from_file(audio_path)
    # 重采样
    if audio_segment.sample_rate != target_sr:
        audio_segment.resample(target_sample_rate=target_sr)
        is_change = True
    if is_change:
        audio_segment.to_wav_file(audio_path)


# 过滤非法的字符
def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str + in_str[i]
    return out_str


# 判断是否为中文字符或者英文字符
def is_uchar(uchar):
    if uchar == ' ': return True
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    if u'\u0030' <= uchar <= u'\u0039':
        return False
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    if uchar in ["'"]:
        return True
    if uchar in ['-', ',', '.', '>', '?']:
        return False
    return False


# 生成噪声的数据列表
def create_noise(path, noise_manifest_path, is_change_frame_rate=True, target_sr=16000):
    if not os.path.exists(path):
        logger.info('噪声音频文件为空，已跳过！')
        return
    json_lines = []
    logger.info('正在创建噪声数据列表，路径：%s，请等待 ...' % path)
    for file in tqdm(os.listdir(path)):
        audio_path = os.path.join(path, file)
        try:
            # 噪声的标签可以标记为空
            text = ""
            # 重新调整音频格式并保存
            if is_change_frame_rate:
                change_rate(audio_path, target_sr=target_sr)
            f_wave = wave.open(audio_path, "rb")
            duration = f_wave.getnframes() / f_wave.getframerate()
            json_lines.append(
                json.dumps(
                    {
                        'audio_filepath': audio_path.replace('\\', '/'),
                        'duration': duration,
                        'text': text
                    },
                    ensure_ascii=False))
        except Exception as e:
            continue
    with open(noise_manifest_path, 'w', encoding='utf-8') as f_noise:
        for json_line in json_lines:
            f_noise.write(json_line + '\n')


# 获取全部字符
def count_manifest(counter, manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            for char in line["text"].replace('\n', ''):
                counter.update(char)
    if os.path.exists(manifest_path.replace('train', 'test')):
        with open(manifest_path.replace('train', 'test'), 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                for char in line["text"].replace('\n', ''):
                    counter.update(char)


def create_manifest_binary(train_manifest_path, test_manifest_path):
    """
    生成数据列表的二进制文件
    :param train_manifest_path: 训练列表的路径
    :param test_manifest_path: 测试列表的路径
    :return:
    """
    for manifest_path in [train_manifest_path, test_manifest_path]:
        dataset_writer = DatasetWriter(manifest_path)
        with open(train_manifest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = line.replace('\n', '')
            dataset_writer.add_data(line)
        dataset_writer.close()


def opus_to_wav(opus_path, save_wav_path, rate=16000):
    audio_segment = AudioSegment.from_file(opus_path)
    audio_segment.resample(rate)
    audio_segment.to_wav_file(save_wav_path)
