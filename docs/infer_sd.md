# 说话人日志语音识别


我们可以使用这个脚本使用模型进行预测，如果如何还没导出模型，需要执行[导出模型](./export_model.md)操作把模型参数导出为预测模型，通过传递音频文件的路径进行识别，通过参数`--audio_path`指定需要预测的音频路径。支持中文数字转阿拉伯数字，将参数`--is_itn`设置为True即可。默认情况下，如果音频大于30秒，会通过VAD分割音频，再对短音频进行识别，拼接结果，最终得到长语音识别结果。
```shell script
python infer_sd_asr.py --audio_path=./dataset/test_long.wav
```

输出结果：
```
2024-10-13 15:09:51.098 | INFO     | masr.utils.utils:print_arguments:13 - ----------- 额外配置参数 -----------
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - audio_db_path: audio_db/
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - audio_path: dataset/test_long.wav
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - decoder: ctc_greedy_search
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - decoder_configs: configs/decoder.yml
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - is_itn: False
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - model_dir: models/ConformerModel_fbank/inference_model/
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - punc_device_id: -1
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - punc_model_dir: None
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - search_audio_db: False
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - speaker_num: None
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - use_gpu: True
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - use_punc: False
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - vector_configs: models/CAMPPlus_Fbank/cam++.yml
2024-10-13 15:09:51.099 | INFO     | masr.utils.utils:print_arguments:15 - vector_model_path: models/CAMPPlus_Fbank/best_model/
2024-10-13 15:09:51.100 | INFO     | masr.utils.utils:print_arguments:15 - vector_threshold: 0.6
2024-10-13 15:09:51.100 | INFO     | masr.utils.utils:print_arguments:16 - ------------------------------------------------
2024-10-13 15:09:51.100 | INFO     | masr.utils.utils:print_arguments:19 - ----------- 模型参数配置 -----------
2024-10-13 15:09:51.100 | INFO     | masr.utils.utils:print_arguments:32 - model_name: ConformerModel
2024-10-13 15:09:51.100 | INFO     | masr.utils.utils:print_arguments:23 - preprocess_conf:
2024-10-13 15:09:51.100 | INFO     | masr.utils.utils:print_arguments:30 - 	feature_method: fbank
2024-10-13 15:09:51.101 | INFO     | masr.utils.utils:print_arguments:26 - 	method_args:
2024-10-13 15:09:51.101 | INFO     | masr.utils.utils:print_arguments:28 - 		num_mel_bins: 80
2024-10-13 15:09:51.101 | INFO     | masr.utils.utils:print_arguments:32 - sample_rate: 16000
2024-10-13 15:09:51.101 | INFO     | masr.utils.utils:print_arguments:32 - streaming: True
2024-10-13 15:09:51.101 | INFO     | masr.utils.utils:print_arguments:33 - ------------------------------------------------
2024-10-13 15:09:51.102 | INFO     | masr.utils.utils:print_arguments:19 - ----------- 解码器参数配置 -----------
2024-10-13 15:09:51.102 | INFO     | masr.utils.utils:print_arguments:23 - attention_rescoring_args:
2024-10-13 15:09:51.102 | INFO     | masr.utils.utils:print_arguments:30 - 	beam_size: 10
2024-10-13 15:09:51.103 | INFO     | masr.utils.utils:print_arguments:30 - 	ctc_weight: 0.3
2024-10-13 15:09:51.103 | INFO     | masr.utils.utils:print_arguments:30 - 	reverse_weight: 1.0
2024-10-13 15:09:51.103 | INFO     | masr.utils.utils:print_arguments:23 - ctc_prefix_beam_search_args:
2024-10-13 15:09:51.103 | INFO     | masr.utils.utils:print_arguments:30 - 	beam_size: 10
2024-10-13 15:09:51.103 | INFO     | masr.utils.utils:print_arguments:33 - ------------------------------------------------
2024-10-13 15:09:51.744 | INFO     | masr.infer_utils.inference_predictor:__init__:38 - 已加载模型：models/ConformerModel_fbank/inference_model/inference.pth
2024-10-13 15:09:51.789 | INFO     | masr.predict:__init__:98 - 流式VAD模型已加载完成
2024-10-13 15:09:52.749 | INFO     | masr.predict:__init__:104 - 预测器已准备完成！
2024-10-13 15:09:53.531 | INFO     | mvector.utils.utils:print_arguments:17 - ----------- 配置文件参数 -----------
2024-10-13 15:09:53.531 | INFO     | mvector.utils.utils:print_arguments:20 - dataset_conf:
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:23 - 	dataLoader:
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		num_workers: 8
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:23 - 	dataset:
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		max_duration: 3
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		min_duration: 0.3
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		sample_rate: 16000
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		target_dB: -20
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		use_dB_normalization: True
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:27 - 	enroll_list: dataset/cn-celeb-test/enroll_list.txt
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:23 - 	eval_conf:
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		batch_size: 8
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		max_duration: 20
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:27 - 	is_use_pksampler: False
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:27 - 	sample_per_id: 4
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:23 - 	sampler:
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		batch_size: 64
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:25 - 		drop_last: True
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:27 - 	train_list: dataset/train_list.txt
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:27 - 	trials_list: dataset/cn-celeb-test/trials_list.txt
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:20 - loss_conf:
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:27 - 	loss: AAMLoss
2024-10-13 15:09:53.532 | INFO     | mvector.utils.utils:print_arguments:23 - 	loss_args:
2024-10-13 15:09:53.533 | INFO     | mvector.utils.utils:print_arguments:25 - 		easy_margin: False
2024-10-13 15:09:53.533 | INFO     | mvector.utils.utils:print_arguments:25 - 		label_smoothing: 0.0
2024-10-13 15:09:53.533 | INFO     | mvector.utils.utils:print_arguments:25 - 		margin: 0.2
2024-10-13 15:09:53.533 | INFO     | mvector.utils.utils:print_arguments:25 - 		scale: 32
2024-10-13 15:09:53.533 | INFO     | mvector.utils.utils:print_arguments:23 - 	margin_scheduler_args:
2024-10-13 15:09:53.533 | INFO     | mvector.utils.utils:print_arguments:25 - 		final_margin: 0.3
2024-10-13 15:09:53.533 | INFO     | mvector.utils.utils:print_arguments:25 - 		initial_margin: 0.0
2024-10-13 15:09:53.533 | INFO     | mvector.utils.utils:print_arguments:27 - 	use_margin_scheduler: True
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:20 - model_conf:
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:23 - 	classifier:
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:25 - 		classifier_type: Cosine
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:25 - 		num_blocks: 0
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:25 - 		num_speakers: 2796
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:27 - 	model: CAMPPlus
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:23 - 	model_args:
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:25 - 		embd_dim: 192
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:20 - optimizer_conf:
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:27 - 	optimizer: Adam
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:23 - 	optimizer_args:
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:25 - 		lr: 0.001
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:25 - 		weight_decay: 1e-05
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:27 - 	scheduler: WarmupCosineSchedulerLR
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:23 - 	scheduler_args:
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:25 - 		max_lr: 0.001
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:25 - 		min_lr: 1e-05
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:25 - 		warmup_epoch: 5
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:20 - preprocess_conf:
2024-10-13 15:09:53.534 | INFO     | mvector.utils.utils:print_arguments:27 - 	feature_method: Fbank
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:23 - 	method_args:
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:25 - 		num_mel_bins: 80
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:25 - 		sample_frequency: 16000
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:27 - 	use_hf_model: False
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:20 - train_conf:
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:27 - 	enable_amp: False
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:27 - 	log_interval: 100
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:27 - 	max_epoch: 60
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:27 - 	use_compile: False
2024-10-13 15:09:53.535 | INFO     | mvector.utils.utils:print_arguments:30 - ------------------------------------------------
2024-10-13 15:09:53.535 | INFO     | mvector.data_utils.featurizer:__init__:51 - 使用【Fbank】提取特征
2024-10-13 15:09:53.698 | INFO     | mvector.models:build_model:20 - 成功创建模型：CAMPPlus，参数为：{'embd_dim': 192}
2024-10-13 15:09:53.993 | INFO     | mvector.utils.checkpoint:load_pretrained:50 - 成功加载预训练模型：models/CAMPPlus_Fbank/best_model/model.pth
2024-10-13 15:09:53.995 | INFO     | mvector.predict:__init__:62 - 成功加载模型参数：models/CAMPPlus_Fbank/best_model/model.pth
2024-10-13 15:09:55.789 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 0, 'text': '一个一辆破及布买车沟两个钻次出满人东摘', 'start': 0.0, 'end': 4.29}
2024-10-13 15:10:01.500 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 1, 'text': '先生就只有医疗记不测', 'start': 4.79, 'end': 7.42}
2024-10-13 15:10:07.551 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 0, 'text': '太秀了邱着一辆急护货车还绑在他里大高的谷上我们都斯营救', 'start': 7.42, 'end': 14.92}
2024-10-13 15:10:13.462 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 1, 'text': '理刚刚的党个原则谁是一把水', 'start': 15.44, 'end': 19.61}
2024-10-13 15:10:13.662 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 2, 'text': '据哈东的统志捡了里达救市事作县长现场着一把手数据是发手', 'start': 20.04, 'end': 28.13}
2024-10-13 15:10:13.738 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 1, 'text': '的手长爱特平他的', 'start': 30.36, 'end': 33.73}
2024-10-13 15:10:13.771 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 3, 'text': '对反对', 'start': 33.73, 'end': 35.23}
2024-10-13 15:10:13.804 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 2, 'text': '他的生长还藏的了吧板不班怎么说他老婆', 'start': 35.23, 'end': 40.48}
2024-10-13 15:10:13.840 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 1, 'text': '来籍欧阳街值得的前妻', 'start': 40.48, 'end': 45.51}
2024-10-13 15:10:13.888 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 4, 'text': '最后再说女儿个少马受那敬专访这救', 'start': 50.2, 'end': 64.34}
2024-10-13 15:10:13.934 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 3, 'text': '陈请泉更上来', 'start': 64.62, 'end': 68.24}
2024-10-13 15:10:13.972 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 4, 'text': '我不认为整个', 'start': 68.82, 'end': 71.62}
2024-10-13 15:10:14.009 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 3, 'text': '那么你管这么快干嘛而真已产下危机人', 'start': 71.62, 'end': 76.72}
2024-10-13 15:10:14.065 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 4, 'text': '与天下为己任那的网就可以人在上眼伤你们瞒在这个山水撞元抓的了让我人很没面子我求着能姆的板人发', 'start': 77.49, 'end': 94.28}
2024-10-13 15:10:14.100 | INFO     | masr.predict:predict_sd_asr:308 - 说话人识别结果：{'speaker': 3, 'text': '在镇说娃华施已经开过会员的决定', 'start': 107.03, 'end': 113.04}
消耗时间：21352ms, 识别结果: [{'speaker': 0, 'text': '一个一辆破及布买车沟两个钻次出满人东摘', 'start': 0.0, 'end': 4.29}, {'speaker': 1, 'text': '先生就只有医疗记不测', 'start': 4.79, 'end': 7.42}, {'speaker': 0, 'text': '太秀了邱着一辆急护货车还绑在他里大高的谷上我们都斯营救', 'start': 7.42, 'end': 14.92}, {'speaker': 1, 'text': '理刚刚的党个原则谁是一把水', 'start': 15.44, 'end': 19.61}, {'speaker': 2, 'text': '据哈东的统志捡了里达救市事作县长现场着一把手数据是发手', 'start': 20.04, 'end': 28.13}, {'speaker': 1, 'text': '的手长爱特平他的', 'start': 30.36, 'end': 33.73}, {'speaker': 3, 'text': '对反对', 'start': 33.73, 'end': 35.23}, {'speaker': 2, 'text': '他的生长还藏的了吧板不班怎么说他老婆', 'start': 35.23, 'end': 40.48}, {'speaker': 1, 'text': '来籍欧阳街值得的前妻', 'start': 40.48, 'end': 45.51}, {'speaker': 4, 'text': '最后再说女儿个少马受那敬专访这救', 'start': 50.2, 'end': 64.34}, {'speaker': 3, 'text': '陈请泉更上来', 'start': 64.62, 'end': 68.24}, {'speaker': 4, 'text': '我不认为整个', 'start': 68.82, 'end': 71.62}, {'speaker': 3, 'text': '那么你管这么快干嘛而真已产下危机人', 'start': 71.62, 'end': 76.72}, {'speaker': 4, 'text': '与天下为己任那的网就可以人在上眼伤你们瞒在这个山水撞元抓的了让我人很没面子我求着能姆的板人发', 'start': 77.49, 'end': 94.28}, {'speaker': 3, 'text': '在镇说娃华施已经开过会员的决定', 'start': 107.03, 'end': 113.04}]

进程已结束，退出代码为 0
```
