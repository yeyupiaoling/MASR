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
2024-10-13 15:09:51.103 | INFO     | masr.decoders.beam_search_decoder:__init__:37 - ======================================================================
2024-10-13 15:09:51.103 | INFO     | masr.decoders.beam_search_decoder:__init__:38 - 初始化解码器...
2024-10-13 15:09:51.103 | INFO     | masr.decoders.beam_search_decoder:__init__:44 - language model: model path = lm/zh_giga.no_cna_cmn.prune01244.klm, is_character_based = True, max_order = 5, dict_size = 0
2024-10-13 15:09:51.103 | INFO     | masr.decoders.beam_search_decoder:__init__:49 - 初始化解码器完成!
2024-10-13 15:09:51.103 | INFO     | masr.decoders.beam_search_decoder:__init__:50 - ======================================================================
2024-10-13 15:09:51.103 | INFO     | masr.infer_utils.inference_predictor:__init__:38 - 已加载模型：models/ConformerModel_fbank/inference_model/inference.pth
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
2024-12-22 16:22:38.464 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 0, 'text': '破集部满山沟里面乱钻四出骂人都占', 'start': 0.0, 'end': 4.29}
2024-12-22 16:22:44.534 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 1, 'text': '你们身上就只有一辆吉普车', 'start': 4.79, 'end': 7.42}
2024-12-22 16:22:50.758 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 0, 'text': '太穷了就这一辆吉普车还绑在她李大刚的屁股上我们都是骑自行车', 'start': 7.42, 'end': 14.92}
2024-12-22 16:22:57.076 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 1, 'text': '李大刚知不知道党的组织原则谁是一把手', 'start': 15.44, 'end': 19.61}
2024-12-22 16:22:57.463 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 2, 'text': '据汉农同志讲李达康就是这么强势他说县长县长是一把手他做书记书记是一把手', 'start': 20.04, 'end': 28.13}
2024-12-22 16:22:57.613 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 1, 'text': '那他太又当了省长我还得听他的老', 'start': 30.36, 'end': 33.73}
2024-12-22 16:22:57.729 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 3, 'text': '', 'start': 33.73, 'end': 35.23}
2024-12-22 16:22:57.975 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 2, 'text': '他这个省长还当得了吗不管怎么说他老婆总是出现', 'start': 35.23, 'end': 40.48}
2024-12-22 16:22:58.187 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 1, 'text': '便是前妻欧阳军使得前期', 'start': 40.48, 'end': 45.51}
2024-12-22 16:22:58.672 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 4, 'text': '哥哥我最后我再说一句啊能不能帮我个小忙说把那人就全放了别再追究了', 'start': 50.2, 'end': 64.34}
2024-12-22 16:22:58.809 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 3, 'text': '这陈青泉跟你有商务往来', 'start': 64.62, 'end': 68.24}
2024-12-22 16:22:58.940 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 4, 'text': '我不认这人他是个有钱的人', 'start': 68.82, 'end': 71.62}
2024-12-22 16:22:59.134 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 3, 'text': '那你管这么快干嘛呀真以天下为己任', 'start': 71.62, 'end': 76.72}
2024-12-22 16:22:59.720 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 4, 'text': '以天下为己任那是你们的事儿我就是一商人在商言商你们在这个山水庄园抓走人了让我这个人很没面子所以我求求你能不能把人放', 'start': 77.49, 'end': 94.28}
2024-12-22 16:22:59.952 | INFO     | masr.predict:predict_sd_asr:324 - 说话人识别结果：{'speaker': 3, 'text': '这事你还真说完了是成为已经开过会员的决定', 'start': 107.03, 'end': 113.04}
消耗时间：25434ms, 识别结果: [{'speaker': 0, 'text': '破集部满山沟里面乱钻四出骂人都占', 'start': 0.0, 'end': 4.29}, {'speaker': 1, 'text': '你们身上就只有一辆吉普车', 'start': 4.79, 'end': 7.42}, {'speaker': 0, 'text': '太穷了就这一辆吉普车还绑在她李大刚的屁股上我们都是骑自行车', 'start': 7.42, 'end': 14.92}, {'speaker': 1, 'text': '李大刚知不知道党的组织原则谁是一把手', 'start': 15.44, 'end': 19.61}, {'speaker': 2, 'text': '据汉农同志讲李达康就是这么强势他说县长县长是一把手他做书记书记是一把手', 'start': 20.04, 'end': 28.13}, {'speaker': 1, 'text': '那他太又当了省长我还得听他的老', 'start': 30.36, 'end': 33.73}, {'speaker': 3, 'text': '', 'start': 33.73, 'end': 35.23}, {'speaker': 2, 'text': '他这个省长还当得了吗不管怎么说他老婆总是出现', 'start': 35.23, 'end': 40.48}, {'speaker': 1, 'text': '便是前妻欧阳军使得前期', 'start': 40.48, 'end': 45.51}, {'speaker': 4, 'text': '哥哥我最后我再说一句啊能不能帮我个小忙说把那人就全放了别再追究了', 'start': 50.2, 'end': 64.34}, {'speaker': 3, 'text': '这陈青泉跟你有商务往来', 'start': 64.62, 'end': 68.24}, {'speaker': 4, 'text': '我不认这人他是个有钱的人', 'start': 68.82, 'end': 71.62}, {'speaker': 3, 'text': '那你管这么快干嘛呀真以天下为己任', 'start': 71.62, 'end': 76.72}, {'speaker': 4, 'text': '以天下为己任那是你们的事儿我就是一商人在商言商你们在这个山水庄园抓走人了让我这个人很没面子所以我求求你能不能把人放', 'start': 77.49, 'end': 94.28}, {'speaker': 3, 'text': '这事你还真说完了是成为已经开过会员的决定', 'start': 107.03, 'end': 113.04}]

进程已结束，退出代码为 0
```
