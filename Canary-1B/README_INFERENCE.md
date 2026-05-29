# Canary-1B 推理指导

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [文件目录](#文件目录)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备权重](#准备权重)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能](#模型推理性能)
- [公网地址说明](#公网地址说明)

## 概述

Canary-1B 是 NVIDIA 发布的多语言多任务语音模型，采用 FastConformer 编码器和 Transformer 解码器。该模型支持英语、德语、西班牙语、法语 4 种语言的自动语音识别（ASR），并支持英语与德语/西班牙语/法语之间的语音到文本翻译（AST），输出可选择带或不带标点和大小写（PnC）。本文档介绍该模型基于昇腾 NPU 的推理指导。

> 说明：本文档适配对象为 Hugging Face `nvidia/canary-1b` 仓库中的原始 `canary-1b.nemo` 权重，不包含 `canary-1b-flash`、`canary-1b-v2`。

- 版本说明：

  ```text
  url=https://github.com/NVIDIA-NeMo/NeMo.git
  branch=main
  commit_id=44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe
  model_name=Canary-1B
  ```

## 输入输出数据

- 输入数据

  支持 16 kHz 单声道 wav/flac 等音频文件。推理脚本支持直接传入一个或多个本地音频文件路径；评测使用 JSONL manifest。

- 输出数据

  输出为输入音频对应的识别文本或翻译文本。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

| 配套 | 版本 |
|---|---|
| 固件与驱动 | 25.5.1+ |
| CANN | 8.5.1 |
| Python | 3.11.14 |
| PyTorch / torch_npu | 2.9.0 |
| torchaudio | 2.9.0 |

说明：Atlas 800I A2 推理卡请以 CANN 版本选择实际固件与驱动版本。

## 文件目录

```text
Canary-1B
├── README_INFERENCE.md                 # 推理指导文档
├── README.md                           # 模型适配说明
├── infer.py                            # 单条或多条音频推理脚本
├── eval_canary.py                      # 精度和性能评测脚本
├── prepare_eval_data.py                # LibriSpeech/MLS/FLEURS 评测数据准备脚本
├── weights
│   └── canary-1b
│       └── canary-1b.nemo              # 下载后的模型权重
├── test_data
│   └── demo.wav                        # 下载后的单条测试音频
├── eval_data                           # 评测数据目录，按需生成
└── eval_results                        # 推理/评测结果目录，按需生成
```

## 快速上手

### 获取源码

1. 获取适配源码。

   ```bash
   git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch
   git checkout master
   cd ACL_PyTorch/built-in/audio/Canary-1B
   ```

2. 安装依赖。

   ```bash
   pip install torch==2.9.0 torch_npu==2.9.0 torchaudio==2.9.0
   pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe"
   pip install soundfile==0.13.1 librosa sentencepiece huggingface_hub jiwer sacrebleu openai-whisper
   ```

### 准备权重

1. 下载 `canary-1b.nemo` 权重。

   原始权重地址：`https://huggingface.co/nvidia/canary-1b/resolve/main/canary-1b.nemo`

   ```bash
   mkdir -p weights/canary-1b
   wget -O weights/canary-1b/canary-1b.nemo \
     https://huggingface.co/nvidia/canary-1b/resolve/main/canary-1b.nemo
   ```

### 准备数据集

1. 准备单条通用英文语音测试文件。

   数据地址：`https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav`。

   ```bash
   mkdir -p test_data
   wget -O test_data/demo.wav \
     https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav
   ```

2. 准备 LibriSpeech test-clean 性能/精度评测数据。使用 `prepare_eval_data.py` 下载数据并生成 manifest。

   数据集地址：`https://www.openslr.org/12`。

   ```bash
   python prepare_eval_data.py \
     --task librispeech \
     --data_dir eval_data \
     --librispeech_dir eval_data/librispeech_raw
   ```

   参数说明：

   - `task`：数据准备任务类型，`librispeech` 表示只准备 LibriSpeech test-clean。
   - `data_dir`：生成的 wav、manifest 和 meta 文件保存目录。
   - `librispeech_dir`：LibriSpeech 原始压缩包和解压目录；若本地已存在则复用，否则自动下载。

   生成的 manifest 默认路径：

   ```text
   eval_data/librispeech_test_clean/manifest_asr_en.jsonl
   ```

3. 准备多语种 ASR 评测数据。使用 `prepare_eval_data.py` 下载数据并生成 manifest。

   数据集地址：`https://huggingface.co/datasets/facebook/multilingual_librispeech`。命令同时默认生成 LibriSpeech test-clean manifest，LibriSpeech 地址为 `https://www.openslr.org/12`。

   ```bash
   python prepare_eval_data.py \
     --task asr \
     --data_dir eval_data \
     --asr_parquet_dir eval_data/mls_parquet \
     --asr_configs german,spanish,french \
     --librispeech_dir eval_data/librispeech_raw \
     --asr_minutes 0
   ```

   参数说明：

   - `task`：数据准备任务类型，`asr` 表示准备 MLS 多语种 ASR 数据，并默认包含 LibriSpeech test-clean。
   - `data_dir`：生成的 wav、manifest 和 meta 文件保存目录。
   - `asr_parquet_dir`：MLS parquet 文件保存或复用目录，目录结构为 `<asr_parquet_dir>/<config>/<split>-00000-of-00001.parquet`。
   - `asr_configs`：需要准备的 MLS 语言配置，多个配置以英文逗号分隔。
   - `librispeech_dir`：LibriSpeech 原始压缩包和解压目录。
   - `asr_minutes`：每个 ASR 数据集的音频时长上限，`0` 表示使用完整 split。

   生成的 ASR manifest 默认路径：

   ```text
   eval_data/librispeech_test_clean/manifest_asr_en.jsonl
   eval_data/mls_test_german/manifest_asr_de.jsonl
   eval_data/mls_test_spanish/manifest_asr_es.jsonl
   eval_data/mls_test_french/manifest_asr_fr.jsonl
   ```

4. 准备多语种 AST 评测数据。使用 `prepare_eval_data.py` 下载数据并生成 manifest。

   数据集地址：`https://huggingface.co/datasets/google/fleurs`。

   ```bash
   python prepare_eval_data.py \
     --task ast \
     --data_dir eval_data \
     --fleurs_parquet_dir eval_data/fleurs_parquet \
     --fleurs_split test \
     --fleurs_limit 0 \
     --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en
   ```

   参数说明：

   - `task`：数据准备任务类型，`ast` 表示准备 FLEURS 语音到文本翻译数据。
   - `data_dir`：生成的 wav、manifest 和 meta 文件保存目录。
   - `fleurs_parquet_dir`：FLEURS parquet 文件保存或复用目录，目录结构为 `<fleurs_parquet_dir>/<config>/<split>-00000-of-00001.parquet`。
   - `fleurs_split`：FLEURS 数据集 split，精度评测使用 `test`。
   - `fleurs_limit`：每个翻译方向的样本数上限，`0` 表示使用完整 split。
   - `ast_directions`：AST 翻译方向，多个方向以英文逗号分隔，格式为 `<source_lang>-<target_lang>`。

   生成的 AST manifest 默认路径：

   ```text
   eval_data/fleurs/en-de/manifest_ast_en_de.jsonl
   eval_data/fleurs/en-es/manifest_ast_en_es.jsonl
   eval_data/fleurs/en-fr/manifest_ast_en_fr.jsonl
   eval_data/fleurs/de-en/manifest_ast_de_en.jsonl
   eval_data/fleurs/es-en/manifest_ast_es_en.jsonl
   eval_data/fleurs/fr-en/manifest_ast_fr_en.jsonl
   ```

### 模型推理

1. 执行单条 ASR 推理。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python infer.py \
     --model weights/canary-1b/canary-1b.nemo \
     --audio test_data/demo.wav \
     --device npu \
     --task asr \
     --source_lang en \
     --target_lang en \
     --pnc yes \
     --batch_size 1 \
     --beam_size 1
   ```

   参数说明：

   - `model`：Hugging Face 模型名、本地 `.nemo` 文件路径或包含 `canary-1b.nemo` 的目录。
   - `audio`：一个或多个输入音频文件路径。
   - `device`：推理设备，支持 `npu`、`cpu`、`cuda`。
   - `task`：任务类型，ASR 使用 `asr`，AST 可使用 `ast` 或 `s2t_translation`。
   - `source_lang`：源语言，支持 `en`、`de`、`es`、`fr`。
   - `target_lang`：目标语言，支持 `en`、`de`、`es`、`fr`。
   - `pnc`：是否输出标点和大小写，支持 `yes`、`no`。
   - `batch_size`：批大小。
   - `beam_size`：解码 beam 大小；吞吐测试常用 `1`，公开精度口径常用 `5`。

2. 执行单条 AST 推理。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python infer.py \
     --model weights/canary-1b/canary-1b.nemo \
     --audio /path/to/en_audio.wav \
     --device npu \
     --task ast \
     --source_lang en \
     --target_lang de \
     --pnc yes \
     --batch_size 1 \
     --beam_size 1
   ```

3. 性能测试。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python eval_canary.py \
     --model weights/canary-1b/canary-1b.nemo \
     --device npu \
     --manifest eval_data/librispeech_test_clean/manifest_asr_en.jsonl \
     --performance_mode \
     --batch_size 64 \
     --beam_size 1 \
     --num_workers 0 \
     --output_dir eval_results/npu_librispeech_test_clean_perf_bs64_beam1
   ```

   参数说明：

   - `performance_mode`：开启性能计时路径。使用 Hugging Face Open ASR Leaderboard 的 NeMo 计时方式：按音频时长降序排序、先 warmup、正式计时使用 audio filepath list、使用 `bfloat16`，并输出 `RTFx`。
   - `num_workers`：DataLoader worker 数，默认 1。若环境 `/dev/shm` 较小，建议设置为 `0`，避免多进程 worker 触发 shared memory bus error。
   - `compute_dtype`：计算精度，支持 `auto`、`float32`、`float16`、`bfloat16`；性能模式下 NPU/CUDA 的 `auto` 默认为 `bfloat16`。
   - `decoding_strategy`：解码策略，支持 `auto`、`beam`、`greedy`、`greedy_batch`；性能模式下 `beam_size=1` 默认使用 `greedy_batch`。

4. 精度测试。

   a）执行 LibriSpeech test-clean 英文 ASR 精度评测。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python eval_canary.py \
     --model weights/canary-1b/canary-1b.nemo \
     --device npu \
     --manifest eval_data/librispeech_test_clean/manifest_asr_en.jsonl \
     --batch_size 16 \
     --beam_size 5 \
     --output_dir eval_results/npu_librispeech_test_clean_bs16_beam5
   ```

   b）执行 MLS ASR 多语种评测。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python eval_canary.py \
     --model weights/canary-1b/canary-1b.nemo \
     --device npu \
     --manifest \
       eval_data/mls_test_german/manifest_asr_de.jsonl \
       eval_data/mls_test_spanish/manifest_asr_es.jsonl \
       eval_data/mls_test_french/manifest_asr_fr.jsonl \
     --batch_size 16 \
     --beam_size 5 \
     --output_dir eval_results/npu_mls_asr_bs16_beam5
   ```

   c）执行 FLEURS AST 多方向评测。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python eval_canary.py \
     --model weights/canary-1b/canary-1b.nemo \
     --device npu \
     --manifest \
       eval_data/fleurs/en-de/manifest_ast_en_de.jsonl \
       eval_data/fleurs/en-es/manifest_ast_en_es.jsonl \
       eval_data/fleurs/en-fr/manifest_ast_en_fr.jsonl \
       eval_data/fleurs/de-en/manifest_ast_de_en.jsonl \
       eval_data/fleurs/es-en/manifest_ast_es_en.jsonl \
       eval_data/fleurs/fr-en/manifest_ast_fr_en.jsonl \
     --batch_size 16 \
     --beam_size 5 \
     --output_dir eval_results/npu_fleurs_ast_bs16_beam5
   ```

   精度结果保存在 `output_dir` 下：

   ```text
   run_env.json
   *.tsv
   *.metrics.json
   summary.metrics.json
   ```

## 模型推理性能

### 性能

| 硬件 | 数据集 | 指标 | 得分 | 竞品 |
|---|---|---|---:|---:|
| Atlas 800I A2 | LibriSpeech test-clean | RTF | 0.005652242997176402 | 0.0042491714115747425 |

### 精度

硬件：Atlas 800I A2

| 任务类型 | 语言  | 数据集                   | 指标   | 得分  | 竞品  |
| -------- | ----- | ------------------------ | ------ | ----- | ----- |
| ASR      | de    | Multilingual LibriSpeech | WER(%) | 3.83  | 4.19  |
| ASR      | es    | Multilingual LibriSpeech | WER(%) | 2.30  | 3.15  |
| ASR      | fr    | Multilingual LibriSpeech | WER(%) | 3.69  | 4.12  |
| AST      | en-de | FLEURS                   | BLEU   | 31.41 | 32.15 |
| AST      | en-es | FLEURS                   | BLEU   | 22.69 | 22.66 |
| AST      | en-fr | FLEURS                   | BLEU   | 39.84 | 40.76 |
| AST      | de-en | FLEURS                   | BLEU   | 33.50 | 33.98 |
| AST      | es-en | FLEURS                   | BLEU   | 21.78 | 21.80 |
| AST      | fr-en | FLEURS                   | BLEU   | 30.29 | 30.95 |

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 模型权重 | NVIDIA Canary-1B Hugging Face 模型仓 | https://huggingface.co/nvidia/canary-1b |
| 开源代码仓 | NVIDIA NeMo 源码 | https://github.com/NVIDIA-NeMo/NeMo |
| 公开性能参考 | Hugging Face Open ASR Leaderboard | https://github.com/huggingface/open_asr_leaderboard |
| 数据集 | LibriSpeech | https://www.openslr.org/12 |
| 数据集 | FLEURS | https://huggingface.co/datasets/google/fleurs |
| 数据集 | MLS | https://huggingface.co/datasets/facebook/multilingual_librispeech |
