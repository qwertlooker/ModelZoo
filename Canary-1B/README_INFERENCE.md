# Canary-1B 推理指导

- [概述](#概述)

- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  - [获取源码](#获取源码)
  - [准备权重](#准备权重)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)

- [模型推理性能](#模型推理性能)

- [公网地址说明](#公网地址说明)

## 概述

Canary-1B 是 NVIDIA发布的多语言多任务语音模型，采用 FastConformer 编码器和 Transformer 解码器。该模型支持英语、德语、西班牙语、法语 4 种语言的自动语音识别（ASR），并支持英语与德语/西班牙语/法语之间的语音到文本翻译（AST），输出可选择带或不带标点和大小写（PnC）。本文档介绍该模型基于昇腾 NPU 推理指导。

> 说明：本文档适配对象为 Hugging Face `nvidia/canary-1b` 仓库中的原始 `canary-1b.nemo` 权重，不包含 `canary-1b-flash`、`canary-1b-v2` 或 Riva/NIM 服务化镜像。

- 参考论文：
  - Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition
  - Attention Is All You Need

- 参考实现：

  ```text
  url=https://github.com/NVIDIA-NeMo/NeMo.git
  branch=main
  commit_id=44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe
  model_name=Canary-1B
  ```

  适配昇腾 AI 处理器的实现：

  ```text
  url=https://gitcode.com/Ascend/ModelZoo-PyTorch
  branch=master
  code_path=ACL_PyTorch/built-in/audio/Canary-1B
  ```

  通过 Git 获取对应代码的方法如下：

  ```bash
  git clone {repository_url}        # 克隆仓库代码
  cd {repository_name}              # 切换到模型代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的 commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

### 输入输出数据

- 输入数据

  支持 16 kHz 单声道 wav/flac 等音频文件。推理脚本支持直接传入一个或多个本地音频文件路径。

- 输出数据

  输出为输入音频对应的识别文本或翻译文本。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

| 配套 | 版本 | 环境准备指导 |
|---|---|---|
| 固件与驱动 | 与 CANN 配套版本 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN | 8.5.0 或与 torch-npu 匹配版本 | - |
| Python | 3.11 / 3.12 | - |
| PyTorch | 与 torch-npu 配套版本 | - |
| torch_npu | 与 CANN/PyTorch 配套版本 | - |
| NeMo | `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe` 对应源码或兼容版本 | - |
| 硬件 | Atlas 800T A2, Atlas 800I A2 | - |

安装依赖前请先完成 CANN、PyTorch、torch-npu 的安装。NeMo 作为运行依赖通过 pip 从指定 commit 安装，推理用户无需手动克隆 NeMo 源码；如需离线部署，可提前下载 NeMo 源码或 wheel 包并在离线环境安装。NeMo ASR 相关依赖可按如下方式安装：

```bash
pip install torch torch-npu
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe"
pip install soundfile librosa sentencepiece huggingface_hub jiwer sacrebleu openai-whisper
```

如使用本仓历史完整环境，可参考 `requirements.txt`；正式部署建议优先安装推理和评测所需最小依赖，避免引入无关包导致版本冲突。

## 快速上手

### 获取源码

1. 获取适配源码。

   ```bash
   git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch
   git checkout master
   cd ACL_PyTorch/built-in/audio/Canary-1B
   ```

2. 安装 NeMo 依赖，pip 会自动拉取指定 commit。

   ```bash
   pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe"
   ```

### 准备权重

1. 下载 `canary-1b.nemo` 权重。

   原始权重地址：

   ```text
   https://huggingface.co/nvidia/canary-1b/resolve/main/canary-1b.nemo
   ```

   ```bash
   ./scripts/download_weights.sh weights/canary-1b
   ```

### 准备数据集

1. 准备单条 smoke test 音频。

   ```bash
   ./scripts/download_test_data.sh test_data
   ```

   目录结构请参考：

   ```text
   Canary-1B
   ├── infer.py
   ├── scripts
   │   ├── download_test_data.sh
   │   ├── download_weights.sh
   │   ├── eval_canary.py
   │   └── prepare_eval_data.py
   ├── test_data
   │   └── dummy_1s_16k.wav
   └── weights
       └── canary-1b
           └── canary-1b.nemo
   ```

2. 准备 LibriSpeech test-clean 性能/精度评测数据。

   ```bash
   python scripts/prepare_eval_data.py \
     --task librispeech \
     --data_dir eval_data \
     --librispeech_dir eval_data/librispeech_raw
   ```

   生成的 manifest 默认路径：

   ```text
   eval_data/librispeech_test_clean/manifest_asr_en.jsonl
   ```

3. 准备多语种 ASR/AST 评测数据（可选）。

   ```bash
   python scripts/prepare_eval_data.py \
     --task all \
     --data_dir eval_data \
     --asr_parquet_dir eval_data/mls_parquet \
     --asr_configs german,spanish,french \
     --librispeech_dir eval_data/librispeech_raw \
     --fleurs_parquet_dir eval_data/fleurs_parquet \
     --asr_minutes 30 \
     --fleurs_split test \
     --fleurs_limit 50 \
     --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en
   ```

### 模型推理

1. 执行单条 ASR 推理。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python infer.py \
     --model weights/canary-1b/canary-1b.nemo \
     --audio test_data/dummy_1s_16k.wav \
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

2. 执行 AST 推理示例。

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

   性能模式用于尽量贴近 Hugging Face Open ASR Leaderboard 的 NeMo 计时方式：按音频时长降序排序、先 warmup、正式计时使用 audio filepath list、NPU/CUDA 默认使用 `bfloat16`，并输出 `RTFx=audio_seconds/elapsed_seconds`。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python scripts/eval_canary.py \
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

   - `performance_mode`：开启性能计时路径。
   - `num_workers`：DataLoader worker 数。若环境 `/dev/shm` 较小，建议设置为 `0`，避免多进程 worker 触发 shared memory bus error。
   - `compute_dtype`：计算精度，支持 `auto`、`float32`、`float16`、`bfloat16`；性能模式下 NPU/CUDA 的 `auto` 默认为 `bfloat16`。
   - `decoding_strategy`：解码策略，支持 `auto`、`beam`、`greedy`、`greedy_batch`；性能模式下 `beam_size=1` 默认使用 `greedy_batch`。

4. 精度测试。

   a）执行 LibriSpeech test-clean 英文 ASR 精度评测。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python scripts/eval_canary.py \
     --model weights/canary-1b/canary-1b.nemo \
     --device npu \
     --manifest eval_data/librispeech_test_clean/manifest_asr_en.jsonl \
     --batch_size 16 \
     --beam_size 5 \
     --output_dir eval_results/npu_librispeech_test_clean_bs16_beam5
   ```

   b）执行 MLS/FLEURS 多任务评测。

   ```bash
   ASCEND_RT_VISIBLE_DEVICES=0 python scripts/eval_canary.py \
     --model weights/canary-1b/canary-1b.nemo \
     --device npu \
     --batch_size 16 \
     --beam_size 5 \
     --output_dir eval_results/npu_all_bs16_beam5
   ```

   精度结果保存在 `output_dir` 下：

   ```text
   run_env.json
   *.tsv
   *.metrics.json
   summary.metrics.json
   ```

# 模型推理性能

## 性能

以下性能数据基于 LibriSpeech test-clean manifest，单卡 NPU，`--performance_mode --beam_size 1 --batch_size 64 --num_workers 0`，NPU/CUDA 自动使用 `bfloat16`，性能模式默认使用 `greedy_batch` 解码。RTF 越低越好，RTFx 越高越好。

| Model | Card | 数据集 | Batch Size | Beam Size | RTF | RTFx |
|---|---|---|---:|---:|---:|---:|
| Canary-1B | Ascend 910/910B | LibriSpeech test-clean | 64 | 1 | 0.009084 | 110.08 |

说明：Hugging Face Open ASR Leaderboard 中 `nvidia/canary-1b` 的公开 A100 参考 RTFx 为 235.34。该公开值来自 NVIDIA A100-SXM4-80GB / CUDA 环境，只作为量级参考，不作为 NPU 强制通过线。若使用 `batch_size=128` 获得更高吞吐，建议在报告中同时列出 `batch_size=64` 对齐口径结果和 `batch_size=128` 本机最大吞吐结果，并说明 batch、dtype、解码策略、warmup 和 `num_workers` 配置。

## 精度

| Model | 数据集 | Card | Batch Size | Beam Size | WER% |
|---|---|---|---:|---:|---:|
| Canary-1B | LibriSpeech test-clean | Ascend 910/910B | 64 | 1 | 1.4728 |

公开参考：Hugging Face Open ASR Leaderboard 中 `nvidia/canary-1b` 的 LibriSpeech clean WER 为 1.48，Average WER 为 6.50。NVIDIA 模型卡中的 ASR/AST 公开精度实验使用 `beam width=5`、`length penalty=1.0`；若要严格对齐公开模型卡精度表，请使用 `--beam_size 5` 并固定数据集、normalizer 和解码参数。

# 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 模型权重 | NVIDIA Canary-1B Hugging Face 模型仓 | https://huggingface.co/nvidia/canary-1b |
| 开源代码仓 | NVIDIA NeMo 源码 | https://github.com/NVIDIA-NeMo/NeMo |
| 公开性能参考 | Hugging Face Open ASR Leaderboard | https://github.com/huggingface/open_asr_leaderboard |
| 数据集 | LibriSpeech | https://www.openslr.org/12 |
| 数据集 | FLEURS | https://huggingface.co/datasets/google/fleurs |
| 数据集 | MLS | https://huggingface.co/datasets/facebook/multilingual_librispeech |
