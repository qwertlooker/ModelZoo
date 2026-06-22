# DiariZen 推理指导

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

DiariZen 是 BUT-FIT 发布的说话人日志分轨模型，基于 WavLM 分割网络与 WeSpeaker embedding，输出 RTTM。本文档介绍该模型基于昇腾 NPU 的推理指导。

> 说明：本文档适配对象为 `BUT-FIT/diarizen-wavlm-large-s80-md` checkpoint，不包含 `large-s80-md-v2`、base 或 pruning checkpoint。

- 版本说明：

  ```text
  url=https://github.com/BUTSpeechFIT/DiariZen
  commit_id=a60b18151dbbe246e4199d8ef5cd2ece3872ea94
  model_name=DiariZen
  model=BUT-FIT/diarizen-wavlm-large-s80-md@a9b1b0e7974d96dcfd63af417e9da7ad8714040f
  embedding=pyannote/wespeaker-voxceleb-resnet34-LM@837717ddb9ff5507820346191109dc79c958d614
  dscore=nryant/dscore@e02f949ac6592279300a2c33d03daf9e0c12fd27
  reference=Ascend-SACT/BUTSpeechFIT-DiariZen@7961b5ab79b1232b9da367f14f8cd4f592694465
  ```

## 输入输出数据

- 输入数据

  支持一个或多个音频文件，或包含 `id`/`audio_path` 字段的 JSONL manifest。

- 输出数据

  输出为每个 session 一个 RTTM 文件，以及 `run.meta.json`。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

  | 配套 | 版本/要求 |
  |---|---|
  | 硬件 | 支持目标 CANN 的 Atlas 推理服务器 |
  | CANN、驱动、固件 | CANN 8.2.0 及其配套驱动/固件 |
  | Python | 3.10 |
  | PyTorch / torchaudio / torch-npu | 2.5.1 |
  | ONNX Runtime | CPU：`onnxruntime==1.22.1`；NPU：`onnxruntime-cann==1.22.1` |
  | NumPy | 1.26.4 |

  说明：Atlas 800I A2 推理卡请以 CANN 版本选择实际固件与驱动版本。

## 文件目录

```text
BUTSpeechFIT-DiariZen
├── infer.py                            # 推理脚本
├── prepare_eval_data.py                # 评测数据准备脚本
├── score_diarization.py                # DER 评测脚本
├── patches/0001-add-explicit-npu-pipeline-device.patch
├── requirements.txt
├── README_INFERENCE.md                 # 推理指导文档
├── NPU_ADAPTATION.md                   # NPU 适配文档
└── ACCEPTANCE_PLAN.md                  # 验收计划
```

## 快速上手

### 获取源码

1. 获取源码并应用适配补丁。

   ```bash
   git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git source
   git -C source checkout a60b18151dbbe246e4199d8ef5cd2ece3872ea94
   git -C source submodule update --init --recursive
   git -C source worktree add --detach ../upstream-original \
     a60b18151dbbe246e4199d8ef5cd2ece3872ea94
   git -C source worktree add --detach ../upstream-npu \
     a60b18151dbbe246e4199d8ef5cd2ece3872ea94
   git -C upstream-original submodule update --init --recursive
   git -C upstream-npu submodule update --init --recursive
   git -C upstream-npu apply --check \
     ../patches/0001-add-explicit-npu-pipeline-device.patch
   git -C upstream-npu apply \
     ../patches/0001-add-explicit-npu-pipeline-device.patch
   ```

2. 创建并安装 CPU 原始环境。

   ```bash
   python3.10 -m venv .venv-cpu-original
   source .venv-cpu-original/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.5.1 torchaudio==2.5.1 \
     --index-url https://download.pytorch.org/whl/cpu
   python -m pip install onnxruntime==1.22.1
   python -m pip install -r upstream-original/pyannote-audio/requirements.txt
   python -m pip install -r upstream-original/dscore/requirements.txt
   python -m pip install -r requirements.txt
   python -m pip install -e upstream-original/pyannote-audio --no-deps
   python -m pip install -e upstream-original --no-deps
   deactivate
   ```

3. 创建并安装 CPU patch 后环境。

   ```bash
   python3.10 -m venv .venv-cpu-patched
   source .venv-cpu-patched/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.5.1 torchaudio==2.5.1 \
     --index-url https://download.pytorch.org/whl/cpu
   python -m pip install onnxruntime==1.22.1
   python -m pip install -r upstream-npu/pyannote-audio/requirements.txt
   python -m pip install -r upstream-npu/dscore/requirements.txt
   python -m pip install -r requirements.txt
   python -m pip install -e upstream-npu/pyannote-audio --no-deps
   python -m pip install -e upstream-npu --no-deps
   deactivate
   ```

4. 创建并安装 NPU 环境。

   ```bash
   python3.10 -m venv .venv-npu
   source .venv-npu/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.5.1 torchaudio==2.5.1 torch-npu==2.5.1 \
     -i https://mirrors.huaweicloud.com/repository/pypi/simple
   python -m pip install onnxruntime-cann==1.22.1
   python -m pip install -r upstream-npu/pyannote-audio/requirements.txt
   python -m pip install -r upstream-npu/dscore/requirements.txt
   python -m pip install -r requirements.txt
   python -m pip install -e upstream-npu/pyannote-audio --no-deps
   python -m pip install -e upstream-npu --no-deps
   ```

5. 执行 NPU 导入门禁检查。

   ```bash
   python - <<'PY'
   import onnxruntime as ort
   import torch
   import torch_npu
   from diarizen.pipelines.inference import DiariZenPipeline
   assert "CANNExecutionProvider" in ort.get_available_providers()
   print(torch.__version__, ort.__version__, torch.randn(1).to("npu").device)
   PY
   ```

### 准备权重

1. 下载主模型和 embedding 权重。

   主模型地址：`https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md`
   embedding 地址：`https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM`

   ```bash
   huggingface-cli download BUT-FIT/diarizen-wavlm-large-s80-md \
     --revision a9b1b0e7974d96dcfd63af417e9da7ad8714040f \
     --local-dir weights/diarizen-wavlm-large-s80-md
   huggingface-cli download pyannote/wespeaker-voxceleb-resnet34-LM \
     pytorch_model.bin \
     --revision 837717ddb9ff5507820346191109dc79c958d614 \
     --local-dir weights/wespeaker-voxceleb-resnet34-LM

   find weights -type f -print0 | sort -z | xargs -0 sha256sum
   ```

   主模型目录必须包含 `config.toml`、`pytorch_model.bin` 和 `plda/`。

### 准备数据集

1. 准备功能验证样例。

   ```bash
   mkdir -p eval_data/functional
   printf 'EN2002a %s\n' "$PWD/source/example/EN2002a_30s.wav" \
     > eval_data/functional/wav.scp
   python prepare_eval_data.py \
     --wav_scp eval_data/functional/wav.scp \
     --output_manifest eval_data/functional/manifest.jsonl \
     --dataset upstream-example \
     --split functional
   ```

   参数说明：

   - `wav_scp`：`<session_id> <audio_path>` 格式的音频列表文件。
   - `output_manifest`：生成的 JSONL manifest 路径。
   - `dataset`：数据集名称标签。
   - `split`：数据集 split 标签。

2. 准备 L2 正式评测数据。

   ```bash
   python prepare_eval_data.py \
     --wav_scp eval_data/ami/wav.scp \
     --reference_rttm eval_data/ami/reference.rttm \
     --uem eval_data/ami/all.uem \
     --output_manifest eval_data/ami/manifest.jsonl \
     --dataset AMI \
     --split SDM-eval
   ```

   参数说明：

   - `reference_rttm`：reference RTTM 文件路径。
   - `uem`：UEM 文件路径，可选。

### 模型推理

1. 执行未应用 patch 的原始 CPU baseline 推理。

   ```bash
   source .venv-cpu-original/bin/activate
   mkdir -p results/original_cpu
   python - <<'PY'
   from pathlib import Path
   from diarizen.pipelines.inference import DiariZenPipeline

   pipeline = DiariZenPipeline(
       diarizen_hub=Path("weights/diarizen-wavlm-large-s80-md"),
       embedding_model="weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin",
       rttm_out_dir="results/original_cpu",
   )
   pipeline("source/example/EN2002a_30s.wav", sess_name="EN2002a")
   PY
   deactivate
   ```

2. 执行应用 patch 后的 CPU 回归推理。

   ```bash
   source .venv-cpu-patched/bin/activate
   python infer.py \
     --manifest eval_data/functional/manifest.jsonl \
     --model_dir weights/diarizen-wavlm-large-s80-md \
     --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
     --device cpu \
     --output_dir results/patched_cpu
   deactivate
   ```

3. 执行 NPU 推理。

   ```bash
   source .venv-npu/bin/activate
   python infer.py \
     --manifest eval_data/functional/manifest.jsonl \
     --model_dir weights/diarizen-wavlm-large-s80-md \
     --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
     --device npu \
     --output_dir results/npu
   ```

   参数说明：

   - `manifest`：JSONL manifest 路径。
   - `model_dir`：主模型权重目录。
   - `embedding_model`：WeSpeaker embedding 权重文件路径。
   - `device`：推理设备，支持 `npu`、`cpu`。
   - `output_dir`：RTTM 和 `run.meta.json` 输出目录。

4. 执行正式 DER 评测。

   ```bash
   python score_diarization.py \
     --dscore_dir source/dscore \
     --reference_rttm eval_data/ami/reference.rttm \
     --system_rttm results/npu/*.rttm \
     --uem eval_data/ami/all.uem \
     --collar 0.0 \
     --output results/npu/der.txt
   ```

   参数说明：

   - `dscore_dir`：vendored dscore 目录路径。
   - `reference_rttm`：reference RTTM 文件路径。
   - `system_rttm`：待评测的 system RTTM 文件，支持 glob。
   - `uem`：UEM 文件路径。
   - `collar`：DER collar（秒），官方口径使用 `0.0`。
   - `output`：DER 报告输出路径。

## 模型推理性能

NPU L2 性能测试示例：

```bash
mkdir -p results/npu
/usr/bin/time -v -o results/npu/l2.time.txt python infer.py \
  --manifest eval_data/ami/manifest.jsonl \
  --model_dir weights/diarizen-wavlm-large-s80-md \
  --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
  --device npu \
  --output_dir results/npu
```

`results/npu/run.meta.json` 提供 elapsed/RTF/provider。性能数据待正式验收，详见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 开源代码仓 | DiariZen 官方源码 | https://github.com/BUTSpeechFIT/DiariZen |
| 模型权重 | BUT-FIT/diarizen-wavlm-large-s80-md | https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md |
| 模型权重 | pyannote/wespeaker-voxceleb-resnet34-LM | https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM |
| 参考适配 | Ascend-SACT/BUTSpeechFIT-DiariZen | https://gitcode.com/Ascend-SACT/BUTSpeechFIT-DiariZen |

DER 口径见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，设备边界见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
