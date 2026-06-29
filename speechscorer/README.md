# speechscorer 推理指导

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

speechscorer 使用语音模型输出分布的 entropy/perplexity 对 utterance 评分。本文档介绍该模型基于昇腾 NPU 的推理指导。

- 版本说明：

  ```text
  upstream=https://github.com/yaya-sy/speechscorer.git
  commit_id=bbe0be772b37f472994d5a97f809214fd67a2c8e
  reference=https://gitcode.com/Ascend-SACT/speechscorer
  reference_commit=f1d6e3ee3d0f113c610a969e6fde4a29af3216d1
  hubert_processor=facebook/hubert-large-ls960-ft@ece5fabbf034c1073acae96d5401b25be96709d8
  speechocean762=jimbozhang/speechocean762@613968e3b0b789fc33936fb5eba1973176ba7d11
  ```

## 输入输出数据

- 输入：单个 WAV/FLAC/MP3，或包含这些文件的单层目录。
- 原始评测数据：SpeechOcean762 `test` split 2,500 条。
- 输出：包含 `utterance_id`、`entropy`、`perplexity` 的 CSV。
- `prepare_eval_data.py` 将 `test/wav.scp` 固定成独立音频目录和 manifest。
- `evaluate_results.py` 计算与人工 `total` 的 Pearson/Spearman，并可比较 CPU/CUDA 与 NPU。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

  | 配套 | 版本/要求 |
  |---|---|
  | 硬件 | 支持目标 CANN 的 Atlas 推理服务器 |
  | CANN、驱动、固件 | 参考适配为 CANN 8.2.RC1，实际按 torch-npu 配套表选择 |
  | Python | 3.10 |
  | PyTorch / torchaudio / torch-npu | 2.5.1 |
  | Transformers | 4.30.0 |
  | fairseq | 0.12.2，仅 `hubert-mlm` 原始公开路径需要 |

## 文件目录

```text
speechscorer
├── patches/0001-add-explicit-device-selection.patch
├── prepare_eval_data.py
├── evaluate_results.py
├── requirements.txt
└── README.md
```

> 说明：
> - `prepare_eval_data.py`：将 SpeechOcean762 的 Kaldi 格式 `test/wav.scp`
>   转换为扁平 WAV 目录 + JSONL manifest，并校验音频可读性和人工标签完整性。
>   SpeechOcean762 原始格式为嵌套的 Kaldi data dir，本脚本将其规范化为
>   `infer.py` 可直接读取的本地文件结构。
> - `evaluate_results.py`：对 scorer 输出 CSV 计算与人工 `total` 分数的
>   Pearson/Spearman 相关性，同时支持 CPU/CUDA 与 NPU 两组结果的逐 utterance
>   数值比较（entropy/perplexity 误差和排序 Spearman）。

## 快速上手

### 获取源码

1. 获取源码并应用适配补丁。

   ```bash
   git clone https://github.com/yaya-sy/speechscorer.git source
   git -C source checkout bbe0be772b37f472994d5a97f809214fd67a2c8e
   git -C source worktree add --detach ../upstream-original \
     bbe0be772b37f472994d5a97f809214fd67a2c8e
   git -C source worktree add --detach ../upstream-npu \
     bbe0be772b37f472994d5a97f809214fd67a2c8e
   git -C upstream-npu apply --check \
     ../patches/0001-add-explicit-device-selection.patch
   git -C upstream-npu apply \
     ../patches/0001-add-explicit-device-selection.patch
   ```

2. 创建原始 CPU baseline 环境（需独立 venv，CPU PyTorch 与 torch-npu 不可共存，且原始/patched 的 editable 安装需要隔离）。

   ```bash
   python3.10 -m venv .venv-cpu-original
   source .venv-cpu-original/bin/activate
   python -m pip install --upgrade "pip<24.1"
   python -m pip install torch==2.5.1 torchaudio==2.5.1 \
     --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   python -m pip install -e upstream-original --no-deps
   deactivate
   ```

3. 创建 patch 后 CPU 回归环境。

   ```bash
   python3.10 -m venv .venv-cpu-patched
   source .venv-cpu-patched/bin/activate
   python -m pip install --upgrade "pip<24.1"
   python -m pip install torch==2.5.1 torchaudio==2.5.1 \
     --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   python -m pip install -e upstream-npu --no-deps
   deactivate
   ```

4. 创建 NPU 环境（不得安装 CPU 索引 wheel）。

   ```bash
   python3.10 -m venv .venv-npu
   source .venv-npu/bin/activate
   python -m pip install --upgrade "pip<24.1"
   python -m pip install torch==2.5.1 torchaudio==2.5.1 torch-npu==2.5.1 \
     -i https://mirrors.huaweicloud.com/repository/pypi/simple
   python -m pip install -r requirements.txt
   python -m pip install -e upstream-npu --no-deps
   ```

5. NPU 环境安装后执行导入门禁。

   ```bash
   python - <<'PY'
   import fairseq
   import soundfile
   import torch
   import torch_npu
   import transformers
   from speechscorer.main import SCORERES
   print(torch.__version__, transformers.__version__, sorted(SCORERES))
   print(torch.randn(1).to("npu").device)
   PY
   ```

   CPU 环境执行同类检查时省略 `torch_npu` 和 NPU tensor。

### 准备权重

1. 下载 HuBERT-Large fairseq checkpoint（原始 SpeechOcean 图对应权重）。

   ```bash
   mkdir -p weights
   # fairseq checkpoint（约 3.8 GB）
   wget -O weights/hubert_large_ll60k.pt \
     https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt

   # HuggingFace processor（在线路径）
   huggingface-cli download facebook/hubert-large-ls960-ft \
     --revision ece5fabbf034c1073acae96d5401b25be96709d8 \
     --local-dir weights/hubert-large-ls960-ft
   sha256sum weights/hubert_large_ll60k.pt
   ```

   **离线替代**（在可联网机器预下载后传输到 NPU 服务器）：

   ```bash
   mkdir -p weights/hubert-large-ls960-ft
   curl -L --fail -o weights/hubert-large-ls960-ft/preprocessor_config.json \
     https://huggingface.co/facebook/hubert-large-ls960-ft/resolve/ece5fabbf034c1073acae96d5401b25be96709d8/preprocessor_config.json
   curl -L --fail -o weights/hubert-large-ls960-ft/config.json \
     https://huggingface.co/facebook/hubert-large-ls960-ft/resolve/ece5fabbf034c1073acae96d5401b25be96709d8/config.json
   sha256sum weights/hubert_large_ll60k.pt weights/hubert-large-ls960-ft/*.json
   ```

   该 checkpoint 下载约 3.8 GB；正式报告必须记录实际 SHA256。

2. 可选：下载 `whisper-clm` smoke 权重。

   ```bash
   huggingface-cli download openai/whisper-base.en \
     --revision 911407f4214e0e1d82085af863093ec0b66f9cd6 \
     --local-dir weights/whisper-base.en
   ```

   **离线替代**：

   ```bash
   mkdir -p weights/whisper-base.en
   curl -L --fail -o weights/whisper-base.en/config.json \
     https://huggingface.co/openai/whisper-base.en/resolve/911407f4214e0e1d82085af863093ec0b66f9cd6/config.json
   curl -L --fail -o weights/whisper-base.en/model.safetensors \
     https://huggingface.co/openai/whisper-base.en/resolve/911407f4214e0e1d82085af863093ec0b66f9cd6/model.safetensors
   curl -L --fail -o weights/whisper-base.en/tokenizer.json \
     https://huggingface.co/openai/whisper-base.en/resolve/911407f4214e0e1d82085af863093ec0b66f9cd6/tokenizer.json
   sha256sum weights/whisper-base.en/*
   ```

### 准备数据集

1. 下载 SpeechOcean762 并生成 manifest。

   ```bash
   git clone https://github.com/jimbozhang/speechocean762.git \
     eval_data/speechocean762
   git -C eval_data/speechocean762 checkout \
     613968e3b0b789fc33936fb5eba1973176ba7d11

   python prepare_eval_data.py \
     --dataset_dir eval_data/speechocean762 \
     --output_dir eval_data/speechocean762-test
   ```

   参数说明：

   - `dataset_dir`：SpeechOcean762 仓库目录。
   - `output_dir`：生成的音频目录、manifest 和 meta 文件保存目录。

   生成的文件：

   ```text
   eval_data/speechocean762-test/wavs/
   eval_data/speechocean762-test/manifest.jsonl
   eval_data/speechocean762-test/manifest.jsonl.meta.json
   ```

### 模型推理

1. 执行未应用 patch 的原始 CPU baseline 推理。

   ```bash
   source .venv-cpu-original/bin/activate
   cd upstream-original
   speechscore \
     --audio ../eval_data/speechocean762-test/wavs \
     --model_checkpoint ../weights/hubert_large_ll60k.pt \
     --processor_checkpoint ../weights/hubert-large-ls960-ft \
     --scorer hubert-mlm \
     --padding longest \
     --batch_size 8
   cd ..
   mkdir -p results
   cp upstream-original/results/results.csv results/original_cpu.csv
   deactivate
   ```

2. 执行应用 patch 后的同设备 CPU 回归推理。

   ```bash
   source .venv-cpu-patched/bin/activate
   cd upstream-npu
   speechscore \
     --audio ../eval_data/speechocean762-test/wavs \
     --model_checkpoint ../weights/hubert_large_ll60k.pt \
     --processor_checkpoint ../weights/hubert-large-ls960-ft \
     --scorer hubert-mlm \
     --padding longest \
     --device cpu \
     --batch_size 8 \
     --output_csv ../results/patched_cpu.csv
   cd ..
   deactivate
   ```

3. 验证 patch 前后 CPU 一致性。

   ```bash
   source .venv-cpu-patched/bin/activate
   python evaluate_results.py \
     --results results/patched_cpu.csv \
     --baseline results/original_cpu.csv \
     --manifest eval_data/speechocean762-test/manifest.jsonl \
     --output results/original_vs_patched_cpu.json
   deactivate
   ```

4. 执行 NPU 推理。

   ```bash
   source .venv-npu/bin/activate
   cd upstream-npu
   speechscore \
     --audio ../eval_data/speechocean762-test/wavs \
     --model_checkpoint ../weights/hubert_large_ll60k.pt \
     --processor_checkpoint ../weights/hubert-large-ls960-ft \
     --scorer hubert-mlm \
     --padding longest \
     --device npu \
     --batch_size 8 \
     --output_csv ../results/npu.csv
   cd ..
   ```

   参数说明：

   - `audio`：输入音频文件或目录。
   - `model_checkpoint`：HuBERT fairseq checkpoint 路径。
   - `processor_checkpoint`：HuBERT processor 目录。
   - `scorer`：评分器，原始公开路径使用 `hubert-mlm`。
   - `padding`：填充策略，原始路径使用 `longest`。
   - `device`：推理设备，支持 `npu`、`cpu`、`cuda`。
   - `batch_size`：批大小。
   - `output_csv`：输出 CSV 路径。

5. 计算人工分数相关性并比较后端。

   ```bash
   python evaluate_results.py \
     --results results/npu.csv \
     --baseline results/patched_cpu.csv \
     --manifest eval_data/speechocean762-test/manifest.jsonl \
     --output results/cpu_vs_npu.json
   ```

## 模型推理性能

1. 性能测试（以 NPU 为例，用 `/usr/bin/time -v` 包裹全量推理）。

   ```bash
   mkdir -p results
   cd upstream-npu
   /usr/bin/time -v -o ../results/npu.time.txt speechscore \
     --audio ../eval_data/speechocean762-test/wavs \
     --model_checkpoint ../weights/hubert_large_ll60k.pt \
     --processor_checkpoint ../weights/hubert-large-ls960-ft \
     --scorer hubert-mlm \
     --padding longest \
     --device npu \
     --batch_size 8 \
     --output_csv ../results/npu_perf.csv
   ```

   原始 CPU/CUDA 和 patch 后同设备使用相同 batch 和独立日志。

## 适配与精度口径

### 适配事实

正式 patch 把设备选择改为显式 `--device npu/cpu/cuda`，默认 NPU，并增加 `--output_csv`；NPU 路径直接导入 `torch_npu`，缺依赖时暴露错误。模型和输入仍通过 upstream `.to(self.device)` 迁移，CPU/CUDA 算法不变，不需要修改 site-packages。两条评分路径必须分离：`whisper-clm` 仅作 smoke，`hubert-mlm` 是原始公开主线，不得混写。fairseq 依赖较旧，必须实际导入验证，不能仅靠 `pip install` 成功判断可用。

### 官方指标边界

upstream 未发布可对齐的 Pearson/Spearman 数值，只提供人工总分与模型分数的散点图。因此正式迁移只比较 CPU/CUDA 与 NPU 的逐样本分数一致性，不与 upstream 散点图直接对齐。

### 迁移对齐门禁

使用同一 checkpoint、manifest、batch 和 padding，比较 CPU/CUDA 与 NPU 的逐样本分数：

- entropy 逐样本最大绝对误差 `<= 1e-4`、平均绝对误差 `<= 1e-5`；
- perplexity 逐样本相对误差 `<= 1e-4`；
- 逐样本 Spearman 相关 `>= 0.9999`；
- 与人工 total 的相关性差 `<= 0.001`。

这些阈值是迁移初始门禁，不是 upstream 官方容差。

### 性能评测方法

记录 batch 1/4/8/16 的 RTF、samples、`/usr/bin/time -v` 资源占用和峰值 RSS/HBM，正式轮次至少重复 3 次并报告 RTF 中位数。upstream 未发布与当前 Atlas 路径直接可比的硬件性能数值，因此报告 NPU/CPU RTF 比值。

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 模型权重 | HuBERT-Large fairseq checkpoint | https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt |
| 开源代码仓 | speechscorer 源码 | https://github.com/yaya-sy/speechscorer |
| 数据集 | SpeechOcean762 | https://github.com/jimbozhang/speechocean762 |
| 参考适配 | Ascend-SACT 参考仓 | https://gitcode.com/Ascend-SACT/speechscorer |
