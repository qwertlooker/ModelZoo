# speechscorer 推理指导

## 概述

speechscorer 使用语音模型输出分布的 entropy/perplexity 对 utterance 评分。当前交付保留两条明确分离的路径：

- `whisper-clm`：上游默认入口，用于轻量功能 smoke；
- `hubert-mlm`：上游 README 图和 SpeechOcean762 notebook 实际使用的公开演示路径，是原始结果对齐主线。

版本边界：

```text
upstream=https://github.com/yaya-sy/speechscorer.git
commit=bbe0be772b37f472994d5a97f809214fd67a2c8e
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

| 配套 | 版本/要求 |
|---|---|
| 硬件 | 支持目标 CANN 的 Atlas 推理服务器 |
| CANN、驱动、固件 | 参考适配为 CANN 8.2.RC1，实际按 torch-npu 配套表选择 |
| Python | 3.10 |
| PyTorch / torchaudio / torch-npu | 2.5.1 |
| Transformers | 4.30.0 |
| fairseq | 0.12.2，仅 `hubert-mlm` 原始公开路径需要 |

`hubert-mlm` 的 fairseq 依赖较旧。必须在目标 Python/PyTorch 组合中实际完成导入和端到端验证；安装失败时不能改用 `whisper-clm` 冒充原始公开路径。

## 文件目录

```text
speechscorer
├── patches/0001-add-explicit-device-selection.patch
├── prepare_eval_data.py
├── evaluate_results.py
├── requirements.txt
├── README_INFERENCE.md
├── NPU_ADAPTATION.md
└── ACCEPTANCE_PLAN.md
```

执行时另外创建 `source/`、未应用 patch 的 `upstream-original/` 和应用 patch 的
`upstream-npu/`，避免覆盖原始 baseline。

## 快速上手

### 获取源码和安装依赖

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

原始与 patch 后 CPU baseline 分别使用独立环境：

```bash
python3.10 -m venv .venv-cpu-original
source .venv-cpu-original/bin/activate
python -m pip install --upgrade "pip<24.1"
python -m pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e upstream-original --no-deps
deactivate

python3.10 -m venv .venv-cpu-patched
source .venv-cpu-patched/bin/activate
python -m pip install --upgrade "pip<24.1"
python -m pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e upstream-npu --no-deps
deactivate
```

NPU 环境不得安装 CPU 索引 wheel：

```bash
python3.10 -m venv .venv-npu
source .venv-npu/bin/activate
python -m pip install --upgrade "pip<24.1"
python -m pip install torch==2.5.1 torchaudio==2.5.1 torch-npu==2.5.1 \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
python -m pip install -r requirements.txt
python -m pip install -e upstream-npu --no-deps
```

NPU 环境安装后执行导入门禁：

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

原始 SpeechOcean 图对应 HuBERT-Large fairseq checkpoint：

```bash
mkdir -p weights
wget -O weights/hubert_large_ll60k.pt \
  https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt

huggingface-cli download facebook/hubert-large-ls960-ft \
  --revision ece5fabbf034c1073acae96d5401b25be96709d8 \
  --local-dir weights/hubert-large-ls960-ft
sha256sum weights/hubert_large_ll60k.pt
```

该 checkpoint 下载约 3.8 GB；正式报告必须记录实际 SHA256。

可选的 `whisper-clm` smoke 权重：

```bash
huggingface-cli download openai/whisper-base.en \
  --revision 911407f4214e0e1d82085af863093ec0b66f9cd6 \
  --local-dir weights/whisper-base.en
```

### 准备数据集

```bash
git clone https://github.com/jimbozhang/speechocean762.git \
  eval_data/speechocean762
git -C eval_data/speechocean762 checkout \
  613968e3b0b789fc33936fb5eba1973176ba7d11

python prepare_eval_data.py \
  --dataset_dir eval_data/speechocean762 \
  --output_dir eval_data/speechocean762-test
```

生成：

```text
eval_data/speechocean762-test/wavs/
eval_data/speechocean762-test/manifest.jsonl
eval_data/speechocean762-test/manifest.jsonl.meta.json
```

### 模型推理

未应用 patch 的原始 CPU baseline：

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

应用 patch 后的同设备 CPU 回归：

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

先验证 patch 前后 CPU：

```bash
source .venv-cpu-patched/bin/activate
python evaluate_results.py \
  --results results/patched_cpu.csv \
  --baseline results/original_cpu.csv \
  --manifest eval_data/speechocean762-test/manifest.jsonl \
  --output results/original_vs_patched_cpu.json
deactivate
```

NPU candidate：

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

计算人工分数相关性并比较后端：

```bash
python evaluate_results.py \
  --results results/npu.csv \
  --baseline results/patched_cpu.csv \
  --manifest eval_data/speechocean762-test/manifest.jsonl \
  --output results/cpu_vs_npu.json
```

## 模型推理性能

上游未发布固定硬件上的 latency/RTF/吞吐数值。正式报告记录加载时间、音频总时长、RTF、batch 1/4/8/16、峰值 HBM/RSS；性能不能替代 SpeechOcean 数值和相关性对齐。

全量 L2 三组推理命令用 `/usr/bin/time -v -o` 包裹，例如 NPU：

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

由 manifest 的 `total_audio_seconds` 计算 RTF，并记录 samples/s、峰值 RSS/HBM。
原始 CPU/CUDA和 patch 后同设备使用相同 batch 和独立日志；正式 batch 重复 3 次。

| 项目 | 官方/当前状态 |
|---|---|
| SpeechOcean 公开数值相关性 | 官方未发布，只提供 HuBERT-MLM 散点图 |
| CPU/CUDA HuBERT-MLM 全量结果 | 待 3.8 GB checkpoint 实测 |
| NPU 数值、相关性和性能 | 待验收 |

## 公网地址说明

- 官方源码：<https://github.com/yaya-sy/speechscorer>
- SpeechOcean762：<https://github.com/jimbozhang/speechocean762>
- HuBERT checkpoint：<https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt>
- 参考适配：<https://gitcode.com/Ascend-SACT/speechscorer>

详细口径见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，适配事实见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
