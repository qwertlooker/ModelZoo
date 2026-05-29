# Canary-1B NPU 适配说明

## 1. 适配目标

将 NVIDIA NeMo Canary-1B 推理样例整理为规范的 CPU/NPU 融合脚本：

- 默认使用 `--device npu`；
- CPU 验证显式使用 `--device cpu`；
- 不使用 `auto` / `use_gpu`；
- 不在代码中写死 `npu:0` / `cuda:0`；
- 实际 NPU 卡号由环境变量控制，例如 `ASCEND_RT_VISIBLE_DEVICES=0`。

## 2. 上游与 patch

- 上游仓库：<https://github.com/NVIDIA-NeMo/NeMo.git>
- 基准 commit：`44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`
- 本次没有修改上游已有文件，因此没有 `.patch` 文件。
- `Canary-1B/infer.py` 是当前适配新增脚本，不进入 patch。

如后续上游源码需要修改，则在 `Canary-1B/upstream/` 内生成 patch：

```bash
git -C Canary-1B/upstream diff -- <upstream_existing_file> > Canary-1B/patches/0001-xxx.patch
git -C Canary-1B/upstream apply --check ../patches/0001-xxx.patch
```

## 3. 环境准备

### 3.1 CPU 验证环境

当前环境没有系统 `pip`，使用 `uv` 创建虚拟环境并安装依赖：

```bash
uv venv Canary-1B/.venv-cpu --python 3.12
uv pip install --python Canary-1B/.venv-cpu/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com \
  "torch==2.9.1" "torchaudio==2.9.1" "nemo-toolkit[asr]" \
  librosa soundfile sentencepiece huggingface_hub
```

当前验证环境版本：

```text
Python 3.12.3
torch 2.9.1+cu128
torchaudio 2.9.1+cu128
nemo-toolkit 2.7.3
```

### 3.2 NPU 推理环境

NPU 环境中请安装与 CANN 匹配的 `torch` / `torch-npu`：

```bash
pip install torch torch-npu
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe"
pip install soundfile librosa sentencepiece huggingface_hub
```

如果使用本地 NeMo 源码 `/home/Canary-1B-Adapt/NeMo`，推荐安装 ASR extra，而不是只安装某一个 requirements 文件：

```bash
cd /home/Canary-1B-Adapt/NeMo
python -m pip install -e ".[asr]"
```

依赖关系说明：

- `requirements_asr.txt` 不包含 `requirements_lightning.txt`。
- `requirements_lightning.txt` 解决 `lightning.pytorch`、Hydra、OmegaConf 等 NeMo core/lightning 依赖。
- `requirements_asr.txt` 解决 ASR 领域依赖，例如 `lhotse`、`librosa`、`soundfile`、`jiwer`、`sacrebleu`。
- 如果手工按 requirements 安装，Canary-1B ASR 推理至少需要基础依赖、common、lightning、asr 四组：

```bash
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements.txt
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements_common.txt
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements_lightning.txt
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements_asr.txt
```

NPU 环境需要特别注意 `torch` / `torch-npu` 版本与 CANN 匹配；如已安装可用的 NPU 版 PyTorch，安装 NeMo 依赖时避免被 pip 自动升级或替换。

## 4. 权重下载

官方权重：<https://huggingface.co/nvidia/canary-1b>

当前适配版本明确为原始 `nvidia/canary-1b` 的 `canary-1b.nemo` 权重；不是 `nvidia/canary-1b-flash`，也不是 `nvidia/canary-1b-v2`。本地验证权重 SHA256：`b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a`。

可通过 `huggingface_hub.snapshot_download` 下载 `canary-1b.nemo`，按需设置 Gitee HF endpoint：<https://hf-api.gitee.com>。

```python
import os
os.environ["HF_HOME"] = "~/.cache/gitee-ai"
os.environ["HF_ENDPOINT"] = "https://hf-api.gitee.com"

from huggingface_hub import snapshot_download
snapshot_download("nvidia/canary-1b", allow_patterns=["canary-1b.nemo"], local_dir="Canary-1B/weights/canary-1b")
```

下载后校验 SHA256：

```text
b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a
```

推理时指定本地权重：

```bash
--model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo
```

本次在 ModelScope 以 `canary-1b` / `nvidia/canary-1b` 检索未找到同名公开模型。

## 5. 官方参考指标与评测口径

官方/公开指标是 NPU 适配的重要参考，但不是简单的单值通过线。适配报告必须同时记录：来源链接、数据集、split、normalizer/后处理、decode 参数、测试硬件和 batch 策略。

来源：

- NVIDIA Canary-1B model card：<https://huggingface.co/nvidia/canary-1b>
- Hugging Face Open ASR Leaderboard：<https://hf-audio-open-asr-leaderboard.hf.space/>
- Open ASR Leaderboard 代码/说明：<https://github.com/huggingface/open_asr_leaderboard>

关键口径：

- NVIDIA model card 的 ASR/AST 精度结果使用 `beam width=5`、`length penalty=1.0`。
- ASR 使用 WER，并用官方 `openai-whisper` 的 `whisper.normalizers.EnglishTextNormalizer` 处理参考文本和预测文本；仅安装 `whisper_normalizer` 不算满足官方路径。
- 适配/评测脚本遵守根目录《模型NPU 适配标准流程.md》的项目级脚本严格失败原则：必需依赖统一前置 import，缺依赖或 NeMo 官方预期字段缺失时直接报错，不使用宽泛 `try/except`、`hasattr/getattr` 静默降级。
- AST 使用 BLEU，并使用数据集原始标点和大小写。
- 原始 `nvidia/canary-1b` model card 未发布单独的硬件延迟/吞吐表；公开速度参考使用 Open ASR Leaderboard 的 RTFx。
- Open ASR Leaderboard 中 `nvidia/canary-1b` 的公开参考：Average WER `6.50`，RTFx `235.34`。该榜单评测硬件为 NVIDIA A100-SXM4-80GB GPU；该值只作为公开 GPU 量级参考，不作为 NPU 通过线。

完整官方精度表和性能参考表见 `README.md` 与 `ACCEPTANCE_PLAN.md`。NPU 验收应另外记录本机 `elapsed_seconds`、`rtf`、`RTFx=audio_seconds/elapsed_seconds`、最大可用 `batch_size`、`beam_size`、峰值 HBM/RSS，并优先判断同 checkpoint、同数据、同脚本下 NPU 相对 CPU/CUDA 是否退化。

## 6. 测试数据准备

生成最小 smoke-test wav：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import soundfile as sf
out = Path("Canary-1B/test_data/dummy_1s_16k.wav")
out.parent.mkdir(parents=True, exist_ok=True)
sr = 16000
t = np.arange(sr, dtype=np.float32) / sr
sf.write(out, 0.1 * np.sin(2 * np.pi * 440 * t), sr)
print(out)
PY
```

输出：

```text
Canary-1B/test_data/dummy_1s_16k.wav
```

该样例仅用于链路验证，不用于准确率评估。若要验证识别质量，请使用下面的 MLS / LibriSpeech / FLEURS manifest 流程。

### 6.1 MLS / LibriSpeech / FLEURS 评测数据准备

`prepare_eval_data.py` 已按在线/离线混合要求实现：

- `--asr_parquet_dir Canary-1B/eval_data/mls_parquet`：复用或下载 `facebook/multilingual_librispeech` 的 `german/spanish/french` test parquet。
- `--librispeech_dir Canary-1B/eval_data/librispeech_raw`：保留并复用/下载 OpenSLR LibriSpeech `test-clean`，用于性能测试。
- `--fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet`：复用或下载 FLEURS 各语言 `test-00000-of-00001.parquet`。
- `--offline`：禁止联网，缺失文件立即报具体路径。
- MLS/LibriSpeech/FLEURS 使用 `Audio(decode=False)` + `soundfile`，不依赖 `torchcodec`。

推荐最小验收数据：

```bash
python Canary-1B/prepare_eval_data.py \
  --task all \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --asr_minutes 30 \
  --fleurs_split test \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en
```

离线复用同一批文件：

```bash
python Canary-1B/prepare_eval_data.py \
  --task all \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --offline \
  --asr_minutes 30 \
  --fleurs_split test \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en
```

输出 manifest 和 metadata；CPU/NPU 评测必须复用同一份 manifest。详细手动下载命令见 `EVAL_FLEURS_MLS.md`。

## 7. 推理脚本用法

### CPU ASR 验证

```bash
Canary-1B/.venv-cpu/bin/python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --device cpu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --batch_size 1
```

### NPU ASR

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --device npu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --pnc yes
```

### 语音翻译 AST

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio /path/to/en.wav \
  --device npu \
  --task ast \
  --source_lang en \
  --target_lang de \
  --pnc yes
```

## 8. 上游更新处理

上游更新时必须重新执行：

```bash
git -C Canary-1B/upstream fetch origin main
git -C Canary-1B/upstream rev-parse origin/main
grep -RIn "cuda\|gpu\|npu\|to(device)\|torch.load\|nccl" Canary-1B/upstream/nemo/collections/asr
```

重点检查：

- `nemo/collections/asr/models/aed_multitask_models.py`
- `nemo/collections/asr/parts/mixins/transcription.py`
- `nemo/collections/asr/parts/preprocessing/`
- `examples/asr/transcribe_speech.py`

如新增硬编码 CUDA 节点，按标准流程生成 patch 并补充验证记录。
