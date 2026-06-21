# Canary-1B NPU 适配文档

本文保留 Canary-1B NPU 适配过程中的版本边界、上游代码分析、设备适配、环境/权重准备、推理命令和实际验证记录。

文档分工：

- `README_INFERENCE.md`：面向上库/用户的推理指导，单独保留，不在此处重复。
- `NPU_ADAPTATION.md`：只记录适配实现与验证事实。
- `ACCEPTANCE_PLAN.md`：记录 MLS / LibriSpeech / FLEURS 评测方案、分层验收、通过条件、报告模板和已完成 NPU 结果。

## 目录

- [1. 上游版本与代码分析](#1-上游版本与代码分析)
- [2. NPU 适配与运行说明](#2-npu-适配与运行说明)
- [3. 验证记录](#3-验证记录)

## 1. 上游版本与代码分析

### 1. 上游信息

- 上游仓库：<https://github.com/NVIDIA-NeMo/NeMo.git>
- 分支：`main`
- 基准 commit：`44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`
- commit 信息：`ci: remove build-docs and build-test-publish-wheel workflows (#15685)`
- 检查日期：2026-05-23
- 模型权重：<https://huggingface.co/nvidia/canary-1b>
- 版本边界：当前适配的是原始 `nvidia/canary-1b` / `canary-1b.nemo`；不包含 `nvidia/canary-1b-flash` 或 `nvidia/canary-1b-v2`。已验证权重 SHA256：`b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a`。
- 本地上游副本：`Canary-1B/upstream/`（已通过 `git clone --depth 1` 获取）

### 2. 当前目录状态

当前 `Canary-1B/` 原有文件：

- `infer.py`：原始推理 demo，依赖 `torch_npu.contrib.transfer_to_npu`，并包含硬编码音频/缓存路径。
- `README.md`：NPU 运行说明，但要求手工移动脚本并修改路径。
- `requirements.txt`：当前环境导出的依赖，范围明显大于 Canary/NeMo ASR 推理最小依赖。

本次新增/调整：

- `infer.py`：改为当前适配目录维护的参数化 CPU/NPU 融合推理脚本，默认 `--device npu`。
- `patches/README.md`：说明本次没有上游源码 patch。
- `NPU_ADAPTATION.md`：整合后的适配分析、迁移说明和验证记录。
- `.gitignore`：加入 `Canary-1B/upstream/`。

### 3. 与上游匹配情况

Canary-1B 通过 NeMo `EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')` 加载。上游 `nemo/collections/asr/models/aed_multitask_models.py` 的推理链路使用 `trcfg._internal.device`、`tensor.to(device)` 和模型自身 device 传递，未发现必须为 Canary 单独修改上游源码的 `.cuda()` 硬编码节点。

因此本次适配不修改 NeMo 上游已有文件，不生成 `.patch`；交付当前模型目录新增的 `infer.py`、评测/数据准备脚本和整合文档。后续如发现某个 NeMo 版本在 Canary 推理链路中新增硬编码 CUDA/NCCL 节点，应先在 `Canary-1B/upstream/` 对应文件修改，再生成 patch。

### 4. 现有代码审视

| 文件 | 结论 | 说明 |
|---|---|---|
| `infer.py` | 已重写 | 默认 NPU，支持 `--device cpu` 验证；无 `auto/use_gpu`；不写死 `npu:0/cuda:0`；音频、任务、语言和模型路径参数化。 |
| `README.md` | 已更新 | 补充基准 commit、无需 patch、运行方式和验证方式。 |
| `requirements.txt` | 保留但不建议作为最小依赖 | 包含 CUDA/服务端/训练相关大量依赖，正式部署建议按 README 中最小依赖安装。 |
| `patches/` | 无上游 patch | 因未修改 NeMo 上游已有文件，仅保留 README 说明。 |
| `prepare_eval_data.py` / `eval_canary.py` | 已新增 | 提供评测数据准备和评测脚本。 |

### 5. 设备适配点

1. `infer.py::_resolve_device`：仅当 `--device npu` 时导入 `torch_npu` 注册后端；返回 `torch.device('npu')`，不绑定卡号。
2. `EncDecMultiTaskModel.from_pretrained(..., map_location=device)`：加载时按目标设备映射权重。
3. `model.to(device)`：显式迁移模型。
4. `model.transcribe(...)`：输入通过 manifest 显式传入 `taskname/source_lang/target_lang/pnc`，由 NeMo dataloader 和模型内部 device 机制处理 batch。

### 6. 风险与限制

- 当前未在本机真实 NPU 上执行端到端推理；已完成静态检查、`py_compile`、CPU 环境搭建和 CPU 推理启动验证。
- 已通过 HF 镜像下载 `canary-1b.nemo`，并完成当前环境 CPU smoke test，输出 `[0]  I'm a part of that.`。
- Canary-1B 约 1B 参数，NPU 显存、CANN/torch-npu/torch 版本需要匹配。
- NeMo 主分支持续变化；如果上游更新，应重新检查 `EncDecMultiTaskModel`、`ASRTranscriptionMixin` 和音频预处理链路。
- `requirements.txt` 非最小依赖，可能引入无关 CUDA 包；部署时优先安装与当前 CANN/torch-npu 匹配的 PyTorch、torch-npu 和 NeMo ASR 依赖。

### 7. 上游版本检查记录

- 2026-05-23：重新执行 `git clone --depth 1 https://github.com/NVIDIA-NeMo/NeMo.git Canary-1B/upstream` 成功。
- 2026-05-23：`git -C Canary-1B/upstream rev-parse HEAD` 输出 `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`。
- 2026-05-23：`git -C Canary-1B/upstream ls-remote origin refs/heads/main` 确认远端 `main` 同为 `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`。
- 2026-05-23：检查 `nemo/collections/asr/models/aed_multitask_models.py`，确认 Canary 主要推理代码使用 device 传递，无需 patch。

## 2. NPU 适配与运行说明

### 1. 适配目标

将 NVIDIA NeMo Canary-1B 推理样例整理为规范的 CPU/NPU 融合脚本：

- 默认使用 `--device npu`；
- CPU 验证显式使用 `--device cpu`；
- 不使用 `auto` / `use_gpu`；
- 不在代码中写死 `npu:0` / `cuda:0`；
- 实际 NPU 卡号由环境变量控制，例如 `ASCEND_RT_VISIBLE_DEVICES=0`。

### 2. 上游与 patch

- 上游仓库：<https://github.com/NVIDIA-NeMo/NeMo.git>
- 基准 commit：`44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`
- 本次没有修改上游已有文件，因此没有 `.patch` 文件。
- `Canary-1B/infer.py` 是当前适配新增脚本，不进入 patch。

如后续上游源码需要修改，则在 `Canary-1B/upstream/` 内生成 patch：

```bash
git -C Canary-1B/upstream diff -- <upstream_existing_file> > Canary-1B/patches/0001-xxx.patch
git -C Canary-1B/upstream apply --check ../patches/0001-xxx.patch
```

### 3. 环境准备

#### 3.1 CPU 验证环境

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

#### 3.2 NPU 推理环境

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

### 4. 权重下载

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

### 5. 评测口径摘要

Canary-1B 的正式精度/性能验收口径统一维护在 `ACCEPTANCE_PLAN.md`：

- ASR 使用 WER；英文文本按官方 Whisper `EnglishTextNormalizer` 处理。
- AST 使用 sacreBLEU，保留数据集原始标点和大小写。
- 官方精度对齐使用 `beam_size=5`、`length_penalty=1.0`。
- 性能模式单独记录 `elapsed_seconds`、`rtf`、`RTFx=audio_seconds/elapsed_seconds`、`batch_size`、`beam_size`、峰值 HBM/RSS。

适配脚本只保留必要运行说明；数据集下载、MLS/LibriSpeech/FLEURS 评测命令、通过条件和报告字段见 `ACCEPTANCE_PLAN.md`。

### 6. 测试数据准备

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

该样例仅用于链路验证，不用于准确率评估。MLS / LibriSpeech / FLEURS manifest 准备和正式评测流程见 `ACCEPTANCE_PLAN.md`。

### 7. 推理脚本用法

#### CPU ASR 验证

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

#### NPU ASR

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

#### 语音翻译 AST

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

### 8. 上游更新处理

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

## 3. 验证记录

### 1. 静态验证

检查日期：2026-05-23

```bash
find Canary-1B -maxdepth 3 -type f | sort
git status --short
Canary-1B/.venv-cpu/bin/python -m py_compile Canary-1B/infer.py
```

结果：`py_compile` 通过。

### 2. 上游 clone 验证

```bash
git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/NVIDIA-NeMo/NeMo.git Canary-1B/upstream
git -C Canary-1B/upstream rev-parse HEAD
git -C Canary-1B/upstream ls-remote origin refs/heads/main
```

结果：

```text
44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe
44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe
```

本地 upstream HEAD 与远端 `main` 一致。

### 3. patch 验证

本次适配未修改 NeMo 上游已有文件，因此没有 `.patch` 文件需要 `git apply --check`。

如后续新增 patch，执行：

```bash
for p in Canary-1B/patches/*.patch; do
  git -C Canary-1B/upstream apply --check "../patches/$(basename "$p")"
done
```

### 4. CPU 环境准备验证

当前系统缺少 `python3-pip` / `ensurepip`，已使用 `uv` 创建 CPU 验证虚拟环境：

#### 4.1 依赖文件关系说明

NeMo `requirements/` 目录下的文件不是互相全部包含的关系。特别注意：

- `requirements_asr.txt` **不包含** `requirements_lightning.txt`。
- 只执行 `pip install -r requirements_lightning.txt` 只能解决 `lightning`、`hydra-core`、`omegaconf` 等训练/框架依赖，不能解决 ASR 依赖，例如 `lhotse`。
- 只执行 `pip install -r requirements_asr.txt` 只能解决 ASR 领域依赖，例如 `lhotse`、`librosa`、`soundfile`、`jiwer`、`sacrebleu` 等，不能解决 `lightning`。
- `pip install -e "NeMo[asr]"` 或 `pip install "nemo-toolkit[asr]"` 才会通过 NeMo 的 `setup.py` 组合安装基础依赖、`requirements_common.txt`、`requirements_lightning.txt` 和 `requirements_asr.txt`。

NeMo 源码中 `setup.py` 的组合关系如下：

| 安装项/文件 | 作用 | 是否包含其他 requirements |
|---|---|---|
| `requirements.txt` | NeMo 基础依赖，如 `torch`、`numpy`、`huggingface_hub` 等 | 基础安装依赖 |
| `requirements_lightning.txt` | NeMo core/lightning 依赖，如 `lightning`、`hydra-core`、`omegaconf`、`torchmetrics`、`transformers` | 不包含 ASR |
| `requirements_common.txt` | 通用数据/文本依赖，如 `datasets`、`sentencepiece`、`pandas` | 不包含 lightning/ASR |
| `requirements_asr.txt` | ASR 依赖，如 `lhotse`、`librosa`、`soundfile`、`jiwer`、`sacrebleu` | 不包含 lightning/common/base |
| `requirements_audio.txt` | 通用音频处理/评估依赖，如 `lhotse`、`librosa`、`pesq`、`pystoi` | 不等同于 ASR 完整依赖 |
| `requirements_tts.txt` | TTS 依赖 | NeMo extra `tts` 会叠加 ASR/common |
| `requirements_slu.txt` | SLU 依赖 | NeMo extra `slu` 会叠加 ASR |
| `requirements_test.txt` | 测试/格式化依赖 | 仅测试开发使用 |
| `requirements_docs.txt` | 文档构建依赖 | 仅文档使用 |
| `requirements_cu12.txt` / `requirements_cu13.txt` | NVIDIA CUDA 附加依赖 | NPU 环境通常不使用 |
| `requirements_run.txt` | `nemo_run` 相关依赖 | 与 Canary 推理无直接关系 |
| `requirements_speechlm2.txt` | SpeechLM2 相关依赖 | 与 Canary-1B ASR smoke test 无直接关系 |

因此，Canary-1B ASR 推理建议使用以下二选一方式安装依赖。

**方式 A：从 NeMo 源码安装 ASR extra（推荐）**

```bash
cd /home/Canary-1B-Adapt/NeMo
python -m pip install -e ".[asr]"
```

**方式 B：手工按文件安装（适合不能 editable install 的环境）**

```bash
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements.txt
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements_common.txt
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements_lightning.txt
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements_asr.txt
```

如果使用 NPU，还必须额外安装与当前 CANN/驱动匹配的 `torch` / `torch-npu`，不要让上述命令覆盖已验证可用的 NPU 版 PyTorch。

#### 4.2 CPU 依赖安装记录

```bash
uv venv Canary-1B/.venv-cpu --python 3.12
uv pip install --python Canary-1B/.venv-cpu/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com \
  "torch==2.9.1" "torchaudio==2.9.1" "nemo-toolkit[asr]" \
  librosa soundfile sentencepiece huggingface_hub
```

依赖检查：

```bash
Canary-1B/.venv-cpu/bin/python - <<'PY'
import torch, torchaudio, nemo, soundfile, librosa, huggingface_hub
import lightning.pytorch, lhotse
print(torch.__version__, torchaudio.__version__)
PY
```

结果：

```text
torch 2.9.1+cu128
torchaudio 2.9.1+cu128
nemo / lightning / lhotse / soundfile / librosa / huggingface_hub 均可导入
```

### 5. 权重下载验证

权重来源：

- 官方：<https://huggingface.co/nvidia/canary-1b>
- 本次成功使用 HF 镜像：<https://hf-mirror.com/nvidia/canary-1b>

ModelScope 检索结果：使用 ModelScope API / `modelscope` SDK 以 `canary-1b`、`nvidia/canary-1b`、`canary` 检索，未找到同名公开模型。

下载命令：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  curl -k -L --fail --retry 10 --retry-delay 5 -C - \
  -o Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  https://hf-mirror.com/nvidia/canary-1b/resolve/main/canary-1b.nemo
```

结果：

```text
Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo
文件大小：3.8G
SHA256：b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a
```

说明：当前代理环境下 `hf-mirror.com` 经代理会出现 TLS EOF，取消代理环境变量后可正常下载。

### 6. 测试数据准备验证

执行：

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

结果：

```text
Canary-1B/test_data/dummy_1s_16k.wav
```

说明：该文件为 1 秒 16 kHz 单声道正弦波，仅用于 smoke test，不用于识别准确率评估。

#### 6.1 MLS / LibriSpeech / FLEURS 评测数据准备脚本验证

当前脚本已补齐在线/离线混合参数：

```bash
Canary-1B/.venv-cpu/bin/python -m py_compile Canary-1B/prepare_eval_data.py

# 离线缺失检查示例：应报出缺失的本地 MLS/LibriSpeech/FLEURS parquet 路径，不访问远端
python Canary-1B/prepare_eval_data.py \
  --task all \
  --data_dir /tmp/canary_eval_data \
  --asr_parquet_dir /tmp/canary_eval_data/mls_parquet \
  --asr_configs german \
  --librispeech_dir /tmp/canary_eval_data/librispeech_raw \
  --fleurs_parquet_dir /tmp/canary_eval_data/fleurs_parquet \
  --offline \
  --asr_limit 1 \
  --fleurs_limit 1 \
  --ast_directions en-de
```

要求：

- FLEURS 文件固定在 `<fleurs_parquet_dir>/<config>/<split>-00000-of-00001.parquet`，已有则复用。
- MLS 文件固定在 `<asr_parquet_dir>/{german,spanish,french}/test-00000-of-00001.parquet`，已有则复用。
- LibriSpeech 性能测试文件固定在 `<librispeech_dir>/test-clean.tar.gz` 或 `<librispeech_dir>/LibriSpeech/test-clean/`，已有则复用。
- metadata 记录 `asr_parquet_dir` / `librispeech_dir` / `fleurs_parquet_dir` 和 `offline`。
- MLS/LibriSpeech/FLEURS 禁用 HF `Audio` 自动解码，避免 `torchcodec` 依赖。

完整在线/离线/手动下载命令见 `ACCEPTANCE_PLAN.md` 的“MLS / LibriSpeech / FLEURS 验证测试方案”章节。

### 7. 当前环境 CPU 推理验证

执行命令：

```bash
/usr/bin/time -f 'elapsed=%E maxrss=%MKB' timeout 900 \
  Canary-1B/.venv-cpu/bin/python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --device cpu \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --task asr \
  --source_lang en \
  --target_lang en \
  --batch_size 1
```

结果：CPU 端到端推理成功。

```text
[0]  I'm a part of that.
elapsed=0:17.83 maxrss=9042820KB
exit_code=0
```

说明：

- 测试音频是 1 秒 16 kHz 正弦波，只用于 smoke test，不代表识别准确率；
- CPU 推理成功验证了本地 `.nemo` 权重加载、音频读取、manifest 构造、CPU device 路径和 `model.transcribe()` 调用链路；
- 最大 RSS 约 8.6 GiB，Canary-1B 在 CPU 上只适合功能验证，不适合作为性能路径。

### 8. NPU 功能验证命令

有 NPU 和本地权重后执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --device npu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --batch_size 1
```

预期：

- 模型加载到 NPU；
- 输出识别文本；
- 无 `Expected all tensors to be on the same device` 等设备不匹配错误。

### 9. AST 验证命令

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

预期：输出德语翻译文本。

### 10. 当前限制

- 当前环境没有 NPU，未执行 NPU 端到端验证。
- 已通过 HF 镜像下载权重并完成 CPU smoke test；若切换网络或镜像，需重新校验 SHA256。
- Canary-1B 约 1B 参数，CPU 推理即使权重下载完成也可能较慢；建议使用短音频进行 smoke test。

### 11. 完整验收方案

当前 `dummy_1s_16k.wav` 仅用于 smoke test，不能证明 Canary-1B 的功能完整性、性能或精度。完整验收请执行 `ACCEPTANCE_PLAN.md`，至少覆盖：

- ASR：英语、德语、西班牙语、法语；
- AST：英语 ↔ 德语/西班牙语/法语 6 个方向；
- PnC：`yes/no`；
- batch：`1/4/8` 或记录最大可用 batch；
- 精度：ASR WER、AST BLEU；
- 性能：RTF/RTFx、加载时间、峰值内存/HBM、连续运行稳定性。

正式验收报告建议保存到 `Canary-1B/validation_reports/`，模板见 `ACCEPTANCE_PLAN.md`。

当前交付状态：**S2，固定权重的 CPU L0 端到端链路通过；升级到 S3 仍缺同
manifest 的 NPU 精度对齐，升级到 S4 仍缺最低正式验收清单全部结果**。
