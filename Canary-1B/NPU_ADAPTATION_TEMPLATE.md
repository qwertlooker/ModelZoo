# Canary-1B NPU 适配整合模板

> 本文整合原 Canary-1B 适配过程中的分析、适配说明、推理指导、数据评测、验证记录和验收计划。后续模型适配优先复用本文结构；将 `Canary-1B` 示例值替换为目标模型的实际版本、脚本、数据集和验证结果即可。

## 0. 模板使用约定

1. **先填版本边界，再写运行命令**：必须固定源码、权重、辅助模型、数据集版本和检查日期。
2. **保留严格失败原则**：必需依赖前置导入；缺依赖、官方字段缺失、上游版本不兼容、官方评测组件不可用时直接失败，不做静默 fallback。
3. **区分上游 patch 与适配新增文件**：上游已有文件的修改进入 `patches/*.patch`；新增 `infer.py`、评测脚本和本文档直接放模型目录。
4. **NPU 设备不写死卡号**：默认 `--device npu`，CPU 验证显式 `--device cpu`；实际卡号由 `ASCEND_RT_VISIBLE_DEVICES` 控制。
5. **smoke test 不等于验收**：短音频或 dummy 数据只证明链路可运行；正式验收必须覆盖功能、精度、性能、稳定性和报告留档。

## 1. 项目概述

| 字段 | 填写内容 |
|---|---|
| 模型名称 | Canary-1B |
| 任务类型 | 多语言 ASR；英语与德/西/法之间 AST |
| 原始模型 | `nvidia/canary-1b` |
| 当前适配权重 | `canary-1b.nemo` |
| 明确排除 | `canary-1b-flash`、`canary-1b-v2` |
| 适配目标 | 在保持 NeMo 官方加载、prompt、tokenizer、解码和评测口径的前提下支持昇腾 NPU 推理与评测 |

后续模型可按以下格式替换：

```text
model_name=<目标模型精确名称>
task=<ASR/TTS/SE/diarization/...>
source_repo=<上游源码仓库或模型仓库>
source_commit=<固定 commit/tag/release>
weight_repo=<权重仓库或下载地址>
weight_file=<具体 checkpoint 文件>
excluded_variants=<不在本次适配范围内的同系列变体>
```

## 2. 版本边界与上游信息

### 2.1 Canary-1B 示例

| 项目 | 值 |
|---|---|
| 上游源码 | <https://github.com/NVIDIA-NeMo/NeMo.git> |
| 分支 | `main` |
| 基准 commit | `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe` |
| commit 信息 | `ci: remove build-docs and build-test-publish-wheel workflows (#15685)` |
| 检查日期 | 2026-05-23 |
| 权重来源 | <https://huggingface.co/nvidia/canary-1b> |
| 权重 SHA256 | `b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a` |
| 本地 upstream | `Canary-1B/upstream/`，不提交 |

### 2.2 后续模型必须记录

```bash
git clone <upstream_repo> <model_dir>/upstream
git -C <model_dir>/upstream rev-parse HEAD
git -C <model_dir>/upstream ls-remote origin <default_branch>
sha256sum <weight_file>
```

并在文档中同步记录：

- 源码 repo、分支、commit/tag；
- 权重 repo、文件名、版本或校验值；
- tokenizer、codec、vocoder、speaker embedding 等辅助组件版本；
- 明确排除的同系列变体；
- 检查日期和远端最新状态。

## 3. 目录结构与交付件

### 3.1 Canary-1B 当前结构

```text
Canary-1B/
├── README.md                    # 原模型目录说明；仅保留简要入口和文件索引
├── NPU_ADAPTATION_TEMPLATE.md   # 本整合模板：分析、适配、验证、验收合一
├── infer.py                     # 单条/多条音频推理脚本
├── eval_canary.py               # 精度和性能评测脚本
├── prepare_eval_data.py         # LibriSpeech/MLS/FLEURS 数据准备脚本
├── utils.py                     # 推理和评测复用工具
├── requirements.txt             # 当前环境依赖记录，非最小部署清单
├── patches/
│   └── README.md                # 本次未修改 NeMo 上游源码
├── weights/                     # 本地权重，通常不提交大文件
└── test_data/                   # smoke test 数据
```

### 3.2 后续模型建议

- 保留一个整合文档，覆盖“分析 + 适配 + 验证 + 验收”；避免为同一内容维护多份 Markdown。
- 如需面向上库交付，可从本文第 6、7、8 节抽取精简版 `README_INFERENCE.md`；但源信息仍以整合文档为准。
- 所有脚本命令必须可从模型目录或仓库根目录复现，避免绝对路径。

## 4. 代码适配分析

### 4.1 Canary-1B 结论

Canary-1B 通过 NeMo `EncDecMultiTaskModel` 恢复 `.nemo` 权重。当前基准 commit 下，主要推理链路使用模型 device、`tensor.to(device)` 和内部 `trcfg._internal.device` 传递，未发现必须为 Canary 单独修改的 `.cuda()` 硬编码节点。

因此本次适配：

- 不修改 NeMo 上游已有文件；
- 不生成 `.patch`；
- 仅维护当前模型目录中的 `infer.py`、`eval_canary.py`、`prepare_eval_data.py`、`utils.py` 和文档。

### 4.2 扫描命令模板

```bash
grep -RIn "cuda\|gpu\|npu\|to(device)\|torch.load\|map_location\|nccl" \
  <model_dir>/upstream <model_dir> \
  --exclude-dir=.git --exclude-dir=.venv --exclude='*.log'
```

需要重点处理：

- `.cuda()`、`torch.cuda.*`、`device="cuda"`；
- `torch.load(..., map_location="cuda")`；
- `backend="nccl"`；
- 训练专用分布式逻辑误入推理链路；
- 隐式远程下载或 CPU/GPU fallback。

若后续上游源码需要修改：

```bash
git -C <model_dir>/upstream diff -- <upstream_existing_file> > <model_dir>/patches/0001-xxx.patch
git -C <model_dir>/upstream apply --check ../patches/0001-xxx.patch
```

## 5. NPU 适配实现规范

### 5.1 设备选择

```python
import torch


def resolve_device(device_name: str) -> torch.device:
    if device_name == "npu":
        import torch_npu  # noqa: F401  # 只在 NPU 路径注册后端
    return torch.device(device_name)
```

要求：

- `infer.py` 默认 `--device npu`；
- CPU 验证使用 `--device cpu`；
- 不使用 `auto` / `use_gpu`；
- 不写死 `npu:0` / `cuda:0`；
- 卡号通过环境变量控制。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python infer.py --device npu ...
```

### 5.2 Canary-1B 适配点

1. 仅当 `--device npu` 时导入 `torch_npu`。
2. `EncDecMultiTaskModel.restore_from(..., map_location=device)` 按目标设备加载。
3. `model.to(device)` 显式迁移模型。
4. `model.transcribe(...)` 使用 manifest 或音频路径传入任务、语言和 PnC 字段。
5. `eval_canary.py` 保持官方评测依赖和解码参数，不使用简化替代。

## 6. 环境准备

### 6.1 NPU 推理环境

| 配套 | Canary-1B 示例版本 |
|---|---|
| 固件与驱动 | 25.5.1+ |
| CANN | 8.5.1 |
| Python | 3.11.14 |
| PyTorch / torch_npu | 2.9.0 |
| torchaudio | 2.9.0 |

安装示例：

```bash
pip install torch==2.9.0 torch_npu==2.9.0 torchaudio==2.9.0
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe"
pip install soundfile librosa sentencepiece huggingface_hub jiwer sacrebleu openai-whisper
```

如使用本地 NeMo：

```bash
cd /path/to/NeMo
python -m pip install -e ".[asr]"
```

### 6.2 CPU 验证环境示例

```bash
uv venv Canary-1B/.venv-cpu --python 3.12
uv pip install --python Canary-1B/.venv-cpu/bin/python \
  "torch==2.9.1" "torchaudio==2.9.1" "nemo-toolkit[asr]" \
  librosa soundfile sentencepiece huggingface_hub jiwer sacrebleu openai-whisper
```

已验证环境记录：

```text
Python 3.12.3
torch 2.9.1+cu128
torchaudio 2.9.1+cu128
nemo-toolkit 2.7.3
```

## 7. 权重与测试数据

### 7.1 权重下载

```bash
mkdir -p Canary-1B/weights/canary-1b
wget -O Canary-1B/weights/canary-1b/canary-1b.nemo \
  https://huggingface.co/nvidia/canary-1b/resolve/main/canary-1b.nemo
sha256sum Canary-1B/weights/canary-1b/canary-1b.nemo
```

当前已验证 SHA256：

```text
b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a
```

### 7.2 smoke test 数据

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

该数据只用于链路检查，不用于精度验收。

### 7.3 正式评测数据

Canary-1B 评测数据分为：

| 用途 | 数据集 | 任务 | 指标 |
|---|---|---|---|
| 英文 ASR/性能 | LibriSpeech test-clean | ASR en→en | WER、RTF、RTFx |
| 多语种 ASR | Multilingual LibriSpeech test | ASR de/es/fr | WER |
| 多方向翻译 | FLEURS test | AST en↔de/es/fr | BLEU |

推荐最小验收数据准备：

```bash
python Canary-1B/prepare_eval_data.py \
  --task all \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --asr_split test \
  --asr_minutes 30 \
  --asr_pnc no \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --fleurs_split test \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes
```

离线复用时加 `--offline`，缺文件应直接报具体路径，不联网兜底。

## 8. 推理与评测命令

### 8.1 单条 ASR 推理

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b/canary-1b.nemo \
  --audio Canary-1B/test_data/demo.wav \
  --device npu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --pnc yes \
  --batch_size 1
```

CPU 验证只替换设备：

```bash
python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b/canary-1b.nemo \
  --audio Canary-1B/test_data/demo.wav \
  --device cpu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --pnc yes
```

### 8.2 性能评测

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b/canary-1b.nemo \
  --manifest Canary-1B/eval_data/librispeech_test_clean/manifest_asr_en.jsonl \
  --device npu \
  --batch_size 8 \
  --beam_size 1 \
  --warmup_batches 2 \
  --output_dir Canary-1B/eval_results/perf_librispeech
```

### 8.3 精度评测

ASR：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b/canary-1b.nemo \
  --manifest Canary-1B/eval_data/mls_test_german/manifest_asr_de.jsonl \
  --device npu \
  --batch_size 8 \
  --beam_size 5 \
  --length_penalty 1.0 \
  --output_dir Canary-1B/eval_results/asr_de
```

AST：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b/canary-1b.nemo \
  --manifest Canary-1B/eval_data/fleurs/en-de/manifest_ast_en_de.jsonl \
  --device npu \
  --batch_size 8 \
  --beam_size 5 \
  --length_penalty 1.0 \
  --output_dir Canary-1B/eval_results/ast_en_de
```

## 9. 官方口径与参考指标

### 9.1 评测口径

- ASR 指标：WER；英文按官方 Whisper `EnglishTextNormalizer` 处理。
- AST 指标：BLEU；保留数据集原始标点和大小写。
- 官方精度对齐：`beam_size=5`，`length_penalty=1.0`。
- 性能模式：通常使用 `beam_size=1`，记录 RTF/RTFx、batch size、精度、硬件和峰值内存。

### 9.2 Canary-1B 公开参考

| 任务 | 数据集 | 指标 | 官方参考 |
|---|---|---|---|
| ASR | MCV-16.1 test | WER | En 7.97 / De 4.61 / Es 3.99 / Fr 6.53 |
| ASR | MLS test | WER | En 3.06 / De 4.19 / Es 3.15 / Fr 4.12 |
| AST | FLEURS test | BLEU | En→De 32.15 / En→Es 22.66 / En→Fr 40.76 / De→En 33.98 / Es→En 21.80 / Fr→En 30.95 |
| AST | CoVoST-v2 test | BLEU | De→En 37.67 / Es→En 40.70 / Fr→En 40.42 |

Open ASR Leaderboard 公开参考：Average WER `6.50`，RTFx `235.34`。该值来自 A100 GPU 环境，只作量级参考，不作为 NPU 硬性通过线。

## 10. 验证记录模板

### 10.1 静态与脚本检查

```bash
python -m py_compile Canary-1B/infer.py Canary-1B/eval_canary.py Canary-1B/prepare_eval_data.py Canary-1B/utils.py
git -C Canary-1B/upstream apply --check ../patches/*.patch  # 有 patch 时执行
python Canary-1B/infer.py --help
python Canary-1B/eval_canary.py --help
python Canary-1B/prepare_eval_data.py --help
```

记录表：

| 检查项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| py_compile | `<命令>` | `<通过/失败及错误>` | `<日期>` |
| patch check | `<命令或无 patch>` | `<通过/不适用>` | `<日期>` |
| help | `<命令>` | `<通过/失败及错误>` | `<日期>` |
| 权重校验 | `sha256sum ...` | `<sha256>` | `<日期>` |
| 数据可读性 | `<命令>` | `<样本数/时长>` | `<日期>` |

### 10.2 Canary-1B 已完成记录

- 已通过 HF 镜像下载 `canary-1b.nemo` 并校验 SHA256。
- 已完成当前环境 CPU smoke test，示例输出：`[0]  I'm a part of that.`
- 当前本仓库未记录本机真实 NPU 端到端复验日志；正式验收需补齐第 11 节报告字段。

## 11. 分层验收计划

| 层级 | 目标 | 数据 | 通过条件 |
|---|---|---|---|
| L0 静态检查 | 文件、版本、patch、help、py_compile | 无需大模型或少量 dummy | 脚本可编译；patch 可应用；版本边界清晰；无硬编码卡号或静默 fallback |
| L1 链路 smoke | 模型加载、单样本推理、NPU 设备迁移 | 1-3 条短音频 | CPU/NPU 均可运行；输出文本；无隐式 CPU fallback |
| L2 子集精度/性能 | 小规模可复现精度与吞吐 | MLS 30 分钟、LibriSpeech 30 分钟、FLEURS 每方向 50 条 | 指标无明显退化；记录 RTF/RTFx、batch、beam、内存 |
| L3 全量验收 | 正式上库/上线依据 | LibriSpeech、MLS、FLEURS 全量或项目指定数据 | 与公开参考或 CPU/CUDA 基线同口径对齐；稳定性和资源满足交付要求 |

### 11.1 报告字段

正式报告建议保存到 `<model_dir>/validation_reports/`，至少包含：

```text
model_name:
source_repo:
source_commit:
weight_repo:
weight_file:
weight_sha256:
hardware:
driver_firmware:
CANN:
python:
torch:
torch_npu:
torchaudio:
script_commit:
command:
dataset:
split:
num_samples:
audio_seconds:
batch_size:
beam_size:
length_penalty:
precision:
elapsed_seconds:
rtf:
rtfx:
peak_hbm_or_rss:
metric_name:
metric_value:
reference_value:
pass_or_fail:
known_limitations:
```

## 12. Canary-1B NPU 适配结果示例

硬件：Atlas 800I A2。

### 12.1 性能

| 数据集 | 指标 | NPU 结果 | 公开 GPU 参考 |
|---|---|---:|---:|
| LibriSpeech test-clean | RTF | 0.005652242997176402 | 0.0042491714115747425 |

### 12.2 精度

| 任务 | 方向 | 数据集 | 指标 | NPU 结果 | 公开参考 |
|---|---|---|---|---:|---:|
| ASR | de | MLS | WER(%) | 3.83 | 4.19 |
| ASR | es | MLS | WER(%) | 2.30 | 3.15 |
| ASR | fr | MLS | WER(%) | 3.69 | 4.12 |
| AST | en-de | FLEURS | BLEU | 31.41 | 32.15 |
| AST | en-es | FLEURS | BLEU | 22.69 | 22.66 |
| AST | en-fr | FLEURS | BLEU | 39.84 | 40.76 |
| AST | de-en | FLEURS | BLEU | 33.50 | 33.98 |
| AST | es-en | FLEURS | BLEU | 21.78 | 21.80 |
| AST | fr-en | FLEURS | BLEU | 30.29 | 30.95 |

## 13. 经验复用清单

- 固定精确版本边界，避免把同系列不同 checkpoint 混为一个适配对象。
- 优先复用官方模型恢复、tokenizer、prompt、decoder 和 metric。
- 性能和精度使用不同 beam 设置时必须分开报告。
- 评测数据准备支持在线下载、本地复用和离线严格失败三种模式。
- 所有结果记录命令、环境、数据集、样本数、音频时长和输出目录。
- 后续模型若只需要一份文档，就以本文为主；不要再把相同信息拆成多份互相引用的 Markdown。

## 14. 公网地址

| 类型 | 地址 |
|---|---|
| NeMo 源码 | <https://github.com/NVIDIA-NeMo/NeMo.git> |
| Canary-1B 权重 | <https://huggingface.co/nvidia/canary-1b> |
| LibriSpeech | <https://www.openslr.org/12> |
| Multilingual LibriSpeech | <https://huggingface.co/datasets/facebook/multilingual_librispeech> |
| FLEURS | <https://huggingface.co/datasets/google/fleurs> |
| Open ASR Leaderboard | <https://hf-audio-open-asr-leaderboard.hf.space/> |
