# MOSS-Speech 推理指导

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [文件目录](#文件目录)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备权重与 Space 代码](#准备权重与-space-代码)
  - [准备测试数据](#准备测试数据)
  - [模型推理](#模型推理)
  - [流程复查命令](#流程复查命令)
- [模型推理性能与精度验收](#模型推理性能与精度验收)
- [公网地址说明](#公网地址说明)

## 概述

MOSS-Speech 是 OpenMOSS/FNLP 发布的语音对话大模型，推理链路包含主模型、MOSS-Speech-Codec、Hugging Face Space 中的 CosyVoice/TTS 相关代码与示例 prompt 音频。本文档给出当前适配目录中 `infer.py` 的 NPU 推理指导。

> 说明：仓库内原始 `README.md` 保留为既有实现参考，不再作为本次细化的主推理文档。本文件只描述当前适配后的规范化推理流程。

- 版本说明：

  ```text
  model_name=MOSS-Speech
  main_model_url=https://modelscope.cn/models/openmoss/MOSS-Speech
  main_model_git=https://www.modelscope.cn/openmoss/MOSS-Speech.git
  main_model_branch=master
  main_model_commit_id=270d64296cafb94ca1f35b14b8d7918a1c4a2dc0

  codec_url=https://modelscope.cn/models/AI-ModelScope/MOSS-Speech-Codec
  codec_git=https://www.modelscope.cn/AI-ModelScope/MOSS-Speech-Codec.git
  codec_branch=master
  codec_commit_id=a5423645a66476da761bbbdbc2003ae34e3c31c4

  space_url=https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech
  space_branch=main
  space_commit_id=92a89018a8aa6b36f08c366c2659c76ffdc3f980
  check_date=2026-06-01
  ```

当前适配边界：只覆盖 `openmoss/MOSS-Speech` 主模型 + `AI-ModelScope/MOSS-Speech-Codec` + `OpenMOSS-Team/MOSS-Speech` Space 代码的单请求推理；不覆盖 `MOSS-TTSD-v0.5`、CosyVoice 训练、TensorRT、ONNX 导出或其他派生变体。

## 输入输出数据

- 输入数据

  - 文本 prompt：通过 `--prompt` 传入。
  - decoder prompt audio：生成音频时通过 `--prompt_audio` 传入，默认使用 Space 自带 `assets/prompt_cn.wav`。

- 输出数据

  - `--output_modality text`：输出文本并保存到 `text_*.txt`。
  - `--output_modality audio`：输出 wav 音频并保存到 `audio_*.wav`。

## 推理环境准备

- 该模型需要以下插件与驱动。具体 PyTorch / torch-npu 版本请按实际 CANN 版本选择。

  **表 1** 版本配套表

| 配套 | 版本 |
|---|---|
| 昇腾 NPU 驱动/固件 | >=25.0.RC1.1 商发版本 |
| CANN Toolkit / Kernel / NNAL | >=8.2.RC1 商发版本 |
| Python | 建议 3.10 或 3.11 |
| PyTorch / torch-npu / torchaudio | 与 CANN 匹配 |
| transformers / modelscope | 以主模型 remote code 可加载为准 |

安装基础依赖：

```bash
pip install torch torch-npu torchaudio
pip install transformers accelerate modelscope soundfile librosa gradio spaces diffusers
```

说明：`MOSS-Speech/requirements.txt` 是历史环境冻结，包含训练、服务和 CUDA/NVIDIA 相关包；NPU 环境中安装时应避免覆盖已匹配的 `torch` / `torch-npu`。

## 文件目录

```text
MOSS-Speech
├── README.md                           # 原始既有实现参考，保持不改
├── README_INFERENCE.md                 # 当前推理指导文档
├── infer.py                            # 单请求文本/音频生成脚本
├── ANALYSIS.md                         # 上游与设备相关代码分析
├── NPU_ADAPTATION.md                   # NPU 适配说明
├── NPU_VALIDATION.md                   # 验证记录
├── ACCEPTANCE_PLAN.md                  # 分层验收计划
├── patches
│   └── README.md                       # 上游 patch 管理说明
├── upstream                            # HF Space 源码，按需下载，默认不提交
│   └── assets/prompt_cn.wav
├── weights                             # 主模型与 codec，按需下载，默认不提交
│   ├── MOSS-Speech
│   └── MOSS-Speech-Codec
└── outputs                             # 推理输出目录，按需生成
```

## 快速上手

### 获取源码

1. 获取适配源码。

   ```bash
   git clone https://github.com/qwertlooker/ModelZoo.git
   cd ModelZoo/MOSS-Speech
   ```

2. 安装依赖。

   ```bash
   pip install torch torch-npu torchaudio
   pip install transformers accelerate modelscope soundfile librosa gradio spaces diffusers
   ```

### 准备权重与 Space 代码

1. 下载 HF Space 代码。

   ```bash
   cd /path/to/ModelZoo
   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
     https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech \
     MOSS-Speech/upstream
   git -C MOSS-Speech/upstream rev-parse HEAD
   ```

   期望 commit：

   ```text
   92a89018a8aa6b36f08c366c2659c76ffdc3f980
   ```

2. 下载主模型与 codec。

   ```bash
   cd /path/to/ModelZoo
   python - <<'PY'
from modelscope import snapshot_download
snapshot_download('openmoss/MOSS-Speech', local_dir='MOSS-Speech/weights/MOSS-Speech')
snapshot_download('AI-ModelScope/MOSS-Speech-Codec', local_dir='MOSS-Speech/weights/MOSS-Speech-Codec')
PY
   ```

3. 记录权重校验值。

   ```bash
   find MOSS-Speech/weights -type f -print0 | sort -z | \
     xargs -0 sha256sum > MOSS-Speech/weights/SHA256SUMS.txt
   ```

### 准备测试数据

默认可直接使用 Space 自带 prompt 音频：

```text
MOSS-Speech/upstream/assets/prompt_cn.wav
MOSS-Speech/upstream/assets/prompt_en.wav
```

如果使用自定义 decoder prompt audio，请准备单声道或可被 `torchaudio` / `soundfile` 解码的 wav 文件，并通过 `--prompt_audio` 指定。

### 模型推理

1. NPU 文本输入生成音频。

   ```bash
   cd /path/to/ModelZoo
   ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-Speech/infer.py \
     --model MOSS-Speech/weights/MOSS-Speech \
     --codec MOSS-Speech/weights/MOSS-Speech-Codec \
     --space_dir MOSS-Speech/upstream \
     --prompt_audio MOSS-Speech/upstream/assets/prompt_cn.wav \
     --prompt "请用一句话介绍武汉的樱花。" \
     --output_modality audio \
     --output_dir MOSS-Speech/outputs \
     --device npu
   ```

2. NPU 文本响应。

   ```bash
   cd /path/to/ModelZoo
   ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-Speech/infer.py \
     --model MOSS-Speech/weights/MOSS-Speech \
     --codec MOSS-Speech/weights/MOSS-Speech-Codec \
     --space_dir MOSS-Speech/upstream \
     --prompt "Hello!" \
     --output_modality text \
     --output_dir MOSS-Speech/outputs \
     --device npu \
     --max_new_tokens 64
   ```

3. CPU smoke test。

   ```bash
   cd /path/to/ModelZoo
   python MOSS-Speech/infer.py \
     --model MOSS-Speech/weights/MOSS-Speech \
     --codec MOSS-Speech/weights/MOSS-Speech-Codec \
     --space_dir MOSS-Speech/upstream \
     --prompt "Hello!" \
     --output_modality text \
     --output_dir MOSS-Speech/outputs_cpu \
     --device cpu \
     --max_new_tokens 64
   ```

参数说明：

- `model`：主模型目录或可由 Transformers remote code 加载的模型 id。
- `codec`：MOSS-Speech-Codec 目录或模型 id。
- `space_dir`：HF Space 源码目录，用于提供 `cosyvoice/`、`utils/` 和 prompt 音频。
- `matcha_dir`：可选 Matcha-TTS 目录；默认使用 `--space_dir/Matcha-TTS`。
- `prompt_audio`：音频生成时使用的 decoder prompt wav。
- `output_modality`：`audio` 或 `text`。
- `device`：`npu`、`cpu` 或 `cuda`。默认 `npu`，实际 NPU 卡号通过 `ASCEND_RT_VISIBLE_DEVICES` 控制。

### 流程复查命令

提交或交付前建议重新执行：

```bash
cd /path/to/ModelZoo

git ls-remote --symref https://www.modelscope.cn/openmoss/MOSS-Speech.git HEAD
git ls-remote --symref https://www.modelscope.cn/AI-ModelScope/MOSS-Speech-Codec.git HEAD
git ls-remote --symref https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech HEAD

git -C MOSS-Speech/upstream rev-parse HEAD
python3 -m py_compile MOSS-Speech/infer.py
python3 MOSS-Speech/infer.py --help

grep -RIn "cuda\|device_map\|torch_npu\|istft\|bfloat16\|cached_download" \
  MOSS-Speech/upstream --exclude-dir=.git | head -200
```

如果后续修改上游已有文件，必须生成 patch 并检查：

```bash
git -C MOSS-Speech/upstream diff -- <upstream_file> > MOSS-Speech/patches/0001-xxx.patch
git -C MOSS-Speech/upstream apply --check ../patches/0001-xxx.patch
```

## 模型推理性能与精度验收

当前仓库未内置 MOSS-Speech 性能/精度自动评测脚本。验收按 `ACCEPTANCE_PLAN.md` 执行，至少记录以下字段：

| 指标 | 说明 |
|---|---|
| 成功率 | 请求成功数 / 总请求数 |
| 文本质量 | 输出非空，人工相关性评分 |
| 音频有效性 | wav 可读、采样率正确、时长 > 0、非全零、无 NaN/Inf |
| 端到端耗时 | 单请求 wall time |
| RTF | 生成耗时 / 输出音频时长 |
| 内存 | 峰值 HBM 与 CPU RSS |
| 对齐基线 | 同 checkpoint、同输入、同生成参数下 CPU/CUDA 输出 |

分层验收建议：

| 层级 | 数据规模 | 通过条件 |
|---|---:|---|
| L0 | 1-2 条 prompt | 脚本无异常，文本非空或音频可读 |
| L1 | 10 条中英 prompt | text/audio 两类输出均正常 |
| L2 | 50 条 prompt + 2 个 prompt audio | 人工听感无严重崩坏，ASR 回识别无系统性退化 |
| L3 | 100+ 条连续请求 | 无随机崩溃或明显内存泄漏，性能报告完整 |

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 主模型 | MOSS-Speech ModelScope 仓 | https://modelscope.cn/models/openmoss/MOSS-Speech |
| Codec | MOSS-Speech-Codec ModelScope 仓 | https://modelscope.cn/models/AI-ModelScope/MOSS-Speech-Codec |
| Space 代码 | OpenMOSS-Team/MOSS-Speech HF Space | https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech |
| 适配参考 | 原始 MOSS-Speech README 保留在本目录 | MOSS-Speech/README.md |
