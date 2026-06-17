# Canary-1B NPU 适配文档（整合版）

本文是 Canary-1B 后续复用的“适配”文档，合并并保留原 `ANALYSIS.md`、`NPU_ADAPTATION.md`、`NPU_VALIDATION.md`、`EVAL_FLEURS_MLS.md` 和 `ADAPTATION_CASE_SUMMARY.md` 中与版本边界、代码分析、环境、权重、数据、推理、评测、验证记录、案例结果相关的细节。

> 文档边界：`README_INFERENCE.md` 作为面向上库/用户的推理指导单独保留，不在本次整合中改动；正式验收设计和报告模板见 `ACCEPTANCE_PLAN.md`。

## 目录

- [1. 适配案例总览](#1-适配案例总览)
- [2. 上游版本与代码分析](#2-上游版本与代码分析)
- [3. NPU 适配与运行说明](#3-npu-适配与运行说明)
- [4. 验证记录](#4-验证记录)
- [5. MLS / LibriSpeech / FLEURS 验证测试方案](#5-mls--librispeech--fleurs-验证测试方案)

## 1. 适配案例总览

### Canary-1B 昇腾 NPU 适配案例总结

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [适配环境](#适配环境)
- [适配工作说明](#适配工作说明)
- [适配实施过程](#适配实施过程)
  - [模型与上游版本确认](#模型与上游版本确认)
  - [权重与依赖准备](#权重与依赖准备)
  - [数据与评测口径准备](#数据与评测口径准备)
  - [NPU 推理链路适配](#npu-推理链路适配)
  - [性能与精度验证](#性能与精度验证)
- [适配结果](#适配结果)
- [经验总结](#经验总结)
- [公网地址说明](#公网地址说明)

#### 概述

本案例围绕 NVIDIA Canary-1B 多语言多任务语音模型开展昇腾 NPU 推理适配。Canary-1B 基于 FastConformer 编码器和 Transformer 解码器，支持英语、德语、西班牙语、法语的自动语音识别（ASR），以及英语与德语/西班牙语/法语之间的语音到文本翻译（AST）。

适配目标不是重写模型结构，而是在保持 NeMo 官方模型加载、特征提取、任务提示、解码和评测口径的前提下，将推理执行设备从通用 CPU/CUDA 路径扩展到昇腾 NPU，并完成单样本推理、批量评测、性能计时和精度对齐验证。

本案例适配对象为 Hugging Face `nvidia/canary-1b` 中的原始 `canary-1b.nemo` 权重，不包含 `canary-1b-flash` 或 `canary-1b-v2`。

版本基线如下：

```text
url=https://github.com/NVIDIA-NeMo/NeMo.git
branch=main
commit_id=44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe
model_name=Canary-1B
weight=canary-1b.nemo
```

#### 输入输出数据

- 输入数据

  输入为 16 kHz 单声道音频，格式可为 wav、flac 等本地音频文件。批量评测使用 JSONL manifest 组织样本，每行描述一个音频样本及其任务信息。

  ASR 任务的关键字段包括音频路径、音频时长、参考文本、源语言、目标语言、任务类型和是否保留标点大小写。AST 任务在此基础上将目标语言设置为翻译输出语言。

- 输出数据

  输出为模型生成的文本。ASR 输出为源语音对应的转写文本；AST 输出为目标语言翻译文本。评测阶段会同时保存预测文本、参考文本、样本级耗时、数据集级指标和运行环境信息。

#### 适配环境

本案例使用的 NPU 推理环境配套如下。

| 配套 | 版本 |
|---|---|
| 固件与驱动 | 25.5.1+ |
| CANN | 8.5.1 |
| Python | 3.11.14 |
| PyTorch / torch_npu | 2.9.0 |
| torchaudio | 2.9.0 |

环境准备的核心原则是：PyTorch、torch_npu 与 CANN 版本必须匹配；NeMo ASR 依赖按官方 extra 安装；音频解码、文本指标和分词相关依赖作为必需依赖显式安装。缺失依赖时直接暴露原始错误，不增加静默兼容层或自动降级路径。

#### 适配工作说明

本次适配大致完成了以下工作：

1. 固定 NeMo 上游 commit 和 Canary-1B 原始权重，避免因上游接口变动导致适配结果不可复现。
2. 梳理 Canary-1B 的 ASR/AST 输入字段、语言代码、PnC 开关和解码参数，保持与官方任务定义一致。
3. 增加 NPU 设备选择与模型迁移逻辑，使用 `ASCEND_RT_VISIBLE_DEVICES` 控制实际卡号，不在代码中写死 `npu:0`。
4. 将推理过程整理为无梯度、评估模式、批量输入的执行链路，支持单音频验证和 manifest 批量评测。
5. 在性能模式中使用按音频时长排序、warmup、正式计时、`bfloat16` 计算和 RTF/RTFx 统计，便于与公开性能口径对照。
6. 在精度模式中保留官方解码路径：`beam_size=5`、`length_penalty=1.0`，ASR 使用 WER，AST 使用 BLEU。
7. 准备 LibriSpeech、Multilingual LibriSpeech 和 FLEURS 三类评测数据，分别覆盖英文 ASR 性能、多语种 ASR 精度和多方向 AST 精度。
8. 输出可追溯的评测结果，包括运行环境、逐样本预测、数据集指标汇总和性能统计。

#### 适配实施过程

##### 模型与上游版本确认

适配前先确认模型权重与上游代码的匹配关系。Canary-1B 的 `.nemo` 文件内部包含模型结构配置、tokenizer 信息和训练时的任务提示配置，因此适配时不应手工改写模型拓扑，也不应替换 tokenizer 或 prompt formatter。

关键确认点包括：

- 使用原始 `nvidia/canary-1b` 权重文件 `canary-1b.nemo`；
- 校验权重 SHA256，保证 CPU/NPU 使用同一 checkpoint；
- 固定 NeMo commit，确保模型恢复、音频前处理和解码接口稳定；
- 明确本案例不覆盖 Canary Flash 或 Canary v2，避免混用不同模型族的配置和指标。

##### 权重与依赖准备

权重准备采用本地 `.nemo` 文件方式。这样做可以避免推理时隐式访问公网，也便于在离线 NPU 环境中复现。权重目录中只需要保存原始 `.nemo` 文件，不需要展开或转换为其它格式。

依赖准备重点如下：

- PyTorch 与 torch_npu 必须使用与 CANN 匹配的版本；
- NeMo 使用 ASR extra，保证模型恢复、音频特征、tokenizer、解码器和指标相关组件完整；
- `soundfile`、`librosa`、`sentencepiece`、`jiwer`、`sacrebleu`、`openai-whisper` 等作为评测链路依赖；
- NPU 后端注册只在 NPU 路径中引入，CPU 验证路径不要求安装 torch_npu；
- 不使用 CPU fallback、第三方近似 normalizer 或简化 BLEU/WER 逻辑替代官方评测路径。

##### 数据与评测口径准备

评测数据分为三类：

| 用途 | 数据集 | 任务 | 指标 |
|---|---|---|---|
| 性能/英文 ASR 验证 | LibriSpeech test-clean | ASR en→en | RTF、RTFx、WER |
| 多语种 ASR 精度 | Multilingual LibriSpeech test | ASR de/es/fr | WER |
| 多方向翻译精度 | FLEURS test | AST en↔de/es/fr | BLEU |

Manifest 采用逐行 JSON 结构，核心字段包括：

```text
audio_filepath: 本地音频路径
duration: 音频时长，单位秒
text: 参考文本
task: asr 或 ast
source_lang: 源语言代码，en/de/es/fr
target_lang: 目标语言代码，en/de/es/fr
pnc: 是否输出标点和大小写
```

数据准备时优先复用本地文件；如离线模式下文件缺失，则直接报出缺失路径。MLS 和 FLEURS 的 parquet 音频字段按未解码二进制读取，再使用 `soundfile` 统一解码，以减少对额外媒体解码后端的依赖。

评测口径保持如下约束：

- ASR 精度使用 WER；英文文本按 Whisper 官方 normalizer 处理后计算；
- AST 精度使用 sacreBLEU，并保留数据集原始标点和大小写；
- 官方精度对齐使用 `beam_size=5` 和 `length_penalty=1.0`；
- 性能测试使用 `beam_size=1`，并在 NPU 上优先使用 `bfloat16`；
- 公开 GPU 榜单数据仅作为量级参考，不作为 NPU 适配硬性通过线。

##### NPU 推理链路适配

NPU 适配的主体工作集中在设备、精度、批处理和解码链路上。

设备侧处理方式如下：

- 通过运行参数选择 `npu`、`cpu` 或 `cuda`；
- NPU 路径中注册 torch_npu 后端；
- 使用逻辑设备名 `npu` 迁移模型，不绑定物理卡号；
- 实际使用哪张卡由 `ASCEND_RT_VISIBLE_DEVICES` 控制；
- 推理阶段进入 eval 模式，并关闭梯度计算。

模型执行链路保持 NeMo 官方流程：

1. 从 `.nemo` 恢复模型配置与权重；
2. 根据任务类型设置 ASR 或 AST prompt；
3. 按样本语言设置源语言、目标语言和 PnC 选项；
4. 将音频路径列表送入官方转写/翻译接口；
5. 使用官方 tokenizer 与解码器生成文本；
6. 收集预测文本并写入评测结果。

解码侧根据目标不同选择策略：

- 性能模式：`beam_size=1`，优先使用批量 greedy 解码，减少搜索开销；
- 精度模式：`beam_size=5`，使用 beam search，对齐 NVIDIA model card 公开精度口径；
- length penalty 保持 1.0，避免因搜索参数变化影响 BLEU/WER 对比。

精度侧处理方式如下：

- NPU/CUDA 性能模式默认使用 `bfloat16`；
- 精度验证可显式切换 `float32`、`float16` 或 `bfloat16`；
- 不对模型输出做额外规则修补；
- 不在 NPU 不支持时自动切回 CPU，避免掩盖适配问题。

##### 性能与精度验证

验证分为三层：

1. 链路验证：使用短音频样本确认模型加载、音频读取、设备迁移、任务提示和文本生成流程可运行。
2. 性能验证：使用 LibriSpeech test-clean，按音频时长排序，先 warmup 再正式计时，统计总音频时长、总耗时、RTF 和 RTFx。
3. 精度验证：使用 MLS 与 FLEURS，分别计算多语种 ASR WER 和多方向 AST BLEU，并与公开参考值做同口径对照。

性能统计中：

```text
RTF = elapsed_seconds / audio_seconds
RTFx = audio_seconds / elapsed_seconds
```

RTF 越低越好，RTFx 越高越好。报告中同时记录 batch size、beam size、计算精度、硬件型号、驱动/CANN/PyTorch 版本，保证结果可复查。

#### 适配结果

##### 性能

硬件：Atlas 800I A2

| 数据集 | 指标 | NPU 结果 | 公开 GPU 参考 |
|---|---|---:|---:|
| LibriSpeech test-clean | RTF | 0.005652242997176402 | 0.0042491714115747425 |

##### 精度

硬件：Atlas 800I A2

| 任务类型 | 语言/方向 | 数据集 | 指标 | NPU 结果 | 公开参考 |
|---|---|---|---|---:|---:|
| ASR | de | Multilingual LibriSpeech | WER(%) | 3.83 | 4.19 |
| ASR | es | Multilingual LibriSpeech | WER(%) | 2.30 | 3.15 |
| ASR | fr | Multilingual LibriSpeech | WER(%) | 3.69 | 4.12 |
| AST | en-de | FLEURS | BLEU | 31.41 | 32.15 |
| AST | en-es | FLEURS | BLEU | 22.69 | 22.66 |
| AST | en-fr | FLEURS | BLEU | 39.84 | 40.76 |
| AST | de-en | FLEURS | BLEU | 33.50 | 33.98 |
| AST | es-en | FLEURS | BLEU | 21.78 | 21.80 |
| AST | fr-en | FLEURS | BLEU | 30.29 | 30.95 |

结果表明，在保持官方权重、官方解码和公开评测口径基本一致的情况下，Canary-1B 可以在昇腾 NPU 上完成 ASR 与 AST 推理适配。多语种 ASR 指标达到或优于公开参考，AST 指标与公开参考整体接近，性能结果也处于可对照范围。

#### 经验总结

1. Canary-1B 的适配重点不是模型结构修改，而是保证 NeMo 官方恢复、prompt、tokenizer、解码和评测链路在 NPU 上完整跑通。
2. 语音模型评测对数据和文本后处理非常敏感，WER/BLEU 必须明确 normalizer、标点大小写、beam size 和 length penalty。
3. NPU 性能测试需要区分模型计算耗时和数据准备耗时，正式计时前应 warmup，并尽量使用按时长排序后的批量输入。
4. `beam_size=1` 和 `beam_size=5` 服务于不同目标：前者适合吞吐评估，后者适合官方精度对齐，不能混用后直接比较指标。
5. 离线部署时应提前准备 `.nemo` 权重、parquet/音频数据和 manifest，推理阶段不应依赖远程下载。
6. 对 NPU 适配问题应显式失败并暴露原始错误，避免用 CPU fallback、简化 normalizer 或替代指标掩盖真实兼容性问题。

#### 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 模型权重 | NVIDIA Canary-1B Hugging Face 模型仓 | https://huggingface.co/nvidia/canary-1b |
| 开源代码仓 | NVIDIA NeMo 源码 | https://github.com/NVIDIA-NeMo/NeMo |
| 公开性能参考 | Hugging Face Open ASR Leaderboard | https://github.com/huggingface/open_asr_leaderboard |
| 数据集 | LibriSpeech | https://www.openslr.org/12 |
| 数据集 | FLEURS | https://huggingface.co/datasets/google/fleurs |
| 数据集 | Multilingual LibriSpeech | https://huggingface.co/datasets/facebook/multilingual_librispeech |

## 2. 上游版本与代码分析

### Canary-1B NPU 适配分析

#### 1. 上游信息

- 上游仓库：<https://github.com/NVIDIA-NeMo/NeMo.git>
- 分支：`main`
- 基准 commit：`44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`
- commit 信息：`ci: remove build-docs and build-test-publish-wheel workflows (#15685)`
- 检查日期：2026-05-23
- 模型权重：<https://huggingface.co/nvidia/canary-1b>
- 版本边界：当前适配的是原始 `nvidia/canary-1b` / `canary-1b.nemo`；不包含 `nvidia/canary-1b-flash` 或 `nvidia/canary-1b-v2`。已验证权重 SHA256：`b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a`。
- 本地上游副本：`Canary-1B/upstream/`（已通过 `git clone --depth 1` 获取）

#### 2. 当前目录状态

当前 `Canary-1B/` 原有文件：

- `infer.py`：原始推理 demo，依赖 `torch_npu.contrib.transfer_to_npu`，并包含硬编码音频/缓存路径。
- `README.md`：NPU 运行说明，但要求手工移动脚本并修改路径。
- `requirements.txt`：当前环境导出的依赖，范围明显大于 Canary/NeMo ASR 推理最小依赖。

本次新增/调整：

- `infer.py`：改为当前适配目录维护的参数化 CPU/NPU 融合推理脚本，默认 `--device npu`。
- `patches/README.md`：说明本次没有上游源码 patch。
- `NPU_ADAPTATION.md`：整合后的适配分析、迁移说明、验证记录、评测方案和案例总结。
- `.gitignore`：加入 `Canary-1B/upstream/`。

#### 3. 与上游匹配情况

Canary-1B 通过 NeMo `EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')` 加载。上游 `nemo/collections/asr/models/aed_multitask_models.py` 的推理链路使用 `trcfg._internal.device`、`tensor.to(device)` 和模型自身 device 传递，未发现必须为 Canary 单独修改上游源码的 `.cuda()` 硬编码节点。

因此本次适配不修改 NeMo 上游已有文件，不生成 `.patch`；交付当前模型目录新增的 `infer.py`、评测/数据准备脚本和整合文档。后续如发现某个 NeMo 版本在 Canary 推理链路中新增硬编码 CUDA/NCCL 节点，应先在 `Canary-1B/upstream/` 对应文件修改，再生成 patch。

#### 4. 现有代码审视

| 文件 | 结论 | 说明 |
|---|---|---|
| `infer.py` | 已重写 | 默认 NPU，支持 `--device cpu` 验证；无 `auto/use_gpu`；不写死 `npu:0/cuda:0`；音频、任务、语言和模型路径参数化。 |
| `README.md` | 已更新 | 补充基准 commit、无需 patch、运行方式和验证方式。 |
| `requirements.txt` | 保留但不建议作为最小依赖 | 包含 CUDA/服务端/训练相关大量依赖，正式部署建议按 README 中最小依赖安装。 |
| `patches/` | 无上游 patch | 因未修改 NeMo 上游已有文件，仅保留 README 说明。 |
| `prepare_eval_data.py` / `eval_canary.py` | 已新增 | 提供评测数据准备和评测脚本。 |

#### 5. 设备适配点

1. `infer.py::_resolve_device`：仅当 `--device npu` 时导入 `torch_npu` 注册后端；返回 `torch.device('npu')`，不绑定卡号。
2. `EncDecMultiTaskModel.from_pretrained(..., map_location=device)`：加载时按目标设备映射权重。
3. `model.to(device)`：显式迁移模型。
4. `model.transcribe(...)`：输入通过 manifest 显式传入 `taskname/source_lang/target_lang/pnc`，由 NeMo dataloader 和模型内部 device 机制处理 batch。

#### 6. 风险与限制

- 当前未在本机真实 NPU 上执行端到端推理；已完成静态检查、`py_compile`、CPU 环境搭建和 CPU 推理启动验证。
- 已通过 HF 镜像下载 `canary-1b.nemo`，并完成当前环境 CPU smoke test，输出 `[0]  I'm a part of that.`。
- Canary-1B 约 1B 参数，NPU 显存、CANN/torch-npu/torch 版本需要匹配。
- NeMo 主分支持续变化；如果上游更新，应重新检查 `EncDecMultiTaskModel`、`ASRTranscriptionMixin` 和音频预处理链路。
- `requirements.txt` 非最小依赖，可能引入无关 CUDA 包；部署时优先安装与当前 CANN/torch-npu 匹配的 PyTorch、torch-npu 和 NeMo ASR 依赖。

#### 7. 上游版本检查记录

- 2026-05-23：重新执行 `git clone --depth 1 https://github.com/NVIDIA-NeMo/NeMo.git Canary-1B/upstream` 成功。
- 2026-05-23：`git -C Canary-1B/upstream rev-parse HEAD` 输出 `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`。
- 2026-05-23：`git -C Canary-1B/upstream ls-remote origin refs/heads/main` 确认远端 `main` 同为 `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`。
- 2026-05-23：检查 `nemo/collections/asr/models/aed_multitask_models.py`，确认 Canary 主要推理代码使用 device 传递，无需 patch。

## 3. NPU 适配与运行说明

### Canary-1B NPU 适配说明

#### 1. 适配目标

将 NVIDIA NeMo Canary-1B 推理样例整理为规范的 CPU/NPU 融合脚本：

- 默认使用 `--device npu`；
- CPU 验证显式使用 `--device cpu`；
- 不使用 `auto` / `use_gpu`；
- 不在代码中写死 `npu:0` / `cuda:0`；
- 实际 NPU 卡号由环境变量控制，例如 `ASCEND_RT_VISIBLE_DEVICES=0`。

#### 2. 上游与 patch

- 上游仓库：<https://github.com/NVIDIA-NeMo/NeMo.git>
- 基准 commit：`44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`
- 本次没有修改上游已有文件，因此没有 `.patch` 文件。
- `Canary-1B/infer.py` 是当前适配新增脚本，不进入 patch。

如后续上游源码需要修改，则在 `Canary-1B/upstream/` 内生成 patch：

```bash
git -C Canary-1B/upstream diff -- <upstream_existing_file> > Canary-1B/patches/0001-xxx.patch
git -C Canary-1B/upstream apply --check ../patches/0001-xxx.patch
```

#### 3. 环境准备

##### 3.1 CPU 验证环境

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

##### 3.2 NPU 推理环境

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

#### 4. 权重下载

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

#### 5. 官方参考指标与评测口径

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

#### 6. 测试数据准备

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

##### 6.1 MLS / LibriSpeech / FLEURS 评测数据准备

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

输出 manifest 和 metadata；CPU/NPU 评测必须复用同一份 manifest。详细手动下载命令见本文“MLS / LibriSpeech / FLEURS 验证测试方案”章节。

#### 7. 推理脚本用法

##### CPU ASR 验证

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

##### NPU ASR

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

##### 语音翻译 AST

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

#### 8. 上游更新处理

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

## 4. 验证记录

### Canary-1B NPU 验证记录

#### 1. 静态验证

检查日期：2026-05-23

```bash
find Canary-1B -maxdepth 3 -type f | sort
git status --short
Canary-1B/.venv-cpu/bin/python -m py_compile Canary-1B/infer.py
```

结果：`py_compile` 通过。

#### 2. 上游 clone 验证

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

#### 3. patch 验证

本次适配未修改 NeMo 上游已有文件，因此没有 `.patch` 文件需要 `git apply --check`。

如后续新增 patch，执行：

```bash
for p in Canary-1B/patches/*.patch; do
  git -C Canary-1B/upstream apply --check "../patches/$(basename "$p")"
done
```

#### 4. CPU 环境准备验证

当前系统缺少 `python3-pip` / `ensurepip`，已使用 `uv` 创建 CPU 验证虚拟环境：

##### 4.1 依赖文件关系说明

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

##### 4.2 CPU 依赖安装记录

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

#### 5. 权重下载验证

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

#### 6. 测试数据准备验证

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

##### 6.1 MLS / LibriSpeech / FLEURS 评测数据准备脚本验证

当前脚本已补齐在线/离线混合参数：

```bash
Canary-1B/.venv-cpu/bin/python -m py_compile Canary-1B/prepare_eval_data.py

### 离线缺失检查示例：应报出缺失的本地 MLS/LibriSpeech/FLEURS parquet 路径，不访问远端
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

完整在线/离线/手动下载命令见本文“MLS / LibriSpeech / FLEURS 验证测试方案”章节。

#### 7. 当前环境 CPU 推理验证

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

#### 8. NPU 功能验证命令

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

#### 9. AST 验证命令

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

#### 10. 当前限制

- 当前环境没有 NPU，未执行 NPU 端到端验证。
- 已通过 HF 镜像下载权重并完成 CPU smoke test；若切换网络或镜像，需重新校验 SHA256。
- Canary-1B 约 1B 参数，CPU 推理即使权重下载完成也可能较慢；建议使用短音频进行 smoke test。

#### 11. 完整验收方案

当前 `dummy_1s_16k.wav` 仅用于 smoke test，不能证明 Canary-1B 的功能完整性、性能或精度。完整验收请执行 `ACCEPTANCE_PLAN.md`，至少覆盖：

- ASR：英语、德语、西班牙语、法语；
- AST：英语 ↔ 德语/西班牙语/法语 6 个方向；
- PnC：`yes/no`；
- batch：`1/4/8` 或记录最大可用 batch；
- 精度：ASR WER、AST BLEU；
- 性能：RTF/RTFx、加载时间、峰值内存/HBM、连续运行稳定性。

正式验收报告建议保存到 `Canary-1B/validation_reports/`，模板见 `ACCEPTANCE_PLAN.md`。

## 5. MLS / LibriSpeech / FLEURS 验证测试方案

### Canary-1B MLS / LibriSpeech / FLEURS 验证测试方案

按要求将流程拆成两步：

1. **准备数据**：`prepare_eval_data.py` 只负责下载数据、转 16 kHz wav、写 JSONL manifest。
2. **评测**：`eval_canary.py` 只读取已准备好的 manifest，使用与 `infer.py` 相同的 NeMo `model.transcribe()` 机制做推理，再计算 WER/BLEU。

这样 CPU/CUDA/NPU 评测可以复用同一份 wav 和 manifest，避免每次评测重复下载或抽样不一致。

#### 0. 官方参考指标

来源：

- NVIDIA Canary-1B model card：<https://huggingface.co/nvidia/canary-1b>
- Hugging Face Open ASR Leaderboard：<https://hf-audio-open-asr-leaderboard.hf.space/>
- Open ASR Leaderboard 代码/说明：<https://github.com/huggingface/open_asr_leaderboard>

##### 0.1 官方精度数据

NVIDIA model card 说明 ASR/AST 公开结果使用 `beam width=5`、`length penalty=1.0`。ASR 使用 WER，并用 whisper-normalizer 归一化参考和预测文本；AST 使用 BLEU，并保留数据集原始标点和大小写。

| 任务 | 数据集 | 指标 | 官方参考 |
|---|---|---|---|
| ASR | MCV-16.1 test | WER | En 7.97 / De 4.61 / Es 3.99 / Fr 6.53 |
| ASR | MLS test | WER | En 3.06 / De 4.19 / Es 3.15 / Fr 4.12 |
| AST | FLEURS test | BLEU | En→De 32.15 / En→Es 22.66 / En→Fr 40.76 / De→En 33.98 / Es→En 21.80 / Fr→En 30.95 |
| AST | CoVoST-v2 test | BLEU | De→En 37.67 / Es→En 40.70 / Fr→En 40.42 |
| AST | mExpresso test | BLEU | En→De 23.84 / En→Es 35.74 / En→Fr 28.29 |

##### 0.2 公开性能数据

原始 `nvidia/canary-1b` model card 没有单独发布硬件延迟/吞吐表。当前可引用的公开性能参考是 Hugging Face Open ASR Leaderboard 的 RTFx。该榜单说明开源模型评测在 NVIDIA A100-SXM4-80GB GPU、CUDA 12.6、PyTorch 2.4.0 下运行，batch size 尽量使用 64，显存不足时自适应降低。

截至 2026-05-26，`nvidia/canary-1b` 公开参考为：

| 指标 | 值 |
|---|---:|
| Average WER | 6.50 |
| RTFx | 235.34 |
| AMI WER | 13.90 |
| Earnings22 WER | 12.19 |
| GigaSpeech WER | 10.12 |
| LibriSpeech clean WER | 1.48 |
| LibriSpeech other WER | 2.93 |
| SPGISpeech WER | 2.06 |
| Tedlium WER | 3.56 |
| VoxPopuli WER | 5.79 |

上述 RTFx 只作为公开 GPU 量级参考，不是 NPU 通过线。本仓库评测输出的 `elapsed_seconds`、`rtf` 可换算 `RTFx = audio_seconds / elapsed_seconds`，并应和 `beam_size`、`batch_size`、设备、峰值内存一起记录。

#### 1. 前置依赖

```bash
pip install datasets soundfile librosa tqdm jiwer sacrebleu openai-whisper
```

- 数据准备需要：`datasets soundfile librosa tqdm`。
- 评测需要：`jiwer sacrebleu openai-whisper`。ASR WER 固定走官方 Whisper `EnglishTextNormalizer` 路径（`from whisper.normalizers import EnglishTextNormalizer`）；依赖缺失或导入失败时会在脚本启动导入阶段直接抛出原始异常。仅安装 `whisper_normalizer` 不视为满足官方路径，且不使用本地 fallback normalizer。
- 如 Hugging Face 访问慢，可设置 `HF_ENDPOINT` / `HF_HOME`；但评测数据推荐使用下面的显式本地目录参数，便于离线迁移。

##### 1.1 评测脚本 import / 依赖规范

项目级流程规范详见根目录《模型NPU 适配标准流程.md》的“项目级脚本严格失败原则”。`Canary-1B/eval_canary.py` 作为本模型评测入口必须遵守该项目级规范：

1. 除设备后端探测类 import（例如仅 `--device npu` 才需要的 `torch_npu`）外，评测依赖统一放在文件顶部导入，禁止在 metric 计算阶段临时 import 后再 fallback。
2. ASR WER 只能使用官方路径 `from whisper.normalizers import EnglishTextNormalizer`；不得改用 `whisper_normalizer` 包、regex/basic normalizer 或其他静默替代实现。
3. 任一必需依赖缺失时脚本应直接失败并暴露原始异常；不要用宽泛 `try/except` 包装成兼容路径，不要吞掉异常，不要继续推理后再给出不可对齐官方口径的指标。
4. 对 NeMo 配置、版本字段和解码配置使用当前官方预期字段；字段缺失表示环境或版本不匹配，应立即报错，不添加 `hasattr/getattr` 式静默兼容。

#### 2. 准备数据

> 当前 `prepare_eval_data.py` 已支持在线/离线混合模式：ASR 精度使用 `--asr_parquet_dir` 指定 `facebook/multilingual_librispeech` parquet 保存目录，性能测试保留 LibriSpeech `test-clean`，使用 `--librispeech_dir` 指定 OpenSLR tar/解压目录，FLEURS 使用 `--fleurs_parquet_dir` 指定 parquet 保存目录；目标文件已存在时直接复用，缺失时在线下载到该目录，`--offline` 下缺失则直接报具体路径且不联网。
>
> MLS/LibriSpeech/FLEURS 都不再依赖 `torchcodec` 自动解码：脚本将 HF `Audio` 列 cast 为 `decode=False`，再用 `soundfile` 读取 bytes/path 写 16 kHz wav。

##### 2.0 推荐目录结构

```text
Canary-1B/eval_data/mls_parquet/
  german/test-00000-of-00001.parquet
  spanish/test-00000-of-00001.parquet
  french/test-00000-of-00001.parquet
Canary-1B/eval_data/librispeech_raw/
  test-clean.tar.gz
  LibriSpeech/test-clean/
Canary-1B/eval_data/fleurs_parquet/
  en_us/test-00000-of-00001.parquet
  de_de/test-00000-of-00001.parquet
  es_419/test-00000-of-00001.parquet
  fr_fr/test-00000-of-00001.parquet
```

MLS 日志应看到 `loading local MLS parquet: ...` 或 `downloading MLS parquet to ...`；LibriSpeech 日志应看到 `using existing LibriSpeech directory/archive` 或 `downloading LibriSpeech test-clean to ...`；FLEURS 日志应看到 `loading local FLEURS parquet: ...` 或 `downloading FLEURS parquet to ...`。

##### 2.1 最小验收数据：MLS 30 分钟 + LibriSpeech test-clean 30 分钟 + FLEURS 每方向 50 条

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

生成 manifest；同时生成同名 `.meta.json`。MLS 的 `dataset` 应为 `facebook/multilingual_librispeech`，LibriSpeech 用于性能测试并记录 `purpose`：

```text
Canary-1B/eval_data/librispeech_test_clean/manifest_asr_en.jsonl
Canary-1B/eval_data/mls_test_german/manifest_asr_de.jsonl
Canary-1B/eval_data/mls_test_spanish/manifest_asr_es.jsonl
Canary-1B/eval_data/mls_test_french/manifest_asr_fr.jsonl
Canary-1B/eval_data/fleurs/en-de/manifest_ast_en_de.jsonl
Canary-1B/eval_data/fleurs/en-es/manifest_ast_en_es.jsonl
Canary-1B/eval_data/fleurs/en-fr/manifest_ast_en_fr.jsonl
Canary-1B/eval_data/fleurs/de-en/manifest_ast_de_en.jsonl
Canary-1B/eval_data/fleurs/es-en/manifest_ast_es_en.jsonl
Canary-1B/eval_data/fleurs/fr-en/manifest_ast_fr_en.jsonl
```

##### 2.2 准备 ASR MLS test + LibriSpeech test-clean 全量

```bash
python Canary-1B/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --asr_split test \
  --asr_minutes 0 \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --asr_pnc no
```

##### 2.3 只准备 MLS ASR test 全量（不含性能用 LibriSpeech）

```bash
python Canary-1B/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --asr_split test \
  --asr_minutes 0 \
  --no-include_librispeech_test_clean \
  --asr_pnc no
```

##### 2.4 只准备性能测试用 LibriSpeech test-clean 全量

```bash
python Canary-1B/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --asr_configs "" \
  --include_librispeech_test_clean \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --librispeech_minutes 0 \
  --asr_pnc no
```

##### 2.5 只准备 AST FLEURS test 全量

```bash
python Canary-1B/prepare_eval_data.py \
  --task ast \
  --data_dir Canary-1B/eval_data \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --fleurs_split test \
  --fleurs_limit 0 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes
```

##### 2.6 离线复用本地数据

当上述目录已经由在线脚本或手动命令准备好后，离线环境使用同一命令加 `--offline`：

```bash
python Canary-1B/prepare_eval_data.py \
  --task all \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --offline \
  --asr_configs german,spanish,french \
  --asr_minutes 30 \
  --fleurs_split test \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en
```

离线模式不会访问 Hugging Face/OpenSLR；缺失文件会直接报类似：

```text
Offline mode enabled and MLS parquet is missing: .../german/test-00000-of-00001.parquet
Offline mode enabled and LibriSpeech data is missing: .../LibriSpeech/test-clean or .../test-clean.tar.gz
Offline mode enabled and FLEURS parquet is missing: .../en_us/test-00000-of-00001.parquet
```

##### 2.7 手动命令行下载到脚本指定目录

MLS ASR 三种语言 test parquet：

```bash
mkdir -p Canary-1B/eval_data/mls_parquet/{german,spanish,french}

curl -L -o Canary-1B/eval_data/mls_parquet/german/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/german/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/mls_parquet/spanish/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/spanish/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/mls_parquet/french/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/french/test-00000-of-00001.parquet
```

FLEURS 四种语言 test parquet：

```bash
mkdir -p Canary-1B/eval_data/fleurs_parquet/{en_us,de_de,es_419,fr_fr}

curl -L -o Canary-1B/eval_data/fleurs_parquet/en_us/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/en_us/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/fleurs_parquet/de_de/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/de_de/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/fleurs_parquet/es_419/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/es_419/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/fleurs_parquet/fr_fr/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/fr_fr/test-00000-of-00001.parquet
```

手动下载后再运行第 2.6 节 `--offline` 命令，脚本会直接复用本地文件，不重复下载。

#### 3. 评测

评测脚本默认读取第 2.1 节的标准 manifest 列表；也可以用 `--manifest` 显式指定一个或多个 manifest。

##### 3.1 `beam_size` / `batch_size` 选择

- `beam_size` 是 Transformer decoder 的 beam search 宽度，不是 batch 大小：
  - `beam_size=1` 等价于 greedy decode，只保留 1 条候选，速度最快，适合 smoke test、吞吐测试和日常调试。
  - `beam_size=5` 每步保留 5 条候选，通常精度更好，但 decoder 计算量和显存占用都会增加。
- NVIDIA Canary-1B model card 的公开 ASR/AST 精度表使用 `beam width=5`、`length penalty=1.0`；因此正式精度对齐建议使用 `--beam_size 5`。
- NVIDIA model card 的普通 transcribe 示例使用 `batch_size=16`；本地 NPU/CUDA 性能评测应优先尝试 `--batch_size 16`，如显存不足再降到 `8/4/2/1`。
- `batch_size=1 + beam_size=5` 是最保守但很慢的组合，适合小规模 CPU/NPU 精度对齐，不适合完整吞吐评测。CPU 全量评测尤其慢，建议只做 smoke test 或小子集基线。

推荐参数：

| 场景 | 推荐参数 | 说明 |
|---|---|---|
| 精度对齐公开指标 | `--beam_size 5 --batch_size 16` | OOM 时将 batch 依次降到 `8/4/2/1` |
| NPU/CUDA 吞吐测试 | `--performance_mode --beam_size 1 --batch_size 64/128` | 对齐 Open ASR Leaderboard 计时口径：duration 降序、warmup、audio list、bf16 |
| CPU 小子集基线 | `--beam_size 5 --batch_size 1` | 仅用于精度口径一致；全量会很慢 |
| 快速 smoke test | `--beam_size 1 --batch_size 1` | 只验证链路是否跑通 |

##### 3.2 一次评测全部已准备任务（推荐：NPU 精度模式）

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_all_bs16_beam5
```

如出现 OOM，保持 `--beam_size 5` 不变，优先下调 `--batch_size 8/4/2/1`。

##### 3.3 NPU 吞吐/速度模式

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --manifest Canary-1B/eval_data/librispeech_test_clean/manifest_asr_en.jsonl \
  --performance_mode \
  --batch_size 64 \
  --beam_size 1 \
  --output_dir Canary-1B/eval_results/npu_librispeech_test_clean_perf_bs64_beam1
```

`--performance_mode` 是专门的性能计时路径，用于尽量贴近 Hugging Face Open ASR Leaderboard 的 NeMo 评测方式：按音频时长降序排序，先 warmup `--warmup_batches` 个 batch，正式计时只统计完整音频列表转写，默认在 NPU/CUDA 上使用 `bfloat16`，并向 Canary audio-list `transcribe()` 传入 `pnc=nopnc`、`num_workers=1`。该模式会额外输出 `rtfx = audio_seconds / elapsed_seconds`。如需强制精度类型，可显式传 `--compute_dtype float32|float16|bfloat16`。

性能模式下 `--beam_size 1` 默认使用 NeMo AED `greedy_batch` 解码策略，避免仍走 beam decoder 的额外开销。如需强制使用 beam decoder 做完全可比实验，可加 `--decoding_strategy beam`。

##### 3.4 只评测 ASR

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --manifest \
    Canary-1B/eval_data/mls_test_german/manifest_asr_de.jsonl \
    Canary-1B/eval_data/mls_test_spanish/manifest_asr_es.jsonl \
    Canary-1B/eval_data/mls_test_french/manifest_asr_fr.jsonl \
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_asr_mls_bs16_beam5
```

##### 3.5 只评测 FLEURS AST 六个方向

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --manifest \
    Canary-1B/eval_data/fleurs/en-de/manifest_ast_en_de.jsonl \
    Canary-1B/eval_data/fleurs/en-es/manifest_ast_en_es.jsonl \
    Canary-1B/eval_data/fleurs/en-fr/manifest_ast_en_fr.jsonl \
    Canary-1B/eval_data/fleurs/de-en/manifest_ast_de_en.jsonl \
    Canary-1B/eval_data/fleurs/es-en/manifest_ast_es_en.jsonl \
    Canary-1B/eval_data/fleurs/fr-en/manifest_ast_fr_en.jsonl \
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_ast_fleurs_bs16_beam5
```

#### 4. CPU/CUDA/NPU 对比

准备数据只跑一次。之后三种设备分别运行评测脚本，保持同一批 manifest 和同一解码参数。精度对齐时固定 `--beam_size 5`；性能对比时可另外跑 `--beam_size 1`。

```bash
### CPU 小子集/保守基线。全量会很慢，不建议作为吞吐路径。
python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device cpu \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/cpu_all

### NPU 精度模式。OOM 时只下调 batch_size，保持 beam_size=5。
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_all_bs16_beam5

### NPU 吞吐模式。
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 16 \
  --beam_size 1 \
  --output_dir Canary-1B/eval_results/npu_all_bs16_beam1
```

对比：

- ASR：`*_metrics.json` 中的 `wer_percent`。
- AST：`*_metrics.json` 中的 `bleu`。
- 性能：`elapsed_seconds`、`rtf`。

#### 5. 通过条件

##### ASR：MLS test

- 最小规模：30 分钟。
- 推荐规模：全量约 5 小时。
- 主通过条件：同一数据、同一脚本、同一 `beam_size=5` 下，NPU WER 相对 CPU/CUDA 不劣化。
- 若直接对公开值，`WER <= 公开值 + max(公开值 * 10%, 0.5)`。

##### AST：FLEURS En↔De/Es/Fr

- 最小规模：每方向 50 条。
- 推荐规模：FLEURS test 全量。
- 主通过条件：同一数据、同一脚本、同一 `beam_size=5` 下，NPU BLEU 相对 CPU/CUDA 下降 ≤ 0.5。
- 若直接对公开值，`BLEU >= 公开值 - max(公开值 * 10%, 1.0)`。

FLEURS 公开参考：

| 方向 | BLEU |
|---|---:|
| En→De | 32.15 |
| En→Es | 22.66 |
| En→Fr | 40.76 |
| De→En | 33.98 |
| Es→En | 21.80 |
| Fr→En | 30.95 |

#### 6. 输出文件

评测输出目录中包含：

- `<tag>.tsv`：逐样本 `sample_id / audio_path / duration / reference / hypothesis`。
- `<tag>.metrics.json`：单个 manifest 指标。
- `summary.metrics.json`：汇总指标。
- `run_env.json`：Python、torch、NeMo、设备和命令行参数记录。
- `*.jsonl.meta.json`：数据准备元信息，包含 dataset/config/split/limit、本地数据目录和 `offline`，便于确认 FLEURS 使用的是 `test` split 且复用同一批本地文件。
