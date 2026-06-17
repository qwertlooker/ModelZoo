# Canary-1B 昇腾 NPU 适配案例总结

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

## 概述

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

## 输入输出数据

- 输入数据

  输入为 16 kHz 单声道音频，格式可为 wav、flac 等本地音频文件。批量评测使用 JSONL manifest 组织样本，每行描述一个音频样本及其任务信息。

  ASR 任务的关键字段包括音频路径、音频时长、参考文本、源语言、目标语言、任务类型和是否保留标点大小写。AST 任务在此基础上将目标语言设置为翻译输出语言。

- 输出数据

  输出为模型生成的文本。ASR 输出为源语音对应的转写文本；AST 输出为目标语言翻译文本。评测阶段会同时保存预测文本、参考文本、样本级耗时、数据集级指标和运行环境信息。

## 适配环境

本案例使用的 NPU 推理环境配套如下。

| 配套 | 版本 |
|---|---|
| 固件与驱动 | 25.5.1+ |
| CANN | 8.5.1 |
| Python | 3.11.14 |
| PyTorch / torch_npu | 2.9.0 |
| torchaudio | 2.9.0 |

环境准备的核心原则是：PyTorch、torch_npu 与 CANN 版本必须匹配；NeMo ASR 依赖按官方 extra 安装；音频解码、文本指标和分词相关依赖作为必需依赖显式安装。缺失依赖时直接暴露原始错误，不增加静默兼容层或自动降级路径。

## 适配工作说明

本次适配大致完成了以下工作：

1. 固定 NeMo 上游 commit 和 Canary-1B 原始权重，避免因上游接口变动导致适配结果不可复现。
2. 梳理 Canary-1B 的 ASR/AST 输入字段、语言代码、PnC 开关和解码参数，保持与官方任务定义一致。
3. 增加 NPU 设备选择与模型迁移逻辑，使用 `ASCEND_RT_VISIBLE_DEVICES` 控制实际卡号，不在代码中写死 `npu:0`。
4. 将推理过程整理为无梯度、评估模式、批量输入的执行链路，支持单音频验证和 manifest 批量评测。
5. 在性能模式中使用按音频时长排序、warmup、正式计时、`bfloat16` 计算和 RTF/RTFx 统计，便于与公开性能口径对照。
6. 在精度模式中保留官方解码路径：`beam_size=5`、`length_penalty=1.0`，ASR 使用 WER，AST 使用 BLEU。
7. 准备 LibriSpeech、Multilingual LibriSpeech 和 FLEURS 三类评测数据，分别覆盖英文 ASR 性能、多语种 ASR 精度和多方向 AST 精度。
8. 输出可追溯的评测结果，包括运行环境、逐样本预测、数据集指标汇总和性能统计。

## 适配实施过程

### 模型与上游版本确认

适配前先确认模型权重与上游代码的匹配关系。Canary-1B 的 `.nemo` 文件内部包含模型结构配置、tokenizer 信息和训练时的任务提示配置，因此适配时不应手工改写模型拓扑，也不应替换 tokenizer 或 prompt formatter。

关键确认点包括：

- 使用原始 `nvidia/canary-1b` 权重文件 `canary-1b.nemo`；
- 校验权重 SHA256，保证 CPU/NPU 使用同一 checkpoint；
- 固定 NeMo commit，确保模型恢复、音频前处理和解码接口稳定；
- 明确本案例不覆盖 Canary Flash 或 Canary v2，避免混用不同模型族的配置和指标。

### 权重与依赖准备

权重准备采用本地 `.nemo` 文件方式。这样做可以避免推理时隐式访问公网，也便于在离线 NPU 环境中复现。权重目录中只需要保存原始 `.nemo` 文件，不需要展开或转换为其它格式。

依赖准备重点如下：

- PyTorch 与 torch_npu 必须使用与 CANN 匹配的版本；
- NeMo 使用 ASR extra，保证模型恢复、音频特征、tokenizer、解码器和指标相关组件完整；
- `soundfile`、`librosa`、`sentencepiece`、`jiwer`、`sacrebleu`、`openai-whisper` 等作为评测链路依赖；
- NPU 后端注册只在 NPU 路径中引入，CPU 验证路径不要求安装 torch_npu；
- 不使用 CPU fallback、第三方近似 normalizer 或简化 BLEU/WER 逻辑替代官方评测路径。

### 数据与评测口径准备

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

### NPU 推理链路适配

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

### 性能与精度验证

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

## 适配结果

### 性能

硬件：Atlas 800I A2

| 数据集 | 指标 | NPU 结果 | 公开 GPU 参考 |
|---|---|---:|---:|
| LibriSpeech test-clean | RTF | 0.005652242997176402 | 0.0042491714115747425 |

### 精度

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

## 经验总结

1. Canary-1B 的适配重点不是模型结构修改，而是保证 NeMo 官方恢复、prompt、tokenizer、解码和评测链路在 NPU 上完整跑通。
2. 语音模型评测对数据和文本后处理非常敏感，WER/BLEU 必须明确 normalizer、标点大小写、beam size 和 length penalty。
3. NPU 性能测试需要区分模型计算耗时和数据准备耗时，正式计时前应 warmup，并尽量使用按时长排序后的批量输入。
4. `beam_size=1` 和 `beam_size=5` 服务于不同目标：前者适合吞吐评估，后者适合官方精度对齐，不能混用后直接比较指标。
5. 离线部署时应提前准备 `.nemo` 权重、parquet/音频数据和 manifest，推理阶段不应依赖远程下载。
6. 对 NPU 适配问题应显式失败并暴露原始错误，避免用 CPU fallback、简化 normalizer 或替代指标掩盖真实兼容性问题。

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 模型权重 | NVIDIA Canary-1B Hugging Face 模型仓 | https://huggingface.co/nvidia/canary-1b |
| 开源代码仓 | NVIDIA NeMo 源码 | https://github.com/NVIDIA-NeMo/NeMo |
| 公开性能参考 | Hugging Face Open ASR Leaderboard | https://github.com/huggingface/open_asr_leaderboard |
| 数据集 | LibriSpeech | https://www.openslr.org/12 |
| 数据集 | FLEURS | https://huggingface.co/datasets/google/fleurs |
| 数据集 | Multilingual LibriSpeech | https://huggingface.co/datasets/facebook/multilingual_librispeech |
