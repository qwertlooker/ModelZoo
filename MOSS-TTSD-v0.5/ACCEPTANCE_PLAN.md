# MOSS-TTSD-v0.5 完整验收方案

本文记录 MOSS-TTSD-v0.5 基于昇腾 NPU 的分层验收方案、数据准备、功能矩阵、质量/性能指标、稳定性要求、命令模板和报告模板。推理指导见 `README_INFERENCE.md`，适配实现与验证事实见 `NPU_ADAPTATION.md`。

## 目录

- [0. 验收目标与范围](#0-验收目标与范围)
- [0.1 权重下载与固定](#01-权重下载与固定)
- [0.2 Attention 后端与 flash-attn / TorchCodec 约束](#02-attention-后端与-flash-attn--torchcodec-约束)
- [1. 原始模型能力与验收覆盖](#1-原始模型能力与验收覆盖)
- [2. 验收环境与前置检查](#2-验收环境与前置检查)
- [3. 数据集选择与 JSONL 规范](#3-数据集选择与-jsonl-规范)
- [4. 功能验收](#4-功能验收)
- [5. 精度与质量验收](#5-精度与质量验收)
- [6. 性能验收](#6-性能验收)
- [7. 稳定性与异常验收](#7-稳定性与异常验收)
- [8. 命令模板汇总](#8-命令模板汇总)
- [9. 报告模板](#9-报告模板)
- [10. 最终准入标准](#10-最终准入标准)

## 0. 验收目标与范围

本方案用于验收 `OpenMOSS/MOSS-TTSD` tag `v0.5` 原项目代码在应用 `patches/0001-adapt-v0.5-inference-to-npu.patch` 后的昇腾 NPU 推理结果。验收不只看是否生成 WAV，而是覆盖原始模型声明的对话式双说话人 TTS/TTSD 能力、NPU 设备适配正确性、端到端性能、可懂度、音色保持、自然度和人工听测。

**模型边界**

- 源码：`OpenMOSS/MOSS-TTSD` tag `v0.5` / commit `0e078c62389922d3aa873ce182daf31142860b18`。
- 模型：`fnlp/MOSS-TTSD-v0.5`（HF <https://huggingface.co/fnlp/MOSS-TTSD-v0.5>）或同内容别名 `OpenMOSS-Team/MOSS-TTSD-v0.5`；ModelScope 地址为 <https://modelscope.cn/models/openmoss/MOSS-TTSD-v0.5>。本次记录 HF HEAD `8527b9136b6afefe2252ae597cecea2e80e7ebeb`、ModelScope HEAD `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59`。
- Codec：原项目 `XY_Tokenizer` + `fnlp/XY_Tokenizer_TTSD_V0` 的 `xy_tokenizer.ckpt`（HF <https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0>；ModelScope <https://modelscope.cn/models/openmoss/XY_Tokenizer_TTSD_V0>）；本次记录 HF HEAD `c83433728e698ed0698e88cb5096bc221fb8f8c5`、ModelScope HEAD `79082154409f5e883d9487c4d4b4be363323b039`。
- 原项目入口：`upstream/inference.py`、`upstream/generation_utils.py`、`upstream/modeling_asteroid.py`、`upstream/XY_Tokenizer/`；不新增旁路推理脚本。
- 支持输入：JSONL，每行至少包含 `text`、`prompt_audio_speaker1`、`prompt_text_speaker1`、`prompt_audio_speaker2`、`prompt_text_speaker2`，可选 `base_path`。
- 支持输出：按样本输出 `output_*.wav`；输出采样率以原项目/codec 实际返回为准，验收报告中必须记录。
- 不包含：MOSS-TTSD v0.7、v1.0、SGLang 路径、未固定版本的一键包改动、服务化部署压测。

**验收分层**

| 层级 | 目的 | 数据规模 | 必跑条件 | 结论用途 |
|---|---|---:|---|---|
| L0 smoke | 验证 patch 后原项目 `inference.py` 可加载模型、使用 NPU 并输出可读 WAV | 1-2 条官方 examples | 每次改动必跑 | 只证明端到端链路可运行 |
| L1 功能回归 | 覆盖中文、英文、中英混合、双说话人、prompt 切换、normalize、长短文本和错误暴露 | 10-30 条 | 每次交付必跑 | 判断功能完整性 |
| L2 推荐性能/质量 | 与 CPU/CUDA 源路径同 checkpoint、同 JSONL、同 seed 对齐；做可懂度/音色/自然度评估 | 50-200 条 | 正式验收必跑 | 判断 NPU 适配是否可接受 |
| L3 完整复现/发布 | 多数据集、多说话人、多语言场景、人工盲测与公开指标报告 | 500+ 条 | 有资源时跑 | 发布级报告和对外声明 |

## 0.1 权重下载与固定

在 `MOSS-TTSD-v0.5/upstream/` 下执行：

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p weights/MOSS-TTSD-v0.5 XY_Tokenizer/weights

hf download fnlp/MOSS-TTSD-v0.5 \
  --revision 8527b9136b6afefe2252ae597cecea2e80e7ebeb \
  --local-dir weights/MOSS-TTSD-v0.5

hf download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt \
  --revision c83433728e698ed0698e88cb5096bc221fb8f8c5 \
  --local-dir XY_Tokenizer/weights
```

ModelScope 可选下载命令（国内镜像；在同一目录下执行）：

```bash
python -m pip install -U modelscope
mkdir -p weights/MOSS-TTSD-v0.5 XY_Tokenizer/weights

modelscope download --model openmoss/MOSS-TTSD-v0.5 \
  --local_dir weights/MOSS-TTSD-v0.5

modelscope download --model openmoss/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt \
  --local_dir XY_Tokenizer/weights
```

codec checkpoint 固定 URL：<https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/resolve/c83433728e698ed0698e88cb5096bc221fb8f8c5/xy_tokenizer.ckpt>。下载后必须把模型权重和 `xy_tokenizer.ckpt` 的 SHA256 写入验收报告：

```bash
cd MOSS-TTSD-v0.5/upstream
find weights/MOSS-TTSD-v0.5 -type f -maxdepth 1 -print0 | sort -z | xargs -0 sha256sum
sha256sum XY_Tokenizer/weights/xy_tokenizer.ckpt
```

通过条件：权重来源、revision、SHA256 可复现；如果使用 ModelScope 镜像或组织别名，必须记录是否与 HF 固定 revision 文件内容一致。

## 0.2 Attention 后端与 `flash-attn` / TorchCodec 约束

`flash-attn` 官方包面向 CUDA/ROCm GPU kernel，当前不作为 Ascend NPU 验收依赖安装。NPU 验收命令固定显式使用 `--attn_implementation sdpa`；如目标 torch-npu 组合不支持 `sdpa`，可显式改为 `eager` 复测并在报告中记录。不得在代码中静默回退，也不得把 CUDA/ROCm `flash_attention_2` 作为 NPU 必需路径。

TorchAudio 2.9+ 的 `torchaudio.load` / `torchaudio.save` 会强依赖 TorchCodec；本适配不把 `torchcodec` 作为 NPU 验收依赖，而是通过 patch 将 prompt 音频读取与 WAV 写出改为 `soundfile`。若日志仍出现 `TorchCodec is required for load_with_torchcodec` 或 `TorchCodec is required for save_with_torchcodec`，视为 patch 未应用或路径未覆盖，需先修正后再验收。

## 1. 原始模型能力与验收覆盖

### 1.1 原始能力拆解

| 能力 | 原项目表现 | 本适配验收要求 |
|---|---|---|
| 对话式 TTS/TTSD | 输入 `[S1]...[S2]...` 形式长对话文本 | 至少验证短句、多轮对话、长文本三类长度 |
| 双说话人 | 每个样本提供 S1/S2 prompt audio/text | 必须验证 S1/S2 均出声，声纹可区分，speaker tag 不丢失 |
| 中文 | 官方 example 包含中文长对话 | L1 至少 2 条；L2 至少 20 条或 10 分钟输出 |
| 英文 | 官方 example 包含英文长对话 | L1 至少 2 条；L2 至少 20 条或 10 分钟输出 |
| 中英混合 | 原模型可处理混合文本 | L1 至少 2 条；需检查中英切换无明显乱码/跳读 |
| prompt 复用 | 支持不同 speaker prompt | L1 至少 2 组 prompt；L2 覆盖男/女、高/低音色 |
| text normalize | `--use_normalize` 默认开启 | L1 同一 JSONL 分别跑开启/关闭或至少覆盖开启路径，数字/英文缩写/标点可解释 |
| 设备选择 | 原项目 CUDA/CPU，patch 增加 NPU | 必须验证 `--device npu` 无 CPU 静默回退、无 CUDA 硬编码 |
| attention backend | 原项目默认 `flash_attention_2` | NPU 固定 `sdpa`，必要时显式 `eager` 复测 |
| 音频 I/O | 原项目使用 torchaudio | patch 后不得触发 TorchCodec 强依赖 |

### 1.2 不可接受的验收方式

- 只检查输出目录里有 WAV 文件，而不检查可读性、时长、采样率和质量。
- 用全零/纯噪声 WAV、短于 0.5 秒的异常输出或明显截断输出通过验收。
- NPU 命令失败后自动改跑 CPU 并宣称 NPU 通过。
- 因缺少 `flash-attn`、`torchcodec`、模型字段或官方依赖而静默 fallback 到非官方路径。
- 只在 1 条官方 example 上得出“功能完整”或“质量通过”结论。

## 2. 验收环境与前置检查

### 2.1 环境记录

验收报告必须记录：

```bash
python -V
pip freeze | grep -E 'torch|torch-npu|torchaudio|transformers|accelerate|soundfile|librosa|numpy|scipy|speechbrain|resemblyzer|wespeaker|funasr|whisper|jiwer|pystoi|pesq|utmos|dnsmos' || true
npu-smi info || true
uname -a
git -C MOSS-TTSD-v0.5/upstream rev-parse HEAD
git -C MOSS-TTSD-v0.5/upstream status --short
sha256sum MOSS-TTSD-v0.5/patches/0001-adapt-v0.5-inference-to-npu.patch
```

最低需记录：运行日期、CANN 版本、驱动/固件版本、NPU 型号、NPU 数量、HBM、Python、torch、torch-npu、transformers、torchaudio、soundfile、源码 commit、patch SHA256、模型权重 SHA256、XY Tokenizer checkpoint SHA256。

### 2.2 patch apply 与源码检查

```bash
git -C MOSS-TTSD-v0.5/upstream reset --hard v0.5
git -C MOSS-TTSD-v0.5/upstream apply --check ../patches/0001-adapt-v0.5-inference-to-npu.patch
git -C MOSS-TTSD-v0.5/upstream apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
python -m py_compile \
  MOSS-TTSD-v0.5/upstream/inference.py \
  MOSS-TTSD-v0.5/upstream/generation_utils.py \
  MOSS-TTSD-v0.5/upstream/gradio_demo.py \
  MOSS-TTSD-v0.5/upstream/podcast_generate.py \
  MOSS-TTSD-v0.5/upstream/modeling_asteroid.py \
  MOSS-TTSD-v0.5/upstream/XY_Tokenizer/inference.py \
  MOSS-TTSD-v0.5/upstream/XY_Tokenizer/utils/helpers.py \
  MOSS-TTSD-v0.5/upstream/XY_Tokenizer/xy_tokenizer/model.py \
  MOSS-TTSD-v0.5/upstream/XY_Tokenizer/xy_tokenizer/nn/quantizer.py
! grep -R -I -E 'torchaudio\.(load|save|info)\(' \
  MOSS-TTSD-v0.5/upstream --exclude-dir=.git
```

通过条件：patch 可干净应用；语法检查通过；除 `torchaudio.functional.resample` 等不触发 TorchCodec 的路径外，不再出现 `torchaudio.load/save/info` 文件 I/O 调用。

### 2.3 依赖安装原则

```bash
cd MOSS-TTSD-v0.5/upstream
pip install torch torch-npu
grep -vE '^flash-attn([<>= ].*)?$' requirements.txt > /tmp/moss-ttsd-v0.5-requirements-npu.txt
pip install -r /tmp/moss-ttsd-v0.5-requirements-npu.txt
pip install -r XY_Tokenizer/requirements.txt
```

项目级脚本严格失败原则适用于本验收：除 NPU 后端注册模块可以按设备条件导入外，必需依赖应前置安装并在缺失时暴露原始错误。不得添加不必要兼容层、静默 fallback、CPU 回退或远程下载回退。

## 3. 数据集选择与 JSONL 规范

### 3.1 JSONL 字段规范

每行样本建议包含：

```json
{
  "id": "case_zh_dialog_001",
  "base_path": "examples",
  "text": "[S1]你好，我们开始测试。[S2]好的，我会用第二个说话人的声音回答。",
  "prompt_audio_speaker1": "zh_spk1_moon.wav",
  "prompt_text_speaker1": "周一到周五，每天早晨七点半到九点半的直播片段。",
  "prompt_audio_speaker2": "zh_spk2_moon.wav",
  "prompt_text_speaker2": "如果大家想听到更丰富更及时的直播内容。",
  "language": "zh",
  "scenario": "short_dialog",
  "expected_speakers": 2
}
```

原项目只要求核心字段；`id/language/scenario/expected_speakers` 等元数据可供验收统计使用。prompt 音频路径相对于 `base_path` 或 JSONL 所在目录时必须在报告中说明。

### 3.2 推荐数据组合

| 数据 | 覆盖 | 规模/难度 | 用途 | 建议 |
|---|---|---:|---|---|
| 原项目 `examples/examples.jsonl` | 官方中文/英文长对话、双说话人 prompt | 2 条，低 | L0 | 必跑，但不能作为完整质量依据 |
| 自建短句 JSONL | 加载、短文本、speaker tag 基础 | 4-8 条，低 | L1 | 覆盖 S1 单句、S2 单句、S1/S2 交替 |
| 自建功能矩阵 JSONL | 中文/英文/中英混合/normalize/长短文本 | 10-30 条，中 | L1 | 每次交付必跑 |
| AISHELL-3 / CSMSC prompt 改造 | 中文 prompt 与可懂度 | 50-200 条，中 | L2/L3 | 固定 speaker prompt 与文本清单 |
| LibriTTS / VCTK prompt 改造 | 英文 prompt 与可懂度 | 50-200 条，中 | L2/L3 | 覆盖男女声和不同口音 |
| 内部播客/对话样本 | 业务真实场景 | 50-500 条，中高 | L2/L3 | 需脱敏并固定版本 |
| 人工 MOS/CMOS/A-B | 主观自然度和偏好 | 20-100 条，高 | L2/L3 | 盲测，至少 5 名听测人 |

### 3.3 分层数据量建议

| 层级 | 建议数据量 | 选择原则 |
|---|---:|---|
| L0 | 官方 `examples/examples.jsonl` 2 条，或抽 1 条 | 只验证端到端调用和 WAV 输出 |
| L1 | 10-30 条；中文/英文/混合各 ≥2，双说话人 ≥4，normalize 特征 ≥4 | 小而全，覆盖所有关键开关和输入形态 |
| L2-min | 50 条或生成音频 ≥20 分钟；中英文各 ≥20 条 | 用于资源有限的正式验收 |
| L2-full | 200 条或生成音频 ≥60 分钟；多 prompt、多长度 | 推荐正式验收目标 |
| L3 | 500+ 条，含公开/内部固定测试集与人工盲测 | 用于发布级报告 |

## 4. 功能验收

### 4.1 L0 smoke test

```bash
cd MOSS-TTSD-v0.5/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_l0_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

输出结构检查：

```bash
python - <<'PY'
import glob, soundfile as sf
paths = sorted(glob.glob('outputs_l0_npu/output_*.wav'))
print('wav_count=', len(paths))
assert len(paths) >= 1
for p in paths:
    data, sr = sf.read(p, always_2d=True)
    dur = len(data) / sr
    print(p, 'sr=', sr, 'shape=', data.shape, 'duration=', dur, 'peak=', abs(data).max())
    assert sr > 0 and dur > 0.5
    assert abs(data).max() > 1e-5
PY
```

通过条件：

- 退出码为 0；
- 输出 WAV 数量与输入有效样本数量一致，或失败样本有明确错误日志且 L0 不允许全部失败；
- WAV 可读、非零时长、非全静音；
- 无 `Expected all tensors to be on the same device`；
- 无 `aclnnFlashAttentionScore` attention mask query/key 长度不一致错误，例如 `[B, 1, L+7, L]`；
- 无 `TorchCodec is required for load_with_torchcodec` / `save_with_torchcodec`；
- 无 CUDA-only / `.cuda()` / 硬编码 CUDA 设备导致的错误；
- 日志中明确使用 NPU，不得静默切到 CPU。

### 4.2 L1 全功能矩阵

准备 `acceptance_l1.jsonl`，至少包含：

| 用例 | language | speaker | text 特征 | prompt 要求 | 样本数 |
|---|---|---|---|---|---:|
| 中文短句 | zh | S1/S2 | 1-2 轮短对话 | 中文 prompt | ≥2 |
| 中文长对话 | zh | S1/S2 | 10 轮以上，标点丰富 | 中文 prompt | ≥2 |
| 英文短句 | en | S1/S2 | 1-2 turns | English prompt | ≥2 |
| 英文长对话 | en | S1/S2 | 10+ turns | English prompt | ≥2 |
| 中英混合 | zh-en | S1/S2 | 英文缩写、数字、中文夹英文 | 任一同语种或跨语种 prompt | ≥2 |
| normalize 数字 | zh/en | S1/S2 | 日期、金额、百分比、缩写 | 与文本语言匹配 | ≥2 |
| prompt 切换 | zh/en | S1/S2 | 同文本不同 prompt | 至少两组 speaker prompt | ≥2 |
| 边界长度 | zh/en | S1/S2 | 很短文本和接近业务上限长文本 | 可复用 prompt | ≥2 |
| speaker tag 边界 | zh/en | S1/S2 | S1 连续多句、S2 连续多句、交替 | 双 prompt | ≥2 |

L1 NPU 命令：

```bash
cd MOSS-TTSD-v0.5/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl acceptance_l1.jsonl \
  --output_dir outputs_l1_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

可选 normalize 对照：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl acceptance_l1.jsonl \
  --output_dir outputs_l1_npu_no_normalize \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42
```

通过条件：

- 所有样本退出码为 0，输出 WAV 数量正确；
- 输出语言与输入文本一致，无整段错语种；
- S1/S2 轮次可听辨，不出现整段单一说话人吞并所有轮次；
- `--use_normalize` 路径可运行，数字/缩写等发音与 normalize 预期一致或问题可解释；
- 中英混合不报错，无明显乱码、长时间静音、重复卡死；
- 长文本若因显存或最大长度失败，必须记录最大可通过长度、错误栈和是否属于业务限制；
- 无设备不一致、CUDA 硬编码、TorchCodec、attention mask 形状错误。

## 5. 精度与质量验收

MOSS-TTSD-v0.5 属生成式语音模型，不能用单一数值等同于通过。正式验收建议同时使用自动指标和人工听测；自动指标只作为筛查和回归检测，最终结论需包含人工主观质量。

### 5.1 指标与工具建议

| 维度 | 指标 | 推荐工具/模型 | 说明 |
|---|---|---|---|
| 可懂度 | ASR 回识别 CER/WER | 固定 ASR 模型；中文可用 Paraformer/Whisper，英文可用 Whisper/Canary 等 | 同一 ASR、同一 normalizer 下比较 CPU/CUDA vs NPU |
| 文本覆盖 | 删除率/插入率/重复率 | ASR transcript + `jiwer` / 自定义统计 | 检查漏读、复读和长文本截断 |
| 音色保持 | speaker embedding cosine / EER | speechbrain ECAPA、WeSpeaker、resemblyzer 等固定版本 | 分别比较输出 S1 对 prompt S1、输出 S2 对 prompt S2 |
| 说话人切换 | speaker switch accuracy、串音率 | VAD + speaker verification，或人工标注 | 检查 `[S1]`/`[S2]` 是否对应正确音色 |
| 自然度 | DNSMOS / UTMOS / NISQA | 固定版本 | 客观参考，不替代人工听测 |
| 音频有效性 | 时长、采样率、峰值、RMS、静音占比 | `soundfile` + `numpy` | 必须检查，防止空音频通过 |
| 主观质量 | MOS / CMOS / A-B preference | 盲测表单 | 至少记录人数、样本数、评分尺度和置信区间 |

项目级严格失败原则：如果官方或选定评测组件不可用，应直接失败并报告缺失依赖或版本，不要用正则/基础规范化器、简化指标或名称相近第三方包替代已声明的正式评测路径。若另设“非官方快速筛查”模式，必须在报告中单独标注，不能作为正式验收结论。

### 5.2 L2 推荐验收门槛

| 项 | 最小规模 | 推荐规模 | 通过条件 |
|---|---:|---:|---|
| 中文可懂度 | ≥20 条或 ≥10 分钟输出 | ≥100 条或 ≥30 分钟输出 | NPU CER 相对 CPU/CUDA 不明显退化；建议绝对差 ≤ 1.0 或相对差 ≤ 10%，二者取宽 |
| 英文可懂度 | ≥20 条或 ≥10 分钟输出 | ≥100 条或 ≥30 分钟输出 | NPU WER 相对 CPU/CUDA 不明显退化；建议绝对差 ≤ 1.0 或相对差 ≤ 10%，二者取宽 |
| 中英混合 | ≥10 条 | ≥50 条 | 主要实体/数字/英文缩写无系统性错误；NPU 不劣于 CPU/CUDA |
| 音色保持 | ≥10 个 prompt pair | ≥30 个 prompt pair | 正确 speaker 相似度高于交叉 speaker；NPU 与 CPU/CUDA 差异在统计噪声内 |
| 说话人切换 | ≥50 个 turn | ≥300 个 turn | 切换错误、串音、漏 speaker tag 无系统性增加 |
| 自然度 | ≥20 条 | ≥100 条 | DNSMOS/UTMOS 与 CPU/CUDA 持平；人工听感无明显退化 |
| 人工 A/B | ≥20 条、≥5 人 | ≥50 条、≥10 人 | NPU 相对 CPU/CUDA preference 不显著低于基线 |

如果没有 CPU/CUDA 可比基线，必须说明原因，并以固定历史 NPU 基线或人工听测作为临时准入；不能宣称“与原始路径一致”。

### 5.3 L3 发布级验收

完整发布级报告建议：

1. 固定源码 tag、patch、模型/codec revision、依赖版本、CANN 和硬件。
2. 固定 L3 JSONL、prompt 音频来源、文本来源、授权和脱敏状态。
3. 中英文分别覆盖短句、长对话、数字、专名、代码切换、跨语言、不同性别/音色 prompt。
4. 端到端输出全部留档：JSONL、命令、日志、WAV、自动指标 CSV、人工评分原始表。
5. 同一数据分别跑 CPU/CUDA（如可用）与 NPU，并使用同 seed、同 attention backend 可行配置、同后处理。
6. 对每类失败样本做归因：文本过长、prompt 质量差、speaker tag 错误、模型能力边界、NPU 适配问题、依赖版本问题。

## 6. 性能验收

### 6.1 必须记录的指标

| 指标 | 含义 | 记录方式 |
|---|---|---|
| 样本数 | 输入 JSONL 行数和成功输出数 | 日志 + 输出目录统计 |
| generated_audio_seconds | 输出 WAV 总时长 | `soundfile` 读取求和 |
| elapsed_seconds | 端到端墙钟耗时 | `/usr/bin/time -v` 或脚本计时 |
| RTF | 推理耗时 / 生成音频时长，越低越好 | `elapsed_seconds / generated_audio_seconds` |
| RTFx | 生成音频时长 / 推理耗时，越高越好 | `generated_audio_seconds / elapsed_seconds` |
| first-run time | 首次加载/编译/冷启动耗时 | 单独记录首轮 |
| steady-state time | 稳定后重复运行耗时 | 至少 3 轮取均值/中位数 |
| HBM | NPU 峰值显存 | `npu-smi info` 或平台监控 |
| RSS | CPU 进程峰值内存 | `/usr/bin/time -v` Maximum resident set size |
| dtype/backend | `bfloat16/float16/float32`、`sdpa/eager` | 命令行参数 |

性能统计示例：

```bash
cd MOSS-TTSD-v0.5/upstream
/usr/bin/time -v bash -lc 'ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl acceptance_l2.jsonl \
  --output_dir outputs_l2_npu_run1 \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize'
```

输出时长统计：

```bash
python - <<'PY'
import glob, soundfile as sf
paths = sorted(glob.glob('outputs_l2_npu_run1/output_*.wav'))
total = 0.0
for p in paths:
    data, sr = sf.read(p, always_2d=True)
    total += len(data) / sr
print('wav_count=', len(paths))
print('generated_audio_seconds=', total)
PY
```

### 6.2 性能通过条件

- 同 checkpoint、同 JSONL、同参数下，NPU 相对 CPU/CUDA 源路径没有功能退化；
- NPU RTF/RTFx 满足业务目标；如未达标，必须给出瓶颈说明（模型加载、codec、attention、生成长度、I/O、首轮编译等）；
- 稳定运行至少 3 轮，不出现显存持续增长、随机 OOM、输出数量波动；
- 记录 `sdpa` 与必要时 `eager` 的差异；不得在报告中混淆不同 backend 的性能。

## 7. 稳定性与异常验收

### 7.1 重复性

同一 JSONL、同一 `--seed 42`、同一设备和依赖版本重复运行 3 次：

```bash
for i in 1 2 3; do
  ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
    --jsonl acceptance_l1.jsonl \
    --output_dir outputs_l1_npu_seed42_run${i} \
    --device npu \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --model_path weights/MOSS-TTSD-v0.5 \
    --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
    --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
    --seed 42 \
    --use_normalize
done
```

通过条件：输出数量一致、总时长差异可解释、无随机崩溃/OOM；生成式模型不要求 WAV 二进制完全一致，但异常差异需记录。

### 7.2 错误暴露

应构造少量负例确认错误可见：缺失权重、错误 prompt 路径、非法 JSONL、缺少核心字段。通过条件是程序暴露清晰错误并以非通过状态结束；不得吞掉错误后输出空目录并返回成功结论。

## 8. 命令模板汇总

### 8.1 L0 NPU

```bash
cd MOSS-TTSD-v0.5/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_l0_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

### 8.2 L2 CPU/NPU 对比

CPU：

```bash
cd MOSS-TTSD-v0.5/upstream
python inference.py \
  --jsonl acceptance_l2.jsonl \
  --output_dir outputs_l2_cpu \
  --device cpu \
  --dtype float32 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

NPU：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl acceptance_l2.jsonl \
  --output_dir outputs_l2_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

如目标 torch-npu 组合不支持 `sdpa`，显式改为：

```bash
--attn_implementation eager
```

并在报告中单独标注，不能与 `sdpa` 性能混算。

## 9. 报告模板

正式验收输出建议保存为 `MOSS-TTSD-v0.5/validation_reports/YYYYMMDD_<device>.md`，并附原始命令、日志、自动指标 CSV、人工听测表和输出 WAV 索引。

```text
层级：L0/L1/L2/L3
模型：MOSS-TTSD-v0.5
源码：OpenMOSS/MOSS-TTSD tag v0.5 / 0e078c62389922d3aa873ce182daf31142860b18
patch：0001-adapt-v0.5-inference-to-npu.patch / SHA256：
模型权重来源/revision/SHA256（HF 8527b9136b6afefe2252ae597cecea2e80e7ebeb 或 ModelScope 2633fdb794b9b6acd2a0c80dae6c2961f7db9d59）：
XY Tokenizer checkpoint 来源/revision/SHA256（HF c83433728e698ed0698e88cb5096bc221fb8f8c5 或 ModelScope 79082154409f5e883d9487c4d4b4be363323b039）：
日期：
硬件：NPU 型号 / 数量 / HBM
驱动/固件/CANN：
Python / torch / torch-npu / transformers / torchaudio / soundfile：
命令：
输入 JSONL：
样本数：输入 / 成功 / 失败
输出目录：
输出 WAV 数量 / 总时长 / 采样率分布：
性能：elapsed / RTF / RTFx / first-run / steady-state / HBM / RSS
可懂度：ASR 模型 / normalizer / CER-WER
文本覆盖：删除率 / 插入率 / 重复率 / 截断样本
音色：speaker embedding 模型 / 同说话人相似度 / 交叉相似度 / EER
说话人切换：turn 数 / 准确率 / 串音率
自然度：DNSMOS / UTMOS / 其他
人工听测：人数 / 样本数 / MOS / CMOS / A-B preference / 置信区间
相对 CPU/CUDA 结论：
问题列表与日志：
最终结论：通过/不通过/需复测
```

## 10. 最终准入标准

正式验收通过需同时满足：

1. L0、L1 全部通过；
2. L2 至少完成最小规模，并提供 CPU/CUDA 或固定历史基线对比；
3. 无设备不一致、CUDA 硬编码、TorchCodec、attention mask 形状、静默 CPU 回退等 NPU 适配问题；
4. 输出 WAV 可读、非全静音、无系统性截断/复读；
5. 可懂度、音色、说话人切换、自然度相对基线无明显退化；
6. 性能指标、环境、权重 SHA256、命令和日志可复现；
7. 所有未通过项有明确问题单、复测条件和风险说明。
