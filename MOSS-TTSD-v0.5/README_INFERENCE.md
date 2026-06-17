# MOSS-TTSD-v0.5 推理指导

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [文件目录](#文件目录)
- [快速上手](#快速上手)
  - [获取源码并应用 patch](#获取源码并应用-patch)
  - [准备权重](#准备权重)
  - [准备测试数据](#准备测试数据)
  - [模型推理](#模型推理)
- [模型推理性能](#模型推理性能)
- [公网地址说明](#公网地址说明)

## 概述

MOSS-TTSD-v0.5 是 OpenMOSS 发布的对话式双说话人文本转语音/文本转语音对话（TTS/TTSD）模型。模型以包含 `[S1]`、`[S2]` 标签的对话文本作为输入，并结合两个说话人的 prompt audio / prompt text 生成自然对话音频。本文档介绍 `OpenMOSS/MOSS-TTSD` tag `v0.5` 基于昇腾 NPU 的推理指导。

> 说明：本文档适配对象为 `OpenMOSS/MOSS-TTSD` tag `v0.5` 原项目代码、`fnlp/MOSS-TTSD-v0.5` 权重和 `fnlp/XY_Tokenizer_TTSD_V0` codec checkpoint。不包含 MOSS-TTSD v0.7、v1.0、SGLang 路径或未固定版本的一键包改动。

- 版本说明：

  ```text
  url=https://github.com/OpenMOSS/MOSS-TTSD.git
  tag=v0.5
  commit_id=0e078c62389922d3aa873ce182daf31142860b18
  model_name=MOSS-TTSD-v0.5
  patch=patches/0001-adapt-v0.5-inference-to-npu.patch
  ```

- 适配原则：
  - 不修改原始 `README.md`。
  - 不新增旁路推理脚本；继续使用原项目已有 `inference.py`，通过 patch 适配 NPU。
  - NPU 默认显式使用 `--device npu`，实际卡号由 `ASCEND_RT_VISIBLE_DEVICES` 控制。
  - NPU attention backend 默认使用 `--attn_implementation sdpa`；如目标 torch-npu 组合不支持，显式改为 `eager` 复测并记录。

## 输入输出数据

- 输入数据

  推理入口读取 JSONL 文件。每行至少包含以下字段：

  ```json
  {
    "base_path": "examples",
    "text": "[S1]你好。[S2]你好，我们开始测试。",
    "prompt_audio_speaker1": "zh_spk1_moon.wav",
    "prompt_text_speaker1": "周一到周五，每天早晨七点半到九点半的直播片段。",
    "prompt_audio_speaker2": "zh_spk2_moon.wav",
    "prompt_text_speaker2": "如果大家想听到更丰富更及时的直播内容。"
  }
  ```

  `prompt_audio_speaker1`、`prompt_audio_speaker2` 可为相对 `base_path` 的路径，也可按原项目逻辑传入可解析的本地路径。prompt 音频建议为清晰人声 WAV，正式验收需固定 prompt 来源、采样率和文本。

- 输出数据

  输出为 `output_*.wav`，保存到 `--output_dir` 指定目录。输出采样率以模型/codec 返回值为准，验收报告必须记录输出 WAV 数量、采样率、总时长、峰值/RMS 和是否可播放。

## 推理环境准备

- 该模型需要以下插件与驱动。实际版本以目标 CANN 与 torch-npu 官方匹配表为准。

  **表 1** 版本配套表

| 配套 | 版本 |
|---|---|
| 固件与驱动 | 25.0.RC1.1+，推荐随 CANN 版本配套升级 |
| CANN Toolkit / Kernel / NNAL | 8.2.RC1+，推荐使用目标机器统一版本 |
| Python | 3.10 / 3.11 |
| PyTorch / torch-npu | 与 CANN 匹配，建议 2.6.0+ |
| transformers / accelerate | 按原项目依赖安装 |
| soundfile / torchaudio | `soundfile` 用于文件读写；`torchaudio.functional.resample` 用于重采样 |

说明：

- `flash-attn` 官方包面向 CUDA/ROCm GPU kernel，当前不作为 Ascend NPU 必需依赖安装。NPU 推理固定显式使用 `--attn_implementation sdpa`；仅 CUDA/ROCm GPU 路径显式使用 `flash_attention_2` 时才安装 `flash-attn`。
- TorchAudio 2.9+ 的 `torchaudio.load` / `torchaudio.save` 会进入 TorchCodec 路径。本适配通过 patch 将 prompt 音频读取和 WAV 写出改为 `soundfile`，不要求额外安装 `torchcodec`。如果仍看到 `TorchCodec is required for load_with_torchcodec` 或 `save_with_torchcodec`，说明 patch 未应用或路径未覆盖。

## 文件目录

```text
MOSS-TTSD-v0.5
├── README_INFERENCE.md                         # 推理指导文档
├── README.md                                   # 模型适配说明
├── NPU_ADAPTATION.md                           # NPU 适配文档与验证记录
├── ACCEPTANCE_PLAN.md                          # 完整验收方案
├── V1_0_DIFF_REFERENCE.md                      # v1.0 差异参考
├── patches
│   ├── README.md                               # patch 使用说明
│   └── 0001-adapt-v0.5-inference-to-npu.patch  # v0.5 NPU 适配 patch
├── upstream                                    # OpenMOSS/MOSS-TTSD tag v0.5 源码
│   ├── inference.py                            # patch 后的推理入口
│   ├── generation_utils.py                     # 生成和音频处理逻辑
│   ├── modeling_asteroid.py                    # Asteroid 生成模型
│   ├── examples
│   │   ├── examples.jsonl                      # 官方中文/英文示例
│   │   ├── zh_spk1_moon.wav
│   │   ├── zh_spk2_moon.wav
│   │   ├── m1.wav
│   │   └── m2.wav
│   ├── weights
│   │   └── MOSS-TTSD-v0.5                      # 下载后的模型权重
│   └── XY_Tokenizer
│       ├── config/xy_tokenizer_config.yaml     # codec 配置
│       └── weights/xy_tokenizer.ckpt           # 下载后的 codec checkpoint
└── validation_reports                          # 验收报告目录，按需生成
```

## 快速上手

### 获取源码并应用 patch

1. 获取适配仓库。

   ```bash
   git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch
   git checkout master
   cd ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5
   ```

2. 准备原项目 tag `v0.5` 代码并应用 patch。

   ```bash
   # 如已存在 upstream，可跳过 clone
   git clone https://github.com/OpenMOSS/MOSS-TTSD.git upstream

   git -C upstream fetch --depth 1 origin tag v0.5
   git -C upstream checkout v0.5
   git -C upstream apply --check ../patches/0001-adapt-v0.5-inference-to-npu.patch
   git -C upstream apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
   ```

3. 安装依赖。

   ```bash
   cd upstream
   pip install torch torch-npu
   grep -vE '^flash-attn([<>= ].*)?$' requirements.txt > /tmp/moss-ttsd-v0.5-requirements-npu.txt
   pip install -r /tmp/moss-ttsd-v0.5-requirements-npu.txt
   pip install -r XY_Tokenizer/requirements.txt
   ```

### 准备权重

1. 下载 MOSS-TTSD-v0.5 权重和 XY Tokenizer checkpoint。

   原始权重地址：

   - MOSS-TTSD-v0.5：`https://huggingface.co/fnlp/MOSS-TTSD-v0.5`
   - XY Tokenizer：`https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0`

   ```bash
   cd MOSS-TTSD-v0.5/upstream
   python -m pip install -U "huggingface_hub[cli]"
   mkdir -p weights/MOSS-TTSD-v0.5 XY_Tokenizer/weights

   hf download fnlp/MOSS-TTSD-v0.5 \
     --revision 8527b9136b6afefe2252ae597cecea2e80e7ebeb \
     --local-dir weights/MOSS-TTSD-v0.5

   hf download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt \
     --revision c83433728e698ed0698e88cb5096bc221fb8f8c5 \
     --local-dir XY_Tokenizer/weights
   ```

2. 国内环境可使用 ModelScope 镜像。

   ```bash
   cd MOSS-TTSD-v0.5/upstream
   python -m pip install -U modelscope
   mkdir -p weights/MOSS-TTSD-v0.5 XY_Tokenizer/weights

   modelscope download --model openmoss/MOSS-TTSD-v0.5 \
     --local_dir weights/MOSS-TTSD-v0.5

   modelscope download --model openmoss/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt \
     --local_dir XY_Tokenizer/weights
   ```

3. 下载后记录 SHA256。

   ```bash
   cd MOSS-TTSD-v0.5/upstream
   find weights/MOSS-TTSD-v0.5 -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum
   sha256sum XY_Tokenizer/weights/xy_tokenizer.ckpt
   ```

### 准备测试数据

1. 使用原项目官方示例作为 smoke test 数据。

   默认示例路径：

   ```text
   MOSS-TTSD-v0.5/upstream/examples/examples.jsonl
   ```

   该文件包含中文和英文双说话人长对话示例，并引用 `examples/` 目录下的 prompt wav。

2. 正式验收时准备 `acceptance_l1.jsonl` / `acceptance_l2.jsonl`。字段、规模和通过条件见 `ACCEPTANCE_PLAN.md`。L1 至少覆盖中文、英文、中英混合、短句、长对话、双说话人、normalize 和 prompt 切换。

### 模型推理

1. 执行 NPU smoke 推理。

   ```bash
   cd MOSS-TTSD-v0.5/upstream
   ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
     --jsonl examples/examples.jsonl \
     --output_dir outputs_npu \
     --device npu \
     --dtype bfloat16 \
     --attn_implementation sdpa \
     --model_path weights/MOSS-TTSD-v0.5 \
     --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
     --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
     --seed 42 \
     --use_normalize
   ```

   参数说明：

   - `jsonl`：输入 JSONL 文件路径。
   - `output_dir`：输出 WAV 保存目录。
   - `device`：推理设备，支持 `npu`、`cpu`、`cuda`。
   - `dtype`：模型权重 dtype，支持 `bfloat16`、`float16`、`float32`；NPU 推荐 `bfloat16`。
   - `attn_implementation`：Transformers attention 后端；NPU 推荐 `sdpa`，必要时显式改为 `eager`。
   - `model_path`：本地模型权重目录或 Hugging Face id。
   - `spt_config_path`：XY Tokenizer 配置路径。
   - `spt_checkpoint_path`：XY Tokenizer checkpoint 路径。
   - `seed`：随机种子。
   - `use_normalize`：启用原项目文本归一化路径。

2. 执行 CPU 功能/质量基线。

   ```bash
   cd MOSS-TTSD-v0.5/upstream
   python inference.py \
     --jsonl examples/examples.jsonl \
     --output_dir outputs_cpu \
     --device cpu \
     --dtype float32 \
     --attn_implementation sdpa \
     --model_path weights/MOSS-TTSD-v0.5 \
     --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
     --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
     --seed 42 \
     --use_normalize
   ```

3. 检查输出 WAV。

   ```bash
   cd MOSS-TTSD-v0.5/upstream
   python - <<'PY'
   import glob, soundfile as sf
   paths = sorted(glob.glob('outputs_npu/output_*.wav'))
   print('wav_count=', len(paths))
   for p in paths:
       data, sr = sf.read(p, always_2d=True)
       dur = len(data) / sr
       print(p, 'sr=', sr, 'shape=', data.shape, 'duration=', round(dur, 3), 'peak=', float(abs(data).max()))
   PY
   ```

   L0 通过条件：输出 WAV 数量与输入有效样本一致、WAV 可读、时长大于 0、非全静音、无设备不一致、无 CUDA 硬编码、无 TorchCodec 报错、无 NPU attention mask 形状错误。

## 模型推理性能

MOSS-TTSD-v0.5 属自回归生成式 TTS/TTSD 模型，性能以生成音频总时长和端到端墙钟时间计算。

```bash
cd MOSS-TTSD-v0.5/upstream
/usr/bin/time -v bash -lc 'ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl acceptance_l2.jsonl \
  --output_dir outputs_l2_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize'
```

统计输出时长：

```bash
python - <<'PY'
import glob, soundfile as sf
paths = sorted(glob.glob('outputs_l2_npu/output_*.wav'))
total = 0.0
for p in paths:
    data, sr = sf.read(p, always_2d=True)
    total += len(data) / sr
print('wav_count=', len(paths))
print('generated_audio_seconds=', total)
PY
```

报告中至少记录：样本数、成功输出数、输出 WAV 总时长、elapsed seconds、`RTF=elapsed/generated_audio_seconds`、`RTFx=generated_audio_seconds/elapsed`、dtype、attention backend、峰值 HBM、CPU RSS、首次加载/编译耗时和稳定推理耗时。完整性能与质量验收口径见 `ACCEPTANCE_PLAN.md`。

## 公网地址说明

| 资源 | 地址 | 说明 |
|---|---|---|
| OpenMOSS/MOSS-TTSD 源码 | <https://github.com/OpenMOSS/MOSS-TTSD> | 使用 tag `v0.5`，commit `0e078c62389922d3aa873ce182daf31142860b18` |
| MOSS-TTSD-v0.5 HF 权重 | <https://huggingface.co/fnlp/MOSS-TTSD-v0.5> | 固定 revision `8527b9136b6afefe2252ae597cecea2e80e7ebeb` |
| MOSS-TTSD-v0.5 ModelScope 权重 | <https://modelscope.cn/models/openmoss/MOSS-TTSD-v0.5> | 国内镜像，记录 HEAD `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59` |
| XY Tokenizer HF checkpoint | <https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0> | 固定 revision `c83433728e698ed0698e88cb5096bc221fb8f8c5` |
| XY Tokenizer ModelScope checkpoint | <https://modelscope.cn/models/openmoss/XY_Tokenizer_TTSD_V0> | 国内镜像，记录 HEAD `79082154409f5e883d9487c4d4b4be363323b039` |
