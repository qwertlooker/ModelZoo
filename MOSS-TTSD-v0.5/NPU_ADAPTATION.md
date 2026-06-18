# MOSS-TTSD-v0.5 NPU 适配文档

本文保留 MOSS-TTSD-v0.5 NPU 适配过程中的版本边界、上游代码分析、设备适配、环境/权重准备、推理命令和验证记录。

文档分工：

- `README_INFERENCE.md`：面向上库/用户的推理指导，单独保留，不在此处重复完整操作手册。
- `NPU_ADAPTATION.md`：只记录适配实现、上游分析、迁移说明和验证事实。
- `ACCEPTANCE_PLAN.md`：记录分层验收、数据集/JSONL 规范、质量/性能指标、通过条件和报告模板。

## 目录

- [1. 上游版本与代码分析](#1-上游版本与代码分析)
- [2. NPU 适配与运行说明](#2-npu-适配与运行说明)
- [3. 验证记录](#3-验证记录)

## 1. 上游版本与代码分析

### 1.1 上游信息

- 上游仓库：<https://github.com/OpenMOSS/MOSS-TTSD>
- 上游版本：tag `v0.5`
- tag commit：`0e078c62389922d3aa873ce182daf31142860b18`
- 检查日期：2026-06-17
- 模型权重：`fnlp/MOSS-TTSD-v0.5` / `OpenMOSS-Team/MOSS-TTSD-v0.5`
- 模型权重地址：HF <https://huggingface.co/fnlp/MOSS-TTSD-v0.5>，ModelScope <https://modelscope.cn/models/openmoss/MOSS-TTSD-v0.5>
- 当前记录模型权重 revision：HF `8527b9136b6afefe2252ae597cecea2e80e7ebeb`，ModelScope `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59`
- Codec：原项目 `XY_Tokenizer` + `fnlp/XY_Tokenizer_TTSD_V0` 的 `xy_tokenizer.ckpt`
- Codec 地址：HF <https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0>，ModelScope <https://modelscope.cn/models/openmoss/XY_Tokenizer_TTSD_V0>
- 当前记录 codec revision：HF `c83433728e698ed0698e88cb5096bc221fb8f8c5`，ModelScope `79082154409f5e883d9487c4d4b4be363323b039`
- 本地上游副本：`upstream/`
- 版本边界：当前只适配 MOSS-TTSD `v0.5`；不包含 MOSS-TTSD v0.7、v1.0、SGLang 路径或未固定版本的一键包改动。

权重 SHA256 尚未在当前环境完成实测记录：正式验收前必须补充 `weights/MOSS-TTSD-v0.5/` 中核心权重文件和 `XY_Tokenizer/weights/xy_tokenizer.ckpt` 的 SHA256。

### 1.2 当前目录状态

当前 `MOSS-TTSD-v0.5/` 主要文件：

- `README_INFERENCE.md`：推理指导文档。
- `README.md`：模型适配说明；按项目约束不修改原始 README。
- `NPU_ADAPTATION.md`：整合后的适配分析、迁移说明和验证记录。
- `ACCEPTANCE_PLAN.md`：完整验收方案。
- `V1_0_DIFF_REFERENCE.md`：v1.0 差异参考。
- `patches/0001-adapt-v0.5-inference-to-npu.patch`：唯一代码适配 patch。
- `patches/README.md`：patch 应用和校验说明。
- `upstream/`：原项目 tag `v0.5` 代码，用于应用 patch、运行推理和校验。

本次适配不新增独立推理代码文件；所有代码改动均进入 patch 并作用于原项目已有文件。

### 1.3 与上游匹配情况

原项目 v0.5 默认面向 CUDA/GPU 推理：

- `inference.py` 根据 `torch.cuda.is_available()` 自动选择 `cuda/cpu`，无显式 NPU 参数。
- `generation_utils.py` 固定 `attn_implementation="flash_attention_2"`，结束时调用 `torch.cuda.empty_cache()`。
- `requirements.txt` 包含 `flash-attn`；该官方包面向 CUDA/ROCm GPU kernel，不作为 Ascend NPU 依赖。
- `XY_Tokenizer/inference.py` 默认 `--device cuda`。
- `XY_Tokenizer/xy_tokenizer/model.py` 的 `encode/decode` 默认 `device=torch.device("cuda")`，即使输入 tensor 已在 NPU，也会创建 CUDA tensor。
- `XY_Tokenizer/xy_tokenizer/nn/quantizer.py` 使用 `torch.autocast('cuda', enabled=False)`。
- `torchaudio.load` / `torchaudio.save` 在 TorchAudio 2.9+ 环境会进入 TorchCodec 路径，缺少或不匹配时会报 `TorchCodec is required for load_with_torchcodec` / `save_with_torchcodec`。
- `modeling_asteroid.py` 自定义 `GenerationMixin._sample` 先记录 shifted 输入原始长度，再裁掉 `channels - 1` 个位置用于初始前向；如果不同步 `cur_len`，NPU `sdpa` 下发 `aclnnFlashAttentionScore` 时可能收到 query/key 长度不一致的 attention mask。
- `inference.py` 原本将 JSONL 的全部样本一次性传给 `process_batch()`；TTSD-eval 等多样本输入会在 Transformers NPU SDPA 的 `repeat_kv` GQA 展开处产生大块临时张量并 OOM。

因此本次必须修改上游已有文件并生成 patch，而不是仅新增外部包装脚本。

### 1.4 现有代码审视

| 文件 | 结论 | 说明 |
|---|---|---|
| `inference.py` | 已 patch | 增加 `--device npu/cpu/cuda`、`--dtype`、`--attn_implementation`、`--batch_size`、模型/codec 路径参数；默认 NPU、`npu_fa`、batch 0（完整 JSONL）；正整数 batch 时按批读取，并在批间清理 accelerator cache；使用 `soundfile` 写 WAV。 |
| `generation_utils.py` | 已 patch | `load_model()` 支持 dtype 和 attention backend；音频读取/写出改为 `soundfile`；按 CUDA/NPU 分支清理显存。 |
| `modeling_asteroid.py` | 已 patch | 裁剪 shifted speech channels 后同步 `cur_len`；注册 `npu_fa` attention/mask backend，prefill/decode 分别调用 NPU PFA/IFA，并直接传递 GQA KV head 数。 |
| `gradio_demo.py` / `podcast_generate.py` | 已 patch | WAV 写出复用 `save_audio_file()`，移除 `torchaudio.save` 路径。 |
| `XY_Tokenizer/inference.py` | 已 patch | 默认设备改为 NPU，增加 NPU/CUDA 可用性检查。 |
| `XY_Tokenizer/utils/helpers.py` | 已 patch | 音频文件读写改为 `soundfile`，继续保留 `torchaudio.functional.resample`。 |
| `XY_Tokenizer/xy_tokenizer/model.py` | 已 patch | `encode/decode` 默认从输入 tensor 推断设备，不再默认 CUDA。 |
| `XY_Tokenizer/xy_tokenizer/nn/quantizer.py` | 已 patch | autocast device_type 使用当前 tensor device。 |
| `requirements.txt` | 已 patch | 删除 CUDA/ROCm 专用 `flash-attn` 依赖，NPU 环境直接安装 patch 后的 `requirements.txt`。 |

### 1.5 设备适配点

1. `inference.py::_resolve_device`：仅当 `--device npu` 时导入 `torch_npu` 注册后端；返回 `torch.device('npu')`，不绑定卡号。
2. `inference.py::_resolve_dtype`：显式支持 `float32`、`float16`、`bfloat16`；NPU 推理推荐 `bfloat16`。
3. `generation_utils.load_model()`：允许传入 `torch_dtype` 和 `attn_implementation`；正式 NPU 入口默认显式传入 `npu_fa`。
4. `model.to(device)`、`spt.to(device)`：模型和 codec 显式迁移到目标设备。
5. `XY_Tokenizer.encode/decode`：默认从输入 tensor 推断设备，避免在 NPU 路径创建 CUDA tensor。
6. `ResidualVQ.forward()`：`torch.autocast(device_type=z.device.type, enabled=False)`，避免 CUDA-only autocast。
7. 音频 I/O：文件读取/写出走 `soundfile`；重采样仍使用 `torchaudio.functional.resample`，该路径不触发 TorchCodec 文件解码。
8. 显存清理：CUDA 使用 `torch.cuda.empty_cache()`，NPU 使用 `torch.npu.empty_cache()`。
9. attention mask：裁剪 shifted speech channels 后重置 `cur_len = input_ids.shape[1]`，保证 `input_ids`、`attention_mask`、cache position 长度一致。
10. NPU GQA attention：`npu_fa` 使用 PFA/IFA 的 `num_key_value_heads` 参数，避免 SDPA/eager 对 KV 执行 `repeat_kv`；复用 Transformers SDPA 的布尔 causal/padding mask 生成逻辑，但禁用 mask-skip，再转换为 NPU 算子的“True 表示屏蔽”语义。
11. 可选评测 batch：`--batch_size 0` 保留原项目完整 JSONL batch；正整数用于限制峰值 HBM，每次 `process_batch()` 返回后清理 allocator cache。

### 1.6 风险与限制

- 当前环境缺少 `torch`、`torch-npu`、模型权重和 NPU/CANN，未在本机执行 CPU/NPU 实推。
- 权重 SHA256 尚未记录；正式验收前必须补充。
- 原项目 v0.5 中仍存在若干宽泛 `try/except` 和失败后继续处理的逻辑；本次 patch 以 NPU 设备适配为目标，没有重构原项目整体错误处理。
- `npu_fa` 依赖目标 `torch-npu` 的 PFA/IFA GQA 接口；当前本地环境无 NPU，尚未完成真实算子精度/性能验证，正式验收必须补齐与 CPU/CUDA、NPU SDPA 小样本基线的结果对齐。
- 即使消除 `repeat_kv`，完整 JSONL batch 仍可能因输入 embedding、logits、KV cache 或 codec 中间张量超过 HBM；此时需实测最大 batch，不能声称 attention 修改可以无条件容纳任意评测集。
- 生成式 TTS/TTSD 不能用“能输出 WAV”作为完整验收；正式验收需按 `ACCEPTANCE_PLAN.md` 做可懂度、音色、自然度和人工听测。
- NeMo/Transformers/PyTorch/TorchAudio 版本持续变化；若上游或依赖升级，应重新检查 `flash-attn`、TorchCodec、attention backend 和 `GenerationMixin` 行为。

### 1.7 上游版本检查记录

- 2026-06-17：确认 `OpenMOSS/MOSS-TTSD` tag `v0.5` commit 为 `0e078c62389922d3aa873ce182daf31142860b18`。
- 2026-06-17：确认当前 main HEAD 已面向 v1.0，不作为本次适配对象。
- 2026-06-17：扫描 v0.5 原项目 CUDA 假设，确认需 patch 原项目已有文件。
- 2026-06-17：固定模型权重和 codec checkpoint 的 HF/ModelScope revision，待正式下载后补充 SHA256。
- 2026-06-18：根据 TTSD-eval NPU OOM 栈确认整集 batch 在 `sdpa_attention.repeat_kv` 处放大 GQA KV。
- 2026-06-18：为避免强制 batch 1 影响吞吐，新增 `npu_fa` PFA/IFA GQA backend，并将 batch 0 定义为保留原始完整 JSONL 行为；正整数 batch 仅作为显式显存上限。

## 2. NPU 适配与运行说明

本章命令默认从 `ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5` 目录开始，后续路径均使用相对路径。

### 2.1 适配目标

将 OpenMOSS/MOSS-TTSD v0.5 推理链路整理为规范的 CPU/NPU 融合路径：

- 默认使用 `--device npu`；
- CPU 验证显式使用 `--device cpu`；
- 不使用 `auto` / `use_gpu`；
- 不在代码中写死 `npu:0` / `cuda:0`；
- 实际 NPU 卡号由环境变量控制，例如 `ASCEND_RT_VISIBLE_DEVICES=0`；
- 必要代码改动通过 patch 交付，不新增旁路推理脚本；
- NPU 路径不依赖 CUDA/ROCm `flash-attn`，不依赖 TorchCodec 文件 I/O。

### 2.2 上游与 patch

- 上游仓库：<https://github.com/OpenMOSS/MOSS-TTSD>
- 基准 tag：`v0.5`
- 基准 commit：`0e078c62389922d3aa873ce182daf31142860b18`
- 当前 patch：`patches/0001-adapt-v0.5-inference-to-npu.patch`

应用与校验：

```bash
git -C upstream reset --hard v0.5
git -C upstream apply --check ../patches/0001-adapt-v0.5-inference-to-npu.patch
git -C upstream apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
```

如后续上游源码需要继续修改，应在 `upstream/` 内修改已有文件并生成新 patch：

```bash
git -C upstream diff -- <upstream_existing_file> > patches/0002-xxx.patch
git -C upstream apply --check ../patches/0002-xxx.patch
```

### 2.3 环境准备

NPU 环境中请先安装与 CANN 匹配的 `torch` / `torch-npu`，再安装 patch 后的原项目依赖：

```bash
cd upstream
pip install torch torch-npu
# patch 已从 requirements.txt 删除 CUDA/ROCm 专用的 flash-attn 依赖。
pip install -r requirements.txt
pip install -r XY_Tokenizer/requirements.txt
```

依赖关系说明：

- `flash-attn` 官方包面向 CUDA/ROCm GPU kernel，不作为 Ascend NPU 依赖。
- NPU 性能推理使用 `--attn_implementation npu_fa`；`sdpa` 保留为小样本对照路径，`eager` 只用于问题定位。
- `soundfile` 已在原项目依赖中声明，本适配用它替代 `torchaudio.load/save` 文件 I/O。
- 如果安装依赖时 pip 试图替换已有 NPU 版 PyTorch，请先固定与 CANN 匹配的 `torch/torch-npu` 版本，再安装其他依赖。不得用临时过滤 requirements 的命令替代 patch。

### 2.4 权重下载

官方权重与 codec：

| 资产 | URL | revision / HEAD | 目标路径 |
|---|---|---|---|
| MOSS-TTSD-v0.5 模型权重 | HF <https://huggingface.co/fnlp/MOSS-TTSD-v0.5>；同内容别名 <https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.5>；ModelScope <https://modelscope.cn/models/openmoss/MOSS-TTSD-v0.5> | HF `8527b9136b6afefe2252ae597cecea2e80e7ebeb`；ModelScope `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59` | `weights/MOSS-TTSD-v0.5/` |
| XY Tokenizer checkpoint | HF <https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0>；ModelScope <https://modelscope.cn/models/openmoss/XY_Tokenizer_TTSD_V0> | HF `c83433728e698ed0698e88cb5096bc221fb8f8c5`；ModelScope `79082154409f5e883d9487c4d4b4be363323b039` | `XY_Tokenizer/weights/xy_tokenizer.ckpt` |
| XY Tokenizer config | 原项目 tag `v0.5` 自带 | `0e078c62389922d3aa873ce182daf31142860b18` | `XY_Tokenizer/config/xy_tokenizer_config.yaml` |

下载命令和 ModelScope 镜像命令见 `README_INFERENCE.md`。正式验收前记录：模型权重来源、HF/ModelScope revision、`model.safetensors` 或等效权重 SHA256、`xy_tokenizer.ckpt` SHA256。

### 2.5 评测口径摘要

MOSS-TTSD-v0.5 的正式质量/性能验收口径统一维护在 `ACCEPTANCE_PLAN.md`：

- 功能：中文、英文、中英混合、双说话人、prompt 切换、normalize、长短文本和异常暴露。
- 可懂度：固定 ASR 模型和 normalizer，统计 CER/WER。
- 音色：固定 speaker embedding 模型，统计 speaker similarity / EER。
- 公共客观评测：默认使用 `OpenMOSS/TTSD-eval`，记录 ACC、SIM、WER；该流程可测评 v0.5 输出，但不是 v0.5 已发布官方指标，必须与 CPU/CUDA 原始路径做同口径对齐。
- 自然度：DNSMOS / UTMOS / NISQA 等作为客观参考，不替代人工听测。
- 主观：MOS / CMOS / A-B preference，记录人数、样本数和置信区间。
- 性能：记录 `elapsed_seconds`、`RTF`、`RTFx`、dtype、attention backend、峰值 HBM/RSS、首次加载/编译耗时和稳定推理耗时。

### 2.6 TTSD-eval 测评说明

`OpenMOSS/TTSD-eval` 可用于 MOSS-TTSD-v0.5 的公共客观测评：其 pipeline 对生成音频做 MMS-FA forced alignment，再按 `[S1]`/`[S2]` 文本标签切分片段，使用 WeSpeaker 计算 speaker attribution ACC 和 speaker similarity SIM，并用 Whisper-large-v3 计算补充 WER。

使用边界：

- v0.5 README/技术报告未发布 v0.5 在 TTSD-eval 上的官方数值；因此验收报告中仍写“v0.5 官方指标未发布”。
- TTSD-eval 结果用于 L2 公共评测和 NPU 迁移对齐：同一 testset、同一 v0.5 checkpoint、同一输入参数，分别生成 CPU/CUDA 与 NPU 音频，再比较 ACC/SIM/WER。
- TTSD-eval 输入 manifest 必须包含 `text`、`output_audio`、`prompt_audio_speaker1`、`prompt_audio_speaker2`。v0.5 推理完成后，需要把 `output_*.wav` 回填为 `output_audio`。
- 若 `git lfs` testset、MMS-FA checkpoint、WeSpeaker 权重或 Whisper 依赖不可用，直接记录失败原因，不用简化指标替代。

详细命令、manifest 生成方式和报告字段见 `ACCEPTANCE_PLAN.md` 的 “OpenMOSS/TTSD-eval 公共评测” 小节。

### 2.7 推理脚本用法

#### NPU 推理

```bash
cd upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl examples/examples.jsonl \
  --batch_size 0 \
  --output_dir outputs_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation npu_fa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

#### CPU 功能/质量基线

```bash
cd upstream
python inference.py \
  --jsonl examples/examples.jsonl \
  --batch_size 0 \
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

#### 输出结构检查

```bash
cd upstream
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

### 2.8 上游更新处理

上游更新时必须重新执行：

```bash
git -C upstream fetch origin
git -C upstream rev-parse origin/main
grep -RIn "cuda\|gpu\|npu\|flash_attention\|torchaudio\.load\|torchaudio\.save\|to(device)" \
  upstream \
  --exclude-dir=.git
```

重点检查：

- `inference.py`
- `generation_utils.py`
- `modeling_asteroid.py`
- `gradio_demo.py`
- `podcast_generate.py`
- `XY_Tokenizer/inference.py`
- `XY_Tokenizer/utils/helpers.py`
- `XY_Tokenizer/xy_tokenizer/model.py`
- `XY_Tokenizer/xy_tokenizer/nn/quantizer.py`

如新增硬编码 CUDA、`flash_attention_2`、TorchCodec 文件 I/O 或 attention mask 长度问题，按标准流程生成 patch 并补充验证记录。

## 3. 验证记录

### 3.1 当前环境验证结果

检查日期：2026-06-17。

| 项 | 结果 |
|---|---|
| 工作目录 | `/home/pei/ModelZoo` |
| 原项目 tag | `OpenMOSS/MOSS-TTSD` tag `v0.5` |
| tag commit | `0e078c62389922d3aa873ce182daf31142860b18` |
| patch | `patches/0001-adapt-v0.5-inference-to-npu.patch` |
| 当前系统 Python | `Python 3.12.3` |
| 当前环境依赖 | 未安装 `torch`、`torch-npu`、模型权重和 `xy_tokenizer.ckpt` |
| CPU 实推 | 未执行，原因：当前环境缺少依赖和权重 |
| NPU 实推 | 未执行，原因：当前环境无 Ascend NPU/CANN 运行条件 |

### 3.2 静态验证

已完成的本地校验：

```bash
git -C upstream reset --hard v0.5
git -C upstream apply --check ../patches/0001-adapt-v0.5-inference-to-npu.patch
git -C upstream apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
python3 -m py_compile \
  upstream/inference.py \
  upstream/generation_utils.py \
  upstream/gradio_demo.py \
  upstream/podcast_generate.py \
  upstream/modeling_asteroid.py \
  upstream/XY_Tokenizer/inference.py \
  upstream/XY_Tokenizer/utils/helpers.py \
  upstream/XY_Tokenizer/xy_tokenizer/model.py \
  upstream/XY_Tokenizer/xy_tokenizer/nn/quantizer.py
! grep -R -I -E 'torchaudio\.(load|save|info)\(' \
  upstream --exclude-dir=.git
git -C upstream reset --hard v0.5
```

结果：patch apply 检查通过；语法检查通过；`torchaudio.load/save/info` 文件 I/O 路径检查通过。

### 3.3 已知 NPU attention mask 报错修复

旧 patch 在 NPU `--attn_implementation sdpa` 下可能报：

```text
aclnnFlashAttentionScore failed
get unsupported atten_mask shape, the shape is [B, 1, L+7, L]
```

根因：`modeling_asteroid.py` 自定义生成循环先记录 shifted 输入原始长度，再裁掉 `channels - 1` 个位置用于初始前向，但没有同步 `cur_len`。最新 patch 在裁剪 `input_ids` / `attention_mask` 后重置 `cur_len = input_ids.shape[1]`，使 `input_ids`、`attention_mask` 与 cache position 长度一致。

### 3.4 NPU `repeat_kv` OOM 与 GQA FlashAttention 修复

TTSD-eval 等多样本 JSONL 旧路径会将全部样本作为一个 batch。Transformers 4.57.6 明确在 NPU SDPA 路径禁用原生 GQA，并在 `sdpa_attention_forward()` 中对 key/value 调用 `repeat_kv()`。因此显存临时分配与 batch size、最长 padding 长度和 KV head 展开倍数共同增长。

`eager_attention_forward()` 同样调用 `repeat_kv()`，并显式创建 attention weights，不能解决该性能/显存问题。本次在 `modeling_asteroid.py` 中通过 Transformers 官方 `AttentionInterface` / `AttentionMaskInterface` 扩展点注册 `npu_fa`：

- prefill 调用 `torch_npu.npu_prompt_flash_attention`；
- query length 为 1 的 decode 调用 `torch_npu.npu_incre_flash_attention`；
- Q/K/V 使用 `BNSD`，分别传入 query head 数与 KV head 数；
- 不改 site-packages，不运行时 monkey patch，不静默回退；
- `--batch_size 0` 保留原始完整 JSONL batch，正整数才按输入顺序切分；
- 每批 `process_batch()` 返回后再调用对应设备的 `empty_cache()`，此时函数内部大 tensor 已失活，可降低长评测中的 allocator 碎片；
- CPU/CUDA 路径继续显式使用 `sdpa` 或原始 CUDA backend。

官方接口说明中，PFA/IFA 的 `num_key_value_heads` 用于 GQA，要求 query head 数可整除 KV head 数。参考：

- <https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000144.html>
- <https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000146.html>
- <https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000453.html>

若 full-JSONL batch 仍 OOM，应记录峰值 HBM并实测 `--batch_size 8/4/2/1` 的最大可用值；若 batch 1 单样本仍 OOM，再按超长样本或更深层 KV cache 优化问题处理。

### 3.5 提交前必跑检查

```bash
git -C upstream reset --hard v0.5
git -C upstream apply --check ../patches/0001-adapt-v0.5-inference-to-npu.patch
git -C upstream apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
python3 -m py_compile \
  upstream/inference.py \
  upstream/generation_utils.py \
  upstream/gradio_demo.py \
  upstream/podcast_generate.py \
  upstream/modeling_asteroid.py \
  upstream/XY_Tokenizer/inference.py \
  upstream/XY_Tokenizer/utils/helpers.py \
  upstream/XY_Tokenizer/xy_tokenizer/model.py \
  upstream/XY_Tokenizer/xy_tokenizer/nn/quantizer.py
! grep -R -I -E 'torchaudio\.(load|save|info)\(' \
  upstream --exclude-dir=.git
git -C upstream reset --hard v0.5
```

### 3.6 NPU 实测待补项

正式验收环境具备权重和 NPU 后，应补充以下记录：

```bash
python -V
pip freeze | grep -E 'torch|torch-npu|torchaudio|transformers|accelerate|soundfile|librosa|numpy|scipy'
npu-smi info || true
uname -a
sha256sum patches/0001-adapt-v0.5-inference-to-npu.patch
cd upstream
find weights/MOSS-TTSD-v0.5 -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum
sha256sum XY_Tokenizer/weights/xy_tokenizer.ckpt
```

然后按 `ACCEPTANCE_PLAN.md` 执行 L0/L1/L2 验收，并将报告保存到 `MOSS-TTSD-v0.5/validation_reports/YYYYMMDD_<device>.md`。
