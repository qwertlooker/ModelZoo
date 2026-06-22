# MOSS-TTSD-v0.5 NPU 适配文档

本文保留 MOSS-TTSD-v0.5 NPU 适配过程中的版本边界、上游代码分析、设备适配、环境/权重准备、推理命令和验证记录。

文档分工：

- `README.md`：面向上库/用户的推理指导，单独保留，不在此处重复完整操作手册。
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
- 独立重放目录：`source/`（Git 管理）、`upstream-original/`（未应用 patch）、
  `upstream-npu/`（应用 patch）
- 目标 NPU 组合：固件/驱动 25.5.1+、CANN 8.5.1、Python 3.11、
  PyTorch/torch-npu/torchaudio 2.9.0、Transformers 4.57.6。
- 版本边界：当前只适配 MOSS-TTSD `v0.5`；不包含 MOSS-TTSD v0.7、v1.0、SGLang 路径或未固定版本的一键包改动。

权重 SHA256 尚未在当前环境完成实测记录：正式验收前必须补充 `weights/MOSS-TTSD-v0.5/` 中核心权重文件和 `XY_Tokenizer/weights/xy_tokenizer.ckpt` 的 SHA256。

### 1.2 当前目录状态

当前 `MOSS-TTSD-v0.5/` 主要文件：

- `README.md`：推理指导文档。
- `README_old.md`：模型适配说明；按项目约束不修改原始 README。
- `NPU_ADAPTATION.md`：整合后的适配分析、迁移说明和验证记录。
- `ACCEPTANCE_PLAN.md`：完整验收方案。
- `V1_0_DIFF_REFERENCE.md`：v1.0 差异参考。
- `patches/0001-adapt-v0.5-inference-to-npu.patch`：唯一代码适配 patch。
- `patches/README.md`：patch 应用和校验说明。
- `prepare_eval_data.py`：生成 TTSD-eval output manifest，并用
  `verify-ttsd-eval` 校验 evaluator commit、testset、权重和评测环境；保留 subset
  子命令用于调试，但正式 L2 使用中英文全量各 50 条。
- `requirements_eval.txt`：固定 TTSD-eval 直接依赖和 WeSpeaker commit；框架
  `torch/torchaudio==2.8.0` 按 CPU/CUDA profile 单独安装。
- `source/`、`upstream-original/`、`upstream-npu/`：分别用于 Git 管理、
  原始 CUDA baseline 和 patch 后 CUDA/NPU。

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
- `inference.py` 原本将 JSONL 的全部样本一次性传给 `process_batch()`；TTSD-eval
  等多样本输入会长时间停留在无进度输出的自回归 `model.generate()`，并放大
  padding、KV cache 和 logits 开销。旧 NPU SDPA 路径还会在 `repeat_kv` GQA
  展开处产生大块临时张量并 OOM。

因此本次必须修改上游已有文件并生成 patch，而不是仅新增外部包装脚本。

### 1.4 现有代码审视

| 文件 | 结论 | 说明 |
|---|---|---|
| `inference.py` | 已 patch | 增加 `--device npu/cpu/cuda` 和有界 `--batch_size`；默认 NPU、batch size 1。逐批生成并打印进度，全部 batch 完成后沿用原编号写 WAV；模型与 codec 路径按脚本目录固定读取。 |
| `generation_utils.py` | 已 patch | `load_model()` 按设备内部选择固定 dtype 和 attention backend：NPU 为 BF16 + PFA/IFA，CUDA 保持 BF16 + `flash_attention_2`，CPU 为 FP32 + SDPA；音频读取/写出改为 `soundfile`。 |
| `modeling_asteroid.py` | 已 patch | 裁剪 shifted speech channels 后同步 `cur_len`；注册内部 NPU Flash Attention backend，prefill/decode 分别调用 PFA/IFA，并直接传递 GQA KV head 数。 |
| `gradio_demo.py` / `podcast_generate.py` | 已 patch | WAV 写出复用 `save_audio_file()`，移除 `torchaudio.save` 路径。 |
| `XY_Tokenizer/inference.py` | 已 patch | 默认设备改为 NPU，增加 NPU/CUDA 可用性检查。 |
| `XY_Tokenizer/utils/helpers.py` | 已 patch | 音频文件读写改为 `soundfile`，继续保留 `torchaudio.functional.resample`。 |
| `XY_Tokenizer/xy_tokenizer/model.py` | 已 patch | `encode/decode` 默认从输入 tensor 推断设备，不再默认 CUDA。 |
| `XY_Tokenizer/xy_tokenizer/nn/quantizer.py` | 已 patch | autocast device_type 使用当前 tensor device。 |
| `requirements.txt` | 已 patch | 删除 CUDA/ROCm 专用 `flash-attn` 依赖，NPU 环境直接安装 patch 后的 `requirements.txt`。 |

### 1.5 设备适配点

1. `inference.py::_resolve_device`：仅当 `--device npu` 时导入 `torch_npu` 注册后端；返回 `torch.device('npu')`，不绑定卡号。
2. `generation_utils.load_model()`：只接收已解析的设备，内部固定选择 dtype 和 attention backend，不增加注意力相关 CLI 参数。
3. `model.to(device)`、`spt.to(device)`：模型和 codec 显式迁移到目标设备。
4. `XY_Tokenizer.encode/decode`：默认从输入 tensor 推断设备，避免在 NPU 路径创建 CUDA tensor。
5. `ResidualVQ.forward()`：`torch.autocast(device_type=z.device.type, enabled=False)`，避免 CUDA-only autocast。
6. 音频 I/O：文件读取/写出走 `soundfile`；重采样仍使用 `torchaudio.functional.resample`，该路径不触发 TorchCodec 文件解码。
7. 显存清理：CUDA 使用 `torch.cuda.empty_cache()`，NPU 使用 `torch.npu.empty_cache()`。
8. attention mask：裁剪 shifted speech channels 后重置 `cur_len = input_ids.shape[1]`，保证 `input_ids`、`attention_mask`、cache position 长度一致。
9. NPU GQA attention：内部 Flash Attention backend 使用 PFA/IFA 的 `num_key_value_heads` 参数，避免 SDPA/eager 对 KV 执行 `repeat_kv`；复用 Transformers SDPA 的布尔 causal/padding mask 生成逻辑，但禁用 mask-skip，再转换为 NPU 算子的“True 表示屏蔽”语义。
10. 长清单生成：`--batch_size` 默认 `1`，保持输入顺序和 `output_N.wav` 编号，
    每批返回后打印 `[Batch i/N]`，避免 TTSD-eval 整集单批造成长时间无日志以及
    过量 padding/KV cache。

### 1.6 风险与限制

- 当前环境缺少 `torch`、`torch-npu`、模型权重和 NPU/CANN，未在本机执行 CPU/NPU 实推。
- 权重 SHA256 尚未记录；正式验收前必须补充。
- 原项目 v0.5 中仍存在若干宽泛 `try/except` 和失败后继续处理的逻辑；本次 patch 以 NPU 设备适配为目标，没有重构原项目整体错误处理。
- NPU Flash Attention 依赖目标 `torch-npu` 的 PFA/IFA GQA 接口；当前本地环境无 NPU，尚未完成真实算子精度/性能验证，正式验收必须补齐与 CPU/CUDA 原始路径的结果对齐。
- 即使消除 `repeat_kv`，过大的 `--batch_size` 仍可能因输入 embedding、logits、
  KV cache 或 codec 中间张量超过 HBM；TTSD-eval 默认保持 `1`，不得用 CPU 回退
  掩盖 NPU 问题。
- 生成式 TTS/TTSD 不能用“能输出 WAV”作为完整验收；正式验收需按 `ACCEPTANCE_PLAN.md` 做可懂度、音色、自然度和人工听测。
- NeMo/Transformers/PyTorch/TorchAudio 版本持续变化；若上游或依赖升级，应重新检查 `flash-attn`、TorchCodec、attention backend 和 `GenerationMixin` 行为。

### 1.7 上游版本检查记录

- 2026-06-17：确认 `OpenMOSS/MOSS-TTSD` tag `v0.5` commit 为 `0e078c62389922d3aa873ce182daf31142860b18`。
- 2026-06-17：确认当前 main HEAD 已面向 v1.0，不作为本次适配对象。
- 2026-06-17：扫描 v0.5 原项目 CUDA 假设，确认需 patch 原项目已有文件。
- 2026-06-17：固定模型权重和 codec checkpoint 的 HF/ModelScope revision，待正式下载后补充 SHA256。
- 2026-06-18：根据 TTSD-eval NPU OOM 栈确认整集 batch 在 `sdpa_attention.repeat_kv` 处放大 GQA KV。
- 2026-06-18：新增内部 PFA/IFA GQA backend，随后收敛接口，仅保留 `--device`；NPU 自动使用 Flash Attention，不再暴露 dtype、attention、batch 或权重路径参数。
- 2026-06-22：根据 TTSD-eval 长时间停在 `Starting batch audio generation...`
  的现场信息，恢复最小 `--batch_size` 参数，默认单样本并增加逐批进度日志。
- 2026-06-22：从干净目录重放 TTSD-eval 固定 commit 和 testset，确认中英文各
  50 条、200 个 prompt WAV；在独立 Python 3.11 + PyTorch/TorchAudio 2.8.0 CPU
  环境完成 pinned dependencies 安装、`pip check`、五个 CLI `--help`、WER fixture、
  MMS-FA/WeSpeaker hash 与模型加载。Whisper-large-v3 全量下载和加载仍由正式验收
  环境执行，不能据此标记 L2 已完成。

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

从干净源码创建原始和 patch 后工作树：

```bash
git clone https://github.com/OpenMOSS/MOSS-TTSD.git source
git -C source checkout 0e078c62389922d3aa873ce182daf31142860b18
git -C source worktree add --detach ../upstream-original \
  0e078c62389922d3aa873ce182daf31142860b18
git -C source worktree add --detach ../upstream-npu \
  0e078c62389922d3aa873ce182daf31142860b18
git -C upstream-npu apply --check \
  ../patches/0001-adapt-v0.5-inference-to-npu.patch
git -C upstream-npu apply \
  ../patches/0001-adapt-v0.5-inference-to-npu.patch
```

禁止对同一个工作树反复 `reset --hard` 后交替充当原始和 patch 路径；这会破坏三组
baseline 的可审计性。

### 2.3 环境准备

目标配套固定为固件/驱动 25.5.1+、CANN 8.5.1、Python 3.11、
PyTorch/torch-npu/torchaudio 2.9.0、Transformers 4.57.6。NPU 环境不得先安装
PyTorch CPU wheel：

```bash
python3.11 -m venv .venv-npu
source .venv-npu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 torch-npu==2.9.0 torchaudio==2.9.0 \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
python -m pip install transformers==4.57.6
python -m pip install -r upstream-npu/requirements.txt
python -m pip install -r upstream-npu/XY_Tokenizer/requirements.txt
python - <<'PY'
import torch
import torch_npu
print(torch.__version__, torch.randn(1).to("npu").device)
PY
deactivate
```

依赖关系说明：

- `flash-attn` 官方包面向 CUDA/ROCm GPU kernel，不作为 Ascend NPU 依赖。
- NPU 推理固定使用 torch-npu PFA/IFA，不提供 attention backend 参数；CPU 使用 SDPA，CUDA 保持原项目 `flash_attention_2`。
- `soundfile` 已在原项目依赖中声明，本适配用它替代 `torchaudio.load/save` 文件 I/O。
- 原始 CUDA 和 patch 后 CUDA 使用两个独立 venv，并安装相同版本的
  PyTorch、Transformers 4.57.6 和 `flash-attn`。CUDA wheel 索引必须按现场 CUDA
  版本选择并记录，不能复用 NPU 环境。
- 如果安装依赖时 pip 试图替换已有 NPU 版 PyTorch，应修正版本约束；不得临时过滤
  requirements 或让 pip 静默替换。
- TTSD-eval 评测器支持 NPU profile：复用 `.venv-npu`（不另建 venv），对 TTSD-eval
  工作树应用 `patches/0002-adapt-ttsd-eval-to-npu.patch` 后，三个评测器
  （`align.py`/`run_similarity.py`/`whisper_asr.py`）通过 `--device npu:0` /
  `--device npu` 在 NPU 上推理；CPU/CUDA profile 仍使用独立 venv 且不应用该补丁。
  `prepare_eval_data.py verify-ttsd-eval --expected_device npu` 会校验补丁已应用且
  patch 后文件 SHA256 匹配。

### 2.4 权重下载

官方权重与 codec：

| 资产 | URL | revision / HEAD | 目标路径 |
|---|---|---|---|
| MOSS-TTSD-v0.5 模型权重 | HF <https://huggingface.co/fnlp/MOSS-TTSD-v0.5>；同内容别名 <https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.5>；ModelScope <https://modelscope.cn/models/openmoss/MOSS-TTSD-v0.5> | HF `8527b9136b6afefe2252ae597cecea2e80e7ebeb`；ModelScope `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59` | `weights/MOSS-TTSD-v0.5/` |
| XY Tokenizer checkpoint | HF <https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0>；ModelScope <https://modelscope.cn/models/openmoss/XY_Tokenizer_TTSD_V0> | HF `c83433728e698ed0698e88cb5096bc221fb8f8c5`；ModelScope `79082154409f5e883d9487c4d4b4be363323b039` | `XY_Tokenizer/weights/xy_tokenizer.ckpt` |
| XY Tokenizer config | 原项目 tag `v0.5` 自带 | `0e078c62389922d3aa873ce182daf31142860b18` | `XY_Tokenizer/config/xy_tokenizer_config.yaml` |

下载命令见 `README.md`。正式验收只使用固定 HF revision 的同一份 cache/
snapshot；原始 CUDA 通过 `HF_HOME` 离线读取，patch 后 CUDA/NPU 通过符号链接读取。
正式验收前记录模型核心权重和 `xy_tokenizer.ckpt` SHA256。

TTSD-eval 还需要独立评测权重，不能只准备主模型和 codec：

| 评测资产 | 固定版本 | 目标路径 |
|---|---|---|
| WeSpeaker | `voxblink2_samresnet100_ft.zip`，SHA256 `ad0873d380acaa7f4256ff37d40217ee31e4955b26a45064a13a14998cc89d16` | `third_party/TTSD-eval/model/voxblink2_samresnet100_ft/` |
| MMS-FA | S3 version ID `dZWoHyjLHoCxDn.KL1FPSlVCD3CPRtOL`，SHA256 `20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680` | `third_party/TTSD-eval/model/checkpoints/model.pt` |
| Whisper-large-v3 | HF revision `06f233fe06e710322aca913c1bc4249a0d71fce1`，`model.safetensors` SHA256 `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` | `third_party/TTSD-eval/model/whisper-large-v3/` |

完整源码、testset、独立环境、下载、校验和离线加载命令见 `README.md`
的“准备 TTSD-eval 工程”。缺少任一环节时，ACC/SIM/WER 闭环未完成。

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

### 2.7 推理和评测边界

正式迁移验收固定三组：

- `upstream-original` + `.venv-cuda-original`：未应用 patch 的原始 CUDA；
- `upstream-npu` + `.venv-cuda-patched`：应用 patch 后的 CUDA 回归；
- `upstream-npu` + `.venv-npu`：NPU candidate。

三组完整命令、输出目录、功能/L2 manifest 和 TTSD-eval evaluator 命令统一维护在
`README.md` 与 `ACCEPTANCE_PLAN.md`。本文件不复制第二套易漂移的操作
手册。

### 2.8 上游更新处理

上游更新时必须重新执行：

```bash
git -C source fetch origin
git -C source rev-parse origin/main
grep -RIn "cuda\|gpu\|npu\|flash_attention\|torchaudio\.load\|torchaudio\.save\|to(device)" \
  source \
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
MODEL_ROOT="$PWD"
CHECK_DIR="$(mktemp -d)"
git clone https://github.com/OpenMOSS/MOSS-TTSD.git "$CHECK_DIR/source"
git -C "$CHECK_DIR/source" checkout \
  0e078c62389922d3aa873ce182daf31142860b18
git -C "$CHECK_DIR/source" apply --check \
  "$MODEL_ROOT/patches/0001-adapt-v0.5-inference-to-npu.patch"
git -C "$CHECK_DIR/source" apply \
  "$MODEL_ROOT/patches/0001-adapt-v0.5-inference-to-npu.patch"
python3 -m py_compile \
  "$CHECK_DIR/source/inference.py" \
  "$CHECK_DIR/source/generation_utils.py" \
  "$CHECK_DIR/source/gradio_demo.py" \
  "$CHECK_DIR/source/podcast_generate.py" \
  "$CHECK_DIR/source/modeling_asteroid.py" \
  "$CHECK_DIR/source/XY_Tokenizer/inference.py" \
  "$CHECK_DIR/source/XY_Tokenizer/utils/helpers.py" \
  "$CHECK_DIR/source/XY_Tokenizer/xy_tokenizer/model.py" \
  "$CHECK_DIR/source/XY_Tokenizer/xy_tokenizer/nn/quantizer.py"
! grep -R -I -E 'torchaudio\.(load|save|info)\(' \
  "$CHECK_DIR/source" --exclude-dir=.git
```

结果：patch apply 检查通过；语法检查通过；`torchaudio.load/save/info` 文件 I/O 路径检查通过。

### 3.3 已知 NPU attention mask 报错修复

旧 SDPA 适配路径在 NPU 上可能报：

```text
aclnnFlashAttentionScore failed
get unsupported atten_mask shape, the shape is [B, 1, L+7, L]
```

根因：`modeling_asteroid.py` 自定义生成循环先记录 shifted 输入原始长度，再裁掉 `channels - 1` 个位置用于初始前向，但没有同步 `cur_len`。最新 patch 在裁剪 `input_ids` / `attention_mask` 后重置 `cur_len = input_ids.shape[1]`，使 `input_ids`、`attention_mask` 与 cache position 长度一致。

### 3.4 NPU `repeat_kv` OOM 与 GQA FlashAttention 修复

TTSD-eval 等多样本 JSONL 旧路径会将全部样本作为一个 batch。Transformers 4.57.6 明确在 NPU SDPA 路径禁用原生 GQA，并在 `sdpa_attention_forward()` 中对 key/value 调用 `repeat_kv()`。因此显存临时分配与 batch size、最长 padding 长度和 KV head 展开倍数共同增长。

`eager_attention_forward()` 同样调用 `repeat_kv()`，并显式创建 attention weights，不能解决该性能/显存问题。本次在 `modeling_asteroid.py` 中通过 Transformers 官方 `AttentionInterface` / `AttentionMaskInterface` 扩展点注册内部 NPU Flash Attention backend：

- prefill 调用 `torch_npu.npu_prompt_flash_attention`；
- query length 为 1 的 decode 调用 `torch_npu.npu_incre_flash_attention`；
- Q/K/V 使用 `BNSD`，分别传入 query head 数与 KV head 数；
- 不改 site-packages，不运行时 monkey patch，不静默回退；
- 推理 CLI 增加 `--device` 和 `--batch_size`，不暴露 attention、dtype 或权重路径参数；
- CPU 使用 SDPA，CUDA 保持原始 `flash_attention_2`。

官方接口说明中，PFA/IFA 的 `num_key_value_heads` 用于 GQA，要求 query head 数可整除 KV head 数。参考：

- <https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000144.html>
- <https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000146.html>
- <https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000453.html>

patch 后入口默认按单样本顺序处理 JSONL，每批完成后打印进度；全部 batch 结束后
按原入口规则保存对应 `output_N.wav`。
TTSD-eval 首批在 NPU 上可能触发算子/图编译；`forkserver`/`resource_tracker`
子进程本身不是死锁证据。应结合 `npu-smi info` 的利用率、HBM 变化和 CANN 日志判断。
若 `--batch_size 1` 的单样本仍 OOM 或超过 10 分钟保持 0 利用率且无日志进展，
再按超长样本、算子编译或更深层 KV cache 问题处理。

### 3.5 提交前必跑检查

```bash
python3 -m py_compile prepare_eval_data.py ../tools/audit_model_delivery.py
python3 ../tools/audit_model_delivery.py .
git -C "$CHECK_DIR/source" reset --hard \
  0e078c62389922d3aa873ce182daf31142860b18
git -C "$CHECK_DIR/source" apply --check \
  "$MODEL_ROOT/patches/0001-adapt-v0.5-inference-to-npu.patch"
```

### 3.6 NPU 实测待补项

正式验收环境具备权重和 NPU 后，应补充以下记录：

```bash
python -V
pip freeze | grep -E 'torch|torch-npu|torchaudio|transformers|accelerate|soundfile|librosa|numpy|scipy'
npu-smi info || true
uname -a
sha256sum patches/0001-adapt-v0.5-inference-to-npu.patch
cd upstream-npu
find weights/MOSS-TTSD-v0.5 -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum
sha256sum XY_Tokenizer/weights/xy_tokenizer.ckpt
```

然后按 `ACCEPTANCE_PLAN.md` 执行功能验证和 L2 精度/性能验收，并将报告保存到
`MOSS-TTSD-v0.5/validation_reports/YYYYMMDD_<device>.md`。

### 3.7 当前完成状态

当前状态：**S1 静态适配完成**。

已具备版本取证、patch、静态 apply/compile 证据、manifest 工具和三组验收命令。
尚缺模型与 codec 实际权重、三组功能输出以及 TTSD-eval 全量 ACC/SIM/WER 和
RTF/RTFx，因此未达到 S2 或 S3。

## 补充说明（来自 README.md）

以下内容原位于 `README.md`，因偏重适配实现与技术解释，迁移至此以便终端用户文档保持简洁。

### 适配原则

- 不修改原始 `README_old.md`。
- 不新增旁路推理脚本；继续使用原项目已有 `inference.py`，通过 patch 适配 NPU。
- NPU 默认显式使用 `--device npu`，实际卡号由 `ASCEND_RT_VISIBLE_DEVICES` 控制。
- NPU 路径内部固定使用 torch-npu Flash Attention：prefill 调用 `npu_prompt_flash_attention`，decode 调用 `npu_incre_flash_attention`，直接传递 GQA 的 KV head 数，不执行 `repeat_kv`。
- 推理入口只新增一个 `--device` 参数，不向用户暴露 dtype、attention backend、batch size 或权重路径等额外开关。

### NPU GQA FlashAttention 与显存说明

如果 TTSD-eval 或其他多样本 JSONL 在以下位置报 NPU OOM：

```text
transformers/integrations/sdpa_attention.py
value = repeat_kv(value, module.num_key_value_groups)
RuntimeError: NPU out of memory
```

原因是 Transformers 4.57.6 的 NPU SDPA 路径暂不使用原生 GQA，会把 Qwen3 的 key/value heads 通过 `repeat_kv` 实体扩展到全部 attention heads。`eager` 也会执行相同展开，并额外显式构造 attention weights，因此不是该问题的性能修复。

当前 patch 在 NPU 设备路径内部固定选择 Flash Attention backend：

- prefill：`torch_npu.npu_prompt_flash_attention`；
- 单 token decode：`torch_npu.npu_incre_flash_attention`；
- Q/K/V 使用 `BNSD` 布局；
- `num_heads` 和 `num_key_value_heads` 分别取 query 与 key 的 head 数，由算子直接处理 GQA；
- 不增加 attention CLI 参数，不修改 Transformers site-packages，也不静默回退到 SDPA/eager。

目标 `torch-npu` 必须同时提供上述两个接口及 `num_key_value_heads` 参数。运行前可检查：

```bash
python - <<'PY'
import inspect
import torch_npu
print(inspect.signature(torch_npu.npu_prompt_flash_attention))
print(inspect.signature(torch_npu.npu_incre_flash_attention))
PY
```

性能评测直接使用 NPU 设备参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl ../third_party/TTSD-eval/testset/ttsd_eval_zh.jsonl \
  --output_dir ../results/ttsd_eval/npu_zh \
  --device npu \
  --batch_size 1 \
  --seed 42 \
  --use_normalize
```

Flash Attention 消除的是 `repeat_kv` 造成的 KV head 实体展开，不保证整份 manifest
作为一个 batch 时的其他张量都能放入 HBM。patch 后入口默认
`--batch_size 1`，逐批生成并显示 `[Batch i/N]` 进度；全部 batch 完成后仍按原
入口规则写出 `output_N.wav`，避免 50 条长音频全部完成前没有任何进度。

若日志停在 `Starting batch audio generation...`：

1. 先执行 `watch -n 1 npu-smi info`。首批可能触发 NPU 算子/图编译；出现
   `multiprocessing.forkserver` / `resource_tracker` 子进程本身不能证明死锁。
2. 若 NPU 利用率或 HBM 持续变化，先等待首批完成；之后应出现
   `Original outputs shape` 和 `[Batch 1/N] completed`。
3. 若超过 10 分钟 NPU 利用率始终为 0、HBM 不变且无新 CANN 日志，终止进程，
   用 `--batch_size 1` 和单条 manifest 重跑。单条仍卡住时再保留完整 Python
   栈、`npu-smi info`、CANN 日志和依赖版本排查，不能静默切到 CPU。

CPU/CUDA/NPU 候选对齐必须使用相同 `--batch_size`。未应用 patch 的原始入口不支持
该参数，仍保留其原生完整 JSONL batch 作为 upstream baseline；报告中必须明确记录
这一运行参数差异，不能把不同 batch 口径写成严格逐样本数值等价。

### 已知问题：Transformers 5.x 不兼容

如果环境安装了 `transformers==5.12.1`，模型加载阶段可能报：

```text
AttributeError: 'list' object has no attribute 'keys'
```

报错位置通常位于 Transformers 的 `get_expanded_tied_weights_keys()`。原因是 Transformers 5.x 要求 `_tied_weights_keys` 为“目标权重到源权重”的字典映射，而 MOSS-TTSD-v0.5 上游 `modeling_asteroid.py` 仍按 Transformers 4.x 接口将其定义为列表。

当前项目仅记录该依赖边界，不修改上游模型代码。请在运行推理或 TTSD-eval 前固定已验证版本：

```bash
pip uninstall -y transformers
pip install "transformers==4.57.6"
python -c "import transformers; print(transformers.__version__)"
```

预期输出为 `4.57.6`。本地已验证该版本可以完成 `AsteroidTTSInstruct.from_pretrained()` 初始化并保持各通道输入 embedding 与 `lm_heads` 的权重绑定。不要通过忽略异常、关闭权重绑定或 CPU 回退绕过该错误。

### 依赖与环境说明

- **TTSD-eval 评测器依赖**：TTSD-eval 不是无权重评测器。ACC/SIM 依赖 MMS-FA 和 WeSpeaker `voxblink2_samresnet100_ft`，WER 依赖 `openai/whisper-large-v3`。评测器只读取已生成的 WAV。
- **Python 版本与 hdbscan**：上游 README 示例使用 Python 3.12，但固定 WeSpeaker commit 依赖的 `hdbscan==0.8.37` 没有 CPython 3.12 manylinux wheel；CPU/CUDA profile 固定使用已完成安装和 import 验证的 Python 3.11，避免不可复现的本地 C 扩展构建。
- **flash-attn**：`flash-attn` 官方包面向 CUDA/ROCm GPU kernel，当前不作为 Ascend NPU 必需依赖安装。NPU 内部固定使用 torch-npu 原生 PFA/IFA；CUDA 路径保持原项目 `flash_attention_2`，CPU 路径使用 SDPA。
- **TorchCodec**：TorchAudio 2.9+ 的 `torchaudio.load` / `torchaudio.save` 会进入 TorchCodec 路径。本适配通过 patch 将 prompt 音频读取和 WAV 写出改为 `soundfile`，不要求额外安装 `torchcodec`。如果仍看到 `TorchCodec is required for load_with_torchcodec` 或 `save_with_torchcodec`，说明 patch 未应用或路径未覆盖。
- **Transformers 5.x `_tied_weights_keys`**：Transformers 5.x 改变了 `_tied_weights_keys` 的数据结构和 `tie_weights()` 接口，而 MOSS-TTSD-v0.5 上游代码仍使用 Transformers 4.x 接口。当前项目不修改该模型定义，因此必须固定 `transformers==4.57.6`。
- **HF_HOME 与符号链接**：原始代码继续使用 repo id `fnlp/MOSS-TTSD-v0.5`，执行时设置同一个 `HF_HOME` 和 `HF_HUB_OFFLINE=1`，从上述固定 revision cache 加载；patch 后代码通过符号链接读取同一 snapshot。
- **batch_size 对齐**：CPU/CUDA/NPU 候选对齐必须使用相同 `--batch_size`。未应用 patch 的原始入口不支持该参数，仍保留其原生完整 JSONL batch 作为 upstream baseline；报告中必须明确记录这一运行参数差异，不能把不同 batch 口径写成严格逐样本数值等价。
