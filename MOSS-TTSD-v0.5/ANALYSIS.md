# MOSS-TTSD-v0.5 适配分析

## 1. 参考原始仓库与版本边界

检查日期：2026-06-17。

| 项 | 当前记录 |
|---|---|
| 原项目源码 | <https://github.com/OpenMOSS/MOSS-TTSD> |
| 原项目版本 | tag `v0.5` |
| tag commit | `0e078c62389922d3aa873ce182daf31142860b18` |
| 当前 main HEAD | `20dbb4fc44819435fee894d644a0402a0fee736a`，已面向 v1.0 |
| 本地 upstream | `MOSS-TTSD-v0.5/upstream/`，仅用于生成/校验 patch，不提交 |
| 模型权重 | `fnlp/MOSS-TTSD-v0.5` / `OpenMOSS-Team/MOSS-TTSD-v0.5`；HF <https://huggingface.co/fnlp/MOSS-TTSD-v0.5>，HF HEAD `8527b9136b6afefe2252ae597cecea2e80e7ebeb`；ModelScope <https://modelscope.cn/models/openmoss/MOSS-TTSD-v0.5>，ModelScope HEAD `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59` |
| XY Tokenizer | 原 v0.5 代码默认 `XY_Tokenizer/config/xy_tokenizer_config.yaml` + `XY_Tokenizer/weights/xy_tokenizer.ckpt`；HF checkpoint <https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0>，HF HEAD `c83433728e698ed0698e88cb5096bc221fb8f8c5`；ModelScope checkpoint <https://modelscope.cn/models/openmoss/XY_Tokenizer_TTSD_V0>，ModelScope HEAD `79082154409f5e883d9487c4d4b4be363323b039` |
| 当前适配对象 | GitHub tag `v0.5` 原项目代码 + MOSS-TTSD-v0.5 权重 + XY Tokenizer checkpoint |
| 明确排除 | MOSS-TTSD v0.7、MOSS-TTSD v1.0、SGLang 路径、未固定版本的一键包改动 |

权重 SHA256 尚未记录：当前环境未下载大权重和 `xy_tokenizer.ckpt`。正式验收前必须补充模型权重与 codec checkpoint 的 SHA256。

## 2. 新约束下的适配策略

根据项目约束，本次重新收敛为：

- 不修改原始 `README.md`。
- 不新增独立推理/下载/验证代码文件。
- 优先使用原项目 tag `v0.5` 已有 `inference.py`、`generation_utils.py`、`XY_Tokenizer` 代码。
- 必要代码改动统一进入 patch：`patches/0001-adapt-v0.5-inference-to-npu.patch`。

## 3. 设备相关扫描结论

v0.5 原项目主要 CUDA 假设：

- `inference.py` 根据 `torch.cuda.is_available()` 自动选择 `cuda/cpu`，无显式 NPU 参数。
- `generation_utils.py` 固定 `attn_implementation="flash_attention_2"`，并在结束时调用 `torch.cuda.empty_cache()`。
- `requirements.txt` 包含 `flash-attn`，但官方 `flash-attn` 包面向 CUDA/ROCm GPU kernel，不作为 Ascend NPU 依赖安装；NPU 适配路径改用显式 `sdpa/eager` attention 后端。
- 新版 TorchAudio 的 `torchaudio.load` 会进入 TorchCodec 路径；NPU 适配不新增 `torchcodec` 依赖，改用原 requirements 已包含的 `soundfile` 读取 prompt 音频，避免 `load_with_torchcodec` 报错。
- `XY_Tokenizer/inference.py` 默认 `--device cuda`。
- `XY_Tokenizer/xy_tokenizer/model.py` 的 `encode/decode` 默认 `device=torch.device("cuda")`，即使输入 tensor 已在 NPU 也会创建 CUDA tensor。
- `XY_Tokenizer/xy_tokenizer/nn/quantizer.py` 使用 `torch.autocast('cuda', enabled=False)`。
- `modeling_asteroid.py` 自定义 `_sample` 先记录 shifted 输入原始长度，再裁掉 `channels - 1` 个位置用于初始前向；若不同步 `cur_len`，NPU `sdpa` 下发的 `aclnnFlashAttentionScore` 会收到 query/key 长度不一致的 attention mask。

当前 patch 对这些原项目已有文件做最小修改：

- 新增显式 `--device npu/cpu/cuda`，默认 NPU；仅在 NPU 路径条件导入 `torch_npu`。
- 新增 `--dtype`、`--attn_implementation` 和模型/codec 路径参数，避免硬编码环境。
- `XY_Tokenizer.encode/decode` 默认从输入 tensor 推断设备。
- quantizer autocast 使用当前 tensor device。
- 清理显存时按 `cuda/npu` 分支调用对应 empty cache。
- 裁剪 shifted speech channels 后重置 `cur_len`，保证 cache position 与初始 `input_ids` / `attention_mask` 长度一致。

## 4. 当前交付件

- `patches/0001-adapt-v0.5-inference-to-npu.patch`：唯一代码适配交付。
- `patches/README.md`：patch 应用和校验说明。
- `README_INFERENCE.md`：基于原项目 `inference.py` 的快速推理说明。
- `NPU_ADAPTATION.md`、`NPU_VALIDATION.md`、`ACCEPTANCE_PLAN.md`：环境、验证和分层验收说明。

## 5. 风险与待验证项

- 当前环境缺少 `torch`、`torch-npu`、模型权重和 NPU/CANN，未执行 CPU/NPU 实推。
- 原项目 v0.5 中仍存在若干宽泛 `try/except` 和失败后继续处理的逻辑；本次 patch 以 NPU 设备适配为目标，没有重构原项目整体错误处理。
- 生成式 TTS/TTSD 不能用“能输出 WAV”作为完整验收；正式验收需按 `ACCEPTANCE_PLAN.md` 做可懂度、音色、自然度和人工听测。
