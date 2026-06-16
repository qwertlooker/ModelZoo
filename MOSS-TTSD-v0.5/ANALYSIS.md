# MOSS-TTSD-v0.5 适配分析

## 1. 参考原始仓库与版本边界

检查日期：2026-06-16。

| 项 | 当前记录 |
|---|---|
| GitHub 源码 | <https://github.com/OpenMOSS/MOSS-TTSD> |
| GitHub 默认分支 | `main` |
| GitHub HEAD | `20dbb4fc44819435fee894d644a0402a0fee736a` |
| 本地 upstream | `MOSS-TTSD-v0.5/upstream/`，仅用于对照，不提交 |
| 模型权重 | `OpenMOSS-Team/MOSS-TTSD-v0.5` / `fnlp/MOSS-TTSD-v0.5` |
| 模型权重 HEAD | `8527b9136b6afefe2252ae597cecea2e80e7ebeb` |
| 辅助 codec | `OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf` / `fnlp/XY_Tokenizer_TTSD_V0_hf` |
| codec HEAD | `c884072fd69ed00b72cd0d43355c06341c4f51a6` |
| 当前适配对象 | MOSS-TTSD-v0.5 + XY Tokenizer TTSD V0 HF remote-code 路径 |
| 明确排除 | MOSS-TTSD v0.7、MOSS-TTSD v1.0、SGLang 端到端服务化路径、历史第三方一键整合包内未验证改动 |

权重 SHA256 尚未记录：本次环境未下载大权重。正式验收下载后必须补充 `model.safetensors` 和 codec `pytorch_model.bin` 的 SHA256。

## 2. 上游状态判断

GitHub `OpenMOSS/MOSS-TTSD` 的当前顶层 README 已面向 v1.0，并在 `legacy/v0.7/` 保留 v0.7 资料；v0.5 的可加载模型实现随 Hugging Face / ModelScope 模型仓库以 remote-code 文件发布，包括：

- `configuration_moss_ttsd.py`
- `modeling_moss_ttsd.py`
- `processing_moss_ttsd.py`
- `model.safetensors`
- `XY_Tokenizer_TTSD_V0_hf` 的 `configuration_xy_tokenizer.py`、`modeling_xy_tokenizer.py`、`feature_extraction_xy_tokenizer.py`、`pytorch_model.bin`

因此，本次细化适配不再要求用户对截图中列出的多个旧文件手工 `cuda -> npu`，而是提供一个固定 revision 的 `infer.py`：

1. 通过 `snapshot_download(..., revision=...)` 固定模型和 codec；
2. 使用本地 snapshot 加载 `AutoProcessor` / `AutoModel`；
3. 显式 `model.to(device)` 和 `processor.audio_tokenizer.to(device)`；
4. 默认 `--device npu`，CPU 验证显式 `--device cpu`；
5. 不使用 `device_map="auto"`，不写死卡号。

## 3. 设备相关代码扫描

本次对下载的 v0.5 remote-code 小文件做静态扫描，主要发现：

```text
xy_modeling_xy_tokenizer.py: with torch.autocast('cuda', enabled=False)
```

该语句位于禁用 autocast 的上下文中，没有直接 `.cuda()`、`torch.cuda.*`、`device="cuda"` 或 `map_location="cuda"` 的执行迁移逻辑。本次不对 remote-code 生成 patch；若 NPU 实测证明该语句触发后端错误，后续应在本地 snapshot 或上游代码中形成明确 patch，并记录影响。

GitHub 当前 top-level README / 示例代码中仍有 CUDA/GPU 示例，例如 `device = "cuda" if torch.cuda.is_available() else "cpu"`、`device_map="auto"` 等。这些属于上游文档示例，不进入本适配默认执行路径。

## 4. 当前适配新增内容

- `infer.py`：统一推理入口。
- `download_weights.py`：固定 revision 权重下载。
- `prepare_test_data.py`：生成最小 JSONL schema 与合成 prompt wav。
- `validate_outputs.py`：结构性输出检查。
- `requirements.txt`：除 torch/torch-npu 外的最小依赖提示。
- `patches/README.md`：说明当前没有上游 patch。
- `README.md`、`README_INFERENCE.md`、`NPU_ADAPTATION.md`、`NPU_VALIDATION.md`、`ACCEPTANCE_PLAN.md`：按项目标准补齐说明。

## 5. 风险与待验证项

- 未在当前环境下载大权重，未记录权重 SHA256。
- 当前环境缺少 `torch`、`torch-npu`、`transformers`、`torchaudio`，因此未执行 CPU/NPU 实推。
- TTS 质量不能用结构性 WAV 检查替代；正式验收必须包含 ASR 回识别、speaker similarity、DNSMOS/UTMOS、人工 MOS/CMOS 或官方 TTSD-eval 口径。
- NPU 上 `sdpa`、`eager`、`flash_attention_2` 的可用性依赖 torch/torch-npu/CANN 组合；默认选择 `sdpa`，失败时应暴露原始错误，不自动改用 CPU 或其他未验证 backend。
