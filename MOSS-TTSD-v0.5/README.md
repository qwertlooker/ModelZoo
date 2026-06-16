---
license: apache-2.0
hardware: NPU
---
# MOSS-TTSD-v0.5 NPU 推理适配

本目录将 MOSS-TTSD-v0.5 的部署说明细化为与 `Canary-1B/` 类似的交付结构：固定版本边界、提供统一 `infer.py`、权重下载脚本、测试数据脚本、验证说明和分层验收方案。

> 版本边界：当前适配对象是 Hugging Face / ModelScope 同步的 `OpenMOSS-Team/MOSS-TTSD-v0.5`（同 `fnlp/MOSS-TTSD-v0.5`）与 `OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf`（同 `fnlp/XY_Tokenizer_TTSD_V0_hf`），不是 MOSS-TTSD v0.7，也不是 v1.0 / SGLang 路径。检查日期：2026-06-16。

## 1. 适配结论

- 上游源码仓库：<https://github.com/OpenMOSS/MOSS-TTSD>
  - 默认分支：`main`
  - 本地 clone/远端 HEAD：`20dbb4fc44819435fee894d644a0402a0fee736a`
  - 当前 GitHub 顶层文档已面向 v1.0；v0.7 位于 `legacy/v0.7/`，v0.5 模型代码以 Hugging Face remote-code snapshot 为准。
- 模型权重：`OpenMOSS-Team/MOSS-TTSD-v0.5`，HEAD `8527b9136b6afefe2252ae597cecea2e80e7ebeb`。
- 辅助 codec：`OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf`，HEAD `c884072fd69ed00b72cd0d43355c06341c4f51a6`。
- 本次不修改 `MOSS-TTSD-v0.5/upstream/` 中的 GitHub 上游已有文件，因此没有 `.patch`。
- `infer.py` 是当前适配新增脚本，默认 `--device npu`；CPU 验证显式使用 `--device cpu`。
- 不使用 `device_map="auto"`，不写死 `npu:0` / `cuda:0`；实际 NPU 卡号由 `ASCEND_RT_VISIBLE_DEVICES` 控制。
- 历史 ModelScope 一键整合包 `xueshanlinghu/MOSS-TTSD-zhenghebao` 可作为旧部署资料参考，但本目录默认以官方 HF 模型/codec snapshot 为适配边界，避免把未验证的一键包内部手工改动混入默认路径。

## 2. 文件说明

| 文件 | 作用 |
|---|---|
| `infer.py` | 统一 CPU/NPU 推理入口，支持 JSONL 批量输入、固定模型/codec revision、输出 manifest 和 RTF/RTFx 报告。 |
| `download_weights.py` | 下载固定 revision 的 MOSS-TTSD-v0.5 和 XY Tokenizer snapshot。 |
| `prepare_test_data.py` | 生成最小 JSONL schema 与合成 prompt wav，仅用于链路冒烟。 |
| `validate_outputs.py` | 检查 `infer.py` 输出 WAV 是否存在、可读、非零时长；不替代音质/精度评测。 |
| `ANALYSIS.md` | 上游、版本边界、设备扫描和当前差距分析。 |
| `NPU_ADAPTATION.md` | 环境、权重、推理和补丁策略说明。 |
| `NPU_VALIDATION.md` | 当前已执行检查、未执行项和 NPU/CPU 验证命令。 |
| `ACCEPTANCE_PLAN.md` | L0/L1/L2/L3 分层验收方案。 |

## 3. 环境准备

NPU 环境中请先安装与 CANN / 驱动匹配的 PyTorch 和 torch-npu，再安装本目录最小依赖。示例版本边界沿用原 README 的 Ascend 约束：驱动/固件 `>=25.0.RC1.1`，CANN Toolkit/Kernel/NNAL `>=8.2.RC1`，PyTorch/torch-npu `>=2.6.0`。实际安装请以目标机器 CANN 版本对应的 torch-npu 发布说明为准。

```bash
# 先安装匹配 CANN 的 torch / torch-npu，再安装通用依赖
pip install torch torch-npu
pip install -r MOSS-TTSD-v0.5/requirements.txt
```

若使用容器，可继续基于原说明中的 `quay.io/ascend/vllm-ascend:v0.10.0rc1`，但本适配脚本不依赖 vLLM/SGLang 服务化路径。

## 4. 权重下载

推荐下载到本地并在推理时使用 `--local_files_only`：

```bash
# 可按需设置镜像 endpoint，但不要替换为未验证的第三方模型仓库
export HF_ENDPOINT=https://hf-mirror.com

python MOSS-TTSD-v0.5/download_weights.py \
  --output_dir MOSS-TTSD-v0.5/weights
```

下载后目录示例：

```text
MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5/
MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf/
```

大权重未随仓提交；正式验收前应记录 `model.safetensors`、`pytorch_model.bin` 等大文件 SHA256。

## 5. 测试数据准备

生成一个最小 JSONL 和两段合成 prompt wav：

```bash
python MOSS-TTSD-v0.5/prepare_test_data.py \
  --output_dir MOSS-TTSD-v0.5/test_data
```

该数据只用于验证 JSONL schema、模型加载、设备迁移、生成和 WAV 保存链路，不用于音质或音色克隆验收。

## 6. 推理命令

### NPU 推理

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-TTSD-v0.5/infer.py \
  --model_path MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5 \
  --codec_path MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf \
  --input_jsonl MOSS-TTSD-v0.5/test_data/smoke.jsonl \
  --output_dir MOSS-TTSD-v0.5/outputs \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --batch_size 1 \
  --text_normalize \
  --local_files_only
```

### CPU 验证

```bash
python MOSS-TTSD-v0.5/infer.py \
  --model_path MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5 \
  --codec_path MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf \
  --input_jsonl MOSS-TTSD-v0.5/test_data/smoke.jsonl \
  --output_dir MOSS-TTSD-v0.5/outputs_cpu \
  --device cpu \
  --dtype float32 \
  --attn_implementation sdpa \
  --batch_size 1 \
  --local_files_only
```

输出：

```text
MOSS-TTSD-v0.5/outputs/manifest.jsonl
MOSS-TTSD-v0.5/outputs/run_report.json
MOSS-TTSD-v0.5/outputs/sample_0000_00.wav
```

结构检查：

```bash
python MOSS-TTSD-v0.5/validate_outputs.py \
  --manifest MOSS-TTSD-v0.5/outputs/manifest.jsonl
```

## 7. 正式验收提醒

MOSS-TTSD 是生成式 TTS/对话语音模型，不能只用 1 条 dummy 样本判定适配完成。正式验收请按 `ACCEPTANCE_PLAN.md`：

- L0：最小链路冒烟；
- L1：中文/英文、单/双说话人、batch、长文本、不同 prompt 的功能矩阵；
- L2：同 checkpoint、同 JSONL 下 CPU/CUDA 源路径与 NPU 的性能/音质/说话人相似度对齐；
- L3：尽量对齐官方/公开 TTSD-eval、ASR 回识别、speaker similarity、DNSMOS/UTMOS 和人工 MOS/CMOS。
