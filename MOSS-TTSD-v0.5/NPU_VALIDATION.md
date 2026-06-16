# MOSS-TTSD-v0.5 验证记录

## 1. 当前环境验证结果

检查日期：2026-06-16。

| 项 | 结果 |
|---|---|
| 工作目录 | `/home/pei/ModelZoo` |
| GitHub upstream clone | `MOSS-TTSD-v0.5/upstream/` |
| GitHub HEAD | `20dbb4fc44819435fee894d644a0402a0fee736a` |
| `git ls-remote --symref origin HEAD` | `refs/heads/main` -> `20dbb4fc44819435fee894d644a0402a0fee736a` |
| HF 模型 HEAD | `OpenMOSS-Team/MOSS-TTSD-v0.5` -> `8527b9136b6afefe2252ae597cecea2e80e7ebeb` |
| HF codec HEAD | `OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf` -> `c884072fd69ed00b72cd0d43355c06341c4f51a6` |
| 当前系统 Python | `Python 3.12.3` |
| 当前环境依赖 | 未安装 `torch`、`torch-npu`、`transformers`、`torchaudio`、`huggingface_hub` |
| 大权重下载 | 未下载 |
| CPU 实推 | 未执行，原因：当前环境缺少依赖和权重 |
| NPU 实推 | 未执行，原因：当前环境无 Ascend NPU/CANN 运行条件 |

本次已完成的本地检查：

```bash
python3 -m py_compile \
  MOSS-TTSD-v0.5/infer.py \
  MOSS-TTSD-v0.5/download_weights.py \
  MOSS-TTSD-v0.5/prepare_test_data.py \
  MOSS-TTSD-v0.5/validate_outputs.py

python3 MOSS-TTSD-v0.5/prepare_test_data.py \
  --output_dir /tmp/moss_ttsd_test_data
```

> 说明：`prepare_test_data.py` 只依赖 Python 标准库，可验证 JSONL 与 WAV schema；它不代表模型音质验收。

## 2. 提交前必跑检查

### 2.1 patch 检查

当前没有 `.patch` 文件。如后续新增 patch，提交前执行：

```bash
for p in MOSS-TTSD-v0.5/patches/*.patch; do
  [ -e "$p" ] || continue
  git -C MOSS-TTSD-v0.5/upstream apply --check "../patches/$(basename "$p")"
done
```

### 2.2 语法检查

```bash
python -m py_compile \
  MOSS-TTSD-v0.5/infer.py \
  MOSS-TTSD-v0.5/download_weights.py \
  MOSS-TTSD-v0.5/prepare_test_data.py \
  MOSS-TTSD-v0.5/validate_outputs.py
```

### 2.3 权重下载与校验

```bash
python MOSS-TTSD-v0.5/download_weights.py \
  --output_dir MOSS-TTSD-v0.5/weights

sha256sum MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5/model.safetensors
sha256sum MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf/pytorch_model.bin
```

把 SHA256 补回 `README.md` / `NPU_ADAPTATION.md` / 验收报告后再对外声明固定权重。

## 3. CPU 验证命令

CPU 基线用于确认同 checkpoint、同 JSONL 下的功能和质量，不用于 NPU 性能结论。

```bash
python MOSS-TTSD-v0.5/prepare_test_data.py \
  --output_dir MOSS-TTSD-v0.5/test_data

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

python MOSS-TTSD-v0.5/validate_outputs.py \
  --manifest MOSS-TTSD-v0.5/outputs_cpu/manifest.jsonl
```

通过条件：

- 退出码为 0；
- `manifest.jsonl` 中每条输入至少有一个 WAV；
- WAV 可读且时长大于 0；
- `run_report.json` 记录 `elapsed_seconds`、`generated_audio_seconds`、`rtf`、`rtfx`。

## 4. NPU 验证命令

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

python MOSS-TTSD-v0.5/validate_outputs.py \
  --manifest MOSS-TTSD-v0.5/outputs/manifest.jsonl
```

通过条件：

- 无 `Expected all tensors to be on the same device`；
- 无 CUDA-only / NCCL-only / `device_map="auto"` 相关错误；
- 无 silent CPU fallback；
- 输出 WAV 可读且非零时长；
- `run_report.json` 记录 NPU 设备环境和 RTF/RTFx。

如 `sdpa` 在目标 torch-npu 组合上不可用，应显式改用 `--attn_implementation eager` 重新跑并在报告中记录，不在脚本内自动降级。

## 5. 正式验收报告模板

```text
模型：MOSS-TTSD-v0.5
模型 revision：8527b9136b6afefe2252ae597cecea2e80e7ebeb
codec revision：c884072fd69ed00b72cd0d43355c06341c4f51a6
模型权重 SHA256：
codec SHA256：
日期：
硬件：NPU 型号 / 数量 / HBM
驱动/固件/CANN：
Python / torch / torch-npu / transformers / torchaudio：
命令：
输入 JSONL：
样本数 / 总目标文本字数 / prompt 总时长：
输出总时长：
elapsed_seconds：
RTF：
RTFx：
峰值 HBM/RSS：
结构检查：通过/失败
ASR 回识别 CER/WER：
说话人相似度：
DNSMOS/UTMOS：
人工 MOS/CMOS 或 A/B 偏好：
结论：通过/不通过/需复测
问题与日志：
```
