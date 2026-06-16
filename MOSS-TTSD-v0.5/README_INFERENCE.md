# MOSS-TTSD-v0.5 推理快速指南

1. 安装匹配 CANN 的 `torch` / `torch-npu`，再执行：

```bash
pip install -r MOSS-TTSD-v0.5/requirements.txt
```

2. 下载固定版本权重：

```bash
python MOSS-TTSD-v0.5/download_weights.py --output_dir MOSS-TTSD-v0.5/weights
```

3. 准备最小 JSONL：

```bash
python MOSS-TTSD-v0.5/prepare_test_data.py --output_dir MOSS-TTSD-v0.5/test_data
```

4. NPU 推理：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-TTSD-v0.5/infer.py \
  --model_path MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5 \
  --codec_path MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf \
  --input_jsonl MOSS-TTSD-v0.5/test_data/smoke.jsonl \
  --output_dir MOSS-TTSD-v0.5/outputs \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --local_files_only
```

5. 验证输出结构：

```bash
python MOSS-TTSD-v0.5/validate_outputs.py --manifest MOSS-TTSD-v0.5/outputs/manifest.jsonl
```

正式验收和报告模板见 `ACCEPTANCE_PLAN.md`、`NPU_VALIDATION.md`。
