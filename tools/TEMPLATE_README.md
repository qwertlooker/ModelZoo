# <MODEL_NAME> 推理指导

## 概述

`<MODEL_NAME>` 面向 `<TASK>` 场景，当前适配目标为 `<UPSTREAM_REPO>`
的 `<UPSTREAM_COMMIT>` 与权重 `<WEIGHT_SOURCE>`。

```text
commit_id=<TARGET_COMMIT_40_HEX>
```

## 输入输出

| 项目 | 说明 |
|---|---|
| 输入 | `<INPUT_FORMAT>` |
| 输出 | `<OUTPUT_FORMAT>` |
| 主要指标 | `<METRIC>` |

## 推理环境

| 组件 | 版本 / 要求 |
|---|---|
| 硬件 | `<ASCEND_DEVICE>` |
| CANN | `<CANN_VERSION_OR_IMAGE_DIGEST>` |
| Python | `<PYTHON_VERSION>` |
| PyTorch / torch-npu | `<TORCH_NPU_VERSION>` |
| 其他依赖 | `<KEY_DEPENDENCIES>` |

```bash
pip install -r requirements.txt
python3 - <<'PY'
import torch
import torch_npu
print(torch.__version__)
print(torch_npu.__version__)
PY
```

## 文件目录

```text
<MODEL_NAME>/
├── README.md
├── infer.py
├── requirements.txt
└── patches
    └── adapt.patch
```

## 快速上手

### 权重准备

```bash
mkdir -p weights
# 固定 revision / commit 下载，并记录 SHA256 或 metadata check。
```

### 数据准备

```bash
python3 prepare_eval_data.py --output test_data --split <SPLIT>
```

### NPU 推理

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 infer.py \
  --device npu \
  --weights weights \
  --input test_data/manifest.jsonl \
  --output results/npu
```

### 评测与比较

```bash
python3 eval_<metric>.py --input results/npu --output results/npu_metrics.json
python3 compare_<metric>.py \
  --baseline results/baseline_metrics.json \
  --candidate results/npu_metrics.json
```

## 模型推理性能

| 数据集 / 输入 | 设备 | batch size | 指标 | 结果 |
|---|---|---:|---|---:|
| `<DATASET>` | `<DEVICE>` | `<BATCH_SIZE>` | `<LATENCY_OR_THROUGHPUT>` | `<VALUE>` |

## 精度

| 数据集 | split | 样本数 | 指标 | 原始 / 官方 | NPU | 结论 |
|---|---|---:|---|---:|---:|---|
| `<DATASET>` | `<SPLIT>` | `<COUNT>` | `<METRIC>` | `<BASELINE>` | `<NPU_RESULT>` | `<PASS_OR_FAIL>` |

## 公网地址

| 类型 | 地址 | 版本 |
|---|---|---|
| 源码 | `<UPSTREAM_REPO>` | `<UPSTREAM_COMMIT>` |
| 权重 | `<WEIGHT_SOURCE>` | `<WEIGHT_REVISION>` |
| 数据集 | `<DATASET_URL>` | `<DATASET_VERSION>` |
