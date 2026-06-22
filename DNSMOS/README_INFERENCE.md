# DNSMOS 推理指导

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [文件目录](#文件目录)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备权重](#准备权重)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能](#模型推理性能)
- [公网地址说明](#公网地址说明)

## 概述

DNSMOS P.835 是 Microsoft 发布的语音质量评估模型，输出 `SIG`、`BAK`、`OVRL` 和 `P808_MOS` 四个分数字段。本文档介绍该模型基于昇腾 NPU 的推理指导，NPU 路径使用 ONNX Runtime `CANNExecutionProvider`，CPU 路径用于同权重数值基线。

> 说明：本文档适配常规及 personalized DNSMOS P.835 本地 ONNX 模型，不包含在线 DNSMOS API。

- 版本说明：

  ```text
  upstream=https://github.com/microsoft/DNS-Challenge.git
  branch=master
  commit=591184a9fcb2cbdec02520fed81a32bbbf9d73ff
  reference=https://gitcode.com/Ascend-SACT/DNSMOS
  reference_commit=d1e4c2c14df9cb935d61dc5f448e655772b12379
  ```

## 输入输出数据

- 输入数据

  支持一个或多个 WAV 文件、递归 WAV 目录，或包含 `audio_path` 字段的 JSONL manifest。

- 输出数据

  逐文件 CSV，以及运行环境和模型校验值 sidecar `*.meta.json`。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

  | 配套 | 版本 |
  |---|---|
  | 硬件 | 支持目标 CANN 的 Atlas 推理服务器 |
  | CANN、驱动、固件 | CANN 8.2.0 及其配套驱动/固件 |
  | Python | 3.10 |
  | ONNX Runtime | CPU：`onnxruntime==1.22.1`；NPU：`onnxruntime-cann==1.22.1` |
  | librosa / NumPy / soundfile | 见 `requirements.txt` |

## 文件目录

```text
DNSMOS
├── README_INFERENCE.md                 # 推理指导文档
├── NPU_ADAPTATION.md                   # NPU 适配文档
├── ACCEPTANCE_PLAN.md                  # 验收计划
├── infer.py                            # 推理脚本
├── prepare_eval_data.py                # 评测数据准备脚本
├── compare_results.py                  # CPU/NPU 结果比较脚本
├── requirements.txt
├── weights                             # 下载后的模型权重
│   ├── DNSMOS
│   │   ├── model_v8.onnx
│   │   └── sig_bak_ovr.onnx
│   └── pDNSMOS
│       └── sig_bak_ovr.onnx
├── eval_data                           # 评测数据目录，按需生成
└── results                             # 推理/比较结果目录，按需生成
```

## 快速上手

### 获取源码

1. 获取官方源码。

   ```bash
   git clone https://github.com/microsoft/DNS-Challenge.git upstream
   git -C upstream checkout 591184a9fcb2cbdec02520fed81a32bbbf9d73ff
   ```

2. 创建 CPU baseline 环境并安装依赖。

   ```bash
   python3.10 -m venv .venv-cpu
   source .venv-cpu/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install onnxruntime==1.22.1
   deactivate
   ```

3. 创建 NPU 环境并安装依赖。

   ```bash
   python3.10 -m venv .venv-npu
   source .venv-npu/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install onnxruntime-cann==1.22.1
   ```

### 准备权重

1. 拷贝官方权重并校验 SHA256。

   ```bash
   mkdir -p weights
   cp -r upstream/DNSMOS/DNSMOS weights/
   cp -r upstream/DNSMOS/pDNSMOS weights/

   sha256sum \
     weights/DNSMOS/model_v8.onnx \
     weights/DNSMOS/sig_bak_ovr.onnx \
     weights/pDNSMOS/sig_bak_ovr.onnx
   ```

   固定 commit 的预期值：

   ```text
   model_v8.onnx             9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91
   DNSMOS/sig_bak_ovr.onnx   269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd
   pDNSMOS/sig_bak_ovr.onnx  9e3a197449ca2177f0997afec3bd6b890117ce2f17b89d6eea7fa0d47272c81c
   ```

### 准备数据集

1. 下载 VCC2018 并生成 manifest。

   数据集地址：`https://datashare.ed.ac.uk/handle/10283/3061`。

   ```bash
   mkdir -p eval_data/vcc2018
   wget -O eval_data/vcc2018.tar.gz \
     https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz
   tar -xzf eval_data/vcc2018.tar.gz -C eval_data/vcc2018

   python prepare_eval_data.py \
     --audio_dir eval_data/vcc2018 \
     --output_manifest eval_data/vcc2018.jsonl \
     --dataset VCC2018 \
     --split submitted-systems \
     --limit 100
   ```

   参数说明：

   - `audio_dir`：递归扫描的 WAV 目录。
   - `output_manifest`：生成的 JSONL manifest 路径。
   - `dataset`：数据集名称，写入 manifest 元数据。
   - `split`：数据集 split 名称，写入 manifest 元数据。
   - `limit`：保留的最大样本数，`0` 表示不限制。

   生成的 manifest 默认路径：

   ```text
   eval_data/vcc2018.jsonl
   eval_data/vcc2018.jsonl.meta.json
   ```

### 模型推理

1. 执行 CPU 基线推理。

   ```bash
   python infer.py \
     --manifest eval_data/vcc2018.jsonl \
     --model_root weights \
     --device cpu \
     --output_csv results/cpu.csv
   ```

2. 执行 NPU 推理。

   ```bash
   python infer.py \
     --manifest eval_data/vcc2018.jsonl \
     --model_root weights \
     --device npu \
     --output_csv results/npu.csv
   ```

   参数说明：

   - `manifest`：输入 JSONL manifest 路径。
   - `model_root`：权重根目录，包含 `DNSMOS` 和 `pDNSMOS` 子目录。
   - `device`：推理设备，`cpu` 使用 `CPUExecutionProvider`，`npu` 使用 `CANNExecutionProvider`。
   - `output_csv`：逐文件分数输出 CSV 路径。
   - `personalized`：使用 personalized DNSMOS P.835 权重，需使用独立输出文件。

3. 比较 CPU 与 NPU 结果。

   ```bash
   python compare_results.py \
     --baseline results/cpu.csv \
     --candidate results/npu.csv \
     --output results/cpu_vs_npu.json
   ```

## 模型推理性能

| 路径 | 数据 | 结果 |
|---|---|---|
| CPU 算法等价性 | 30 秒样例，常规/personalized | 与官方脚本全字段误差 0 |
| CPU 工具闭环 | 同一样例 manifest | RTF 0.076430，本次仅作链路记录 |
| NPU | 同 manifest | 待 CANN 环境验收 |

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 模型权重 | Microsoft DNSMOS 官方源码 | https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS |
| 参考适配 | Ascend-SACT DNSMOS | https://gitcode.com/Ascend-SACT/DNSMOS |
| 论文 | DNSMOS P.835 | https://arxiv.org/abs/2110.01763 |
| 数据集 | VCC2018 | https://datashare.ed.ac.uk/handle/10283/3061 |

适配实现和已执行验证见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
