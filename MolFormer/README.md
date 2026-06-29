# MoLFormer 推理指导

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

MoLFormer-XL-both-10pct 是 IBM 发布的分子表征模型，基于 SMILES 输出官方 Hugging Face 模型的 `pooler_output` embedding。本入口用于验证同 checkpoint 在 CPU/CUDA 与 NPU 上的 embedding 数值一致性。本文档介绍该模型基于昇腾 NPU 的推理指导。

> 说明：模型卡的 11 项 MoleculeNet 指标来自下游 fine-tuning，不能由 feature extraction 直接得到。

- 版本说明：

  ```text
  commit_id=3b9ac434db387fadf2cf99b99def654cbf193841
  model=ibm-research/MoLFormer-XL-both-10pct@7b12d946c181a37f6012b9dc3b002275de070314
  official_code=IBM/molformer@3b9ac434db387fadf2cf99b99def654cbf193841
  deepchem=deepchem/deepchem@046c8b84fdcbf7e1b72bbbbd07fa2502ff9b94dd
  reference=Ascend-SACT/MolFormer@b39184dcb79501f0cd81def11e7b934176194a4c
  ```

## 输入输出数据

- 输入数据

  命令行 SMILES、每行一个 SMILES 的文本，或包含 `id`/`smiles` 的 JSONL manifest。

- 输出数据

  每条记录的 `id`、原 SMILES 和完整 embedding JSONL。`compare_embeddings.py` 比较 shape、cosine、最大/平均绝对误差。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

  | 配套 | 版本/要求 |
  |---|---|
  | 硬件 | 支持目标 CANN 的 Atlas 推理服务器 |
  | CANN、驱动、固件 | 参考适配 CANN 8.1.RC1；实际按 torch-npu 配套表选择 |
  | Python | 3.10 |
  | PyTorch / torch-npu | 2.1.0 / 2.1.0.post13 |
  | Transformers | 4.35.0 |
  | 推理精度 | FP32 |

## 文件目录

```text
MolFormer
├── infer.py
├── prepare_eval_data.py
├── compare_embeddings.py
├── test_data/smiles_functional.txt          # 10 条功能验证输入
├── requirements.txt
└── README.md
```

## 快速上手

### 获取源码

1. 安装 CPU baseline 环境。

   ```bash
   python3.10 -m venv .venv-cpu
   source .venv-cpu/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.1.0 \
     --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   python -m pip install huggingface_hub
   deactivate
   ```

2. 安装 NPU 环境（不得复用 CPU 索引 wheel）。

   ```bash
   python3.10 -m venv .venv-npu
   source .venv-npu/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.1.0 torch-npu==2.1.0.post13 \
     -i https://mirrors.huaweicloud.com/repository/pypi/simple
   python -m pip install -r requirements.txt
   python -m pip install huggingface_hub
   ```

3. NPU 导入检查。

   ```bash
   python - <<'PY'
   import torch
   import torch_npu
   import transformers
   from transformers import AutoModel, AutoTokenizer
   print(torch.__version__, transformers.__version__)
   print(torch.randn(1).to("npu").device)
   PY
   ```

### 准备权重

1. 下载 `MoLFormer-XL-both-10pct` 权重。

   原始权重地址：`https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct`

   ```bash
   huggingface-cli download ibm-research/MoLFormer-XL-both-10pct \
     --revision 7b12d946c181a37f6012b9dc3b002275de070314 \
     --local-dir weights/MoLFormer-XL-both-10pct

   find weights/MoLFormer-XL-both-10pct -maxdepth 1 -type f -print | sort
   find weights/MoLFormer-XL-both-10pct -maxdepth 1 -type f -print0 \
     | sort -z | xargs -0 sha256sum
   ```

   关键文件预期 SHA256：

   ```text
   config.json            3ef9eaac8c7ca6282fd6256ed038d151bd4ff42a4ff855367e0d7197bbc1c284
   model.safetensors      0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6
   modeling_molformer.py  a3f6273bb44709e566ee02987c50372db727de21799c3d297f64b2437f9e32a8
   tokenizer.json         3df1f2219653c44fac9fa03b7f788b372eb2544ecc176737bb9aca8411b471a5
   ```

### 准备数据集

1. 准备功能验证 manifest。

   ```bash
   python prepare_eval_data.py \
     --smiles_file test_data/smiles_functional.txt \
     --output_manifest eval_data/smiles_functional.jsonl \
     --dataset ModelZoo-fixed-SMILES \
     --split functional
   ```

   参数说明：

   - `smiles_file`：每行一个 SMILES 的输入文本。
   - `output_manifest`：生成的 JSONL manifest 路径。
   - `dataset`：数据集名称，写入 manifest metadata。
   - `split`：split 名称，写入 manifest metadata。

2. 准备 L2 降级固定集（IBM 数据不可取得时使用）。

   ```bash
   python prepare_eval_data.py \
     --generate_l1_count 100 \
     --output_manifest eval_data/smiles_l2_fallback.jsonl \
     --dataset ModelZoo-generated-linear-SMILES \
     --split l2-fallback
   test "$(wc -l < eval_data/smiles_l2_fallback.jsonl)" = 100
   ```

3. 准备 IBM 官方 split 数据。

   从 <https://ibm.box.com/v/MoLFormer-data> 手工下载 `Pretrained MoLFormer.zip` 和 `finetune_datasets.zip`，下载后：

   ```bash
   git clone https://github.com/IBM/molformer.git upstream
   git -C upstream checkout 3b9ac434db387fadf2cf99b99def654cbf193841
   unzip "Pretrained MoLFormer.zip" -d upstream/data
   unzip finetune_datasets.zip -d upstream/data

   python prepare_eval_data.py \
     --csv upstream/data/bbbp/test.csv \
     --output_manifest eval_data/bbbp-test.jsonl \
     --dataset BBBP \
     --split test \
     --official_data_dir upstream/data
   ```

   参数说明：

   - `csv`：IBM 官方 split CSV 路径。
   - `official_data_dir`：IBM 解压数据根目录，用于盘点全部 split 行数和 SHA256。

   实际目录名以压缩包内容为准；若与示例不同，应先按 IBM README 放置，再把真实路径传给脚本。

### 模型推理

1. CPU 推理。

   ```bash
   source .venv-cpu/bin/activate
   python infer.py \
     --model weights/MoLFormer-XL-both-10pct \
     --manifest eval_data/smiles_l2_fallback.jsonl \
     --device cpu \
     --batch_size 32 \
     --output results/embeddings_cpu.jsonl
   ```

2. NPU 推理。

   ```bash
   source .venv-npu/bin/activate
   python infer.py \
     --model weights/MoLFormer-XL-both-10pct \
     --manifest eval_data/smiles_l2_fallback.jsonl \
     --device npu \
     --batch_size 32 \
     --output results/embeddings_npu.jsonl
   ```

   参数说明：

   - `model`：本地模型目录路径。
   - `manifest`：输入 JSONL manifest 路径。
   - `device`：推理设备，支持 `cpu`、`npu`。
   - `batch_size`：批大小。
   - `output`：输出 embedding JSONL 路径。

3. CPU/NPU embedding 对齐。

   ```bash
   python compare_embeddings.py \
     --baseline results/embeddings_cpu.jsonl \
     --candidate results/embeddings_npu.jsonl \
     --output results/cpu_vs_npu.json
   ```

## 模型推理性能

官方未发布与本 feature-extraction 入口及当前 Atlas 环境直接可比的硬件性能数值。

示例：

```bash
mkdir -p results
for BATCH in 1 8 32 64; do
  /usr/bin/time -v -o "results/npu_bs${BATCH}.time.txt" python infer.py \
    --model weights/MoLFormer-XL-both-10pct \
    --manifest eval_data/bbbp-test.jsonl \
    --device npu \
    --batch_size "$BATCH" \
    --output "results/npu_bs${BATCH}.jsonl"
done
```

CPU/CUDA 使用同一 manifest 和 batch 矩阵，写入不同文件。每个 embedding 输出均需与相应 baseline 运行 `compare_embeddings.py`。

## 适配与精度口径

### 适配决策

正式路径直接复用官方模型卡的 `AutoModel` + `pooler_output`，只增加显式设备迁移：`--device npu` 时仅 NPU 路径导入 `torch_npu`，模型和输入通过 `.to(device)` 迁移，算法不变。不修改 Transformers 和 IBM remote code，不需要 patch。CPU 与 NPU 使用独立环境，NPU 不复用 CPU wheel。精确复现 IBM 11 项 MoleculeNet 指标需要独立的下游 fine-tuning 环境，不能由本 feature-extraction 入口直接得到。

### 官方 MoleculeNet 指标

IBM 模型卡在标准 MoleculeNet split 上发布以下指标，均为下游 fine-tuning 结果：

分类任务（AUROC）：

| 数据集 | 官方 AUROC |
|---|---:|
| bbbp | 0.717 ± 0.006 |
| bace | 0.829 ± 0.012 |
| clintox | 0.945 ± 0.005 |
| hiv | 0.822 ± 0.003 |
| tox21 | 0.843 ± 0.001 |
| sider | 0.649 ± 0.005 |

回归任务（MAE / RMSE）：

| 数据集 | 官方 MAE | 官方 RMSE |
|---|---:|---:|
| esol | 0.456 ± 0.014 | 0.587 ± 0.018 |
| freesolv | 0.834 ± 0.060 | 1.075 ± 0.078 |
| lipo | 0.515 ± 0.013 | 0.660 ± 0.016 |
| qm7 | 67.6 ± 1.5 | 92.5 ± 2.7 |
| qm8 | 0.0117 ± 0.0001 | 0.0151 ± 0.0001 |

复现口径：RDKit canonicalize、`isomeric=False`、`seed=12345`、`lr=3e-5`、`500 epochs`、分类 `batch=32`、回归 `batch=128`。IBM 模型卡未发布上述每项的精确 split revision、scaffold seed 和完整训练命令，因此这些字段明确记为"官方未发布"，获得作者 recipe 前不得宣称精确复现。

### 迁移对齐门禁

使用同一 checkpoint、manifest、batch 和 seed，比较 CPU 与 NPU 的 `pooler_output` embedding：

- cosine similarity `>= 0.99999`；
- 逐元素最大绝对误差 `<= 1e-4`；
- 逐元素平均绝对误差 `<= 1e-5`。

这些阈值是迁移初始门禁，不是 IBM 官方容差。`compare_embeddings.py` 输出 cosine、max abs error、mean abs error 三项，三项同时满足才算通过。

### 性能评测方法

记录 batch 1/8/32/64 的 samples/s、`/usr/bin/time -v` 资源占用和峰值 HBM/RSS，正式轮次至少重复 3 次并报告 samples/s 中位数。IBM 模型卡只发布 MoleculeNet 精度，未发布与当前 Atlas 路径直接可比的硬件性能数值，因此报告 NPU/CPU samples/s 比值。

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 模型权重 | IBM MoLFormer-XL Hugging Face 模型仓 | https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct |
| 开源代码仓 | IBM molformer 源码 | https://github.com/IBM/molformer |
| 数据集 | IBM MoLFormer 数据 | https://ibm.box.com/v/MoLFormer-data |
| 参考适配 | Ascend-SACT MolFormer | https://gitcode.com/Ascend-SACT/MolFormer |
