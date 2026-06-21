# MoLFormer 推理指导

## 概述

本目录适配 IBM `MoLFormer-XL-both-10pct` 的 SMILES feature extraction，输出官方 Hugging Face 模型的 `pooler_output`。该入口用于验证同 checkpoint 在 CPU/CUDA 与 NPU 上的 embedding 数值一致性；模型卡的 11 项 MoleculeNet 指标来自下游 fine-tuning，不能由少量功能样例直接得到。

```text
model=ibm-research/MoLFormer-XL-both-10pct@7b12d946c181a37f6012b9dc3b002275de070314
official_code=IBM/molformer@3b9ac434db387fadf2cf99b99def654cbf193841
deepchem=deepchem/deepchem@046c8b84fdcbf7e1b72bbbbd07fa2502ff9b94dd
reference=Ascend-SACT/MolFormer@b39184dcb79501f0cd81def11e7b934176194a4c
```

## 输入输出数据

- 输入：命令行 SMILES、每行一个 SMILES 的文本，或包含 `id`/`smiles` 的 JSONL manifest。
- 输出：每条记录的 `id`、原 SMILES 和完整 embedding JSONL。
- `prepare_eval_data.py` 固定 manifest，并可盘点 IBM 官方 split CSV 的行数、字段和 SHA256。
- `compare_embeddings.py` 比较 shape、cosine、最大/平均绝对误差。

## 推理环境准备

| 配套 | 版本/要求 |
|---|---|
| 硬件 | 支持目标 CANN 的 Atlas 推理服务器 |
| CANN、驱动、固件 | 参考适配 CANN 8.1.RC1；实际按 torch-npu 配套表选择 |
| Python | 3.10 |
| PyTorch / torch-npu | 2.1.0 / 2.1.0.post13 |
| Transformers | 4.35.0 |
| 推理精度 | FP32 |

精确复现 IBM 11 项 fine-tuning 指标需建立独立环境，不能让旧版 PyTorch
Lightning/RDKit 覆盖已验证的 NPU 推理环境；当前 L2 优先使用 IBM 官方 split
全量做 feature-extraction 精度和性能对齐。

## 文件目录

```text
MolFormer
├── infer.py
├── prepare_eval_data.py
├── compare_embeddings.py
├── test_data/smiles_functional.txt          # 10 条功能验证输入
├── requirements.txt
├── README_INFERENCE.md
├── NPU_ADAPTATION.md
└── ACCEPTANCE_PLAN.md
```

## 快速上手

### 获取源码和安装依赖

CPU baseline 环境：

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

NPU 环境不得安装 CPU 索引 wheel：

```bash
python3.10 -m venv .venv-npu
source .venv-npu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.0 torch-npu==2.1.0.post13 \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
python -m pip install -r requirements.txt
python -m pip install huggingface_hub
```

NPU 导入检查：

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

CPU 环境执行导入检查时省略 `torch_npu` 和 NPU tensor。

### 准备权重

```bash
huggingface-cli download ibm-research/MoLFormer-XL-both-10pct \
  --revision 7b12d946c181a37f6012b9dc3b002275de070314 \
  --local-dir weights/MoLFormer-XL-both-10pct

find weights/MoLFormer-XL-both-10pct -maxdepth 1 -type f -print | sort
find weights/MoLFormer-XL-both-10pct -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum
```

固定本地目录后，运行时不使用远端模型名，避免 remote-code 漂移。
本次实际下载的关键文件预期值：

```text
config.json            3ef9eaac8c7ca6282fd6256ed038d151bd4ff42a4ff855367e0d7197bbc1c284
model.safetensors      0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6
modeling_molformer.py  a3f6273bb44709e566ee02987c50372db727de21799c3d297f64b2437f9e32a8
tokenizer.json         3df1f2219653c44fac9fa03b7f788b372eb2544ecc176737bb9aca8411b471a5
```

### 准备数据集

仓内 10 条输入用于功能验证：

```bash
python prepare_eval_data.py \
  --smiles_file test_data/smiles_functional.txt \
  --output_manifest eval_data/smiles_functional.jsonl \
  --dataset ModelZoo-fixed-SMILES \
  --split functional
```

若 IBM 数据暂不可取得，可用脚本确定性生成 100 条简单线性 SMILES 作为 L2 降级
固定集；报告必须明确它不是官方 benchmark：

```bash
python prepare_eval_data.py \
  --generate_l1_count 100 \
  --output_manifest eval_data/smiles_l2_fallback.jsonl \
  --dataset ModelZoo-generated-linear-SMILES \
  --split l2-fallback
test "$(wc -l < eval_data/smiles_l2_fallback.jsonl)" = 100
```

IBM 官方 11 项 fine-tuning 复现需要从 <https://ibm.box.com/v/MoLFormer-data> 手工下载 `Pretrained MoLFormer.zip` 和 `finetune_datasets.zip`。该网盘没有在本文档中假装成稳定自动直链。下载后：

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

实际目录名以压缩包内容为准；若与示例不同，应先按 IBM README 放置，再把真实路径传给脚本。生成的 metadata 会固定每个 CSV 的行数和 SHA256。

### 模型推理

CPU：

```bash
source .venv-cpu/bin/activate
python infer.py \
  --model weights/MoLFormer-XL-both-10pct \
  --manifest eval_data/smiles_l2_fallback.jsonl \
  --device cpu \
  --batch_size 32 \
  --output results/embeddings_cpu.jsonl
```

NPU：

```bash
source .venv-npu/bin/activate
python infer.py \
  --model weights/MoLFormer-XL-both-10pct \
  --manifest eval_data/smiles_l2_fallback.jsonl \
  --device npu \
  --batch_size 32 \
  --output results/embeddings_npu.jsonl
```

对齐：

```bash
python compare_embeddings.py \
  --baseline results/embeddings_cpu.jsonl \
  --candidate results/embeddings_npu.jsonl \
  --output results/cpu_vs_npu.json
```

优先使用 IBM 官方 split 全量替换上述降级 manifest。若宣称复现模型卡 11 项表，
还必须在独立环境中按固定 split 运行官方 fine-tuning，不得用 embedding 比较代替。

## 模型推理性能

官方未发布与本 feature-extraction 入口及当前 Atlas 环境直接可比的硬件性能数值。
对同一 L2 manifest 分别以 batch 1/8/32/64 运行三组命令；每个输出的
`*.meta.json` 记录 elapsed 和 samples/s，另用 `/usr/bin/time -v` 记录峰值 RSS，
NPU 记录峰值 HBM。正式 batch 重复 3 次并报告中位数。

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

CPU/CUDA 使用同一 manifest 和 batch 矩阵，写入不同文件。每个 embedding 输出均需
与相应 baseline 运行 `compare_embeddings.py`。

| 路径 | 数据/环境 | 结果 |
|---|---|---|
| CPU feature extraction | 10 条、batch 4、Transformers 4.35.0 | 20.535 samples/s，仅作链路记录 |
| Transformers 4.35 vs 4.57 | 同 checkpoint/manifest | embedding 逐元素误差 0 |
| NPU embedding | 同 manifest | 待验收 |
| L2 IBM split | 官方 split 全量或明确降级固定集 | 待 CPU/CUDA/NPU 精度和性能验收 |

## 公网地址说明

- 模型：<https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct>
- IBM 代码：<https://github.com/IBM/molformer>
- IBM 数据：<https://ibm.box.com/v/MoLFormer-data>
- 参考适配：<https://gitcode.com/Ascend-SACT/MolFormer>

完整指标口径见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，适配决策见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
