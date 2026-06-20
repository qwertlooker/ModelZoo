# MoLFormer 推理指导

## 概述

本目录适配 IBM `MoLFormer-XL-both-10pct` 的 SMILES feature extraction。输出为模型 `pooler_output`，可用于相似性、可视化或下游预测器；不把随机初始化的 DeepChem 训练 loss 当作官方预训练模型精度。

```text
model=https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct
model_commit=7b12d946c181a37f6012b9dc3b002275de070314
official_code=https://github.com/IBM/molformer.git
deepchem=https://github.com/deepchem/deepchem.git
deepchem_commit=046c8b84fdcbf7e1b72bbbbd07fa2502ff9b94dd
reference=https://gitcode.com/Ascend-SACT/MolFormer
reference_commit=b39184dcb79501f0cd81def11e7b934176194a4c
```

版本边界是 10% ZINC + 10% PubChem、46.8M 参数的 checkpoint，不是 `MoLFormer-XL` 全量预训练变体。

## 环境与权重

参考适配环境为 Python 3.10、CANN 8.1.RC1、PyTorch/torch-npu 2.1.0。安装时以当前 CANN 配套表为准：

```bash
pip install torch==2.1.0 \
  --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu==2.1.0.post13 \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
pip install -r requirements.txt
huggingface-cli download ibm-research/MoLFormer-XL-both-10pct \
  --revision 7b12d946c181a37f6012b9dc3b002275de070314 \
  --local-dir weights/MoLFormer-XL-both-10pct
```

记录权重和 remote-code 文件 SHA256，不依赖运行时远端更新。

## 推理

准备 `smiles.txt`，每行一个 canonical SMILES。NPU：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 infer.py \
  --model weights/MoLFormer-XL-both-10pct \
  --input smiles.txt \
  --device npu \
  --batch_size 32 \
  --output embeddings_npu.jsonl
```

CPU：

```bash
python3 infer.py \
  --model weights/MoLFormer-XL-both-10pct \
  --input smiles.txt \
  --device cpu \
  --output embeddings_cpu.jsonl
```

单次也可直接传参：

```bash
python3 infer.py --device cpu --smiles \
  'Cn1c(=O)c2c(ncn2C)n(C)c1=O' \
  'CC(=O)Oc1ccccc1C(=O)O'
```

验收见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，适配决策见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
