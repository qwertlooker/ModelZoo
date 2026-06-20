# MoLFormer 验收计划

## 0. 版本边界

- checkpoint：`ibm-research/MoLFormer-XL-both-10pct@7b12d946c181a37f6012b9dc3b002275de070314`
- IBM 原始代码：`3b9ac434db387fadf2cf99b99def654cbf193841`
- DeepChem 参考：`046c8b84fdcbf7e1b72bbbbd07fa2502ff9b94dd`
- Ascend-SACT 参考：`b39184dcb79501f0cd81def11e7b934176194a4c`
- 不包含论文完整 100% 预训练权重或未固定的下游 checkpoint。

## 1. 原始数据、指标和 checkpoint

模型卡对 `MoLFormer-XL-both-10pct` 公布 11 个 MoleculeNet 下游任务结果：

| 分类数据集（AUROC，越高越好） | BBBP | HIV | BACE | SIDER | ClinTox | Tox21 |
|---|---:|---:|---:|---:|---:|---:|
| 10% ZINC + 10% PubChem | 91.5 | 81.3 | 86.6 | 68.9 | 94.6 | 84.5 |

| 回归数据集 | QM9 MAE | QM8 MAE | ESOL RMSE | FreeSolv RMSE | Lipophilicity RMSE |
|---|---:|---:|---:|---:|---:|
| 10% ZINC + 10% PubChem | 1.7754 | 0.0108 | 0.3295 | 0.2221 | 0.5472 |

来源：<https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct>。

官方 IBM 包使用预先生成的 `train.csv`、`valid.csv`、`test.csv`（QM8/QM9/Lipo
采用相应前缀文件）。模型卡和代码仓未逐项公布这些 split 的样本数，也未说明它们
是否等同于某个 DeepChem random/scaffold split；下载官方
`finetune_datasets.zip` 后必须逐文件统计并记录，不能自行重切分后冒充官方结果。

官方 fine-tuning 脚本将 SMILES 用 RDKit canonicalize，并设置
`isomeric=False`；默认 seed `12345`、学习率 `3e-5`、最多 500 epochs，
batch size 按任务为 32 或 128，以 validation 最优 epoch 对应的 test score
作为结果。分类用 AUROC；QM8/QM9 报 average MAE；ESOL/FreeSolv/Lipophilicity
报 RMSE。

这些是下游 fine-tuning 指标，不是 base checkpoint 做 feature extraction 即可直接
得到的指标。验收必须固定 IBM 原始 split、normalizer、seed、训练参数和
fine-tuned checkpoint；缺少这些条件时不得声称复现官方表。

## 2. 迁移对齐主线

### L1/L2：官方 checkpoint feature extraction

准备固定 canonical SMILES manifest，覆盖：

- 模型卡两条示例；
- 典型长度和接近 202 token 上限；
- padding batch；
- 无效/超长输入作为失败用例。

同 checkpoint、tokenizer、remote code 和 batch 分别跑 CPU/CUDA 与 NPU。按 SMILES 比较：

- embedding shape 完全一致；
- cosine similarity `>= 0.99999`；
- FP32 最大绝对误差 `<= 1e-4`、平均绝对误差 `<= 1e-5`；
- batch 1 与 batch N 的同一样本 embedding 满足相同阈值。

### L3：官方下游结果

使用 IBM 官方 fine-tuning 代码和 MoleculeNet 数据，固定 11 个任务的 split/seed/metric。先在原始 CUDA 路径得到当前环境基线，再在 NPU 使用同配置训练/评测：

- 分类 AUROC：NPU 相对 CUDA 下降不超过 0.5 个百分点；
- 回归：NPU 指标相对 CUDA 劣化不超过 1%，且绝对差异同时报告；
- 只有复现配置与官方一致时才比较官方表值。

## 3. 分层和性能

| 层级 | 范围 |
|---|---|
| L0 | 2 条示例 smoke，只验链路 |
| L1 | 固定 100 条 SMILES，CPU/NPU embedding 对齐 |
| L2 | 固定 MoleculeNet 多数据集样本和 batch/长度矩阵 |
| L3 | 11 个下游任务完整 fine-tuning/evaluation |

记录加载时间、samples/s、batch 1/8/32/64、峰值 HBM、30 轮稳定性。dummy 或在两条训练样本上评估 MAE 不构成精度验收。

## 4. 当前验收状态

- 已通过：模型/IBM/DeepChem/参考仓版本取证；`infer.py` 语法检查；
  模型卡 11 项指标和官方 fine-tuning 参数核对。
- 未执行：checkpoint 下载、CPU/NPU embedding 对齐、IBM 官方 split
  统计、11 项 fine-tuning、性能和稳定性。
- 当前结论：推理入口静态门禁通过；迁移精度未验收。

## 5. 报告模板

```text
模型/remote-code/tokenizer SHA:
数据集、split、seed、manifest:
CPU/CUDA/NPU环境:
embedding cosine/max/mean error:
11任务配置与指标（如执行）:
吞吐、峰值内存、稳定性:
与CUDA/官方值差异:
结论和未完成项:
```
