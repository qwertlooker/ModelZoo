# MoLFormer 验收计划

## 0. 版本边界

- checkpoint：`ibm-research/MoLFormer-XL-both-10pct@7b12d946c181a37f6012b9dc3b002275de070314`
- IBM 原始代码：`3b9ac434db387fadf2cf99b99def654cbf193841`
- DeepChem 参考：`046c8b84fdcbf7e1b72bbbbd07fa2502ff9b94dd`
- Ascend-SACT 参考：`b39184dcb79501f0cd81def11e7b934176194a4c`
- 不包含论文完整 100% 预训练权重或未固定的下游 checkpoint。

## 1. 原始测试集、官方指标和 checkpoint

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

固定 IBM commit 中可直接看到 6 个分类 runner、Lipo runner 和多个 QM9 属性
runner；没有发布与模型卡 11 项表逐项一一对应的完整统一命令清单，且仓库中没有
直接命名的 QM8/ESOL/FreeSolv runner。`finetune_datasets.zip` 还需要从 IBM Box
人工取得。因此当前可以形成完整的 embedding 迁移对齐，但在作者 recipe 未补齐前，
不能声称一个命令即可精确复现 11 项官方表。

官方 fine-tuning 脚本将 SMILES 用 RDKit canonicalize，并设置
`isomeric=False`；默认 seed `12345`、学习率 `3e-5`、最多 500 epochs，
batch size 按任务为 32 或 128，以 validation 最优 epoch 对应的 test score
作为结果。分类用 AUROC；QM8/QM9 报 average MAE；ESOL/FreeSolv/Lipophilicity
报 RMSE。

这些是下游 fine-tuning 指标，不是 base checkpoint 做 feature extraction 即可直接
得到的指标。验收必须固定 IBM 原始 split、normalizer、seed、训练参数和
fine-tuned checkpoint；缺少这些条件时不得声称复现官方表。

## 2. 迁移对齐主线

### 功能验证与 L2：官方 checkpoint feature extraction

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

embedding 阈值是 FP32 初始工程门禁，需用同设备重复运行和 patch 前后回归结果校准，
不是 IBM 官方容差。

可执行入口：

```bash
python prepare_eval_data.py \
  --smiles_file test_data/smiles_functional.txt \
  --output_manifest eval_data/smiles_functional.jsonl \
  --dataset ModelZoo-fixed-SMILES --split functional
python compare_embeddings.py \
  --baseline results/embeddings_cpu.jsonl \
  --candidate results/embeddings_npu.jsonl \
  --output results/cpu_vs_npu.json
```

L2 使用 IBM `finetune_datasets.zip` 的固定公开 split 生成 manifest，至少覆盖
一个分类数据集和一个回归数据集；本次只比较同 checkpoint embedding，不把它描述
为 11 项 fine-tuning 官方指标复现。

## 3. 功能验证与 L2

| 层级 | 范围 |
|---|---|
| 功能验证 | 仓内 10 条固定 SMILES，检查单条、batch、长度和失败用例 |
| L2 | 优先使用 IBM 官方 split 全量；recipe/数据不完整时至少固定一个分类和一个回归公开 split | embedding 精度、samples/s 和资源 |

## 4. 功能矩阵

| 维度 | 必测值 |
|---|---|
| 输入 | 单条、JSONL manifest、padding batch |
| 长度 | 短、典型、接近 202 token |
| batch | 1/8/32/64 |
| 设备 | 原始 CPU/CUDA、NPU |
| 异常 | 无效 SMILES、空输入、超长输入、缺 remote code |

## 5. L2 精度与性能验证

L2 优先对 IBM `finetune_datasets.zip` 中可取得的官方 split 全量执行；若作者统一
recipe 不完整，至少选择一个分类和一个回归公开 split，固定 CSV SHA 和 manifest。
三组使用相同 batch/长度矩阵，报告 embedding cosine/max/mean error。

性能记录模型加载时间、纯推理 elapsed、samples/s、batch 1/8/32/64、峰值 RSS/HBM。
正式 batch 至少重复 3 次并报告中位数；每次输出必须继续通过 embedding 精度比较。
官方没有当前入口的硬件性能值，因此报告 NPU/CPU(CUDA) samples/s 比值，不伪造
speedup 通过线。

## 6. 最低正式验收清单

- [ ] 模型、remote code、tokenizer 文件 SHA 已固定。
- [ ] 仓内 10 条功能 manifest/metadata 已生成并完成 CPU/NPU 功能验证。
- [ ] CPU/CUDA、NPU embedding 分别保存并通过 `compare_embeddings.py`。
- [ ] batch 1 与 batch N 一致性、长度和失败用例通过。
- [ ] IBM `finetune_datasets.zip` 的 split 行数和 SHA 已盘点。
- [ ] L2 至少一个分类和一个回归 split 完成同 manifest 的 CPU/NPU embedding 对齐。
- [ ] L2 batch 矩阵的 samples/s、加载时间、峰值 RSS/HBM 和相对比值已归档。

## 7. 当前验收状态

- 已通过：模型/IBM/DeepChem/参考仓版本取证；实际下载固定 HF checkpoint；
  10 条固定 manifest 的 CPU feature extraction；Transformers 4.35.0/4.57.6
  输出误差 0；模型卡 11 项指标和官方 fine-tuning 参数核对。
- 未执行：NPU 功能验证、IBM 官方 split 统计和 L2 精度/性能对齐。
- 当前结论：S2 CPU feature-extraction 通过；NPU 迁移精度未验收。

## 8. 报告模板

```text
模型/remote-code/tokenizer SHA:
数据集、split、seed、manifest:
CPU/CUDA/NPU环境:
embedding cosine/max/mean error:
L2分类/回归split及样本数:
加载时间、batch矩阵samples/s、峰值RSS/HBM及比值:
与CUDA/官方值差异:
结论和未完成项:
```

## 补充说明（来自 README_INFERENCE.md）

### IBM Box 数据获取

IBM 官方 11 项 fine-tuning 复现需要从 <https://ibm.box.com/v/MoLFormer-data>
手工下载 `Pretrained MoLFormer.zip` 和 `finetune_datasets.zip`。该网盘没有在
本文档中假装成稳定自动直链。

### L2 降级固定集

若 IBM 数据暂不可取得，可用脚本确定性生成 100 条简单线性 SMILES 作为 L2 降级
固定集；报告必须明确它不是官方 benchmark。

### 官方 split 与 fine-tuning 复现要求

优先使用 IBM 官方 split 全量替换上述降级 manifest。若宣称复现模型卡 11 项表，
还必须在独立环境中按固定 split 运行官方 fine-tuning，不得用 embedding 比较代替。

### 性能方法论

对同一 L2 manifest 分别以 batch 1/8/32/64 运行三组命令；每个输出的
`*.meta.json` 记录 elapsed 和 samples/s，另用 `/usr/bin/time -v` 记录峰值 RSS，
NPU 记录峰值 HBM。正式 batch 重复 3 次并报告中位数。

### 当前状态表

| 路径 | 数据/环境 | 结果 |
|---|---|---|
| CPU feature extraction | 10 条、batch 4、Transformers 4.35.0 | 20.535 samples/s，仅作链路记录 |
| Transformers 4.35 vs 4.57 | 同 checkpoint/manifest | embedding 逐元素误差 0 |
| NPU embedding | 同 manifest | 待验收 |
| L2 IBM split | 官方 split 全量或明确降级固定集 | 待 CPU/CUDA/NPU 精度和性能验收 |
