# MoLFormer NPU 适配文档

## 1. 版本边界

- 模型：`ibm-research/MoLFormer-XL-both-10pct`
- HF commit：`7b12d946c181a37f6012b9dc3b002275de070314`
- 参数/精度：46.8M / FP32
- 预训练数据：10% ZINC15 + 10% PubChem；训练前 canonicalize、移除 isomeric 信息、过滤超过 202 token 的分子。
- 官方代码：`IBM/molformer`
- DeepChem HEAD：`046c8b84fdcbf7e1b72bbbbd07fa2502ff9b94dd`
- Ascend-SACT 参考 commit：`b39184dcb79501f0cd81def11e7b934176194a4c`
- 检查日期：2026-06-20。

## 2. 参考实现审查与正式路径

参考仓通过 `deepchem-ascend==0.0.5` 执行两条样本的随机初始化 MLM/回归训练。该流程：

- 没有固定二进制包对应的 DeepChem 源码 commit；
- 示例未从 IBM checkpoint 加载预训练权重；
- 在训练集自身计算 MAE，没有公开测试 split；
- 只能证明特定 wheel 的训练链路，不能证明官方模型结果或迁移精度。

因此正式最小推理路径直接复用官方模型卡的 `AutoModel` + `pooler_output`，只增加显式设备迁移：

- 默认 `--device npu`，仅 NPU 路径导入 `torch_npu`；
- CPU/CUDA/NPU 使用同 tokenizer、remote code、checkpoint 和 pooling；
- 不修改 Transformers/IBM remote code，不需要 patch；
- 输出完整 embedding，便于逐样本数值对齐。

DeepChem 下游训练属于 L3 复现路径。若后续必须交付 DeepChem 训练，应基于固定 DeepChem commit 提交可审查 patch，不再依赖无法追溯的定制 wheel。

## 3. 验证事实

2026-06-20 已完成版本取证、代码静态审查和 `infer.py` 语法检查。当前主机未安装 PyTorch、torch-npu、Transformers，且无 NPU/权重，因此未执行 embedding 数值、MoleculeNet 或性能验收。

用户推理见 [README_INFERENCE.md](README_INFERENCE.md)，官方 11 项下游指标和
迁移对齐方案见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。
