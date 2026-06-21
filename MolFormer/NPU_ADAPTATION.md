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
- `prepare_eval_data.py` 固定 SMILES manifest 并可盘点 IBM split；
  `compare_embeddings.py` 对 shape/cosine/绝对误差执行门禁。
- 仓内 10 条输入只作为功能验证；IBM 数据不可取得时可通过
  `--generate_l1_count 100` 生成 L2 降级固定集，metadata 固定样本数和 SHA256，
  但不得描述为官方 benchmark。

精确复现 IBM 11 项表属于独立 fine-tuning 工作。若后续交付该路径，应基于固定
DeepChem/IBM commit 提交可审查 patch，不再依赖无法追溯的定制 wheel。

## 3. 验证事实

2026-06-20 已完成版本取证、代码静态审查和脚本语法检查。

2026-06-20 补充 CPU clean-path 实测：

- 从固定 HF revision 实际下载 `config`、remote code、tokenizer 和 179 MiB
  `model.safetensors`；
- `model.safetensors` SHA256 为
  `0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6`；
- `prepare_eval_data.py` 对仓内 10 条 SMILES 生成 manifest，SHA256
  `10f19a22c2c72f5f77110ec5287d994b8de4440b4ee4e17b88a6b47f8609243f`；
- 在 Python 3.12.3、PyTorch 2.9.1 CPU、Transformers 4.35.0 上完成 10 条、
  batch 4 推理，embedding shape 均为 `[768]`；
- 用 Transformers 4.57.6 再运行同输入，与 4.35.0 输出逐元素误差 `0.0`；
- `compare_embeddings.py` 自比较和跨上述两个 Transformers 版本比较均通过。

该实测证明 CPU feature-extraction 和新增工具链可运行，不证明文档声明的
PyTorch/torch-npu 2.1 NPU 组合。当前仍未执行 NPU 功能验证和 IBM 官方 split
全量精度/性能对齐。

IBM Box 数据需要人工下载，11 项指标必须走独立 fine-tuning 路径。当前状态是
**S2：CPU feature extraction 实测通过；升级到 S3 仍缺 NPU 同 manifest
精度和性能对齐**。

CPU 与 NPU 使用独立环境；NPU 环境不得复用
`https://download.pytorch.org/whl/cpu` 安装的 CPU wheel。

用户推理见 [README_INFERENCE.md](README_INFERENCE.md)，官方 11 项下游指标和
迁移对齐方案见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。
