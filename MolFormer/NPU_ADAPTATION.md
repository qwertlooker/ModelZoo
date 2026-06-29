# MoLFormer NPU 适配文档

## 1. 版本边界

- 模型：`ibm-research/MoLFormer-XL-both-10pct`
- HF commit：`7b12d946c181a37f6012b9dc3b002275de070314`
- 参数/精度：46.8M / FP32
- 预训练数据：10% ZINC15 + 10% PubChem；训练前 canonicalize、移除 isomeric 信息、过滤超过 202 token 的分子。
- 官方代码：`IBM/molformer`
- DeepChem HEAD：`046c8b84fdcbf7e1b72bbbbd07fa2502ff9b94dd`
- Ascend-SACT 参考 commit：`b39184dcb79501f0cd81def11e7b934176194a4c`
- 检查日期：2026-06-29。

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

用户推理见 [README.md](README.md)，官方 11 项下游指标和
迁移对齐方案见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

权重下载优先使用 `huggingface-cli download`（在线路径），README 已补全等价 `curl` 离线替代命令（见 7.2 离线规则）。单文件 safetensors 模型约 179 MiB，离线下载简单。

## 上库就绪与目标仓对齐

- 目标仓快照：`https://gitcode.com/Ascend/ModelZoo-PyTorch.git`，2026-06-29 重新查询
  `master` HEAD `7a02a6701c971b29df188a0f3241e1efe249d1df`（"modify document"）。
  2026-06-22 审阅快照为 `ec2a7b514973805f66b67c9178d2f5c9e97eee34`；本次不复用历史快照。
- 拟合入路径：`ACL_PyTorch/built-in/nlp/MolFormer`。目标仓 `nlp/` 下不存在该目录，
  本次为新增，不涉及替换或增量更新。
- 最新参考目录：同领域选 `ACL_PyTorch/built-in/nlp/ProtBert_for_Pytorch`（nlp 序列/
  分子-蛋白语言模型，最后实质变更 `6fecdfba7`，2026-06-18，`Protbert_infer.py` +
  `TestProtbert_2onnx.py` + `requirements.txt`），选择原因是同属 nlp 分子/序列模型且含
  infer + ONNX 路径；同领域 `nlp/chronos-2`（`6fecdfba7`，2026-06-18）提供
  `ascend_infer.py`/`eval_accuracy.py`/`eval_performance.py` 精度性能脚本参考。
- 贡献规范与 PR 门禁：`Ascend/modelzoo` HEAD `5eab9a4921c7f12edb555079836429a8f285cd1f`
  的 CONTRIBUTING.md 要求源码、README、参考模型 License、测试用例；AASIST-L 另含
  `modelzoo_level.txt`，但 ProtBert、chronos-2 等目录未提供 LICENSE/modelzoo_level.txt，
  历史目录与当前 PR 门禁存在差异。按贡献规范提交，不跳过也不伪造。
- 上库文件清单（候选）：`README.md`、`infer.py`、`prepare_eval_data.py`、
  `compare_embeddings.py`、`requirements.txt`、`test_data/smiles_functional.txt`；
  上库前补 `LICENSE`、`modelzoo_level.txt`。
- 排除项：`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、`patches/README.md`、`upstream/`、
  `weights/`、`eval_data/`、`eval_results/`、`.codex-reference/`、日志与虚拟环境。
- 许可证：上游 `IBM/molformer` License 上库前核对并拷贝；Transformers/HuggingFace
  remote code 各自 License 在公网地址说明中记录。`modelzoo_level.txt` 须在 NPU 实测后据实填写。

## 补充说明（来自 README.md）

### 独立 fine-tuning 环境与 L2 对齐策略

精确复现 IBM 11 项 fine-tuning 指标需建立独立环境，不能让旧版 PyTorch
Lightning/RDKit 覆盖已验证的 NPU 推理环境；当前 L2 优先使用 IBM 官方 split
全量做 feature-extraction 精度和性能对齐。

### 固定本地权重目录

固定本地目录后，运行时不使用远端模型名，避免 remote-code 漂移。

### CPU 导入检查

CPU 环境执行导入检查时省略 `torch_npu` 和 NPU tensor。
