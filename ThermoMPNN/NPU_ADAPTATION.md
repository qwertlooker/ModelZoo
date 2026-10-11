# ThermoMPNN NPU 适配记录

## 版本边界与来源

检查日期：2026-10-11。用户明确要求当前无 NPU 环境时进行模拟适配；本轮不以 Mock 推进 S2/S3。

- 上游：`https://github.com/Kuhlman-Lab/ThermoMPNN`，默认分支 `main`，远端 HEAD 和本地 `upstream/` 均为 `370f76ec62bd929f7425e311d8df04a0d094990f`。
- 主权重：该 commit 的 `models/thermoMPNN_default.pt`，实际通过固定 URL curl 下载约 38 MiB；SHA256 `af449118ccfb4e34971d802321907ed82a8a035ae80f055bf8fc56009a4a838a`。
- 辅助权重：同 commit 的 `vanilla_model_weights/v_48_020.pt`；SHA256 `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd`。
- 模型范围：ThermoMPNN default 单点突变，FP32，蛋白 batch=1，按蛋白分组的突变列表。排除 ThermoMPNN-D、ThermoMPNN-I、训练和多链复合物输入。
- 功能样例：RCSB 1UBQ，三条真实结构上的突变，无实验标签，仅作功能输入；PDB SHA256 `d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161`。
- 原始测试 CSV：`data_all/testing/mega_test.csv`，SHA256 `3adaa13aae93a575704e50cbb9bec4b84516789558204124dacf43925fb9b68f`。

## 引用聊天四模型逐项分析

引用聊天可读取到 AlphaFold-Multimer、DeepAb、MHCflurry、ThermoMPNN 四个名称，助手正文只有内容引用标记，未包含可复用的分析；本轮重新取证。当前 ACL built-in 树未发现四者同名目录；网页检索未发现不等于证明不存在。

| 模型 | 已取得证据 | 路线判断与验收边界 |
|---|---|---|
| AlphaFold-Multimer | [MindSPONGE 官方支持表](https://www.mindspore.cn/mindsponge/docs/en/stable/user/structure_prediction.html) 明确列出 GPU/Ascend 推理 | 优先审计官方 MindSpore 实现；Google JAX 原仓不能只替换 device。核对 Multimer 权重版本、MSA/template、recycle、relax 和结构指标；单体 MEGA-Fold 不能替代 Multimer 证据。本轮仅分析，未定级 |
| DeepAb | [官方仓](https://github.com/RosettaCommons/DeepAb) 为 PyTorch 网络 + PyRosetta 结构实现；本轮未查到可验证现成 NPU 版本 | torch_npu 迁移网络，PyRosetta 保留 CPU。固定 ensemble、旧环境、Rosetta-DL/PyRosetta 授权、CDR 编号；分别比较网络分布与最终 CDR-H3 RMSD。本轮仅分析，未定级 |
| MHCflurry | [官方 requirements](https://github.com/openvax/mhcflurry/blob/20f03d45358dd104df9df0e28e2e0ccea11c9bfb/requirements.txt) 已使用 PyTorch，HEAD `20f03d45358dd104df9df0e28e2e0ccea11c9bfb`；本轮未查到可验证的官方 NPU 版本 | 当前分支优先 torch_npu；核对 CUDA sizing/计时、LocallyConnected1D/KerasBatchNorm 兼容层。历史 TensorFlow/Keras 路线不能套到最新分支。固定发布包，区分 affinity/processing/presentation、HLA allele、肽长度与 percentile。本轮只分析，仓内其他任务的实现状态由其自身文档负责 |
| ThermoMPNN | [官方仓](https://github.com/Kuhlman-Lab/ThermoMPNN) 提供 ProteinMPNN 迁移学习 GNN，本轮未查到现成 NPU 版本 | 本轮实施 default FP32 在线 torch_npu 模拟适配；核心张量已有 device 传播，新增显式入口。原 benchmarking CUDA 硬编码，split/符号/WT 序列/AF 结构需回归；当前 S1 |

推进建议：AlphaFold-Multimer 先审计已有官方实现；ThermoMPNN、最新 MHCflurry 先做在线 PyTorch 探针；DeepAb 先解决环境和授权，再评估完整流程的加速收益。这是源码结构推断，不是 NPU 性能结论。项目里 MolFormer 仅提供三文档与显式设备交付参考，其历史 CPU 结果不能替代 ThermoMPNN 验收。

## 目标仓快照和最新参考目录

重新执行 `git ls-remote https://gitcode.com/Ascend/ModelZoo-PyTorch.git refs/heads/master`，得到目标仓 commit `7eb681e0b8da5390fa8d8a26e7ea1106468bcd38`，2026-10-10，PaddleOCR-VL-1.5 README/性能脚本更新。本次检查使用独立 clone 的最近 60 个 commit，不复用历史快照。

拟合入路径：`ACL_PyTorch/built-in/nlp/ThermoMPNN`。`git ls-tree -r HEAD ACL_PyTorch/built-in` 未发现 ThermoMPNN 目录，本次为新增；领域按现有蛋白模型 ProtBERT 的 nlp 分类暂定，正式 PR 时需维护者确认。

| 参考目录 | 最后实质变更 commit/date | 选择原因 |
|---|---|---|
| `ACL_PyTorch/built-in/nlp/Laya` | `4ab9116cd4b32d43e28b0f39f665f74dc95062e9`，2026-09-29 | 最近可见 nlp 在线 PyTorch 推理实质合入，Atlas 800I A2，独立验证入口；其图编译改动不适用于本轮模型 |
| `ACL_PyTorch/built-in/cv/OVRCOAT` | `1e095beb42ef4e1248d116f2d9e230b82f25bd37`，2026-10-09 | 最新可见 PyTorch 模型新增，A2、CANN 8.3.RC1、torch 2.1 及源码补丁交付 |
| `ACL_PyTorch/built-in/cv/RSGPNet` | `3d0b05477720c6397985507072b52519fedf01f0`，2026-10-08 | A2、CANN 8.3.RC1、torch 2.1 的同类近期环境参考 |

同领域历史 `nlp/ProtBert_for_Pytorch` 仅参考蛋白模型分类：浅克隆在该路径只可见边界提交 `4da8c11…`（2026-06-29），不能当作其最后实质变更证明，未用于新近排序。当前参考窗口未见蛋白稳定性 GNN 的近期合入。

模拟硬件选 **Atlas 800I A2 单卡**。不把“最新出现硬件”误当成通用默认：近期 CosyVoice3 还有 Ascend 950PR 专用实现，但不据此迁移 ThermoMPNN 到其 vLLM 路线。软件组合采用 CANN 8.3.RC1、PyTorch 2.1.0、torch-npu 2.1.0.post17、Python 3.10，torch-npu patch 版本由当前 Ascend 官方兼容表补足。它是模拟目标方案，未作安装或算子支持实测。

贡献规范：实时查询 `Ascend/modelzoo` HEAD `5eab9a4921c7f12edb555079836429a8f285cd1f` 并读取 CONTRIBUTING.md；目标仓 ACL README 仍指向该文件。要求源码、README、许可证、自测试、CLA 和 built-in `modelzoo_level.txt`。其旧 Python 3.7 流水线、GPU 性能比较与当前项目流程不同，正式上库时需核对适用环境及性能门禁；本轮没有签署 CLA、提交官方 PR 或生成虚假的 OK 状态。

## 实现与设备节点

- 上游 `analysis/custom_inference.py` 自动选择 CUDA/CPU；`analysis/thermompnn_benchmarking.py` 的 metrics 和 labels 硬编码 CUDA。
- 新增 `infer.py` 显式 `--device npu/cpu/cuda`，默认 NPU；仅 NPU 条件导入 torch_npu，设备不可用立即失败，不自动回退。
- 复用 `TransferModelPL.load_from_checkpoint(..., map_location='cpu').model`、官方 `Mutation`、`alt_parse_PDB`、`TransferModel.forward`。所有权重先载入 CPU 后统一迁移；这不是 CPU 推理回退。
- 核心 `tied_featurize` 取模型 parameter 的 device 创建张量，ProteinMPNN 的 gather/topk、one_hot/einsum、LayerNorm、Conv1d 等保持原样。所有算子仍需 NPU 探针确认，不提前优化或改算法。
- 数据准备独立运行，不导入模型；manifest 固定 PDB SHA、ID、链、0-based 索引及标签。Megascale 采用官方 `-ddG_ML` 符号，并保留实验 WT 序列覆盖结构序列的逻辑。
- 数值比较按 ID 对齐且验证样本身份；非有限数、重复 ID、漏样本、标签不一致、阈值失败立即报错。
- 质量指标为明确等价的 pooled Pearson/Spearman/MSE/RMSE/R²；性能写实际 device、至少三次计时、mutations/s、中位数和计时范围。正式验收还要三次独立进程、环境版本和回退日志。
- 报告拒绝 CPU/显式模拟结果，核对预测 SHA、样本数、比较结果与性能次数。收集通过仍不自动赋予 S3。

未修改上游已有文件，因此 **无 patch**，`git apply --check` 不适用；不生成空 patch，不修改 site-packages，不引入静态检查屏蔽注释。CPU/CUDA 路径的模型算法保持相同；相对上游新入口的回归仍需真实 CPU 输出验证。

## 依赖边界

上游 `environment.yaml` 包含 CUDA/训练工具，不能直接用于 NPU；最新 `pyproject.toml` 还选择 torchmetrics 1.9.0，而代码仍调用旧 `squared` 参数。因此固定旧接口兼容的 torchmetrics 0.11.4 和 Lightning 2.2.5，形成待实机导入验证的推理依赖，不通过 monkey patch 处理缺接口。BioPython 1.81 保留 `pairwise2` 的上游导入。

`requirements.txt` 不包含框架 wheel、CUDA、mmseqs2 或训练数据工具；保留 Lightning/wandb/joblib 是上游前置 import 链所需。`constraints-npu.txt` 锁定 torch/torch-npu 防止依赖安装覆盖框架。安装命令见自包含 README。

当前控制机是 WSL Ubuntu 24.04，Python 3.12.3，无 pip、NumPy、PyTorch、SciPy、torch_npu。其 Python 不符合上游 3.10 声明，不能声称在此完成环境配套。未修改系统 Python 或安装系统依赖。clean-room 使用无 pip 的新 venv 仅隔离 stdlib 测试上下文，不作为推荐 NPU 部署环境。

## 验证事实和未执行项

2026-10-11 已执行：

- 查询上游、目标仓和贡献规范远端 HEAD；读取近期实质合入 README，盘点目标目录。
- 固定 URL curl 下载主权重与 1UBQ，实际验证 SHA 和 ATOM 数据；辅助权重来自固定 clone 并计算 SHA。
- 上游源码保持 clean，新增 Python 文件语法编译通过。
- `python3 -m unittest discover -s ThermoMPNN/tests -v`：12 项离线回归通过，包含真实数据准备/比较 CLI 和 Mock 推理编排。Mock 只证明路由、warmup/计时次数及文件契约。
- 实际运行 Megascale CSV 盘点：28,312 行，排除 140 条 wt，28,172 条突变，28 个蛋白；没有 PDB，因此不声称全量数据准备完成。
- 依赖无关入口的 `--help`、功能 manifest 准备、比较失败退出码已执行；详见 clean-room 记录。

未执行/阻塞：

| 命令或门禁 | 结果与原因 | 补验方式 |
|---|---|---|
| `python3 infer.py --help` | 启动时报 `ModuleNotFoundError: No module named 'numpy'` | 在 README 声明的 Python 3.10 配套环境安装依赖及 source，再执行 |
| `python3 eval_metrics.py --help` | 同上 | 安装 NumPy/SciPy 后执行导入和已知指标 fixture |
| `infer.py --device cpu ...` | 未执行真实模型输出；缺框架/版本环境 | 用同 manifest 跑 CPU，并对齐官方单链预处理与原始 dataset 路径 |
| NPU tensor、推理和 L2 | 用户当前无 NPU 硬件 | A2 环境按 README 命令补验，查算子回退日志 |
| NPU 性能 | 无实测 | 同 L2 manifest 至少三次独立启动，记录中位数，绝不报告 Mock 数字 |
| 原始全量 PDB/split 对齐 | 配套 PDB 未准备 | 取得授权的原始结构，复核 140 行 wt 与论文 N 差异后冻结完整 manifest |
| 官方在线自测试、CLA、版权头/modelzoo_level.txt | 无真实验收与维护者确认 | S3 后据实补齐，不复用离线 unittest 冒充在线模型测试 |

## 上库文件清单和排除项

候选文件：`README.md`、`LICENSE`、`infer.py`、`delivery_utils.py`、`prepare_eval_data.py`、`prepare_megascale.py`、`compare_results.py`、`eval_metrics.py`、`report_acceptance.py`、`requirements.txt`、`constraints-npu.txt`、`test_data/mutations.csv`、`tests/test_delivery.py`。

排除：`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、`upstream/`、下载/运行生成的 `source/`、权重、PDB、dataset、manifest、输出、日志、`.venv*/`、`__pycache__/`、`.codex-reference/`。`LICENSE` 原样保留上游 MIT 版权；新增代码的正式目标仓版权头规则需上库前确认，不冒用华为版权归属。

当前候选目录未就绪：最低真实推理路径、NPU L2 和贡献门禁仍有缺失；结构审计不能证明可上库。

## clean-room 重放记录

从精确候选清单复制到 `.codex-reference/thermompnn-20261011/clean-room/`，新建 `.venv-replay`（`--without-pip`）；不复制内部适配/验收文档、历史输出或原 upstream。清除 PYTHONPATH，从候选目录运行：

| 阶段 | 结果 |
|---|---|
| 新 stdlib venv，候选脚本 py_compile | 通过 |
| 数据准备/盘点/比较/报告 `--help` | 通过 |
| candidate tests，Mock 设备与推理编排 | 12 项通过 |
| 固定 commit 获取官方 source、权重和功能 PDB | 实际获取并核对；无 patch |
| 三条 1UBQ manifest | 通过，真实结构输入，无模型输出 |
| `infer.py --help` / `eval_metrics.py --help` | 缺 NumPy，阻塞 |
| 真实 CPU/NPU、指标和性能 | 未验证 |

首次候选目录 curl 主权重下载出现 SSL 连接失败（退出码 35），保留在私有 replay 日志；使用同固定 URL 加 `--retry 3 --retry-all-errors` 后下载成功，SHA 与已固定权重一致。没有替换下载源或关闭 TLS 校验。最终入口变更后再次同步候选代码并重跑 stdlib 测试与语法检查。

## 需求—证据表与状态

| 要求 | 证据 | 状态 |
|---|---|---|
| 固定模型/辅助权重 | 远端 HEAD、实际下载 SHA | 已证实 |
| 静态模拟适配 | 新入口、语法、12 项回归 | 已证实 |
| 数据准备工具 | 真实 PDB manifest、官方 CSV 盘点 | 已证实；全量结构缺失 |
| 原始指标取证 | 官方 README/dataset + 论文/预印本来源，详见验收计划 | 已证实口径；checkpoint 精确可比性待确认 |
| 环境可安装/import | 仅拟定固定配套；当前无关键依赖 | 缺失 |
| CPU/CUDA 原行为回归 | 模型算法未修改，尚无真实输出 | 缺失 |
| NPU 对齐、全量指标和性能 | 无 NPU 实测 | 缺失 |
| 上库候选就绪 | 精确清单，尚未通过真实路径及贡献门禁 | 缺失 |

**当前状态：S1 静态模拟适配完成。** 升级 S2 需要真实 CPU 或 NPU 功能输出；升级 S3 需要完整 NPU L2 精度/性能和可比基线。Mock、数据盘点和静态审计不改变该状态。
