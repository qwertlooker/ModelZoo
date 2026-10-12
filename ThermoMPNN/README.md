# ThermoMPNN 推理指导

## 概述

ThermoMPNN 是 Kuhlman-Lab 基于 ProteinMPNN 迁移学习的图神经网络，用蛋白质结构预测单点氨基酸突变的稳定性变化 ΔΔG。本文介绍该模型基于昇腾 NPU 的推理方法。当前交付为 **S1 静态模拟适配**，尚未完成真实模型推理和 NPU 验收。

参考实现：[Kuhlman-Lab/ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN)，`commit_id=370f76ec62bd929f7425e311d8df04a0d094990f`，权重 `thermoMPNN_default.pt`。当前范围为单点突变推理；ThermoMPNN-D、ThermoMPNN-I 和训练不在本轮交付范围。复用上游模型及预处理，无上游源码修改，无 patch。

## 输入输出数据

输入为本地 PDB、一个明确的链 ID 和单点突变列表；支持同一蛋白的多突变预测。输出为逐样本 `ddG_pred`（kcal/mol，正值表示去稳定化）、原始样本字段与运行 metadata。

CSV 必需字段：`id,pdb,chain,position,wildtype,mutation`，可选 `ddG_true`。
`position` 是上游解析后链序列的 **0-based 索引**，不是 PDB residue number；缺失残基和 insertion code 必须先核对。只接受标准 20 种氨基酸及单链输入，不能混用不同链的残基编号。

评测目录结构、固定归档的获取和离线复用见[准备正式评测数据和报告](#准备正式评测数据和报告)。

## 推理环境准备

| 配套 | 版本 |
|---|---|
| 模拟目标硬件 | Atlas 800I A2，单卡 |
| 固件与驱动 | 与 CANN 8.3.RC1 配套，服务器就绪后记录实际版本 |
| CANN | 8.3.RC1 |
| Python | 3.10.x（上游指定 3.10） |
| PyTorch / torch_npu | 2.1.0 / 2.1.0.post17 |
| NumPy / pandas | 1.26.4 / 2.2.2 |
| Lightning / torchmetrics | 2.2.5 / 0.11.4 |
| BioPython / OmegaConf / SciPy | 1.81 / 2.3.0 / 1.13.1 |

硬件参考官方目标仓 2026-10 的 OVRCOAT/RSGPNet，以及 2026-09 的 Laya 合入记录；CANN/PyTorch 配对依据 Ascend 官方兼容表。上述模型依赖是待实机验证的固定方案。旧版 torchmetrics 保留上游使用的 `MeanSquaredError(squared=...)` 接口；不直接安装上游最新版训练环境。现有系统 Python 满足版本时直接使用，不额外创建 venv。

## 文件目录

```text
ThermoMPNN/
├── README.md
├── LICENSE
├── infer.py
├── baseline_upstream.py
├── delivery_utils.py
├── prepare_eval_data.py
├── prepare_megascale.py
├── compare_results.py
├── eval_metrics.py
├── report_acceptance.py
├── requirements.txt
├── constraints-npu.txt
├── test_data/
│   └── mutations.csv
└── tests/
    ├── test_delivery.py
    └── test_upstream_baseline.py
```

下列命令均从本文件所在的 ThermoMPNN 目录执行。`source/`、`weights/`、`eval_data/` 和输出目录由准备步骤生成，不作为上库源码提交。

## 快速上手

### 获取源码和依赖

```bash
git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git source
git -C source checkout 370f76ec62bd929f7425e311d8df04a0d094990f
git -C source diff --exit-code HEAD
```

在已按 CANN 配套安装的 NPU 环境执行；没有预装框架时，使用官方配套 wheel 安装命令：

```bash
python3 -m pip install torch==2.1.0 torch-npu==2.1.0.post17
python3 -m pip install -c constraints-npu.txt -r requirements.txt
python3 -m pip freeze > environment.freeze.txt
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.ones(1).to('npu').device)"
python3 -c "import sys; sys.path.insert(0, 'source'); from train_thermompnn import TransferModelPL; from protein_mpnn_utils import alt_parse_PDB; import scipy.stats; print('model and evaluator imports OK')"
```

安装源必须提供 CANN 配套框架 wheel；不要从 CPU-only 索引安装后追加 torch-npu。CPU 精度对照复用该环境并显式使用 `--device cpu`。不需要 torchvision、torchaudio、mmseqs2；上游 Lightning 入口前置导入 wandb，因此保留 wandb 依赖，但推理不创建日志服务。卡号由运行环境的 `ASCEND_RT_VISIBLE_DEVICES` 设置，不写入模型代码。

### 准备权重和功能样例

```bash
mkdir -p weights eval_data
curl -L --fail -o weights/thermoMPNN_default.pt https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN/370f76ec62bd929f7425e311d8df04a0d094990f/models/thermoMPNN_default.pt
sha256sum weights/thermoMPNN_default.pt source/vanilla_model_weights/v_48_020.pt
curl -L --fail -o eval_data/1UBQ.pdb https://files.rcsb.org/download/1UBQ.pdb
sha256sum eval_data/1UBQ.pdb
python3 prepare_eval_data.py --csv test_data/mutations.csv --pdb-dir eval_data --dataset RCSB-1UBQ-functional --split demo --output eval_data/functional.jsonl
```

权重 SHA256：`af449118ccfb4e34971d802321907ed82a8a035ae80f055bf8fc56009a4a838a`；辅助权重 SHA256：`c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd`。二者由推理脚本强制检查。1UBQ 本次下载 SHA256：`d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161`；PDB 数据库后续可能更新，应保存实际下载文件。该三条无标签输入只检查功能。

可在联网机器用上述 curl 命令准备，再将固定源码、权重和 PDB 传入目标机器，校验后按相同目录组织；推理不执行自动下载。

### 模型推理和比较

```bash
python3 infer.py --device cpu --checkpoint weights/thermoMPNN_default.pt --manifest eval_data/functional.jsonl --output-dir outputs_cpu
python3 infer.py --device npu --checkpoint weights/thermoMPNN_default.pt --manifest eval_data/functional.jsonl --output-dir outputs_npu
python3 compare_results.py --baseline outputs_cpu/predictions.jsonl --candidate outputs_npu/predictions.jsonl --atol 0.001 --rtol 0.001 --report functional_comparison.json
```

每次使用新的输出目录，避免覆盖证据。比较容差为暂定 FP32 工程值，需用真实 baseline 校准；比较失败返回非零退出码。设备不可用时直接失败。输出 `run.json` 含模型和输入 SHA、实际输出设备、计时范围及三次运行中位数；输出张量位于 NPU 不足以排除内部算子回退，实机时另查运行日志。

### 准备正式评测数据和报告

原始 Megascale 测试 CSV 已随固定源码提供。先盘点，再生成规范 CSV：

```bash
python3 prepare_megascale.py --csv source/data_all/testing/mega_test.csv --inventory mega_inventory.json --output eval_data/mega_normalized.csv
```

盘点得到 28,312 行，其中 140 行 `wt` 被排除，形成 28 个蛋白的 28,172 条突变。论文图中的 N=28,312 不应直接作为本脚本突变数；正式验收前必须复核原始 split 与过滤差异。标签按官方 `MegaScaleDataset` 转为 `-ddG_ML`。

Megascale test 的原始结构固定为 Zenodo **7992926 / v2_230420** 的 `AlphaFold_model_PDBs.zip`（约 14 MB，发布页 MD5 `c29317a75f28d008b2718316995902ef`）。请按发布页要求登记数据使用情况，不要用 RCSB 同名结构替换。

```bash
curl -L --fail --retry 3 -o eval_data/AlphaFold_model_PDBs.zip https://zenodo.org/records/7992926/files/AlphaFold_model_PDBs.zip
python3 prepare_megascale.py --csv source/data_all/testing/mega_test.csv --inventory mega_structures.json --pdb-archive eval_data/AlphaFold_model_PDBs.zip --pdb-dir eval_data/megascale_pdbs
```

脚本先验证归档 MD5，只提取 CSV 所需 PDB，并记录归档与每个 PDB 的 SHA256；缺失、重名或已有文件内容不一致立即失败。离线机器直接复用此 ZIP 执行相同命令，内容一致的 PDB 可复用。生成布局：

```text
eval_data/
├── AlphaFold_model_PDBs.zip
├── mega_normalized.csv   # 运行后生成
├── megascale_pdbs/       # 按 WT_name 映射提取 28 个 .pdb
└── mega_test.jsonl       # 运行后生成，共 28,172 条突变
```

规范脚本固定链 A。保持固定上游 `MegaScaleDataset` 语义：实验 WT 只覆盖总序列 `seq` 并用于突变检查；特征化使用的 `seq_chain_A` 保留结构序列。两者长度必须相同，不擅自修正上游行为。

先运行独立官方 CPU 基线，再验证新增入口的 CPU 输出，最后进行 NPU 对齐。`baseline_upstream.py` 直接读取固定原始 CSV、`dataset_splits/mega_splits.pkl` 的 test split 和官方 `config.yaml`，执行未修改的 `MegaScaleDataset`（含原始 `parse_PDB`）及 `TransferModelPL`；不调用 infer.py 的预处理或规范 CSV。它核验源码/权重/CSV/split SHA、结构 SHA、蛋白集合、实验 WT、每条突变和标签，将官方输出映射到 manifest 的稳定 ID；重复实验保留各自 ID。输出目录内生成独立 `cache/`，不复用旧解析缓存。此入口仅支持完整 Megascale test，功能样例不适用。

```bash
python3 prepare_eval_data.py --csv eval_data/mega_normalized.csv --pdb-dir eval_data/megascale_pdbs --dataset ThermoMPNN-Megascale --split test --output eval_data/mega_test.jsonl
python3 baseline_upstream.py --checkpoint weights/thermoMPNN_default.pt --manifest eval_data/mega_test.jsonl --pdb-dir eval_data/megascale_pdbs --output-dir outputs_mega_upstream_cpu
python3 infer.py --device cpu --checkpoint weights/thermoMPNN_default.pt --manifest eval_data/mega_test.jsonl --output-dir outputs_mega_cpu
python3 compare_results.py --baseline outputs_mega_upstream_cpu/predictions.jsonl --candidate outputs_mega_cpu/predictions.jsonl --atol 0.001 --rtol 0.001 --report upstream_cpu_comparison.json
for run in 1 2 3; do
  python3 infer.py --device npu --checkpoint weights/thermoMPNN_default.pt --manifest eval_data/mega_test.jsonl --output-dir "outputs_mega_npu_${run}" --warmup 1 --runs 3 || exit 1
done
python3 compare_results.py --baseline outputs_mega_upstream_cpu/predictions.jsonl --candidate outputs_mega_npu_1/predictions.jsonl --atol 0.001 --rtol 0.001 --report mega_comparison.json
python3 eval_metrics.py --predictions outputs_mega_upstream_cpu/predictions.jsonl --report upstream_cpu_metrics.json
python3 eval_metrics.py --predictions outputs_mega_npu_1/predictions.jsonl --report mega_metrics.json
python3 report_acceptance.py --run-dir outputs_mega_npu_1 --performance-dirs outputs_mega_npu_1 outputs_mega_npu_2 outputs_mega_npu_3 --comparison mega_comparison.json --metrics mega_metrics.json --output acceptance_report.json
```

指标采用 pooled mutations 的 Pearson、Spearman、MSE、RMSE、R²；SciPy 相关系数为等价数学实现。FireProt、S669、SSYM 等需要其各自的序列对齐与 split 规则，不能直接套用 Megascale 转换器。报告入口只收集证据，拒绝 CPU/显式模拟结果，不自动认证 S3 或上库就绪。上述循环独立启动三个 NPU 进程。报告校验不同 run_id、目录、预测 SHA 及一致的 manifest/权重/软件/设备可见性/计时配置，先取各进程内部耗时中位数，再取进程间中位数，给出 mutations/s 和 proteins/s。不传 `--performance-dirs` 时明确列出独立进程性能验收未完成；旧 run.json 无 run_id 时需重新推理，不能复制结果充数。运行标识只检测证据误复用，真实性仍须实机日志证明。

NPU `run.json` 使用 schema 3：记录 Python、逻辑卡、`ASCEND_HOME_PATH`、实际 `torch.version.cann`、`torch.npu.get_device_name()` 和主机名，性能汇总逐项要求一致。CANN 版本由固定 torch-npu 的环境采集逻辑提供，不能填写文档计划版本冒充实测；缺失或 `not known` 时推理在计时前失败。旧记录缺字段必须重新运行。设备型号和主机名用于防止误混结果，正式验收仍需保存 npu-smi、驱动/固件及 fallback 日志。

### 离线自测试

```bash
python3 -m unittest discover -s tests -v
```

此测试只使用 stdlib/Mock，不加载真实模型，不能替代 NPU 在线自测试。环境安装与真实模型功能尚未验证。

## 模型推理性能

| 模型 | 设备 | 数据集 | 精度 | 吞吐 / 延迟 |
|---|---|---|---|---|
| ThermoMPNN default FP32 | Atlas 800I A2（模拟目标） | Megascale test | 待 NPU 验证 | 待 NPU 验证 |

### 精度说明

尚未产生模型预测或官方指标复现结果；离线测试的常量输出只存在于测试临时目录。正式验收必须同时具备正确 split 的全量指标、同 checkpoint 的可比基线、真实 NPU 性能及环境证据。

## 公网地址说明

| 来源 | 用途 |
|---|---|
| [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) | 官方源码、测试 CSV、split、主权重及 ProteinMPNN 权重，MIT |
| [原始论文](https://doi.org/10.1073/pnas.2314853121) | 模型定义与公开指标 |
| [Megascale](https://doi.org/10.5281/zenodo.7401274) / [FireProt](https://doi.org/10.5281/zenodo.8169288) | 用户自行准备正式测试数据及结构 |
| [RCSB 1UBQ](https://www.rcsb.org/structure/1UBQ) | 功能样例 |
| [Ascend 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) | CANN/PyTorch/torch_npu 配套依据 |
| [目标仓](https://gitcode.com/Ascend/ModelZoo-PyTorch) | A2 模拟硬件及推理交付参考 |

命令中使用的直接访问地址：

- 源码 clone：https://github.com/Kuhlman-Lab/ThermoMPNN.git
- 主权重：https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN/370f76ec62bd929f7425e311d8df04a0d094990f/models/thermoMPNN_default.pt
- 功能 PDB：https://files.rcsb.org/download/1UBQ.pdb
- 固定结构发布页：https://zenodo.org/records/7992926
- 结构 ZIP：https://zenodo.org/records/7992926/files/AlphaFold_model_PDBs.zip
