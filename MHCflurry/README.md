# MHCflurry 推理指导

## 概述

MHCflurry 预测 MHC I 肽结合亲和力、抗原 processing 和 presentation。
本文提供基于昇腾 NPU 的 PyTorch eager 推理入口，使用完整官方 presentation
bundle，复用其中的 affinity 和 processing 子模型及官方后处理。
当前交付为 **S1 静态适配完成，待 NPU 验证**；Mock 不代表模型实测。

参考实现：

```text
url=https://github.com/openvax/mhcflurry
branch=master
commit_id=20f03d45358dd104df9df0e28e2e0ccea11c9bfb
code_version=2.3.14
weight_release=2.3.0
weight_archive=models_class1_presentation.20260928.tar.bz2
license=Apache-2.0
```

2.1.x 的 TensorFlow 路径、训练、扫描完整蛋白和其他权重变体不在本次范围。

## 输入输出数据

输入为 CSV，每条包含唯一 `id`、8–15 个标准氨基酸的 `peptide`。
表头必须非空且不重复，每行列数须与表头一致；纯空白 ID 或 allele 不接受。
affinity 每条必须有一个 `alleles`；presentation 支持每条最多六个以分号分隔的
MHC I 等位基因。processing 无需等位基因。`n_flank`、`c_flank` 必须成对出现，
缺少两列表示无 flank 模式；空 flank 与无 flank 模式不同。

输出为 `predictions.csv`：affinity 单位 nM；processing 和 presentation 分数为
0–1；presentation percentile 为上游 0–100 标度（越低表示越强）。affinity
必须为有限正值，分数和百分位必须在各自范围内，越界结果直接失败。presentation 输出另保留官方
`best_allele`，用于核验最强结合 allele 及同分顺序。`run.json` 记录实际设备、
输入和权重哈希、版本、同步后的耗时及计时边界。编码、集成、百分位和 sklearn
presentation 后处理在 CPU，神经网络拟在 NPU；不得将此描述为全流程纯 NPU。

## 推理环境准备

以下是拟验证配套，尚未实际安装或执行。参考最新 nlp 在线适配 Laya 的
Atlas 800I A2 单卡形态，并与 Ascend/PyTorch 官方配套表交叉核实。

| 配套 | 版本 |
|---|---|
| 硬件 | Atlas 800I A2，单卡，待实测 |
| 固件与驱动 | Laya 参考值 25.5.6；目标机按 CANN/硬件兼容表核实 |
| CANN | 9.1.0 |
| Python | 3.12.13 |
| PyTorch / torch_npu | 2.10.0 / 2.10.0.post4，TorchNPU 26.1.0 配套 |
| MHCflurry | 2.3.14，固定上述源码 commit 并应用 diff.patch |
| NumPy / pandas / scikit-learn | 2.2.6 / 2.3.3 / 1.7.2 |

使用已经配好 CANN 的系统 Python 或基础容器，无需新增 venv。Laya 的基础镜像
参考为 `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1`，本模型不需要 vLLM。
该 tag 的 digest 尚未核实，不作为已冻结或实测环境。正式执行前记录实际
镜像 digest、CANN、驱动固件和 `pip freeze`。不要先安装 CPU-only torch 再追加
torch-npu，也不要直接升级配套框架。

## 文件目录

```text
MHCflurry/
├── README.md
├── LICENSE
├── diff.patch
├── source_manifest.json
├── constraints.txt
├── requirements.txt
├── adapter_utils.py
├── prepare_weights.py
├── prepare_eval_data.py
├── infer.py
├── runtime.py
├── compare_results.py
├── eval_metrics.py
├── test_data/
│   └── smoke.csv
└── tests/
    └── test_adapter.py
```

## 快速上手

所有命令在本模型目录执行。输出目录及 `--report` 文件必须不存在，防止覆盖
CPU/NPU 证据；重跑时使用新的路径。报告入口拒绝已有文件，并以独占方式写入。
以下 Linux/bash 命令包含未来真实设备补验入口；无 NPU 时先执行最后的离线检查。

### 获取源码和安装

```bash
export MODEL_ROOT="$PWD"
git clone https://github.com/openvax/mhcflurry.git upstream
git -C upstream checkout 20f03d45358dd104df9df0e28e2e0ccea11c9bfb
git -C upstream apply --check "$MODEL_ROOT/diff.patch"
git -C upstream apply "$MODEL_ROOT/diff.patch"
python3 -m pip install -c constraints.txt -r requirements.txt
python3 -m pip install --no-deps ./upstream
python3 -m pip check
```

业务依赖不包含 torch/torch-npu；约束文件用于保护环境内的配套版本。
`--no-deps` 只用于已单独列全依赖的上游安装。推理前通过
`source_manifest.json` 校验安装包全部 Python 源码的哈希（仅规范化 CRLF），
不同 commit 或没有补丁将直接失败。撤销补丁使用：

```bash
git -C upstream apply --reverse --check "$MODEL_ROOT/diff.patch"
git -C upstream apply --reverse "$MODEL_ROOT/diff.patch"
python3 -m pip install --no-deps --force-reinstall ./upstream
```

补丁应用在固定源码 checkout；不需要修改 site-packages。

### 准备权重

```bash
mkdir -p weights
curl --fail --location --retry 2 -o weights/models_class1_presentation.20260928.tar.bz2 \
  https://github.com/openvax/mhcflurry/releases/download/2.3.0/models_class1_presentation.20260928.tar.bz2
echo '768e360832b3d45dfe9d5e8cde4ffcb90abdb87773ca65f3a86ba7cc20f614db  weights/models_class1_presentation.20260928.tar.bz2' | sha256sum -c -
python3 prepare_weights.py --directory weights --offline
export MODEL_DIR=$(python3 -c "from pathlib import Path; p=list(Path('weights/extracted').rglob('affinity_predictor/manifest.csv')); assert len(p)==1; print(p[0].resolve().parents[1])")
test -f "$MODEL_DIR/weights.csv"
```

压缩包大小 160,356,140 bytes，SHA 来自官方 GitHub Release asset metadata。
可在联网机器执行 curl 后，将完整压缩包传入模型目录的 `weights/`，目标机复用
上述校验和 `--offline` 命令。在线辅助入口 `python3 prepare_weights.py` 等价调用
curl；`--check-only` 只检查 HTTP；`--dry-run` 只打印固定 URL、路径和 SHA。
解压只允许普通文件和目录，拒绝越界路径、链接及已有目标目录。
保留官方归档：推理默认读取 `weights/models_class1_presentation.20260928.tar.bz2`，
可用 `--weight-archive /absolute/path/to/archive.tar.bz2` 指定离线位置。
每次推理先核验固定归档 SHA，从归档生成可信文件清单，再与 model-dir 全部文件
逐项比较；权重或 manifest 修改、缺失、额外文件均失败。校验不计入性能时间。
旧解压目录只要与官方归档完全一致即可复用。

### 准备数据集

仓内四条肽及人工 AAA/CCC flank 仅用于功能链路验证，无真实标签，不能作为 L2。

```bash
python3 prepare_eval_data.py --input test_data/smoke.csv --output-dir eval_data/smoke \
  --task presentation --dataset mhcflurry-demo --split demo --purpose smoke
```

正式 L2 使用固定公开/业务 cohort；在 `eval_data/l2_source.csv` 准备相同 CSV 字段。
若计算 presentation 指标，必须包含非空、非纯空白的 `sample_id` 和二元 `label`，每个 patient 必须包含
正、负样本。不得重新随机抽负样本或将训练集作为未见测试集。冻结来源、split、
选择规则及 SHA，并使用至少 1,000 条有代表性的真实记录；此下限是工程最低范围，
不是官方 benchmark 的替代。处理入口完全离线：

```bash
python3 prepare_eval_data.py --input eval_data/l2_source.csv --output-dir eval_data/l2 \
  --task presentation --dataset frozen-presentation-cohort --split test --purpose l2
```

该命令生成 `eval_data/l2/manifest.csv` 和 `manifest.meta.json`，准备脚本不加载模型。
官方 2.3.0 comparison-v4 的完整 cohort 是 181,131 条/10 patients，但公开报告包
只有汇总表和绘图素材，没有原始逐条输入，不能从报告包伪造这份 manifest。
同一原始 cohort 尚未取得时，以同 checkpoint CPU 精度对照确认迁移，另报告其
dataset/metric；公开数字只能注明来源作为参考。

### 模型推理

先在目标机确认 NPU 分配及物理到逻辑卡映射，按实际配额设置
`ASCEND_RT_VISIBLE_DEVICES`，不在代码中绑定卡号。只读核实环境后执行张量探针：

```bash
npu-smi info
python3 -c "import torch,torch_npu; x=torch.ones(1,device='npu')+1; torch.npu.synchronize(); print(torch.__version__,torch_npu.__version__,x.device,x.cpu())"
python3 infer.py --manifest eval_data/smoke/manifest.csv --model-dir "$MODEL_DIR" \
  --task presentation --device npu --output-dir outputs_npu_smoke
```

正式精度和性能执行同一固定 manifest、全 ensemble、FP32、eager 和 batch 128。
presentation 按相同的有序 allele 集合组批，保留 allele 顺序和原始行 ID；
实际尾批或小 genotype 组可小于 128。评测仍按原始 patient 分组。
先保留未改设备算法的 CPU 输出，再执行 NPU。输出包含两次 warmup 后的三次
独立计时调用及中位数，另记首次加载加推理耗时；不包含 CSV 写盘。

```bash
python3 infer.py --manifest eval_data/l2/manifest.csv --model-dir "$MODEL_DIR" \
  --task presentation --device cpu --batch-size 128 --warmup 2 --runs 3 --output-dir outputs_cpu_l2
python3 infer.py --manifest eval_data/l2/manifest.csv --model-dir "$MODEL_DIR" \
  --task presentation --device npu --batch-size 128 --warmup 2 --runs 3 --output-dir outputs_npu_l2
python3 compare_results.py --reference outputs_cpu_l2 --candidate outputs_npu_l2 \
  --atol 0.001 --rtol 0.001 --report numeric_report.json
python3 eval_metrics.py --manifest eval_data/l2/manifest.csv --results outputs_cpu_l2 --report cpu_metrics.json
python3 eval_metrics.py --manifest eval_data/l2/manifest.csv --results outputs_npu_l2 \
  --reference-metrics cpu_metrics.json --max-metric-drop 0.001 --report npu_metrics.json
```

容差是暂定 FP32 工程阈值，需 CPU baseline 波动校准后冻结。比较器按稳定 ID
逐列检查，输入、权重、源码、任务或 batch 不同以及非有限数都会失败；阈值失败
非零退出。指标为等 patient 平均 AP/AUROC/PPV@N，PPV 截断平分同分样本。
数值通过不自动升级 S3，还需人工核实 L2 数据资格、正式指标、实际 NPU 算子
执行日志及环境证据，排查未支持算子的隐式 CPU fallback。
推理、数值比较和指标入口共用完整结果校验；独立运行指标入口也会检查
affinity、processing_score、presentation_score、presentation_percentile 和 best_allele，
不会仅凭排序分数放行损坏的预测文件。

presentation 比较还要求 `best_allele` 非空且逐 ID 完全一致；即使数值在容差内，
allele 选择变化仍返回失败。指标报告保留源码 revision、MHCflurry/PyTorch 版本、
task、batch、optimization/compile 配置及实际网络设备；CPU/NPU 配置不一致或
设备记录不符直接失败。旧指标报告缺少这些字段时必须重新生成，不能补填猜测值。

affinity 和 processing 使用同一入口的 `--task affinity` / `--task processing`，
各任务使用独立输出目录。affinity 逐条一个等位基因；processing 支持无需等位基因。
无 flank 的 L2 应另准备不含两列 flank 的 manifest，不可混用有 flank 公开指标。

### 无硬件离线检查

```bash
python3 infer.py --help
python3 prepare_weights.py --dry-run
python3 infer.py --manifest test_data/smoke.csv --model-dir weights/not-loaded \
  --output-dir outputs_dry_run --device npu --dry-run
export MHCFLURRY_UPSTREAM="$PWD/upstream"
python3 -m unittest discover -s tests -v
```

上述检查不生成 NPU 推理或性能证据。`MHCFLURRY_UPSTREAM` 是声明的已打补丁
源码目录，用于 Mock CPU/CUDA/NPU 分支回归；不设置时相关源码测试跳过。

## 模型推理性能

| 模型 | Card | Batch | 指标 | 实测结果 |
|---|---|---:|---|---|
| MHCflurry 2.3.0 presentation bundle | Atlas 800I A2（拟验证） | 128 | 端到端 peptides/s、三次耗时中位数 | 待 NPU 验证 |

## 精度

官方 2.3.0 comparison-v4 在 181,131 条同 cohort 上的等 patient 平均结果：

| 配置 | AP | PPV@N | AUROC | NPU |
|---|---:|---:|---:|---|
| 有 N/C flanks | 0.7986 | 0.7450 | 0.9524 | 待 NPU 验证 |
| 无 flanks | 0.7893 | 0.7373 | 0.9501 | 待 NPU 验证 |

来源为固定源码 `docs/release_model_evaluation.md` 和官方 comparison-v4 报告。
这些是 presentation-label ranking 指标，不能冒充独立定量结合亲和力准确率，
也不能直接用于不同 cohort 的通过线。

## 公网地址说明

| 用途 | 地址 |
|---|---|
| 源码与许可证 | https://github.com/openvax/mhcflurry/tree/20f03d45358dd104df9df0e28e2e0ccea11c9bfb |
| 源码首页 | https://github.com/openvax/mhcflurry |
| Git 下载入口 | https://github.com/openvax/mhcflurry.git |
| 权重 | https://github.com/openvax/mhcflurry/releases/download/2.3.0/models_class1_presentation.20260928.tar.bz2 |
| 官方指标说明 | https://github.com/openvax/mhcflurry/blob/20f03d45358dd104df9df0e28e2e0ccea11c9bfb/docs/release_model_evaluation.md |
| 官方比较报告 | https://github.com/openvax/mhcflurry/releases/download/2.3.0/model-comparison.20260928-v4.tar.gz |
| 官方模型仓参考 | https://gitcode.com/Ascend/ModelZoo-PyTorch/tree/e779117e8cfeba296a7105e26adac0741ab04768/ACL_PyTorch/built-in/nlp/Laya |
| 配套表 | https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md |
| 参考镜像 | https://quay.nju.edu.cn/ascend/vllm-ascend |
| 依赖 metadata | https://pypi.org |
