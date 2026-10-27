# Datadog Toto 2.0 22m

## 概述

本工作区是离线适配准备，默认 NPU，不自动回退 CPU。**尚未运行真实模型或 NPU，未达到 S3。** 固定模型与源代码：https://github.com/DataDog/toto/tree/1d3e1c97ccf615016d38fe910602c905ae31fbb7；权重 Datadog/Toto-2.0-22m revision `685e4ae3e2be8d8998025e53dd98e7fdcb296a89`，许可证 Apache-2.0。

## 输入输出数据

输入 `[batch,variates,512]`，输出 `[batch,variates,horizon]` 中位数及九个分位数；本入口仅支持全部观测值有限的输入，不自动把缺失值置为已观测。 结果含九个分位数和中位数；`evaluate` 给出本集合加权分位损失和分位交叉比例，未实现 BOOM CRPS/MASE。

## 推理环境准备

固定 Toto 源码的 `toto2/pyproject.toml` 和 `dd_unit_scaling/pyproject.toml` 明确要求 Python ≥3.12、torch ≥2.14.0。目标昇腾栈是否有配套版本尚未知；确认存在兼容栈前不要绕过这些要求或安装不匹配版本。 使用匹配设备的 CANN、PyTorch 和 torch_npu；未确定硬件和配套版本，不安装 CPU-only PyTorch 再追加 torch_npu。模型代码、numpy、safetensors 等依赖须按固定上游版本及目标栈配置；缺失时直接报错。非必要不创建 venv。

仅在已确认目标栈满足上述约束后，按固定提交准备源码并安装；安装前核对依赖解析不得替换配套 torch：

```bash
python3 -c 'import sys, torch, torch_npu; from packaging.version import Version; assert sys.version_info >= (3, 12) and Version(torch.__version__) >= Version("2.14")'
git clone https://github.com/DataDog/toto.git toto-source
git -C toto-source checkout 1d3e1c97ccf615016d38fe910602c905ae31fbb7
python3 -m pip install ./toto-source/dd_unit_scaling ./toto-source/toto2
python3 -m pip check
```

入口校验 Toto 模型源码 SHA `62614814939484b05aac1423d6ebf09977543fda75a53e3dadb5a54f8ec6fdbf`。当前未找到已验证的目标昇腾配套栈，不能提前认定这些命令可运行。

## 文件目录

`model.py`、`common.py`、`prepare_series.py` 为工具入口，`test_model.py` 为离线回归；权重、数据、输出不入正式候选。

## 快速上手

以下命令从本目录执行。准备模型和数据的固定文件，校验 SHA；若网络在 NPU 机不可用，在联网机执行后将文件传入本地目录。

```bash
export MODEL_WORK_DIR="$PWD/work"
mkdir -p "$MODEL_WORK_DIR/model" "$MODEL_WORK_DIR/data"
curl -fL 'https://huggingface.co/Datadog/Toto-2.0-22m/resolve/685e4ae3e2be8d8998025e53dd98e7fdcb296a89/model.safetensors' -o "$MODEL_WORK_DIR/model/model.safetensors"
curl -fL 'https://huggingface.co/Datadog/Toto-2.0-22m/resolve/685e4ae3e2be8d8998025e53dd98e7fdcb296a89/config.json' -o "$MODEL_WORK_DIR/model/config.json"
printf '%s  %s\n' '9cd503d82df3aa71747862688f47a31c1d0a4b80f898df6e046189016eaa21dd' "$MODEL_WORK_DIR/model/model.safetensors" | sha256sum -c -
printf '%s  %s\n' 'abeaf0fcd54aaac66757fde69ec3ddb4d3bfdcf96e0c8f767aefd03ab4c9e8d9' "$MODEL_WORK_DIR/model/config.json" | sha256sum -c -
curl -fL 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv' -o "$MODEL_WORK_DIR/data/ETTh1.csv"
printf '%s  %s\n' 'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066' "$MODEL_WORK_DIR/data/ETTh1.csv" | sha256sum -c -
python3 prepare_series.py --dataset "$MODEL_WORK_DIR/data/ETTh1.csv" --output "$MODEL_WORK_DIR/test.npz" --split test --horizon 96
python3 model.py infer --model-dir "$MODEL_WORK_DIR/model" --input "$MODEL_WORK_DIR/test.npz" --output "$MODEL_WORK_DIR/npu.npz" --device npu --task forecast
python3 model.py evaluate --input "$MODEL_WORK_DIR/test.npz" --results "$MODEL_WORK_DIR/npu.npz" --output "$MODEL_WORK_DIR/npu-metrics.json" --metric regression
```

其中 NPU 推理或训练须在硬件及依赖配套后执行。对照运行可显式使用 CPU；比较前按任务确定可接受的阈值。下例的 0.01 仅说明参数，不是验收门槛。

```bash
python3 model.py infer --model-dir "$MODEL_WORK_DIR/model" --input "$MODEL_WORK_DIR/test.npz" --output "$MODEL_WORK_DIR/cpu.npz" --device cpu --task forecast
export MAX_ABSOLUTE_ERROR=0.01
python3 model.py compare --reference "$MODEL_WORK_DIR/cpu.npz" --candidate "$MODEL_WORK_DIR/npu.npz" --max-absolute "$MAX_ABSOLUTE_ERROR" --output "$MODEL_WORK_DIR/compare.json"
```

## 性能与精度

官方 22m 模型卡的 BOOM CRPS 0.363、MASE 0.601；该聚合指标需要 BOOM 完整协议，不能用 ETTh1 的 MSE/MAE/WQL 代替。 当前只提供本地协议，22m、4m 和 Toto 1.0 的权重与输出不同；ETTh1 未覆盖缺失、变长和多分块场景。WQL 是本输入集合的均值分位损失，分母为目标绝对值之和，不是 BOOM 聚合 CRPS。 固定输入 SHA、模型 SHA、任务及 head 后才允许比较。同步计时覆盖模型调用，排除输入准备和结果传回；单次 batch 中位数不是三次独立运行中位数。完整 NPU 精度、三次独立性能、端到端时间和显存待上机测量。

## 公网地址

| 内容 | 具体地址 |
|---|---|
| 上游源码 | https://github.com/DataDog/toto/tree/1d3e1c97ccf615016d38fe910602c905ae31fbb7 |
| 固定权重 | https://huggingface.co/Datadog/Toto-2.0-22m/tree/685e4ae3e2be8d8998025e53dd98e7fdcb296a89 |
| 公开参考 | https://huggingface.co/Datadog/Toto-2.0-22m/blob/685e4ae3e2be8d8998025e53dd98e7fdcb296a89/README.md |
| ETTh1 数据 | https://github.com/zhouhaoyi/ETDataset/blob/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv |
