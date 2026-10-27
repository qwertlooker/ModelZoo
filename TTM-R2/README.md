# IBM Granite TinyTimeMixer R2 512→96

## 概述

本工作区是离线适配准备，默认 NPU，不自动回退 CPU。**尚未运行真实模型或 NPU，未达到 S3。** 上游审计快照：<https://github.com/ibm-granite/granite-tsfm/tree/9d0b839f997eb4918822a43036012ce8c277687c>；实际入口固定 PyPI granite-tsfm 0.3.8 的模型源码 SHA `6c4cbcaa417ea09f643ee83918524e7bca6ac369b6b3500a5073c7ccc868dc14`，不假设该轮子等同审计快照。权重 ibm-granite/granite-timeseries-ttm-r2 revision `d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f`，许可证 Apache-2.0。

参考实现 `commit_id=9d0b839f997eb4918822a43036012ce8c277687c`（固定源码或资料审计快照；实际运行依赖版本仍以上述约束为准）。

## 输入输出数据

输入 `[batch,512,channels]`，标签/输出 `[batch,96,channels]`；使用训练分区 `[0,8640)` 逐通道均值和总体标准差，记录 scaler。评价 MSE/MAE 为标准化量纲。 默认全测试区间、步长 1；仅使用训练区间拟合 scaler。

## 推理环境准备

PyPI `granite-tsfm==0.3.8` 声明 Python ≥3.11、<3.14，PyTorch ≥2.10、<2.11，Transformers ≥4.57.6、<6；本入口固定 Transformers 4.57.6。社区案例使用 torch 2.9 并跳过依赖检查，不满足该包声明，不能据此认定兼容。目标硬件的 CANN/torch_npu 配套能满足声明后，才可安装并进行实际测试。使用匹配设备的 CANN、PyTorch 和 torch_npu；不安装 CPU-only PyTorch 再追加 torch_npu。缺失依赖直接报错，非必要不创建 venv。

目标栈满足版本约束并已核对软件配套后，可安装业务依赖并检查依赖一致性：

```bash
set -e
python3 -c 'import torch, torch_npu; from packaging.version import Version; assert Version("2.10") <= Version(torch.__version__) < Version("2.11")'
python3 -m pip install 'granite-tsfm==0.3.8' 'transformers==4.57.6'
python3 -m pip check
```

安装命令及依赖解析尚未在目标栈运行；若解析试图替换配套 torch，停止并重新核对镜像。

## 文件目录

`model.py`、`common.py`、`prepare_series.py` 为工具入口，`performance.py` 为独立性能汇总，`test_model.py`、`test_performance.py` 为离线回归；权重、数据、输出不入正式候选。

## 快速上手

以下命令从本目录执行。准备模型和数据的固定文件，校验 SHA；若网络在 NPU 机不可用，在联网机执行后将文件传入本地目录。

```bash
export MODEL_WORK_DIR="$PWD/work"
mkdir -p "$MODEL_WORK_DIR/model" "$MODEL_WORK_DIR/data"
curl -fL 'https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/resolve/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/model.safetensors' -o "$MODEL_WORK_DIR/model/model.safetensors"
curl -fL 'https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/resolve/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/config.json' -o "$MODEL_WORK_DIR/model/config.json"
printf '%s  %s\n' 'a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710' "$MODEL_WORK_DIR/model/model.safetensors" | sha256sum -c -
printf '%s  %s\n' '5e2367547c103e92cb8ebc63cbd5ad4d7bf83facad5a2a2ec261fa770ed659d5' "$MODEL_WORK_DIR/model/config.json" | sha256sum -c -
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

ETTh1 context 512/horizon 96，官方 Granite R2 原始结果表 zero-shot MSE 0.363；需逐项核对外部标准化、测试切分、模型分支及 evaluator 后才可比较。 当前只提供本地协议，需要复核 0.3.8 与固定 checkpoint 的配置字段实际使用；不同 R2 分支或 research-r2 不可替代。输入七通道是独立预测接口，不等于已验证跨通道混合。 固定输入 SHA、模型 SHA、任务及 head 后才允许比较。同步计时覆盖模型调用，排除输入准备和结果传回；单次 batch 中位数不是三次独立运行中位数。完整 NPU 精度、三次独立性能、端到端时间和显存待上机测量。

## NPU 独立性能报告

每次 `infer` 成功后额外生成 `<输出文件>.performance.json`，保存逐批同步耗时、总模型调用时间、吞吐、结果 SHA、命令和运行 ID。元信息含 Python、PyTorch、torch_npu、设备名称、进程内逻辑卡号及 `ASCEND_RT_VISIBLE_DEVICES` 原值；物理卡映射仍须结合 `npu-smi info` 核对。`performance.py` 校验至少三次不同运行 ID、相同模型/数据/设备/参数，再汇总独立运行中位数；CPU 报告不能用于 NPU 性能结论。

```bash
set -e
for RUN in 1 2 3; do
  python3 model.py infer --model-dir "$MODEL_WORK_DIR/model" --input "$MODEL_WORK_DIR/test.npz" --output "$MODEL_WORK_DIR/npu-run-$RUN.npz" --device npu --task forecast
done
python3 performance.py --runs "$MODEL_WORK_DIR"/npu-run-*.npz.performance.json --output "$MODEL_WORK_DIR/performance-median.json"
```

统计包括末尾不足 batch 的批次；吞吐单位为输入样本/秒（TimesFM 为通道序列/秒）。这是纯模型调用口径，端到端时间、首次编译和峰值 HBM 仍需在目标 NPU 单独实测。报告校验不能认证人工编辑的文件或证明设备证据真实，发布前须核对原始日志。

本目录 `LICENSE` 保留固定上游参考实现的许可原文；正式自测、完整依赖安装与 NPU 验收仍待完成。

## 公网地址

| 内容 | 具体地址 |
|---|---|
| 上游源码 | https://github.com/ibm-granite/granite-tsfm/tree/9d0b839f997eb4918822a43036012ce8c277687c |
| 固定权重 | https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/tree/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f |
| 公开参考 | https://github.com/ibm-granite/granite-tsfm/blob/9d0b839f997eb4918822a43036012ce8c277687c/notebooks/hfdemo/tinytimemixer/full_benchmarking/combined_results-granite-r2.csv |
| ETTh1 数据 | https://github.com/zhouhaoyi/ETDataset/blob/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv |
| 下载或依赖地址 | https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/resolve/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/config.json |
| 下载或依赖地址 | https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/resolve/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/model.safetensors |
| 下载或依赖地址 | https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv |
| 参考许可证 | https://raw.githubusercontent.com/ibm-granite/granite-tsfm/9d0b839f997eb4918822a43036012ce8c277687c/LICENSE |
| 官方软件配套表 | https://github.com/Ascend/pytorch/blob/94dd5bf251144e6227e2b5228ad9f878878c0bf4/COMPATIBILITY.en.md |
