# AutonLab MOMENT-1-large，重建及训练预测头

## 概述

本工作区是离线适配准备，默认 NPU，不自动回退 CPU。**尚未运行真实模型或 NPU，未达到 S3。** 固定模型与源代码：<https://github.com/moment-timeseries-foundation-model/moment/tree/38f7310ad594100747ca2a8357e9c7ca7d323e0e>；权重 AutonLab/MOMENT-1-large revision `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`，许可证 MIT。

参考实现 `commit_id=38f7310ad594100747ca2a8357e9c7ca7d323e0e`（固定源码或资料审计快照；实际运行依赖版本仍以上述约束为准）。

## 输入输出数据

输入 `[batch,channels,512]`，预测输出 `[batch,channels,192]`；外部标准化只用训练分区。重建模式输出输入长度 512，默认全观测，不是缺失值补全验收。 训练头命令必须实际成功并生成 预测头旁的同名 `.json` 文件 元数据；预测命令会核对该元数据，不接受缺少训练记录的随机头。

## 推理环境准备

固定上游源码 momentfm 0.1.5；需在配套目标栈中确认 torch、torch_npu、Transformers、safetensors 和 numpy。训练与推理均未实际运行。 使用匹配设备的 CANN、PyTorch 和 torch_npu；未确定硬件和配套版本，不安装 CPU-only PyTorch 再追加 torch_npu。模型代码、numpy、safetensors 等依赖须按固定上游版本及目标栈配置；缺失时直接报错。非必要不创建 venv。

确认配套栈后，从固定源码安装并检查依赖；安装前核对解析过程不能替换配套 torch：

```bash
set -e
python3 -c 'import torch, torch_npu'
git clone https://github.com/moment-timeseries-foundation-model/moment.git moment-source
git -C moment-source checkout 38f7310ad594100747ca2a8357e9c7ca7d323e0e
python3 -m pip install ./moment-source 'transformers==4.57.6' safetensors
python3 -m pip check
```

入口校验 MOMENT 模型源码 SHA `70510f443376c9e15bc41326cf7ed3374b1f9e43d9cda2ac448e6004c0943820`；安装及训练尚未在目标栈运行。

## 文件目录

```text
README.md
LICENSE
model.py
prepare_series.py
train_head.py
```

## 快速上手

以下命令从本目录执行。准备模型和数据的固定文件，校验 SHA；若网络在 NPU 机不可用，在联网机执行后将文件传入本地目录。

```bash
export MODEL_WORK_DIR="$PWD/work"
mkdir -p "$MODEL_WORK_DIR/model" "$MODEL_WORK_DIR/data"
curl -fL 'https://huggingface.co/AutonLab/MOMENT-1-large/resolve/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc/model.safetensors' -o "$MODEL_WORK_DIR/model/model.safetensors"
curl -fL 'https://huggingface.co/AutonLab/MOMENT-1-large/resolve/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc/config.json' -o "$MODEL_WORK_DIR/model/config.json"
printf '%s  %s\n' 'a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572' "$MODEL_WORK_DIR/model/model.safetensors" | sha256sum -c -
printf '%s  %s\n' 'b0111650e8718e31d7e89575666aa3e140747d35279ec1a7c9f6ed7616c29433' "$MODEL_WORK_DIR/model/config.json" | sha256sum -c -
curl -fL 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv' -o "$MODEL_WORK_DIR/data/ETTh1.csv"
printf '%s  %s\n' 'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066' "$MODEL_WORK_DIR/data/ETTh1.csv" | sha256sum -c -
python3 prepare_series.py --dataset "$MODEL_WORK_DIR/data/ETTh1.csv" --output "$MODEL_WORK_DIR/train.npz" --split train --horizon 192
python3 prepare_series.py --dataset "$MODEL_WORK_DIR/data/ETTh1.csv" --output "$MODEL_WORK_DIR/test.npz" --split test --horizon 192
python3 train_head.py --model-dir "$MODEL_WORK_DIR/model" --train "$MODEL_WORK_DIR/train.npz" --output "$MODEL_WORK_DIR/head.safetensors" --device npu --epochs 1
export HEAD_SHA256="$(sha256sum "$MODEL_WORK_DIR/head.safetensors" | cut -d ' ' -f 1)"
python3 model.py infer --model-dir "$MODEL_WORK_DIR/model" --input "$MODEL_WORK_DIR/test.npz" --output "$MODEL_WORK_DIR/npu.npz" --device npu --task forecast --head "$MODEL_WORK_DIR/head.safetensors" --head-sha256 "$HEAD_SHA256"
python3 model.py evaluate --input "$MODEL_WORK_DIR/test.npz" --results "$MODEL_WORK_DIR/npu.npz" --output "$MODEL_WORK_DIR/npu-metrics.json" --metric regression
```

其中 NPU 推理或训练须在硬件及依赖配套后执行。对照运行可显式使用 CPU；比较前按任务确定可接受的阈值。下例的 0.01 仅说明参数，不是验收门槛。

```bash
python3 model.py infer --model-dir "$MODEL_WORK_DIR/model" --input "$MODEL_WORK_DIR/test.npz" --output "$MODEL_WORK_DIR/cpu.npz" --device cpu --task forecast --head "$MODEL_WORK_DIR/head.safetensors" --head-sha256 "$HEAD_SHA256"
export MAX_ABSOLUTE_ERROR=0.01
python3 model.py compare --reference "$MODEL_WORK_DIR/cpu.npz" --candidate "$MODEL_WORK_DIR/npu.npz" --max-absolute "$MAX_ABSOLUTE_ERROR" --output "$MODEL_WORK_DIR/compare.json"
```

## 性能与精度

官方教程 large、ETTh1 horizon 192、训练 head 一轮后 MSE 0.422/MAE 0.432；当前 FP32 训练路径尚未证明与教程 AMP 配方及数据处理完全一致。 当前只提供本地协议，官方预训练仅重建头；预测头必须先训练并保存 `head.safetensors` 及同名 `.json` 元数据，入口检查 SHA、训练步数、预测长度和 backbone。该元数据只验证记录一致性，仍需检查训练日志与数据真实性。 固定输入 SHA、模型 SHA、任务及 head 后才允许比较。同步计时覆盖模型调用，排除输入准备和结果传回；单次 batch 中位数不是三次独立运行中位数。完整 NPU 精度、三次独立性能、端到端时间和显存待上机测量。

## NPU 独立性能报告

每次 `infer` 成功后额外生成 推理输出文件旁的 `.performance.json`，保存逐批同步耗时、总模型调用时间、吞吐、结果 SHA、命令和运行 ID。元信息含 Python、PyTorch、torch_npu、设备名称、进程内逻辑卡号及 `ASCEND_RT_VISIBLE_DEVICES` 原值；物理卡映射仍须结合 `npu-smi info` 核对。`model.py performance` 校验至少三次不同运行 ID、相同模型/数据/设备/参数，再汇总独立运行中位数；CPU 报告不能用于 NPU 性能结论。

```bash
set -e
for RUN in 1 2 3; do
  python3 model.py infer --model-dir "$MODEL_WORK_DIR/model" --input "$MODEL_WORK_DIR/test.npz" --output "$MODEL_WORK_DIR/npu-run-$RUN.npz" --device npu --task forecast --head "$MODEL_WORK_DIR/head.safetensors" --head-sha256 "$HEAD_SHA256"
done
python3 model.py performance --runs "$MODEL_WORK_DIR"/npu-run-*.npz.performance.json --output "$MODEL_WORK_DIR/performance-median.json"
```

统计包括末尾不足 batch 的批次；吞吐单位为输入样本/秒（TimesFM 为通道序列/秒）。这是纯模型调用口径，端到端时间、首次编译和峰值 HBM 仍需在目标 NPU 单独实测。报告校验不能认证人工编辑的文件或证明设备证据真实，发布前须核对原始日志。

本目录 `LICENSE` 保留固定上游参考实现的许可原文；正式自测、完整依赖安装与 NPU 验收仍待完成。

## 公网地址

| 内容 | 具体地址 |
|---|---|
| 上游源码 | https://github.com/moment-timeseries-foundation-model/moment/tree/38f7310ad594100747ca2a8357e9c7ca7d323e0e |
| 固定权重 | https://huggingface.co/AutonLab/MOMENT-1-large/tree/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc |
| 公开参考 | https://raw.githubusercontent.com/moment-timeseries-foundation-model/moment/38f7310ad594100747ca2a8357e9c7ca7d323e0e/tutorials/forecasting.ipynb |
| ETTh1 数据 | https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv |
| 下载或依赖地址 | https://github.com/moment-timeseries-foundation-model/moment.git |
| 下载或依赖地址 | https://huggingface.co/AutonLab/MOMENT-1-large/resolve/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc/config.json |
| 下载或依赖地址 | https://huggingface.co/AutonLab/MOMENT-1-large/resolve/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc/model.safetensors |
| 下载或依赖地址 | https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv |
| 参考许可证 | https://raw.githubusercontent.com/moment-timeseries-foundation-model/moment/38f7310ad594100747ca2a8357e9c7ca7d323e0e/LICENSE |
