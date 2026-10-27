# TimesFM 2.5 推理与 ETTh1 评测

## 概述

本目录准备 `google/timesfm-2.5-200m-transformers` 的本地 PyTorch 推理与 ETTh1 评测入口。默认 `--device npu`，设备不可用或缺少 `torch_npu` 时直接失败；CPU 和 CUDA 仅通过显式参数选择。当前没有 NPU 硬件，**没有执行模型推理、NPU 精度或性能验收**。

固定 Transformers `v5.3.0`（源码提交 `aad13b87ed59f2afcfaebc985f403301887a35fc`），权重 revision `5a9806b9b291fad9233b5249d88263f1846304d3`；代码和权重采用 Apache-2.0。此目录不使用 TimesFM 3.0 权重，也不将原生 PyTorch 版权重直接用于 Transformers 入口。

参考实现 `commit_id=aad13b87ed59f2afcfaebc985f403301887a35fc`（固定源码或资料审计快照；实际运行依赖版本仍以上述约束为准）。

## 输入输出

- `prepare` 读取原始 ETTh1 CSV，固定测试区间、通道、历史长度、预测长度和文件 SHA，生成 JSON manifest。
- `infer` 输出 JSONL：第一行是模型、数据、设备等元信息，后续每行包含窗口和通道 ID、原始量纲真值及预测。支持单变量逐通道批量预测；预测长度为 1–128，历史长度最大 16384。
- `evaluate` 在原始数据量纲上计算全部窗口的 MSE 和 MAE；它不等于标准化 MSE，也不等于 BOOM 的 CRPS/MASE。
- `compare` 检查两个结果的输入、模型、数据及顺序相同，再按用户提供的绝对误差阈值判断通过或失败。

## 推理环境

需要目标 NPU 驱动、匹配的 CANN/PyTorch/torch_npu、Python、Transformers 5.3.0 及其依赖。硬件和软件版本尚未确定，兼容性待上机确认。安装 Transformers 时须保留目标镜像中的配套 PyTorch，不能先安装 CPU-only PyTorch 再追加 torch_npu。建议由目标镜像提供基础依赖，确认完整依赖已存在后再执行：

```bash
set -e
python3 -m pip install --no-deps 'transformers==5.3.0'
```

该命令不自动解决依赖；缺失时按选定镜像的版本配套解决，不能将命令视作已验证的完整环境。无需新建虚拟环境，除非目标机器已有不兼容的依赖组合。

## 文件目录

```text
timesfm_npu.py        # 数据准备、推理、评测和比较
performance.py       # 单次计时记录与三次独立 NPU 性能汇总
test_performance.py  # 性能报告的离线负向回归
test_timesfm_npu.py   # 无模型和硬件的工具回归测试
```

## 数据准备与快速上手

以下命令在本目录（`TimesFM-2.5`）执行。下载路径、数据和权重是环境产物，不纳入正式模型文件；下载后校验 SHA。空间需能容纳约 925 MB 权重以及数据和输出文件。

```bash
export MODEL_WORK_DIR="$PWD/timesfm_work"
mkdir -p "$MODEL_WORK_DIR/model" "$MODEL_WORK_DIR/data" "$MODEL_WORK_DIR/results"
curl -fL 'https://huggingface.co/google/timesfm-2.5-200m-transformers/resolve/5a9806b9b291fad9233b5249d88263f1846304d3/config.json' -o "$MODEL_WORK_DIR/model/config.json"
curl -fL 'https://huggingface.co/google/timesfm-2.5-200m-transformers/resolve/5a9806b9b291fad9233b5249d88263f1846304d3/model.safetensors' -o "$MODEL_WORK_DIR/model/model.safetensors"
curl -fL 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv' -o "$MODEL_WORK_DIR/data/ETTh1.csv"
printf '%s  %s\n' '452ecae918f67b2e7d0f2892ab424d1876939e70d077db3708a4fe8ca03a7de5' "$MODEL_WORK_DIR/model/config.json" | sha256sum -c -
printf '%s  %s\n' 'b53f6d52114e2ad786890f3c4637ce05f580b7800d6e24401f88b398b76035ef' "$MODEL_WORK_DIR/model/model.safetensors" | sha256sum -c -
printf '%s  %s\n' 'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066' "$MODEL_WORK_DIR/data/ETTh1.csv" | sha256sum -c -
python3 timesfm_npu.py prepare --dataset "$MODEL_WORK_DIR/data/ETTh1.csv" --output "$MODEL_WORK_DIR/manifest.json" --context 512 --horizon 96
```

默认测试区间为原 CSV 的行索引 `[11520,14400)`，每个窗口向前取 512 点、向后预测 96 点，步长 1，七个通道分开处理。测试窗口允许使用测试开始前的历史值，但预测真值全部在测试区间内。没有外部标准化：输入保持原始值，由模型内部处理。此协议是本项目的明确评测协议，**没有声称其结果等于 BOOM 排行榜**。

2026-10-27 已在本地读取并校验官方 ETTh1 文件，实际有 17,420 行；上述协议得到 2,785 个窗口、19,495 条通道序列。这个数据准备检查没有运行模型。

上机后执行：

```bash
python3 timesfm_npu.py infer --model-dir "$MODEL_WORK_DIR/model" --manifest "$MODEL_WORK_DIR/manifest.json" --output "$MODEL_WORK_DIR/results/npu.jsonl" --device npu --batch-size 8 --warmup 3
python3 timesfm_npu.py evaluate --manifest "$MODEL_WORK_DIR/manifest.json" --results "$MODEL_WORK_DIR/results/npu.jsonl" --output "$MODEL_WORK_DIR/results/npu-metrics.json"
python3 timesfm_npu.py infer --model-dir "$MODEL_WORK_DIR/model" --manifest "$MODEL_WORK_DIR/manifest.json" --output "$MODEL_WORK_DIR/results/cpu.jsonl" --device cpu --batch-size 8 --warmup 3
```

比较前按业务或验收要求设定阈值，不预设通过标准。下面的 `0.01` 只是命令演示，不是已批准的精度门槛：

```bash
export MAX_ABSOLUTE_ERROR=0.01
python3 timesfm_npu.py compare --reference "$MODEL_WORK_DIR/results/cpu.jsonl" --candidate "$MODEL_WORK_DIR/results/npu.jsonl" --max-absolute "$MAX_ABSOLUTE_ERROR" --output "$MODEL_WORK_DIR/results/compare.json"
```

## 性能与精度

每个 batch 的计时在模型调用前后同步设备；输入构造和结果传回主机不计入，首次执行前按参数预热。输出的 batch 中位数仅是**单次运行内部的统计**，不是三次独立运行中位数，最后一个不足 batch 的批次也计入。正式性能需至少独立运行三次、分别保留日志和完整配置，再计算中位数；同时单独测量端到端时间及峰值显存。目前均待 NPU 验证。

ETTh1 结果必须保留数据 SHA、切分、窗口、步长、通道、模型版本及设备。原始值 MSE/MAE 仅用于本协议的 CPU/NPU 对齐及业务参考。官方 BOOM 的 TimesFM 2.5 MASE 0.632、CRPS 0.382 使用原生权重、2048 上下文及特定预测选项；本入口未实现该完整评测，不以其数据宣称复现。

## NPU 独立性能报告

每次 `infer` 成功后额外生成 `<输出文件>.performance.json`，保存逐批同步耗时、总模型调用时间、吞吐、结果 SHA、命令和运行 ID。元信息含 Python、PyTorch、torch_npu、设备名称、进程内逻辑卡号及 `ASCEND_RT_VISIBLE_DEVICES` 原值；物理卡映射仍须结合 `npu-smi info` 核对。`performance.py` 校验至少三次不同运行 ID、相同模型/数据/设备/参数，再汇总独立运行中位数；CPU 报告不能用于 NPU 性能结论。

默认显式使用 SDPA；仅在目标栈确认不支持该路径时使用 `--attention eager` 并记录原因、重新验证精度和性能，不自动回退。加载时固定 FP32，严格检查 Transformers 5.3.0。

```bash
set -e
for RUN in 1 2 3; do
  python3 timesfm_npu.py infer --model-dir "$MODEL_WORK_DIR/model" --manifest "$MODEL_WORK_DIR/manifest.json" --output "$MODEL_WORK_DIR/results/npu-run-$RUN.jsonl" --device npu --batch-size 8 --warmup 3
done
python3 performance.py --runs "$MODEL_WORK_DIR"/results/npu-run-*.jsonl.performance.json --output "$MODEL_WORK_DIR/performance-median.json"
```

统计包括末尾不足 batch 的批次；吞吐单位为输入样本/秒（TimesFM 为通道序列/秒）。这是纯模型调用口径，端到端时间、首次编译和峰值 HBM 仍需在目标 NPU 单独实测。报告校验不能认证人工编辑的文件或证明设备证据真实，发布前须核对原始日志。

本目录 `LICENSE` 保留固定上游参考实现的许可原文；正式自测、完整依赖安装与 NPU 验收仍待完成。

## 公网地址

| 用途 | 地址 |
|---|---|
| 上游 TimesFM | https://github.com/google-research/timesfm/tree/e51928e27119cb17bebc005be2696b75e0a9e688 |
| Transformers 代码 | https://github.com/huggingface/transformers/tree/aad13b87ed59f2afcfaebc985f403301887a35fc |
| 固定模型文件 | https://huggingface.co/google/timesfm-2.5-200m-transformers/tree/5a9806b9b291fad9233b5249d88263f1846304d3 |
| ETTh1 数据 | https://github.com/zhouhaoyi/ETDataset/tree/1d16c8f4f943005d613b5bc962e9eeb06058cf07 |
| BOOM 原始榜单 | https://huggingface.co/spaces/Datadog/BOOM/blob/2bc23d1a3eeebb7ac73a610afcb5db9470c72552/results/leaderboards/BOOM_leaderboard.csv |
| 下载或依赖地址 | https://huggingface.co/google/timesfm-2.5-200m-transformers/resolve/5a9806b9b291fad9233b5249d88263f1846304d3/config.json |
| 下载或依赖地址 | https://huggingface.co/google/timesfm-2.5-200m-transformers/resolve/5a9806b9b291fad9233b5249d88263f1846304d3/model.safetensors |
| 下载或依赖地址 | https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv |
| 参考许可证 | https://raw.githubusercontent.com/huggingface/transformers/aad13b87ed59f2afcfaebc985f403301887a35fc/LICENSE |
