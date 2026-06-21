# DNSMOS 推理指导

## 概述

本目录提供 Microsoft DNSMOS P.835 本地 ONNX 模型在昇腾 NPU 上的推理和迁移对齐入口。模型输出 `SIG`、`BAK`、`OVRL` 和 `P808_MOS`；NPU 路径使用 ONNX Runtime `CANNExecutionProvider`，CPU 路径用于同权重数值基线。

版本边界：

```text
upstream=https://github.com/microsoft/DNS-Challenge.git
branch=master
commit=591184a9fcb2cbdec02520fed81a32bbbf9d73ff
reference=https://gitcode.com/Ascend-SACT/DNSMOS
reference_commit=d1e4c2c14df9cb935d61dc5f448e655772b12379
```

当前适配的是常规及 personalized DNSMOS P.835，不包含在线 DNSMOS API。

## 输入输出数据

- 输入：一个或多个 WAV、递归 WAV 目录，或包含 `audio_path` 字段的 JSONL manifest。
- 输出：逐文件 CSV、运行环境及模型校验值 sidecar `*.meta.json`。
- 数据准备：`prepare_eval_data.py` 验证音频并生成固定 manifest 和 metadata。
- 结果对齐：`compare_results.py` 比较 CPU/CUDA 与 NPU 的七个分数字段。

## 推理环境准备

| 配套 | 版本/要求 |
|---|---|
| 硬件 | 支持目标 CANN 的 Atlas 推理服务器 |
| CANN、驱动、固件 | CANN 8.2.0 及其配套驱动/固件 |
| Python | 3.10 |
| ONNX Runtime | CPU：`onnxruntime==1.22.1`；NPU：`onnxruntime-cann==1.22.1` |
| librosa / NumPy / soundfile | 见 `requirements.txt` |

ONNX Runtime 官方 CANN EP 配套表将 1.22.1 对应到 CANN 8.2.0。CPU 和 NPU
环境必须分开创建，避免 CPU `onnxruntime` 覆盖 CANN 构建。

## 文件目录

```text
DNSMOS
├── infer.py
├── prepare_eval_data.py
├── compare_results.py
├── requirements.txt
├── README_INFERENCE.md
├── NPU_ADAPTATION.md
└── ACCEPTANCE_PLAN.md
```

## 快速上手

### 获取源码和安装依赖

从当前模型目录执行。先克隆固定官方源码：

```bash
git clone https://github.com/microsoft/DNS-Challenge.git upstream
git -C upstream checkout 591184a9fcb2cbdec02520fed81a32bbbf9d73ff
```

CPU baseline 环境：

```bash
python3.10 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install onnxruntime==1.22.1
python - <<'PY'
import onnxruntime as ort
assert "CPUExecutionProvider" in ort.get_available_providers()
print(ort.__version__, ort.get_available_providers())
PY
deactivate
```

NPU 环境：

```bash
python3.10 -m venv .venv-npu
source .venv-npu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install onnxruntime-cann==1.22.1
python - <<'PY'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
assert "CANNExecutionProvider" in ort.get_available_providers()
PY
```

如果目标基础镜像使用内部构建的 CANN EP，应以对应 wheel 替换上述 PyPI 包，并在
验收报告中记录 wheel 文件名和 SHA256；不得同时安装 `onnxruntime`。

### 准备权重

```bash
mkdir -p weights
cp -r upstream/DNSMOS/DNSMOS weights/
cp -r upstream/DNSMOS/pDNSMOS weights/

sha256sum \
  weights/DNSMOS/model_v8.onnx \
  weights/DNSMOS/sig_bak_ovr.onnx \
  weights/pDNSMOS/sig_bak_ovr.onnx
```

固定 commit 的预期值：

```text
model_v8.onnx             9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91
DNSMOS/sig_bak_ovr.onnx   269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd
pDNSMOS/sig_bak_ovr.onnx  9e3a197449ca2177f0997afec3bd6b890117ce2f17b89d6eea7fa0d47272c81c
```

### 准备数据集

VCC2018 只能作为非官方迁移回归集，不能冒充论文隐藏测试集：

```bash
mkdir -p eval_data/vcc2018
wget -O eval_data/vcc2018.tar.gz \
  https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz
tar -xzf eval_data/vcc2018.tar.gz -C eval_data/vcc2018

python prepare_eval_data.py \
  --audio_dir eval_data/vcc2018 \
  --output_manifest eval_data/vcc2018.jsonl \
  --dataset VCC2018 \
  --split submitted-systems \
  --limit 100
```

生成：

```text
eval_data/vcc2018.jsonl
eval_data/vcc2018.jsonl.meta.json
```

### 模型推理

CPU 基线：

```bash
python infer.py \
  --manifest eval_data/vcc2018.jsonl \
  --model_root weights \
  --device cpu \
  --output_csv results/cpu.csv
```

NPU：

```bash
python infer.py \
  --manifest eval_data/vcc2018.jsonl \
  --model_root weights \
  --device npu \
  --output_csv results/npu.csv
```

personalized 路径在两条命令中同时增加 `--personalized`，并使用独立输出文件。

比较结果：

```bash
python compare_results.py \
  --baseline results/cpu.csv \
  --candidate results/npu.csv \
  --output results/cpu_vs_npu.json
```

## 模型推理性能

论文未发布可直接作为当前 NPU 通过线的硬件性能数值。正式报告至少记录样本数、总音频时长、首次和稳定运行耗时、RTF、NPU 型号及 CANN/ONNX Runtime 版本。精度主线和阈值见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

对 L2 同一 manifest 分别在两个环境执行，并保留独立资源日志：

```bash
mkdir -p results
/usr/bin/time -v -o results/cpu.time.txt python infer.py \
  --manifest eval_data/vcc2018.jsonl --model_root weights \
  --device cpu --output_csv results/cpu_perf.csv
/usr/bin/time -v -o results/npu.time.txt python infer.py \
  --manifest eval_data/vcc2018.jsonl --model_root weights \
  --device npu --output_csv results/npu_perf.csv
```

`*.csv.meta.json` 提供 elapsed/RTF/provider，`*.time.txt` 提供峰值 RSS；NPU 峰值 HBM
另由现场监控记录。常规和 personalized 均执行，正式轮次至少重复 3 次。

| 路径 | 数据 | 结果 |
|---|---|---|
| CPU 算法等价性 | 30 秒样例，常规/personalized | 与官方脚本全字段误差 0 |
| CPU 工具闭环 | 同一样例 manifest | RTF 0.076430，本次仅作链路记录 |
| NPU | 同 manifest | 待 CANN 环境验收 |

## 公网地址说明

- 官方源码：<https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS>
- 参考适配：<https://gitcode.com/Ascend-SACT/DNSMOS>
- DNSMOS P.835 论文：<https://arxiv.org/abs/2110.01763>

适配实现和已执行验证见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
