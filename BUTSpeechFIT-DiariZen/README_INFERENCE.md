# DiariZen 推理指导

## 概述

本目录适配 `BUT-FIT/diarizen-wavlm-large-s80-md` 说话人日志分轨 checkpoint，输出 RTTM。分割网络运行在 PyTorch NPU；WeSpeaker embedding 运行在 ONNX Runtime `CANNExecutionProvider`；Kaldi fbank 明确保留为 CPU 前处理。

```text
upstream=BUTSpeechFIT/DiariZen@a60b18151dbbe246e4199d8ef5cd2ece3872ea94
model=BUT-FIT/diarizen-wavlm-large-s80-md@a9b1b0e7974d96dcfd63af417e9da7ad8714040f
embedding=pyannote/wespeaker-voxceleb-resnet34-LM@837717ddb9ff5507820346191109dc79c958d614
dscore=nryant/dscore@e02f949ac6592279300a2c33d03daf9e0c12fd27
reference=Ascend-SACT/BUTSpeechFIT-DiariZen@7961b5ab79b1232b9da367f14f8cd4f592694465
```

不包含 `large-s80-md-v2`、base 或 pruning checkpoint。

## 输入输出数据

- 输入：一个或多个音频，或包含 `id`/`audio_path` 的 JSONL manifest。
- 输出：每个 session 一个 RTTM，以及 `run.meta.json`。
- `prepare_eval_data.py` 从 `wav.scp` 固定音频 manifest，并可校验 RTTM/UEM。
- `score_diarization.py` 使用固定 dscore 计算 DER/JER；默认 collar 0 且保留 overlap。

## 推理环境准备

| 配套 | 版本/要求 |
|---|---|
| 硬件 | 支持目标 CANN 的 Atlas 推理服务器 |
| CANN、驱动、固件 | CANN 8.2.0 及其配套驱动/固件 |
| Python | 3.10 |
| PyTorch / torchaudio / torch-npu | 2.5.1 |
| ONNX Runtime | CPU：`onnxruntime==1.22.1`；NPU：`onnxruntime-cann==1.22.1` |
| NumPy | 1.26.4 |

模型权重使用 CC BY-NC 4.0，正式部署前必须确认非商业使用及数据许可要求。

## 文件目录

```text
BUTSpeechFIT-DiariZen
├── infer.py
├── prepare_eval_data.py
├── score_diarization.py
├── patches/0001-add-explicit-npu-pipeline-device.patch
├── requirements.txt
├── README_INFERENCE.md
├── NPU_ADAPTATION.md
└── ACCEPTANCE_PLAN.md
```

执行时创建 `source/`、未应用 patch 的 `upstream-original/` 和应用 patch 的
`upstream-npu/` 三个目录。

## 快速上手

### 获取源码和安装依赖

```bash
git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git source
git -C source checkout a60b18151dbbe246e4199d8ef5cd2ece3872ea94
git -C source submodule update --init --recursive
git -C source worktree add --detach ../upstream-original \
  a60b18151dbbe246e4199d8ef5cd2ece3872ea94
git -C source worktree add --detach ../upstream-npu \
  a60b18151dbbe246e4199d8ef5cd2ece3872ea94
git -C upstream-original submodule update --init --recursive
git -C upstream-npu submodule update --init --recursive
git -C upstream-npu apply --check \
  ../patches/0001-add-explicit-npu-pipeline-device.patch
git -C upstream-npu apply \
  ../patches/0001-add-explicit-npu-pipeline-device.patch
```

原始和 patch 后 CPU baseline 使用独立环境。不要安装 upstream 根
`requirements.txt` 中的 `onnxruntime-gpu`：

```bash
python3.10 -m venv .venv-cpu-original
source .venv-cpu-original/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install onnxruntime==1.22.1
python -m pip install -r upstream-original/pyannote-audio/requirements.txt
python -m pip install -r upstream-original/dscore/requirements.txt
python -m pip install -r requirements.txt
python -m pip install -e upstream-original/pyannote-audio --no-deps
python -m pip install -e upstream-original --no-deps
deactivate

python3.10 -m venv .venv-cpu-patched
source .venv-cpu-patched/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install onnxruntime==1.22.1
python -m pip install -r upstream-npu/pyannote-audio/requirements.txt
python -m pip install -r upstream-npu/dscore/requirements.txt
python -m pip install -r requirements.txt
python -m pip install -e upstream-npu/pyannote-audio --no-deps
python -m pip install -e upstream-npu --no-deps
deactivate
```

NPU 环境不得安装 CPU 索引 wheel：

```bash
python3.10 -m venv .venv-npu
source .venv-npu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchaudio==2.5.1 torch-npu==2.5.1 \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
python -m pip install onnxruntime-cann==1.22.1
python -m pip install -r upstream-npu/pyannote-audio/requirements.txt
python -m pip install -r upstream-npu/dscore/requirements.txt
python -m pip install -r requirements.txt
python -m pip install -e upstream-npu/pyannote-audio --no-deps
python -m pip install -e upstream-npu --no-deps
```

不要修改 site-packages。执行 NPU 导入门禁：

```bash
python - <<'PY'
import onnxruntime as ort
import torch
import torch_npu
from diarizen.pipelines.inference import DiariZenPipeline
assert "CANNExecutionProvider" in ort.get_available_providers()
print(torch.__version__, ort.__version__, torch.randn(1).to("npu").device)
PY
```

CPU/CUDA baseline 使用上述独立环境和 CPU/CUDA ONNX Runtime，不能与
`onnxruntime-cann` 混装。

### 准备权重

```bash
huggingface-cli download BUT-FIT/diarizen-wavlm-large-s80-md \
  --revision a9b1b0e7974d96dcfd63af417e9da7ad8714040f \
  --local-dir weights/diarizen-wavlm-large-s80-md
huggingface-cli download pyannote/wespeaker-voxceleb-resnet34-LM \
  pytorch_model.bin \
  --revision 837717ddb9ff5507820346191109dc79c958d614 \
  --local-dir weights/wespeaker-voxceleb-resnet34-LM

find weights -type f -print0 | sort -z | xargs -0 sha256sum
```

主模型目录必须包含 `config.toml`、`pytorch_model.bin` 和 `plda/`。

### 准备数据集

功能验证使用上游样例：

```bash
mkdir -p eval_data/functional
printf 'EN2002a %s\n' "$PWD/source/example/EN2002a_30s.wav" \
  > eval_data/functional/wav.scp
python prepare_eval_data.py \
  --wav_scp eval_data/functional/wav.scp \
  --output_manifest eval_data/functional/manifest.jsonl \
  --dataset upstream-example \
  --split functional
```

L2 将正式公开数据的 `wav.scp`、reference RTTM 和可选 UEM 传入：

```bash
python prepare_eval_data.py \
  --wav_scp eval_data/ami/wav.scp \
  --reference_rttm eval_data/ami/reference.rttm \
  --uem eval_data/ami/all.uem \
  --output_manifest eval_data/ami/manifest.jsonl \
  --dataset AMI \
  --split SDM-eval
```

数据许可和 split 必须由使用者根据官方 recipe 固定；不能自行猜测后宣称复现官方表。

### 模型推理

未应用 patch 的原始 CPU baseline：

```bash
source .venv-cpu-original/bin/activate
mkdir -p results/original_cpu
python - <<'PY'
from pathlib import Path
from diarizen.pipelines.inference import DiariZenPipeline

pipeline = DiariZenPipeline(
    diarizen_hub=Path("weights/diarizen-wavlm-large-s80-md"),
    embedding_model="weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin",
    rttm_out_dir="results/original_cpu",
)
pipeline("source/example/EN2002a_30s.wav", sess_name="EN2002a")
PY
deactivate
```

应用 patch 后的同设备 CPU 回归：

```bash
source .venv-cpu-patched/bin/activate
python infer.py \
  --manifest eval_data/functional/manifest.jsonl \
  --model_dir weights/diarizen-wavlm-large-s80-md \
  --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
  --device cpu \
  --output_dir results/patched_cpu
deactivate
```

NPU：

```bash
source .venv-npu/bin/activate
python infer.py \
  --manifest eval_data/functional/manifest.jsonl \
  --model_dir weights/diarizen-wavlm-large-s80-md \
  --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
  --device npu \
  --output_dir results/npu
```

功能样例无 reference RTTM，只比较原始与 patch 后 RTTM 是否完全一致，并确认 NPU
成功生成 RTTM。正式 DER 对齐必须使用带 reference RTTM/UEM 的 L2 数据。

正式 DER（默认保留 overlap，不传 `--ignore_overlaps`）：

```bash
python score_diarization.py \
  --dscore_dir source/dscore \
  --reference_rttm eval_data/ami/reference.rttm \
  --system_rttm results/npu/*.rttm \
  --uem eval_data/ami/all.uem \
  --collar 0.0 \
  --output results/npu/der.txt
```

需要忽略 overlap 的独立非官方模式才显式增加 `--ignore_overlaps`。

## 模型推理性能

优先在 upstream 公布数据集的可取得全量 split 上记录总音频时长、RTF、
分割/embedding 阶段耗时、batch 和峰值 HBM/RSS。`infer.py` 会写
`run.meta.json`；三组命令分别用 `/usr/bin/time -v -o` 保存独立资源日志。官方
README 只发布 DER，未发布与当前 Atlas 路径可直接比较的硬件性能数值。

NPU L2 示例：

```bash
mkdir -p results/npu
/usr/bin/time -v -o results/npu/l2.time.txt python infer.py \
  --manifest eval_data/ami/manifest.jsonl \
  --model_dir weights/diarizen-wavlm-large-s80-md \
  --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
  --device npu \
  --output_dir results/npu
```

`results/npu/run.meta.json` 提供 elapsed/RTF/provider。原始 CUDA、patch 后 CUDA
使用同一 manifest 和独立输出目录/日志；正式轮次至少重复 3 次。

| 项目 | 当前状态 |
|---|---|
| dscore 工具 fixture | collar=0、保留 overlap，DER/JER 0.00 |
| 功能验证模型 RTTM | 待权重环境实测 |
| CUDA/NPU DER | 待正式 reference RTTM/UEM |
| Atlas RTF/HBM | 官方未发布，待验收 |

## 公网地址说明

- 官方源码：<https://github.com/BUTSpeechFIT/DiariZen>
- 主模型：<https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md>
- embedding：<https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM>
- 参考适配：<https://gitcode.com/Ascend-SACT/BUTSpeechFIT-DiariZen>

DER 口径见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，设备边界见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
