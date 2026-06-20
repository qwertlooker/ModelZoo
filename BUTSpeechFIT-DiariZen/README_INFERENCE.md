# DiariZen 推理指导

## 概述

本目录适配 `BUT-FIT/diarizen-wavlm-large-s80-md` 说话人日志分轨 checkpoint，输出 RTTM。

```text
upstream=https://github.com/BUTSpeechFIT/DiariZen.git
branch=main
commit=a60b18151dbbe246e4199d8ef5cd2ece3872ea94
model=https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md
model_commit=a9b1b0e7974d96dcfd63af417e9da7ad8714040f
embedding=https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
embedding_commit=837717ddb9ff5507820346191109dc79c958d614
reference=https://gitcode.com/Ascend-SACT/BUTSpeechFIT-DiariZen
reference_commit=7961b5ab79b1232b9da367f14f8cd4f592694465
```

不包含 `diarizen-wavlm-large-s80-md-v2` 或 base checkpoint。

## 环境与安装

参考 NPU 环境为 Python 3.10、PyTorch/torch-npu 2.5.1；CANN/驱动必须与 torch-npu 配套。另需带 `CANNExecutionProvider` 的 ONNX Runtime。

```bash
git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git upstream
git -C upstream checkout a60b18151dbbe246e4199d8ef5cd2ece3872ea94
git -C upstream apply ../patches/0001-add-explicit-npu-pipeline-device.patch

pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu==2.5.1 \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
pip install -r upstream/pyannote-audio/requirements.txt
pip install -r requirements.txt
pip install -e upstream/pyannote-audio --no-deps
pip install -e upstream --no-deps
```

另行安装与 CANN 匹配、包含 `CANNExecutionProvider` 的 ONNX Runtime。
不要安装 upstream 根目录 `requirements.txt` 中的 `onnxruntime-gpu`，也不要手工
修改已安装的 `torchaudio/compliance/kaldi.py`。当前依赖固定
`numpy==1.26.4`。

## 权重

```bash
huggingface-cli download BUT-FIT/diarizen-wavlm-large-s80-md \
  --local-dir weights/diarizen-wavlm-large-s80-md
huggingface-cli download pyannote/wespeaker-voxceleb-resnet34-LM \
  pytorch_model.bin \
  --local-dir weights/wespeaker-voxceleb-resnet34-LM
```

记录目录内全部配置、模型和 PLDA 文件 SHA256。

## 推理

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 infer.py \
  --audio upstream/example/EN2002a_30s.wav \
  --model_dir weights/diarizen-wavlm-large-s80-md \
  --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
  --device npu \
  --output_dir results_npu
```

CPU/CUDA 对齐只改变 `--device` 和输出目录。RTTM 精度验收见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，设备边界见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
