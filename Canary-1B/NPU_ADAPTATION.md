# Canary-1B NPU 适配说明

## 1. 适配目标

将 NVIDIA NeMo Canary-1B 推理样例整理为规范的 CPU/NPU 融合脚本：

- 默认使用 `--device npu`；
- CPU 验证显式使用 `--device cpu`；
- 不使用 `auto` / `use_gpu`；
- 不在代码中写死 `npu:0` / `cuda:0`；
- 实际 NPU 卡号由环境变量控制，例如 `ASCEND_RT_VISIBLE_DEVICES=0`。

## 2. 上游与 patch

- 上游仓库：<https://github.com/NVIDIA-NeMo/NeMo.git>
- 基准 commit：`44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`
- 本次没有修改上游已有文件，因此没有 `.patch` 文件。
- `Canary-1B/infer.py` 是当前适配新增脚本，不进入 patch。

如后续上游源码需要修改，则在 `Canary-1B/upstream/` 内生成 patch：

```bash
git -C Canary-1B/upstream diff -- <upstream_existing_file> > Canary-1B/patches/0001-xxx.patch
git -C Canary-1B/upstream apply --check ../patches/0001-xxx.patch
```

## 3. 环境准备

### 3.1 CPU 验证环境

当前环境没有系统 `pip`，使用 `uv` 创建虚拟环境并安装依赖：

```bash
uv venv Canary-1B/.venv-cpu --python 3.12
uv pip install --python Canary-1B/.venv-cpu/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com \
  "torch==2.9.1" "torchaudio==2.9.1" "nemo-toolkit[asr]" \
  librosa soundfile sentencepiece huggingface_hub
```

当前验证环境版本：

```text
Python 3.12.3
torch 2.9.1+cu128
torchaudio 2.9.1+cu128
nemo-toolkit 2.7.3
```

### 3.2 NPU 推理环境

NPU 环境中请安装与 CANN 匹配的 `torch` / `torch-npu`：

```bash
pip install torch torch-npu
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe"
pip install soundfile librosa sentencepiece huggingface_hub
```

## 4. 权重下载

官方权重：<https://huggingface.co/nvidia/canary-1b>

当前验证使用 HF 镜像：<https://hf-mirror.com/nvidia/canary-1b>

下载脚本：

```bash
./Canary-1B/scripts/download_weights.sh Canary-1B/weights/canary-1b-hfmirror
```

脚本默认通过 `curl` 从 HF 镜像下载 `canary-1b.nemo`，并校验 SHA256：

```text
b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a
```

推理时指定本地权重：

```bash
--model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo
```

本次在 ModelScope 以 `canary-1b` / `nvidia/canary-1b` 检索未找到同名公开模型。

## 5. 测试数据准备

生成最小 smoke-test wav：

```bash
./Canary-1B/scripts/download_test_data.sh Canary-1B/test_data
```

输出：

```text
Canary-1B/test_data/dummy_1s_16k.wav
```

该样例仅用于链路验证，不用于准确率评估。若要验证识别质量，请替换为真实英文/德文/西班牙文/法文语音。

## 6. 推理脚本用法

### CPU ASR 验证

```bash
Canary-1B/.venv-cpu/bin/python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --device cpu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --batch_size 1
```

### NPU ASR

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --device npu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --pnc yes
```

### 语音翻译 AST

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio /path/to/en.wav \
  --device npu \
  --task ast \
  --source_lang en \
  --target_lang de \
  --pnc yes
```

## 7. 上游更新处理

上游更新时必须重新执行：

```bash
git -C Canary-1B/upstream fetch origin main
git -C Canary-1B/upstream rev-parse origin/main
grep -RIn "cuda\|gpu\|npu\|to(device)\|torch.load\|nccl" Canary-1B/upstream/nemo/collections/asr
```

重点检查：

- `nemo/collections/asr/models/aed_multitask_models.py`
- `nemo/collections/asr/parts/mixins/transcription.py`
- `nemo/collections/asr/parts/preprocessing/`
- `examples/asr/transcribe_speech.py`

如新增硬编码 CUDA 节点，按标准流程生成 patch 并补充验证记录。
