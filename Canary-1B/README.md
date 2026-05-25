---
license: apache-2.0
hardware: NPU
---
# Canary-1B NPU 推理适配

本目录提供 NVIDIA NeMo Canary-1B 在昇腾 NPU 上的推理适配脚本和验证说明。当前适配基于 NeMo `main` 分支 commit `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`。

> 版本说明：当前适配对象是 Hugging Face `nvidia/canary-1b` 仓库中的原始 Canary-1B 权重文件 `canary-1b.nemo`，不是 `nvidia/canary-1b-flash`，也不是 `nvidia/canary-1b-v2`。本地已验证权重 SHA256 为 `b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a`，`.nemo` 内部 `model_config.yaml` 记录的 `nemo_version` 为 `1.23.0`。

## 1. 适配结论

- 上游仓库：<https://github.com/NVIDIA-NeMo/NeMo.git>
- 模型权重：<https://huggingface.co/nvidia/canary-1b>
- 本次不修改 NeMo 上游已有文件，因此没有 `.patch`。
- `infer.py` 是当前适配新增脚本，默认 `--device npu`，CPU 验证使用 `--device cpu`。
- 不写死 `npu:0` / `cuda:0`；实际卡号由 `ASCEND_RT_VISIBLE_DEVICES` 控制。

## 2. 环境准备

### CPU 验证环境

当前环境使用 `uv` 创建 Python 3.12 虚拟环境：

```bash
uv venv Canary-1B/.venv-cpu --python 3.12
uv pip install --python Canary-1B/.venv-cpu/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com \
  "torch==2.9.1" "torchaudio==2.9.1" "nemo-toolkit[asr]" \
  librosa soundfile sentencepiece huggingface_hub
```

> 说明：当前 `requirements.txt` 为历史完整环境导出，包含大量非推理最小依赖；CPU/NPU 部署建议按上面的最小依赖方式安装，并按 CANN 版本替换匹配的 `torch/torch-npu`。

### NPU 推理环境

请先安装与 CANN 匹配的 PyTorch 和 torch-npu，然后安装 NeMo ASR 依赖：

```bash
pip install torch torch-npu
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe"
pip install soundfile librosa sentencepiece huggingface_hub
```

## 3. 权重下载

推荐通过 `huggingface_hub` 使用 Gitee HF endpoint 下载到本地目录：

```bash
./Canary-1B/scripts/download_weights.sh Canary-1B/weights/canary-1b
```

下载脚本默认设置：

```bash
HF_HOME=~/.cache/gitee-ai
HF_ENDPOINT=https://hf-api.gitee.com
```

并等价执行：

```python
from huggingface_hub import snapshot_download
snapshot_download("nvidia/canary-1b", allow_patterns=["canary-1b.nemo"], local_dir="Canary-1B/weights/canary-1b")
```

下载完成后会校验 SHA256。当前环境已成功下载：

```text
Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo
SHA256: b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a
```

如果要改用直接 URL / 旧 `curl` 下载方式：

```bash
CANARY_DOWNLOAD_METHOD=curl CANARY_WEIGHT_URL=<mirror-url> ./Canary-1B/scripts/download_weights.sh Canary-1B/weights/canary-1b
```

本次在 ModelScope 以 `canary-1b` / `nvidia/canary-1b` 检索未找到同名公开模型。

## 4. 测试数据准备

生成一个 1 秒 16 kHz 单声道 wav，用于 smoke test：

```bash
./Canary-1B/scripts/download_test_data.sh Canary-1B/test_data
```

生成文件：

```text
Canary-1B/test_data/dummy_1s_16k.wav
```

该文件不是 ASR 准确率样本，只用于验证模型加载、音频读取、设备迁移和推理调用链路。

## 5. CPU 验证

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

当前机器已使用 HF 镜像权重完成 CPU 验证，输出示例：`[0]  I'm a part of that.` 详见 `NPU_VALIDATION.md`。

## 6. NPU 推理

```bash
cd Canary-1B
ASCEND_RT_VISIBLE_DEVICES=0 python infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --device npu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --pnc yes
```

## 7. 语音翻译 AST

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio /path/to/en.wav \
  --device npu \
  --task ast \
  --source_lang en \
  --target_lang de \
  --pnc yes
```

## 8. 交付文件

- `infer.py`：CPU/NPU 融合推理脚本。
- `scripts/download_weights.sh`：权重下载脚本。
- `scripts/download_test_data.sh`：测试 wav 生成脚本。
- `ANALYSIS.md`：上游版本、代码节点和风险分析。
- `NPU_ADAPTATION.md`：适配和运行说明。
- `NPU_VALIDATION.md`：验证命令与结果记录。
- `ACCEPTANCE_PLAN.md`：参考原始模型功能/性能/精度的完整验收方案，包含数据集选择、分层验收、通过条件和报告模板。
- `EVAL_FLEURS_LIBRISPEECH.md`：LibriSpeech test-clean ASR 与 FLEURS AST 子集/全量验证方案和运行命令。
- `scripts/eval_canary.py`：下载/准备 LibriSpeech、FLEURS 子集，运行 ASR/AST 推理并输出 WER/BLEU 指标的统一评测脚本。
- `patches/README.md`：说明本次无上游源码 patch。
