# Canary-1B NPU 验证记录

## 1. 静态验证

检查日期：2026-05-23

```bash
find Canary-1B -maxdepth 3 -type f | sort
git status --short
Canary-1B/.venv-cpu/bin/python -m py_compile Canary-1B/infer.py
```

结果：`py_compile` 通过。

## 2. 上游 clone 验证

```bash
git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/NVIDIA-NeMo/NeMo.git Canary-1B/upstream
git -C Canary-1B/upstream rev-parse HEAD
git -C Canary-1B/upstream ls-remote origin refs/heads/main
```

结果：

```text
44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe
44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe
```

本地 upstream HEAD 与远端 `main` 一致。

## 3. patch 验证

本次适配未修改 NeMo 上游已有文件，因此没有 `.patch` 文件需要 `git apply --check`。

如后续新增 patch，执行：

```bash
for p in Canary-1B/patches/*.patch; do
  git -C Canary-1B/upstream apply --check "../patches/$(basename "$p")"
done
```

## 4. CPU 环境准备验证

当前系统缺少 `python3-pip` / `ensurepip`，已使用 `uv` 创建 CPU 验证虚拟环境：

### 4.1 依赖文件关系说明

NeMo `requirements/` 目录下的文件不是互相全部包含的关系。特别注意：

- `requirements_asr.txt` **不包含** `requirements_lightning.txt`。
- 只执行 `pip install -r requirements_lightning.txt` 只能解决 `lightning`、`hydra-core`、`omegaconf` 等训练/框架依赖，不能解决 ASR 依赖，例如 `lhotse`。
- 只执行 `pip install -r requirements_asr.txt` 只能解决 ASR 领域依赖，例如 `lhotse`、`librosa`、`soundfile`、`jiwer`、`sacrebleu` 等，不能解决 `lightning`。
- `pip install -e "NeMo[asr]"` 或 `pip install "nemo-toolkit[asr]"` 才会通过 NeMo 的 `setup.py` 组合安装基础依赖、`requirements_common.txt`、`requirements_lightning.txt` 和 `requirements_asr.txt`。

NeMo 源码中 `setup.py` 的组合关系如下：

| 安装项/文件 | 作用 | 是否包含其他 requirements |
|---|---|---|
| `requirements.txt` | NeMo 基础依赖，如 `torch`、`numpy`、`huggingface_hub` 等 | 基础安装依赖 |
| `requirements_lightning.txt` | NeMo core/lightning 依赖，如 `lightning`、`hydra-core`、`omegaconf`、`torchmetrics`、`transformers` | 不包含 ASR |
| `requirements_common.txt` | 通用数据/文本依赖，如 `datasets`、`sentencepiece`、`pandas` | 不包含 lightning/ASR |
| `requirements_asr.txt` | ASR 依赖，如 `lhotse`、`librosa`、`soundfile`、`jiwer`、`sacrebleu` | 不包含 lightning/common/base |
| `requirements_audio.txt` | 通用音频处理/评估依赖，如 `lhotse`、`librosa`、`pesq`、`pystoi` | 不等同于 ASR 完整依赖 |
| `requirements_tts.txt` | TTS 依赖 | NeMo extra `tts` 会叠加 ASR/common |
| `requirements_slu.txt` | SLU 依赖 | NeMo extra `slu` 会叠加 ASR |
| `requirements_test.txt` | 测试/格式化依赖 | 仅测试开发使用 |
| `requirements_docs.txt` | 文档构建依赖 | 仅文档使用 |
| `requirements_cu12.txt` / `requirements_cu13.txt` | NVIDIA CUDA 附加依赖 | NPU 环境通常不使用 |
| `requirements_run.txt` | `nemo_run` 相关依赖 | 与 Canary 推理无直接关系 |
| `requirements_speechlm2.txt` | SpeechLM2 相关依赖 | 与 Canary-1B ASR smoke test 无直接关系 |

因此，Canary-1B ASR 推理建议使用以下二选一方式安装依赖。

**方式 A：从 NeMo 源码安装 ASR extra（推荐）**

```bash
cd /home/Canary-1B-Adapt/NeMo
python -m pip install -e ".[asr]"
```

**方式 B：手工按文件安装（适合不能 editable install 的环境）**

```bash
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements.txt
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements_common.txt
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements_lightning.txt
python -m pip install -r /home/Canary-1B-Adapt/NeMo/requirements/requirements_asr.txt
```

如果使用 NPU，还必须额外安装与当前 CANN/驱动匹配的 `torch` / `torch-npu`，不要让上述命令覆盖已验证可用的 NPU 版 PyTorch。

### 4.2 CPU 依赖安装记录

```bash
uv venv Canary-1B/.venv-cpu --python 3.12
uv pip install --python Canary-1B/.venv-cpu/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com \
  "torch==2.9.1" "torchaudio==2.9.1" "nemo-toolkit[asr]" \
  librosa soundfile sentencepiece huggingface_hub
```

依赖检查：

```bash
Canary-1B/.venv-cpu/bin/python - <<'PY'
import torch, torchaudio, nemo, soundfile, librosa, huggingface_hub
import lightning.pytorch, lhotse
print(torch.__version__, torchaudio.__version__)
PY
```

结果：

```text
torch 2.9.1+cu128
torchaudio 2.9.1+cu128
nemo / lightning / lhotse / soundfile / librosa / huggingface_hub 均可导入
```

## 5. 权重下载验证

权重来源：

- 官方：<https://huggingface.co/nvidia/canary-1b>
- 本次成功使用 HF 镜像：<https://hf-mirror.com/nvidia/canary-1b>

ModelScope 检索结果：使用 ModelScope API / `modelscope` SDK 以 `canary-1b`、`nvidia/canary-1b`、`canary` 检索，未找到同名公开模型。

下载命令：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  curl -k -L --fail --retry 10 --retry-delay 5 -C - \
  -o Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  https://hf-mirror.com/nvidia/canary-1b/resolve/main/canary-1b.nemo
```

结果：

```text
Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo
文件大小：3.8G
SHA256：b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a
```

说明：当前代理环境下 `hf-mirror.com` 经代理会出现 TLS EOF，取消代理环境变量后可正常下载。

## 6. 测试数据准备验证

执行：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import soundfile as sf
out = Path("Canary-1B/test_data/dummy_1s_16k.wav")
out.parent.mkdir(parents=True, exist_ok=True)
sr = 16000
t = np.arange(sr, dtype=np.float32) / sr
sf.write(out, 0.1 * np.sin(2 * np.pi * 440 * t), sr)
print(out)
PY
```

结果：

```text
Canary-1B/test_data/dummy_1s_16k.wav
```

说明：该文件为 1 秒 16 kHz 单声道正弦波，仅用于 smoke test，不用于识别准确率评估。

### 6.1 MLS / LibriSpeech / FLEURS 评测数据准备脚本验证

当前脚本已补齐在线/离线混合参数：

```bash
Canary-1B/.venv-cpu/bin/python -m py_compile Canary-1B/prepare_eval_data.py

# 离线缺失检查示例：应报出缺失的本地 MLS/LibriSpeech/FLEURS parquet 路径，不访问远端
python Canary-1B/prepare_eval_data.py \
  --task all \
  --data_dir /tmp/canary_eval_data \
  --asr_parquet_dir /tmp/canary_eval_data/mls_parquet \
  --asr_configs german \
  --librispeech_dir /tmp/canary_eval_data/librispeech_raw \
  --fleurs_parquet_dir /tmp/canary_eval_data/fleurs_parquet \
  --offline \
  --asr_limit 1 \
  --fleurs_limit 1 \
  --ast_directions en-de
```

要求：

- FLEURS 文件固定在 `<fleurs_parquet_dir>/<config>/<split>-00000-of-00001.parquet`，已有则复用。
- MLS 文件固定在 `<asr_parquet_dir>/{german,spanish,french}/test-00000-of-00001.parquet`，已有则复用。
- LibriSpeech 性能测试文件固定在 `<librispeech_dir>/test-clean.tar.gz` 或 `<librispeech_dir>/LibriSpeech/test-clean/`，已有则复用。
- metadata 记录 `asr_parquet_dir` / `librispeech_dir` / `fleurs_parquet_dir` 和 `offline`。
- MLS/LibriSpeech/FLEURS 禁用 HF `Audio` 自动解码，避免 `torchcodec` 依赖。

完整在线/离线/手动下载命令见 `EVAL_FLEURS_MLS.md`。

## 7. 当前环境 CPU 推理验证

执行命令：

```bash
/usr/bin/time -f 'elapsed=%E maxrss=%MKB' timeout 900 \
  Canary-1B/.venv-cpu/bin/python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --device cpu \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --task asr \
  --source_lang en \
  --target_lang en \
  --batch_size 1
```

结果：CPU 端到端推理成功。

```text
[0]  I'm a part of that.
elapsed=0:17.83 maxrss=9042820KB
exit_code=0
```

说明：

- 测试音频是 1 秒 16 kHz 正弦波，只用于 smoke test，不代表识别准确率；
- CPU 推理成功验证了本地 `.nemo` 权重加载、音频读取、manifest 构造、CPU device 路径和 `model.transcribe()` 调用链路；
- 最大 RSS 约 8.6 GiB，Canary-1B 在 CPU 上只适合功能验证，不适合作为性能路径。

## 8. NPU 功能验证命令

有 NPU 和本地权重后执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --device npu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --batch_size 1
```

预期：

- 模型加载到 NPU；
- 输出识别文本；
- 无 `Expected all tensors to be on the same device` 等设备不匹配错误。

## 9. AST 验证命令

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

预期：输出德语翻译文本。

## 10. 当前限制

- 当前环境没有 NPU，未执行 NPU 端到端验证。
- 已通过 HF 镜像下载权重并完成 CPU smoke test；若切换网络或镜像，需重新校验 SHA256。
- Canary-1B 约 1B 参数，CPU 推理即使权重下载完成也可能较慢；建议使用短音频进行 smoke test。

## 11. 完整验收方案

当前 `dummy_1s_16k.wav` 仅用于 smoke test，不能证明 Canary-1B 的功能完整性、性能或精度。完整验收请执行 `ACCEPTANCE_PLAN.md`，至少覆盖：

- ASR：英语、德语、西班牙语、法语；
- AST：英语 ↔ 德语/西班牙语/法语 6 个方向；
- PnC：`yes/no`；
- batch：`1/4/8` 或记录最大可用 batch；
- 精度：ASR WER、AST BLEU；
- 性能：RTF/RTFx、加载时间、峰值内存/HBM、连续运行稳定性。

正式验收报告建议保存到 `Canary-1B/validation_reports/`，模板见 `ACCEPTANCE_PLAN.md`。
