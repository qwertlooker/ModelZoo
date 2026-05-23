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
print(torch.__version__, torchaudio.__version__)
PY
```

结果：

```text
torch 2.9.1+cu128
torchaudio 2.9.1+cu128
nemo / soundfile / librosa / huggingface_hub 均可导入
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
./Canary-1B/scripts/download_test_data.sh Canary-1B/test_data
```

结果：

```text
Canary-1B/test_data/dummy_1s_16k.wav
```

说明：该文件为 1 秒 16 kHz 单声道正弦波，仅用于 smoke test，不用于识别准确率评估。

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
