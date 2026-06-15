# MOSS-Speech NPU 适配说明

## 1. 适配目标

将 MOSS-Speech 单请求推理整理为规范的 CPU/NPU 融合入口：

- 默认 `--device npu`；
- CPU 验证显式 `--device cpu`；
- 不使用 `auto` / `use_gpu`；
- 不在代码中写死 `npu:0` / `cuda:0`；
- NPU 卡号通过 `ASCEND_RT_VISIBLE_DEVICES=0` 等环境变量控制；
- 不通过 `torch_npu.contrib.transfer_to_npu` 隐式迁移 CUDA 写法。

## 2. 版本边界

- 主模型：ModelScope `openmoss/MOSS-Speech`，Git HEAD `270d64296cafb94ca1f35b14b8d7918a1c4a2dc0`。
- Codec：ModelScope `AI-ModelScope/MOSS-Speech-Codec`，Git HEAD `a5423645a66476da761bbbdbc2003ae34e3c31c4`。
- Space 代码：HF Space `OpenMOSS-Team/MOSS-Speech`，Git HEAD `92a89018a8aa6b36f08c366c2659c76ffdc3f980`。
- 当前适配不覆盖 `MOSS-TTSD-v0.5`、CosyVoice 训练、TensorRT、ONNX 导出或其他模型变体。

## 3. 环境准备

### 3.1 NPU 环境

NPU 环境需安装与 CANN 匹配的 PyTorch 和 `torch-npu`。版本需以实际昇腾环境发布矩阵为准，README 原始约束为：

- 昇腾 NPU 驱动/固件：`>=25.0.RC1.1`；
- CANN Toolkit / Kernel / NNAL：`>=8.2.RC1`；
- 硬件参考：Atlas 800I A2 910B 单卡。

推荐安装顺序：

```bash
python -m venv MOSS-Speech/.venv-npu
source MOSS-Speech/.venv-npu/bin/activate
pip install torch torch-npu torchaudio
pip install transformers accelerate modelscope soundfile librosa gradio spaces diffusers
```

如果使用上游 Space 代码：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech \
  MOSS-Speech/upstream
pip install -r MOSS-Speech/upstream/requirements.txt
```

`requirements.txt` 可能包含 CUDA、训练或服务侧依赖；正式环境应先保护已匹配的 `torch` / `torch-npu`，避免被 pip 替换。

### 3.2 CPU 验证环境

CPU 仅用于加载链路和小规模功能验证，不代表正式性能：

```bash
python -m venv MOSS-Speech/.venv-cpu
source MOSS-Speech/.venv-cpu/bin/activate
pip install torch torchaudio transformers accelerate modelscope soundfile librosa gradio spaces diffusers
```

## 4. 权重与代码下载

主模型和 codec 可用 ModelScope 下载到本地，也可直接使用远端 id。正式验收推荐本地目录并记录 SHA256：

```bash
python - <<'PY'
from modelscope import snapshot_download
snapshot_download('openmoss/MOSS-Speech', local_dir='MOSS-Speech/weights/MOSS-Speech')
snapshot_download('AI-ModelScope/MOSS-Speech-Codec', local_dir='MOSS-Speech/weights/MOSS-Speech-Codec')
PY
find MOSS-Speech/weights -type f -maxdepth 3 -print0 | xargs -0 sha256sum > MOSS-Speech/weights/SHA256SUMS.txt
```

Space 代码：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech \
  MOSS-Speech/upstream
```

## 5. 推理脚本

### 5.1 NPU 文本输入生成音频

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-Speech/infer.py \
  --model MOSS-Speech/weights/MOSS-Speech \
  --codec MOSS-Speech/weights/MOSS-Speech-Codec \
  --space_dir MOSS-Speech/upstream \
  --prompt_audio MOSS-Speech/upstream/assets/prompt_cn.wav \
  --prompt "请用一句话介绍武汉的樱花。" \
  --output_modality audio \
  --output_dir MOSS-Speech/outputs \
  --device npu
```

### 5.2 CPU 文本输出 smoke test

```bash
python MOSS-Speech/infer.py \
  --model MOSS-Speech/weights/MOSS-Speech \
  --codec MOSS-Speech/weights/MOSS-Speech-Codec \
  --space_dir MOSS-Speech/upstream \
  --prompt "Hello!" \
  --output_modality text \
  --output_dir MOSS-Speech/outputs_cpu \
  --device cpu \
  --max_new_tokens 64
```

## 6. Patch 策略

当前未修改上游已有文件，因此 `MOSS-Speech/patches/` 中没有 `.patch`。旧 README 中“手工修改 site-packages”的做法不再作为默认流程，但这些修改项也不是直接判定为不需要：它们分别对应 `cached_download` 版本兼容、Whisper 特征提取设备、Matcha-TTS bf16 dtype、HiFiGAN `torch.istft` NPU 支持等潜在问题。默认流程先严格复现官方链路；只有在固定版本、记录原始错误并验证 patch 后，才将对应修改纳入 `patches/`。若后续确需改动：

1. 固定依赖源码版本；
2. 在可复现源码目录中修改；
3. 生成 patch；
4. 执行 `git apply --check`；
5. 在 `NPU_VALIDATION.md` 中记录触发错误、patch 内容和验证命令。

## 7. 失败原则

- 缺少 `transformers`、`modelscope`、官方 remote code、codec 或 prompt audio 时直接报错。
- NPU 后端不可用或算子不支持时直接报错，不自动切到 CPU。
- 不用第三方“相似”codec、简化 vocoder 或非官方 normalizer 替代官方路径。
