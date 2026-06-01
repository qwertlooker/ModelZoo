---
license: apache-2.0
hardware: NPU
---

# MOSS-Speech NPU 适配

本目录提供语音对话大模型 MOSS-Speech 的 NPU 适配入口和验证文档。当前实现参考 Canary-1B 的交付结构，包含：

- `infer.py`：参数化单请求推理入口，默认 `--device npu`；
- `ANALYSIS.md`：上游与设备相关代码分析；
- `NPU_ADAPTATION.md`：环境、下载、运行和 patch 策略；
- `NPU_VALIDATION.md`：验证记录与待补项；
- `ACCEPTANCE_PLAN.md`：L0/L1/L2/L3 分层验收计划；
- `patches/README.md`：上游 patch 管理说明。

## 1. 版本边界

检查日期：2026-06-01。

- 主模型：ModelScope `openmoss/MOSS-Speech`，Git HEAD `270d64296cafb94ca1f35b14b8d7918a1c4a2dc0`。
- Codec：ModelScope `AI-ModelScope/MOSS-Speech-Codec`，Git HEAD `a5423645a66476da761bbbdbc2003ae34e3c31c4`。
- Space 代码：HF Space `OpenMOSS-Team/MOSS-Speech`，Git HEAD `92a89018a8aa6b36f08c366c2659c76ffdc3f980`。
- 当前适配不包含 `MOSS-TTSD-v0.5`、CosyVoice 训练、TensorRT 或 ONNX 导出。

## 2. 环境准备

NPU 环境需安装与 CANN 匹配的 `torch` / `torch-npu` / `torchaudio`。原始 README 的硬件约束为：驱动/固件 `>=25.0.RC1.1`，CANN Toolkit/Kernel/NNAL `>=8.2.RC1`，参考硬件 Atlas 800I A2 910B 单卡。

```bash
pip install torch torch-npu torchaudio
pip install transformers accelerate modelscope soundfile librosa gradio spaces diffusers
```

`requirements.txt` 是历史完整环境冻结，包含训练、服务和 CUDA 相关包；正式 NPU 环境请避免覆盖已匹配的 `torch` / `torch-npu`。

## 3. 下载代码与权重

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech \
  MOSS-Speech/upstream

python - <<'PY'
from modelscope import snapshot_download
snapshot_download('openmoss/MOSS-Speech', local_dir='MOSS-Speech/weights/MOSS-Speech')
snapshot_download('AI-ModelScope/MOSS-Speech-Codec', local_dir='MOSS-Speech/weights/MOSS-Speech-Codec')
PY
```

下载后记录校验值：

```bash
find MOSS-Speech/weights -type f -print0 | xargs -0 sha256sum > MOSS-Speech/weights/SHA256SUMS.txt
```

## 4. 运行

### NPU 生成音频

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

### CPU smoke test（显式指定）

```bash
python MOSS-Speech/infer.py \
  --model MOSS-Speech/weights/MOSS-Speech \
  --codec MOSS-Speech/weights/MOSS-Speech-Codec \
  --space_dir MOSS-Speech/upstream \
  --prompt "Hello!" \
  --output_modality text \
  --max_new_tokens 64 \
  --device cpu
```

## 5. 适配原则

- 必需依赖保留顶层 import；仅 `torch_npu` 在 `--device npu` 时条件导入。
- 不使用 `transfer_to_npu` 隐式替换 CUDA 写法。
- 不手工修改 site-packages；如确需改 `diffusers`、`transformers` 或 Space 上游源码，必须固定版本并生成 patch。
- 不在默认路径中加入 CPU fallback；NPU 算子不支持、权重缺失或 remote-code 不兼容时直接暴露原始错误。

## 6. 验收

详见 `ACCEPTANCE_PLAN.md`。最小 L0 要求：脚本可加载模型/codec，文本输出非空，音频输出可读且非全零；正式 L2/L3 需记录主观质量、ASR 回识别、端到端延迟、RTF、峰值 HBM/RSS。
