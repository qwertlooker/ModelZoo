# MOSS-TTSD-v0.5 推理指导

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [文件目录](#文件目录)
- [快速上手](#快速上手)
  - [获取源码并应用 patch](#获取源码并应用-patch)
  - [准备权重](#准备权重)
  - [准备测试数据](#准备测试数据)
  - [准备 TTSD-eval 评测环境与权重](#准备-ttsd-eval-评测环境与权重)
  - [模型推理](#模型推理)
- [OpenMOSS/TTSD-eval 测评](#openmossttsd-eval-测评)
- [模型推理性能](#模型推理性能)
- [NPU GQA FlashAttention 与显存说明](#npu-gqa-flashattention-与显存说明)
- [已知问题：Transformers 5.x 不兼容](#已知问题transformers-5x-不兼容)
- [公网地址说明](#公网地址说明)

## 概述

MOSS-TTSD-v0.5 是 OpenMOSS 发布的对话式双说话人文本转语音/文本转语音对话（TTS/TTSD）模型。模型以包含 `[S1]`、`[S2]` 标签的对话文本作为输入，并结合两个说话人的 prompt audio / prompt text 生成自然对话音频。本文档介绍 `OpenMOSS/MOSS-TTSD` tag `v0.5` 基于昇腾 NPU 的推理指导。

> 说明：本文档适配对象为 `OpenMOSS/MOSS-TTSD` tag `v0.5` 原项目代码、`fnlp/MOSS-TTSD-v0.5` 权重和 `fnlp/XY_Tokenizer_TTSD_V0` codec checkpoint。不包含 MOSS-TTSD v0.7、v1.0、SGLang 路径或未固定版本的一键包改动。

- 版本说明：

  ```text
  url=https://github.com/OpenMOSS/MOSS-TTSD.git
  tag=v0.5
  commit_id=0e078c62389922d3aa873ce182daf31142860b18
  model_name=MOSS-TTSD-v0.5
  patch=patches/0001-adapt-v0.5-inference-to-npu.patch
  ```

- 适配原则：
  - 不修改原始 `README.md`。
  - 不新增旁路推理脚本；继续使用原项目已有 `inference.py`，通过 patch 适配 NPU。
  - NPU 默认显式使用 `--device npu`，实际卡号由 `ASCEND_RT_VISIBLE_DEVICES` 控制。
  - NPU 路径内部固定使用 torch-npu Flash Attention：prefill 调用 `npu_prompt_flash_attention`，decode 调用 `npu_incre_flash_attention`，直接传递 GQA 的 KV head 数，不执行 `repeat_kv`。
  - 推理入口只新增一个 `--device` 参数，不向用户暴露 dtype、attention backend、batch size 或权重路径等额外开关。

## 输入输出数据

- 输入数据

  推理入口读取 JSONL 文件。每行至少包含以下字段：

  ```json
  {
    "base_path": "examples",
    "text": "[S1]你好。[S2]你好，我们开始测试。",
    "prompt_audio_speaker1": "zh_spk1_moon.wav",
    "prompt_text_speaker1": "周一到周五，每天早晨七点半到九点半的直播片段。",
    "prompt_audio_speaker2": "zh_spk2_moon.wav",
    "prompt_text_speaker2": "如果大家想听到更丰富更及时的直播内容。"
  }
  ```

  `prompt_audio_speaker1`、`prompt_audio_speaker2` 可为相对 `base_path` 的路径，也可按原项目逻辑传入可解析的本地路径。prompt 音频建议为清晰人声 WAV，正式验收需固定 prompt 来源、采样率和文本。

- 输出数据

  输出为 `output_*.wav`，保存到 `--output_dir` 指定目录。输出采样率以模型/codec 返回值为准，验收报告必须记录输出 WAV 数量、采样率、总时长、峰值/RMS 和是否可播放。

## 推理环境准备

- 该模型需要以下插件与驱动。实际版本以目标 CANN 与 torch-npu 官方匹配表为准。

  **表 1** 版本配套表

| 配套 | 版本 |
|---|---|
| 固件与驱动 | 25.5.1+ |
| CANN Toolkit / Kernel / NNAL | 8.5.1 |
| Python | 3.11 |
| PyTorch / torch-npu / torchaudio | 2.9.0 |
| transformers | 当前上游 v0.5 代码固定使用 `4.57.6`；不要安装 5.x |
| accelerate | 按原项目依赖安装 |
| soundfile / torchaudio | `soundfile` 用于文件读写；`torchaudio.functional.resample` 用于重采样 |

说明：

- `flash-attn` 官方包面向 CUDA/ROCm GPU kernel，当前不作为 Ascend NPU 必需依赖安装。NPU 内部固定使用 torch-npu 原生 PFA/IFA；CUDA 路径保持原项目 `flash_attention_2`，CPU 路径使用 SDPA。
- TorchAudio 2.9+ 的 `torchaudio.load` / `torchaudio.save` 会进入 TorchCodec 路径。本适配通过 patch 将 prompt 音频读取和 WAV 写出改为 `soundfile`，不要求额外安装 `torchcodec`。如果仍看到 `TorchCodec is required for load_with_torchcodec` 或 `save_with_torchcodec`，说明 patch 未应用或路径未覆盖。
- Transformers 5.x 改变了 `_tied_weights_keys` 的数据结构和 `tie_weights()` 接口，而 MOSS-TTSD-v0.5 上游代码仍使用 Transformers 4.x 接口。当前项目不修改该模型定义，因此必须固定 `transformers==4.57.6`。

## 文件目录

```text
MOSS-TTSD-v0.5
├── README_INFERENCE.md                         # 推理指导文档
├── README.md                                   # 模型适配说明
├── NPU_ADAPTATION.md                           # NPU 适配文档与验证记录
├── ACCEPTANCE_PLAN.md                          # 完整验收方案
├── V1_0_DIFF_REFERENCE.md                      # v1.0 差异参考
├── prepare_eval_data.py                        # evaluator manifest 工具
├── patches
│   ├── README.md                               # patch 使用说明
│   └── 0001-adapt-v0.5-inference-to-npu.patch  # v0.5 NPU 适配 patch
├── source                                      # 固定 tag 的 Git 管理目录
├── upstream-original                           # 未应用 patch 的 CUDA baseline
├── upstream-npu                                # 应用 patch 的 CUDA 回归/NPU 路径
│   ├── inference.py                            # patch 后的推理入口
│   ├── generation_utils.py                     # 生成和音频处理逻辑
│   ├── modeling_asteroid.py                    # Asteroid 生成模型
│   ├── examples
│   │   ├── examples.jsonl                      # 官方中文/英文示例
│   │   ├── zh_spk1_moon.wav
│   │   ├── zh_spk2_moon.wav
│   │   ├── m1.wav
│   │   └── m2.wav
│   ├── weights
│   │   └── MOSS-TTSD-v0.5                      # 下载后的模型权重
│   └── XY_Tokenizer
│       ├── config/xy_tokenizer_config.yaml     # codec 配置
│       └── weights/xy_tokenizer.ckpt           # 下载后的 codec checkpoint
└── validation_reports                          # 验收报告目录，按需生成
```

## 快速上手

除获取适配仓库的初始步骤外，后续命令默认从 `ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5` 目录开始，示例只使用相对路径。

### 获取源码并应用 patch

1. 获取适配仓库。

   ```bash
   git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch
   git checkout master
   cd ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5
   ```

2. 准备原项目 tag `v0.5` 的独立原始和 patch 工作树。

   ```bash
   git clone https://github.com/OpenMOSS/MOSS-TTSD.git source
   git -C source checkout 0e078c62389922d3aa873ce182daf31142860b18
   git -C source worktree add --detach ../upstream-original \
     0e078c62389922d3aa873ce182daf31142860b18
   git -C source worktree add --detach ../upstream-npu \
     0e078c62389922d3aa873ce182daf31142860b18
   git -C upstream-npu apply --check \
     ../patches/0001-adapt-v0.5-inference-to-npu.patch
   git -C upstream-npu apply \
     ../patches/0001-adapt-v0.5-inference-to-npu.patch
   ```

3. 安装 NPU 环境。不得从 PyTorch CPU wheel 索引安装框架：

   ```bash
   python3.11 -m venv .venv-npu
   source .venv-npu/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.9.0 torch-npu==2.9.0 torchaudio==2.9.0 \
     -i https://mirrors.huaweicloud.com/repository/pypi/simple
   python -m pip install "transformers==4.57.6"
   python -m pip install -r upstream-npu/requirements.txt
   python -m pip install -r upstream-npu/XY_Tokenizer/requirements.txt
   python - <<'PY'
   import torch
   import torch_npu
   print(torch.__version__, torch.randn(1).to("npu").device)
   PY
   deactivate
   ```

4. 原始 CUDA 和 patch 后 CUDA 使用两个独立环境，均安装相同的 PyTorch、
   Transformers 和 CUDA `flash-attn`。以下 CUDA wheel/索引需按实际 CUDA 版本
   选择，不能用于 NPU 环境：

   ```bash
   python3.11 -m venv .venv-cuda-original
   source .venv-cuda-original/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.9.0 torchaudio==2.9.0
   python -m pip install "transformers==4.57.6" flash-attn
   python -m pip install -r upstream-original/requirements.txt
   python -m pip install -r upstream-original/XY_Tokenizer/requirements.txt
   deactivate

   python3.11 -m venv .venv-cuda-patched
   source .venv-cuda-patched/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.9.0 torchaudio==2.9.0
   python -m pip install "transformers==4.57.6" flash-attn
   python -m pip install -r upstream-npu/requirements.txt
   python -m pip install -r upstream-npu/XY_Tokenizer/requirements.txt
   deactivate
   ```

### 准备权重

1. 下载 MOSS-TTSD-v0.5 权重和 XY Tokenizer checkpoint。

   原始权重地址：

   - MOSS-TTSD-v0.5：`https://huggingface.co/fnlp/MOSS-TTSD-v0.5`
   - XY Tokenizer：`https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0`

   ```bash
   export HF_HOME="$PWD/hf-cache"
   python -m pip install -U "huggingface_hub[cli]"
   mkdir -p assets upstream-npu/weights upstream-npu/XY_Tokenizer/weights \
     upstream-original/XY_Tokenizer/weights

   MODEL_SNAPSHOT=$(python - <<'PY'
   from huggingface_hub import snapshot_download
   print(snapshot_download(
       "fnlp/MOSS-TTSD-v0.5",
       revision="8527b9136b6afefe2252ae597cecea2e80e7ebeb",
   ))
   PY
   )

   hf download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt \
     --revision c83433728e698ed0698e88cb5096bc221fb8f8c5 \
     --local-dir assets/XY_Tokenizer_TTSD_V0

   ln -sfn "$MODEL_SNAPSHOT" upstream-npu/weights/MOSS-TTSD-v0.5
   ln -sfn "$PWD/assets/XY_Tokenizer_TTSD_V0/xy_tokenizer.ckpt" \
     upstream-npu/XY_Tokenizer/weights/xy_tokenizer.ckpt
   ln -sfn "$PWD/assets/XY_Tokenizer_TTSD_V0/xy_tokenizer.ckpt" \
     upstream-original/XY_Tokenizer/weights/xy_tokenizer.ckpt
   ```

   原始代码继续使用 repo id `fnlp/MOSS-TTSD-v0.5`，执行时设置同一个
   `HF_HOME` 和 `HF_HUB_OFFLINE=1`，从上述固定 revision cache 加载；patch 后代码
   通过符号链接读取同一 snapshot。

2. 下载后记录 SHA256。

   ```bash
   cd upstream-npu
   find -L weights/MOSS-TTSD-v0.5 -maxdepth 1 -type f -print0 \
     | sort -z | xargs -0 sha256sum
   sha256sum XY_Tokenizer/weights/xy_tokenizer.ckpt
   ```

### 准备测试数据

1. 使用原项目官方示例作为 smoke test 数据。

   默认示例路径：

   ```text
   upstream-original/examples/examples.jsonl
   ```

   该文件包含中文和英文双说话人长对话示例，并引用 `examples/` 目录下的 prompt wav。

2. 下载固定 TTSD-eval 全量 testset：

   ```bash
   git clone https://github.com/OpenMOSS/TTSD-eval.git third_party/TTSD-eval
   git -C third_party/TTSD-eval checkout \
     dea13b98529dc16dcfb5fe45779ad63ac9238337
   curl -L --fail \
     -o third_party/TTSD-eval/testset.zip \
     https://media.githubusercontent.com/media/OpenMOSS/TTSD-eval/dea13b98529dc16dcfb5fe45779ad63ac9238337/testset.zip
   echo "49ed8338f3e5323c5ffcff01f3480a9c245937256d9197d792c973cba5603e17  third_party/TTSD-eval/testset.zip" \
     | sha256sum -c -
   unzip -oq third_party/TTSD-eval/testset.zip -d third_party/TTSD-eval

   wc -l \
     third_party/TTSD-eval/testset/ttsd_eval_zh.jsonl \
     third_party/TTSD-eval/testset/ttsd_eval_en.jsonl
   ```

   L2 使用上述中文和英文全量各 50 条。

3. 功能验证使用官方 `examples/examples.jsonl` 2 条；L2 使用
   `OpenMOSS/TTSD-eval` 全量。该评测可用于 v0.5 输出，但不代表 v0.5 已发布官方
   指标；正式验收需同时比较 ACC/SIM/WER 和 RTF/RTFx。

### 准备 TTSD-eval 评测环境与权重

TTSD-eval 不是无权重评测器。ACC/SIM 依赖 WeSpeaker
`voxblink2_samresnet100_ft` 和 MMS-FA，WER 依赖
`openai/whisper-large-v3`。三者必须在正式评测前下载，不能用名称相近的模型替代。
评测器只读取三组已生成的 WAV，可使用独立 CUDA/CPU 环境执行，不要安装到 NPU
推理环境中。

1. 创建独立评测环境。TTSD-eval 固定 commit 的依赖要求
   `torch/torchaudio<=2.8.0`，不能复用本指导中的 PyTorch 2.9 推理环境。CUDA wheel
   索引按现场 CUDA 版本选择并记录：

   ```bash
   python3.12 -m venv .venv-ttsd-eval
   source .venv-ttsd-eval/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.8.0 torchaudio==2.8.0
   python -m pip install -r third_party/TTSD-eval/requirements.txt
   python -m pip install --force-reinstall --no-deps \
     "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git@c92349a14d6b426808c4e09b8b12e076864dfc11"
   python -m pip install "transformers==4.57.6" "huggingface_hub[cli]"
   python -m pip freeze > third_party/TTSD-eval/evaluator-pip-freeze.txt
   deactivate
   ```

   上述 WeSpeaker commit 是 TTSD-eval 固定 commit 发布前的上游版本，用于避免其
   `requirements.txt` 中未固定的 Git HEAD 漂移。

2. 下载 WeSpeaker 权重。WeNet 官网链接当前由 ModelScope 官方数据集镜像提供
   实际对象，以下命令从 API 获取短期签名 URL，不把会过期的 URL 写入文档：

   ```bash
   EVAL_ROOT="$PWD/third_party/TTSD-eval"
   export EVAL_ROOT
   mkdir -p "$EVAL_ROOT/model/downloads"

   python3 - <<'PY'
   import json
   import os
   import pathlib
   import urllib.request

   api = (
       "https://modelscope.cn/api/v1/datasets/"
       "wenet/wespeaker_pretrained_models/oss/tree"
   )
   with urllib.request.urlopen(api) as response:
       entries = json.load(response)["Data"]
   entry = next(
       item for item in entries
       if item["Key"] == "voxblink2_samresnet100_ft.zip"
   )
   if entry["Size"] != 186890839:
       raise RuntimeError(f"unexpected WeSpeaker archive size: {entry['Size']}")
   output = (
       pathlib.Path(os.environ["EVAL_ROOT"])
       / "model/downloads/voxblink2_samresnet100_ft.zip"
   )
   urllib.request.urlretrieve(entry["Url"], output)
   print(output)
   PY

   echo "ad0873d380acaa7f4256ff37d40217ee31e4955b26a45064a13a14998cc89d16  $EVAL_ROOT/model/downloads/voxblink2_samresnet100_ft.zip" \
     | sha256sum -c -
   unzip -oq "$EVAL_ROOT/model/downloads/voxblink2_samresnet100_ft.zip" \
     -d "$EVAL_ROOT/model"
   echo "5aeee438ca23c0ca6e341bab6c6bf7f465497e1dc323bb1bc1074d6a0c778b11  $EVAL_ROOT/model/voxblink2_samresnet100_ft/avg_model.pt" \
     | sha256sum -c -
   ```

3. 下载 MMS-FA checkpoint。`tools/align.py` 将 torch hub 目录设置为
   `model/`，因此文件必须位于 `model/checkpoints/model.pt`：

   ```bash
   EVAL_ROOT="$PWD/third_party/TTSD-eval"
   mkdir -p "$EVAL_ROOT/model/checkpoints"
   curl -L --fail --retry 3 \
     -o "$EVAL_ROOT/model/checkpoints/model.pt" \
     "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt?versionId=dZWoHyjLHoCxDn.KL1FPSlVCD3CPRtOL"
   test "$(stat -c %s "$EVAL_ROOT/model/checkpoints/model.pt")" = "1262047414"
   sha256sum "$EVAL_ROOT/model/checkpoints/model.pt" \
     | tee "$EVAL_ROOT/model/checkpoints/model.pt.sha256"
   ```

4. 下载固定 revision 的 Whisper-large-v3。正式评测通过本地路径加载，禁止运行时
   从浮动的 `main` 下载：

   ```bash
   source .venv-ttsd-eval/bin/activate
   EVAL_ROOT="$PWD/third_party/TTSD-eval"
   export EVAL_ROOT
   python - <<'PY'
   import os
   from huggingface_hub import snapshot_download

   snapshot_download(
       repo_id="openai/whisper-large-v3",
       revision="06f233fe06e710322aca913c1bc4249a0d71fce1",
       local_dir=os.path.join(os.environ["EVAL_ROOT"], "model/whisper-large-v3"),
       allow_patterns=[
           "added_tokens.json",
           "config.json",
           "generation_config.json",
           "merges.txt",
           "model.safetensors",
           "normalizer.json",
           "preprocessor_config.json",
           "special_tokens_map.json",
           "tokenizer.json",
           "tokenizer_config.json",
           "vocab.json",
       ],
   )
   PY
   find "$EVAL_ROOT/model/whisper-large-v3" -type f -print0 \
     | sort -z | xargs -0 sha256sum \
     > "$EVAL_ROOT/model/whisper-large-v3.sha256"
   deactivate
   ```

下载后必须保留 WeSpeaker、MMS-FA、Whisper 的 SHA256 文件和
`evaluator-pip-freeze.txt`，并在验收报告中记录；缺少任一权重时只能标记为待验收。

### 模型推理

1. 执行未应用 patch 的原始 CUDA baseline。

   ```bash
   source .venv-cuda-original/bin/activate
   export HF_HOME="$(pwd)/hf-cache"
   export HF_HUB_OFFLINE=1
   cd upstream-original
   CUDA_VISIBLE_DEVICES=0 python inference.py \
     --jsonl examples/examples.jsonl \
     --output_dir outputs_original_cuda \
     --seed 42 \
     --use_normalize
   cd ..
   deactivate
   ```

   参数说明：

   - `jsonl`：输入 JSONL 文件路径。
   - `output_dir`：输出 WAV 保存目录。
   - `device`：仅 patch 后入口提供，支持 `npu`、`cpu`、`cuda`。
   - `batch_size`：仅 patch 后入口提供，每批生成的 JSONL 样本数，默认 `1`。
     TTSD-eval 建议保持 `1`；增大前必须观察峰值 HBM 和单批耗时。
   - `seed`：随机种子。
   - `use_normalize`：启用原项目文本归一化路径。

   模型权重、codec 配置和 checkpoint 按本文约定的目录读取。NPU 固定使用 BF16 + torch-npu Flash Attention，CPU 固定使用 FP32 + SDPA，CUDA 保持 BF16 + `flash_attention_2`，不额外暴露注意力参数。

2. 执行应用 patch 后的同设备 CUDA 回归。

   ```bash
   source .venv-cuda-patched/bin/activate
   cd upstream-npu
   python inference.py \
     --jsonl examples/examples.jsonl \
     --output_dir outputs_patched_cuda \
     --device cuda \
     --batch_size 2 \
     --seed 42 \
     --use_normalize
   cd ..
   deactivate
   ```

3. 执行 NPU candidate。

   ```bash
   source .venv-npu/bin/activate
   cd upstream-npu
   python inference.py \
     --jsonl examples/examples.jsonl \
     --output_dir outputs_npu \
     --device npu \
     --batch_size 2 \
     --seed 42 \
     --use_normalize
   cd ..
   ```

4. 检查三组输出 WAV。原始与 patch 后 CUDA 先做同设备回归，再比较 patch 后
   CUDA 与 NPU。

   ```bash
   python - <<'PY'
   from pathlib import Path
   import soundfile as sf

   groups = {
       "original_cuda": Path("upstream-original/outputs_original_cuda"),
       "patched_cuda": Path("upstream-npu/outputs_patched_cuda"),
       "npu": Path("upstream-npu/outputs_npu"),
   }
   for name, directory in groups.items():
       paths = sorted(directory.glob("output_*.wav"))
       if len(paths) != 2:
           raise RuntimeError(f"{name}: expected 2 wav files, got {len(paths)}")
       for path in paths:
           data, sr = sf.read(path, always_2d=True)
           if not sr or not len(data) or not (abs(data).max() > 0):
               raise RuntimeError(f"invalid output: {path}")
           print(name, path, sr, len(data) / sr, float(abs(data).max()))
   PY
   ```

   功能验证通过条件：输出 WAV 数量与输入有效样本一致、WAV 可读、时长大于 0、
   非全静音、无设备不一致、无 CUDA 硬编码、无 TorchCodec 报错、无 NPU
   attention mask 形状错误。


## OpenMOSS/TTSD-eval 测评

`OpenMOSS/TTSD-eval` 可以用于 MOSS-TTSD-v0.5 的公共客观测评。它要求输入 JSONL 至少包含：

- `text`：带 `[S1]` / `[S2]` 标签的 dialogue script；
- `output_audio`：待评测的生成音频；
- `prompt_audio_speaker1` / `prompt_audio_speaker2`：两位说话人的参考音频。

使用口径：v0.5 官方正式质量指标仍为“未发布”；TTSD-eval 结果用于公共评测和 NPU 迁移对齐，不得挪用 v1.0 论文指标作为 v0.5 通过线。

以下命令对中文和英文各 50 条执行三组生成。TTSD-eval JSONL 中的 prompt 路径相对
`testset/`，而 v0.5 codec checkpoint 相对模型工作树；因此先把同一个 testset
`audio/` 链接到两个工作树，再分别从工作树执行。三组输出不能覆盖：

```bash
cd ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5
MODEL_ROOT="$PWD"

ln -sfn "$MODEL_ROOT/third_party/TTSD-eval/testset/audio" \
  "$MODEL_ROOT/upstream-original/audio"
ln -sfn "$MODEL_ROOT/third_party/TTSD-eval/testset/audio" \
  "$MODEL_ROOT/upstream-npu/audio"
mkdir -p results/ttsd_eval

for LANG in zh en; do
  MANIFEST="$MODEL_ROOT/third_party/TTSD-eval/testset/ttsd_eval_${LANG}.jsonl"

  source "$MODEL_ROOT/.venv-cuda-original/bin/activate"
  (
    cd "$MODEL_ROOT/upstream-original"
    HF_HOME="$MODEL_ROOT/hf-cache" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
      python inference.py \
        --jsonl "$MANIFEST" \
        --output_dir "$MODEL_ROOT/results/ttsd_eval/original_cuda_${LANG}" \
        --seed 42 \
        --use_normalize
  )
  deactivate

  source "$MODEL_ROOT/.venv-cuda-patched/bin/activate"
  (
    cd "$MODEL_ROOT/upstream-npu"
    CUDA_VISIBLE_DEVICES=0 python inference.py \
      --jsonl "$MANIFEST" \
      --output_dir "$MODEL_ROOT/results/ttsd_eval/patched_cuda_${LANG}" \
      --device cuda \
      --batch_size 1 \
      --seed 42 \
      --use_normalize
  )
  deactivate

  source "$MODEL_ROOT/.venv-npu/bin/activate"
  (
    cd "$MODEL_ROOT/upstream-npu"
    ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
      --jsonl "$MANIFEST" \
      --output_dir "$MODEL_ROOT/results/ttsd_eval/npu_${LANG}" \
      --device npu \
      --batch_size 1 \
      --seed 42 \
      --use_normalize
  )
  deactivate
done
```

推理完成后生成六份互不覆盖的 evaluator manifest。工具会检查每个
`output_N.wav` 是否存在，并写入 manifest SHA256：

```bash
for LANG in zh en; do
  for GROUP in original_cuda patched_cuda npu; do
    python prepare_eval_data.py attach-output \
      --input_jsonl "third_party/TTSD-eval/testset/ttsd_eval_${LANG}.jsonl" \
      --output_jsonl "results/ttsd_eval/${GROUP}_${LANG}.jsonl" \
      --output_dir "results/ttsd_eval/${GROUP}_${LANG}" \
      --path_root third_party/TTSD-eval/testset
  done
done
```

正式 ACC/SIM/WER 继续使用固定 commit 的 TTSD-eval 原始
`tools/align.py`、`tools/split.py`、`tools/run_similarity.py`、
`wer/whisper_asr.py` 和 `wer/run_wer.py`。逐份运行的精确命令见
`ACCEPTANCE_PLAN.md`；不得用简化相似度或其他 ASR 替代。

## 模型推理性能

MOSS-TTSD-v0.5 属自回归生成式 TTS/TTSD 模型，L2 性能以中英文全量生成音频总时长
和端到端墙钟时间计算。以下展示 NPU 中文命令；原始 CUDA、patch 后 CUDA、英文
split 使用相同参数和独立日志/输出目录：

```bash
cd ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5
MODEL_ROOT="$PWD"
source .venv-npu/bin/activate
mkdir -p results/performance
cd upstream-npu
/usr/bin/time -v -o "$MODEL_ROOT/results/performance/npu_zh.time.txt" \
  env ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl "$MODEL_ROOT/third_party/TTSD-eval/testset/ttsd_eval_zh.jsonl" \
  --output_dir "$MODEL_ROOT/results/performance/npu_zh" \
  --device npu \
  --batch_size 1 \
  --seed 42 \
  --use_normalize
cd "$MODEL_ROOT"
```

统计输出时长：

```bash
python - <<'PY'
from pathlib import Path
import soundfile as sf

paths = sorted(Path("results/performance/npu_zh").glob("output_*.wav"))
total = 0.0
for p in paths:
    data, sr = sf.read(p, always_2d=True)
    total += len(data) / sr
print('wav_count=', len(paths))
print('generated_audio_seconds=', total)
PY
```

报告中至少记录三组输入样本数、成功输出数、输出 WAV 总时长、elapsed seconds、
`RTF=elapsed/generated_audio_seconds`、`RTFx=generated_audio_seconds/elapsed`、
固定 dtype/attention 路径、峰值 HBM 和 CPU RSS。每组至少重复 3 次并报告中位数。
完整性能与质量验收口径见 `ACCEPTANCE_PLAN.md`。

## NPU GQA FlashAttention 与显存说明

如果 TTSD-eval 或其他多样本 JSONL 在以下位置报 NPU OOM：

```text
transformers/integrations/sdpa_attention.py
value = repeat_kv(value, module.num_key_value_groups)
RuntimeError: NPU out of memory
```

原因是 Transformers 4.57.6 的 NPU SDPA 路径暂不使用原生 GQA，会把 Qwen3 的 key/value heads 通过 `repeat_kv` 实体扩展到全部 attention heads。`eager` 也会执行相同展开，并额外显式构造 attention weights，因此不是该问题的性能修复。

当前 patch 在 NPU 设备路径内部固定选择 Flash Attention backend：

- prefill：`torch_npu.npu_prompt_flash_attention`；
- 单 token decode：`torch_npu.npu_incre_flash_attention`；
- Q/K/V 使用 `BNSD` 布局；
- `num_heads` 和 `num_key_value_heads` 分别取 query 与 key 的 head 数，由算子直接处理 GQA；
- 不增加 attention CLI 参数，不修改 Transformers site-packages，也不静默回退到 SDPA/eager。

目标 `torch-npu` 必须同时提供上述两个接口及 `num_key_value_heads` 参数。运行前可检查：

```bash
python - <<'PY'
import inspect
import torch_npu
print(inspect.signature(torch_npu.npu_prompt_flash_attention))
print(inspect.signature(torch_npu.npu_incre_flash_attention))
PY
```

性能评测直接使用 NPU 设备参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl ../third_party/TTSD-eval/testset/ttsd_eval_zh.jsonl \
  --output_dir ../results/ttsd_eval/npu_zh \
  --device npu \
  --batch_size 1 \
  --seed 42 \
  --use_normalize
```

Flash Attention 消除的是 `repeat_kv` 造成的 KV head 实体展开，不保证整份 manifest
作为一个 batch 时的其他张量都能放入 HBM。patch 后入口默认
`--batch_size 1`，逐批生成并显示 `[Batch i/N]` 进度；全部 batch 完成后仍按原
入口规则写出 `output_N.wav`，避免 50 条长音频全部完成前没有任何进度。

若日志停在 `Starting batch audio generation...`：

1. 先执行 `watch -n 1 npu-smi info`。首批可能触发 NPU 算子/图编译；出现
   `multiprocessing.forkserver` / `resource_tracker` 子进程本身不能证明死锁。
2. 若 NPU 利用率或 HBM 持续变化，先等待首批完成；之后应出现
   `Original outputs shape` 和 `[Batch 1/N] completed`。
3. 若超过 10 分钟 NPU 利用率始终为 0、HBM 不变且无新 CANN 日志，终止进程，
   用 `--batch_size 1` 和单条 manifest 重跑。单条仍卡住时再保留完整 Python
   栈、`npu-smi info`、CANN 日志和依赖版本排查，不能静默切到 CPU。

CPU/CUDA/NPU 候选对齐必须使用相同 `--batch_size`。未应用 patch 的原始入口不支持
该参数，仍保留其原生完整 JSONL batch 作为 upstream baseline；报告中必须明确记录
这一运行参数差异，不能把不同 batch 口径写成严格逐样本数值等价。

## 已知问题：Transformers 5.x 不兼容

如果环境安装了 `transformers==5.12.1`，模型加载阶段可能报：

```text
AttributeError: 'list' object has no attribute 'keys'
```

报错位置通常位于 Transformers 的 `get_expanded_tied_weights_keys()`。原因是 Transformers 5.x 要求 `_tied_weights_keys` 为“目标权重到源权重”的字典映射，而 MOSS-TTSD-v0.5 上游 `modeling_asteroid.py` 仍按 Transformers 4.x 接口将其定义为列表。

当前项目仅记录该依赖边界，不修改上游模型代码。请在运行推理或 TTSD-eval 前固定已验证版本：

```bash
pip uninstall -y transformers
pip install "transformers==4.57.6"
python -c "import transformers; print(transformers.__version__)"
```

预期输出为 `4.57.6`。本地已验证该版本可以完成 `AsteroidTTSInstruct.from_pretrained()` 初始化并保持各通道输入 embedding 与 `lm_heads` 的权重绑定。不要通过忽略异常、关闭权重绑定或 CPU 回退绕过该错误。

## 公网地址说明

| 资源 | 地址 | 说明 |
|---|---|---|
| OpenMOSS/MOSS-TTSD 源码 | <https://github.com/OpenMOSS/MOSS-TTSD> | 使用 tag `v0.5`，commit `0e078c62389922d3aa873ce182daf31142860b18` |
| OpenMOSS/TTSD-eval | <https://github.com/OpenMOSS/TTSD-eval> | L2 公共客观评测，记录 ACC/SIM/WER；不是 v0.5 已发布官方指标 |
| WeSpeaker 代码与权重索引 | <https://github.com/wenet-e2e/wespeaker>；<https://modelscope.cn/datasets/wenet/wespeaker_pretrained_models> | 代码固定 commit `c92349a14d6b426808c4e09b8b12e076864dfc11`；下载 `voxblink2_samresnet100_ft.zip` |
| MMS-FA checkpoint | <https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt?versionId=dZWoHyjLHoCxDn.KL1FPSlVCD3CPRtOL> | 固定 S3 version ID；目标路径为 `model/checkpoints/model.pt` |
| Whisper-large-v3 | <https://huggingface.co/openai/whisper-large-v3> | 固定 revision `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| MOSS-TTSD-v0.5 HF 权重 | <https://huggingface.co/fnlp/MOSS-TTSD-v0.5> | 固定 revision `8527b9136b6afefe2252ae597cecea2e80e7ebeb` |
| MOSS-TTSD-v0.5 ModelScope 权重 | <https://modelscope.cn/models/openmoss/MOSS-TTSD-v0.5> | 国内镜像，记录 HEAD `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59` |
| XY Tokenizer HF checkpoint | <https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0> | 固定 revision `c83433728e698ed0698e88cb5096bc221fb8f8c5` |
| XY Tokenizer ModelScope checkpoint | <https://modelscope.cn/models/openmoss/XY_Tokenizer_TTSD_V0> | 国内镜像，记录 HEAD `79082154409f5e883d9487c4d4b4be363323b039` |
