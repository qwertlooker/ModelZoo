# MOSS-TTSD-v0.5 推理指导

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [文件目录](#文件目录)
- [快速上手](#快速上手)
  - [获取源码并应用 patch](#获取源码并应用-patch)
  - [准备权重](#准备权重)
  - [准备测试数据](#准备测试数据)
  - [准备 TTSD-eval 工程](#准备-ttsd-eval-工程)
  - [模型推理](#模型推理)
- [OpenMOSS/TTSD-eval 测评](#openmossttsd-eval-测评)
- [模型推理性能](#模型推理性能)
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

  `prompt_audio_speaker1`、`prompt_audio_speaker2` 可为相对 `base_path` 的路径，也可按原项目逻辑传入可解析的本地路径。

- 输出数据

  输出为 `output_*.wav`，保存到 `--output_dir` 指定目录。输出采样率以模型/codec 返回值为准。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

  | 配套 | 版本 |
  |---|---|
  | 硬件 | Atlas 800I A2 |
  | 固件与驱动 | 25.5.1+ |
  | CANN Toolkit / Kernel / NNAL | 8.5.1 |
  | Python | 3.11 |
  | PyTorch / torch-npu / torchaudio | 2.9.0 |
  | transformers | 4.57.6（不要安装 5.x） |
  | soundfile | 用于文件读写 |

说明：Atlas 800I A2 推理卡请以 CANN 版本选择实际固件与驱动版本。

## 文件目录

```text
MOSS-TTSD-v0.5
├── README.md                                   # 推理指导文档
├── prepare_eval_data.py                        # evaluator manifest/准备门禁工具
├── requirements_eval.txt                       # 固定 TTSD-eval 直接依赖
├── patches
│   ├── 0001-adapt-v0.5-inference-to-npu.patch  # v0.5 NPU 适配 patch
│   └── 0002-adapt-ttsd-eval-to-npu.patch       # TTSD-eval NPU 适配 patch
├── upstream-npu                                # 应用 patch 后的 NPU 路径（下载后）
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
└── third_party
    └── TTSD-eval                               # 下载后的评测工程
```

## 快速上手

后续命令默认从 `ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5` 目录开始，示例只使用相对路径。

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

3. 安装 NPU 环境。

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

4. （可选）原始 CUDA 和 patch 后 CUDA 对照组使用两个独立环境，均安装相同的 PyTorch、Transformers 和 CUDA `flash-attn`。本地不具备 CUDA 时可跳过本步，仅运行 NPU 组完成迁移验收。CUDA wheel/索引需按实际 CUDA 版本选择，不能用于 NPU 环境：

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

5. （可选）CPU 对照组使用独立环境，安装 CPU-only PyTorch。CPU 推理使用 patch 后代码，固定 FP32 + SDPA，不依赖 CUDA 或 NPU：

   ```bash
   python3.11 -m venv .venv-cpu
   source .venv-cpu/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.9.0 torchaudio==2.9.0 \
     --index-url https://download.pytorch.org/whl/cpu
   python -m pip install "transformers==4.57.6"
   python -m pip install -r upstream-npu/requirements.txt
   python -m pip install -r upstream-npu/XY_Tokenizer/requirements.txt
   python - <<'PY'
   import torch
   print(torch.__version__, torch.randn(1).device)
   PY
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

2. 下载后记录 SHA256。

   ```bash
   cd upstream-npu
   find -L weights/MOSS-TTSD-v0.5 -maxdepth 1 -type f -print0 \
     | sort -z | xargs -0 sha256sum
   sha256sum XY_Tokenizer/weights/xy_tokenizer.ckpt
   ```

### 准备测试数据

1. 使用原项目官方示例作为 smoke test 数据：

   ```text
   upstream-original/examples/examples.jsonl
   ```

   该文件包含中文和英文双说话人长对话示例，并引用 `examples/` 目录下的 prompt WAV。

2. L2 使用 `OpenMOSS/TTSD-eval` 中文、英文全量各 50 条。

### 准备 TTSD-eval 工程

TTSD-eval 支持三种评测 profile，NPU 为必跑项，CUDA/CPU 为可选对照。NPU profile 复用推理环境 `.venv-npu`（`torch/torchaudio==2.9.0 + torch-npu==2.9.0`），并需对 TTSD-eval 工作树应用 `patches/0002-adapt-ttsd-eval-to-npu.patch`；CUDA/CPU profile 使用独立 venv（`torch/torchaudio==2.8.0`）。以下命令均从 `ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5` 执行。

#### 获取固定源码和 testset

1. 从干净目录获取评测器固定提交。

   ```bash
   MODEL_ROOT="$PWD"
   EVAL_ROOT="$MODEL_ROOT/third_party/TTSD-eval"
   test ! -e "$EVAL_ROOT"
   mkdir -p "$EVAL_ROOT"

   git -C "$EVAL_ROOT" init
   git -C "$EVAL_ROOT" remote add origin \
     https://github.com/OpenMOSS/TTSD-eval.git
   git -C "$EVAL_ROOT" fetch --depth 1 origin \
     dea13b98529dc16dcfb5fe45779ad63ac9238337
   git -C "$EVAL_ROOT" checkout --detach FETCH_HEAD
   test "$(git -C "$EVAL_ROOT" rev-parse HEAD)" = \
     "dea13b98529dc16dcfb5fe45779ad63ac9238337"
   test -z "$(git -C "$EVAL_ROOT" status --short --untracked-files=no)"
   ```

2. `testset.zip` 在仓库中为 Git LFS pointer，直接下载固定对象到未跟踪的 `model/downloads/`：

   ```bash
   mkdir -p "$EVAL_ROOT/model/downloads"
   curl -L --fail --retry 5 --retry-all-errors \
     -o "$EVAL_ROOT/model/downloads/testset.zip" \
     https://media.githubusercontent.com/media/OpenMOSS/TTSD-eval/dea13b98529dc16dcfb5fe45779ad63ac9238337/testset.zip
   echo "49ed8338f3e5323c5ffcff01f3480a9c245937256d9197d792c973cba5603e17  $EVAL_ROOT/model/downloads/testset.zip" \
     | sha256sum -c -
   test "$(stat -c %s "$EVAL_ROOT/model/downloads/testset.zip")" = "71138324"
   unzip -oq "$EVAL_ROOT/model/downloads/testset.zip" -d "$EVAL_ROOT"
   ```

3. 使用仓内正式工具检查 evaluator commit、受版本控制文件、archive、两个 manifest 的 50+50 样本及 200 个 prompt WAV：

   ```bash
   python3 prepare_eval_data.py verify-ttsd-eval \
     --eval_root "$EVAL_ROOT" \
     --scope source-data \
     --report results/ttsd_eval_setup/source_data.json
   ```

#### 创建评测环境

系统需预先提供 Python 3.11 venv、Git、FFmpeg 和 libsndfile；Ubuntu/Debian 可按现场权限安装：

```bash
sudo apt-get install -y python3.11-venv git ffmpeg libsndfile1
```

**NPU profile**（必跑）复用推理环境，无需独立 venv；先对 TTSD-eval 工作树应用设备适配补丁，再安装评测直接依赖（不安装 torch/torchaudio）：

```bash
source .venv-npu/bin/activate
git -C "$EVAL_ROOT" apply "$PWD/patches/0002-adapt-ttsd-eval-to-npu.patch"
python -m pip install -r requirements_eval.txt
python -m pip check
python -m pip freeze > results/ttsd_eval_setup/evaluator-pip-freeze.txt
python -c 'import torch, torch_npu; print(torch.__version__, torch.npu.is_available(), torch.npu.device_count())'
deactivate
```

**CUDA profile**（可选对照，CUDA 12.8 wheel）：

```bash
python3.11 -m venv .venv-ttsd-eval
source .venv-ttsd-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_eval.txt
python -m pip check
python -m pip freeze > results/ttsd_eval_setup/evaluator-pip-freeze.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
deactivate
```

**CPU profile**（可选对照）只替换框架安装命令（必须在报告中记录 evaluator 为 CPU/FP32）：

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

`requirements_eval.txt` 固定 TTSD-eval 的直接依赖和 WeSpeaker commit，避免原 `requirements.txt` 中版本范围及 `wespeaker.git` HEAD 漂移。

#### 下载三类评测权重

1. 下载 WeSpeaker 权重。

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

2. 下载固定 S3 version ID 的 MMS-FA checkpoint：

   ```bash
   EVAL_ROOT="$PWD/third_party/TTSD-eval"
   mkdir -p "$EVAL_ROOT/model/checkpoints"
   curl -L --fail --retry 5 --retry-all-errors \
     -o "$EVAL_ROOT/model/checkpoints/model.pt" \
     "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt?versionId=dZWoHyjLHoCxDn.KL1FPSlVCD3CPRtOL"
   echo "20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680  $EVAL_ROOT/model/checkpoints/model.pt" \
     | sha256sum -c -
   test "$(stat -c %s "$EVAL_ROOT/model/checkpoints/model.pt")" = "1262047414"
   ```

3. 下载固定 revision 的 Whisper-large-v3，并写入 revision marker。正式评测只从本地目录离线加载。以下使用 NPU 环境（必跑）；CUDA/CPU profile 改用 `.venv-ttsd-eval`：

   ```bash
   source .venv-npu/bin/activate
   EVAL_ROOT="$PWD/third_party/TTSD-eval"
   export EVAL_ROOT
   python - <<'PY'
   import os
   from pathlib import Path
   from huggingface_hub import snapshot_download

   revision = "06f233fe06e710322aca913c1bc4249a0d71fce1"
   model_dir = Path(os.environ["EVAL_ROOT"]) / "model/whisper-large-v3"
   snapshot_download(
       repo_id="openai/whisper-large-v3",
       revision=revision,
       local_dir=model_dir,
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
   (model_dir / "REVISION").write_text(revision + "\n", encoding="utf-8")
   PY
   echo "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95  $EVAL_ROOT/model/whisper-large-v3/model.safetensors" \
     | sha256sum -c -
   test "$(stat -c %s "$EVAL_ROOT/model/whisper-large-v3/model.safetensors")" = \
     "3087130976"
   deactivate
   ```

#### 完整预检

1. 在离线模式下执行完整结构、hash、依赖版本和 import 门禁，并保存机器可读证据。以下为 NPU profile（必跑）；CUDA profile 改用 `.venv-ttsd-eval` 和 `--expected_device cuda`，CPU profile 改用 `.venv-ttsd-eval` 和 `--expected_device cpu`，二者均为可选对照。

   ```bash
   source .venv-npu/bin/activate
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   python prepare_eval_data.py verify-ttsd-eval \
     --eval_root "$EVAL_ROOT" \
     --scope full \
     --expected_device npu \
     --report results/ttsd_eval_setup/full.json

   for ENTRY in \
     tools/align.py tools/split.py tools/run_similarity.py \
     wer/whisper_asr.py wer/run_wer.py; do
     python "$EVAL_ROOT/$ENTRY" --help >/dev/null
   done

   mkdir -p results/ttsd_eval_setup/fixture
   printf '%s\n' \
     '{"text":"[S1]hello world","asr_res":"hello world"}' \
     > results/ttsd_eval_setup/fixture/wer_input.jsonl
   python "$EVAL_ROOT/wer/run_wer.py" \
     --lang en \
     --input_jsonl results/ttsd_eval_setup/fixture/wer_input.jsonl \
     --output_jsonl results/ttsd_eval_setup/fixture/wer_output.jsonl \
     --metrics_txt results/ttsd_eval_setup/fixture/wer.txt
   grep -q '"wer": 0.0' results/ttsd_eval_setup/fixture/wer_output.jsonl
   deactivate
   ```

2. 在正式全量评测前逐个加载三类权重（占用约 5 GiB 以上主机内存）。以下为 NPU profile（必跑）；CUDA/CPU profile 改用 `.venv-ttsd-eval`：

   ```bash
   source .venv-npu/bin/activate
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   export EVAL_ROOT="$PWD/third_party/TTSD-eval"
   python - <<'PY'
   import gc
   import os

   import torch
   import wespeaker
   from torchaudio.pipelines import MMS_FA
   from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

   root = os.environ["EVAL_ROOT"]
   torch.hub.set_dir(os.path.join(root, "model"))
   mms = MMS_FA.get_model()
   del mms
   gc.collect()

   speaker = wespeaker.load_model(
       os.path.join(root, "model/voxblink2_samresnet100_ft")
   )
   del speaker
   gc.collect()

   whisper_path = os.path.join(root, "model/whisper-large-v3")
   processor = AutoProcessor.from_pretrained(whisper_path, local_files_only=True)
   whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
       whisper_path,
       local_files_only=True,
       use_safetensors=True,
       low_cpu_mem_usage=True,
   )
   print(type(processor).__name__, type(whisper).__name__)
   PY
   deactivate
   ```

### 模型推理

迁移验收以 NPU 推理为必跑项；下方步骤 1、2 为可选 CUDA 对照组，本地不具备 CUDA
时可跳过，仅执行步骤 3 的 NPU 推理即可完成功能验证。

1. （可选）执行未应用 patch 的原始 CUDA baseline。

   ```bash
   source .venv-cuda-original/bin/activate
   export HF_HOME="$(pwd)/hf-cache"
   export HF_HUB_OFFLINE=1
   cd upstream-original
   python inference.py \
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
   - `batch_size`：仅 patch 后入口提供，每批生成的 JSONL 样本数，默认 `1`。TTSD-eval 建议保持 `1`。
   - `seed`：随机种子。
   - `use_normalize`：启用原项目文本归一化路径。

   模型权重、codec 配置和 checkpoint 按本文约定的目录读取。NPU 固定使用 BF16 + torch-npu Flash Attention，CPU 固定使用 FP32 + SDPA，CUDA 保持 BF16 + `flash_attention_2`，不额外暴露注意力参数。

2. （可选）执行应用 patch 后的同设备 CUDA 回归。

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

3. 执行 NPU 推理（必跑）。

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

4. （可选）执行 CPU 推理。CPU 路径使用 patch 后代码，固定 FP32 + SDPA。

   ```bash
   source .venv-cpu/bin/activate
   cd upstream-npu
   python inference.py \
     --jsonl examples/examples.jsonl \
     --output_dir outputs_cpu \
     --device cpu \
     --batch_size 2 \
     --seed 42 \
     --use_normalize
   cd ..
   deactivate
   ```

5. 检查已生成组的输出 WAV。NPU 组必查；若运行了 CUDA 或 CPU 对照组，再比较各组输出。

   ```bash
   python - <<'PY'
   from pathlib import Path
   import soundfile as sf

   groups = {
       "original_cuda": Path("upstream-original/outputs_original_cuda"),
       "patched_cuda": Path("upstream-npu/outputs_patched_cuda"),
       "cpu": Path("upstream-npu/outputs_cpu"),
       "npu": Path("upstream-npu/outputs_npu"),
   }
   if not groups["npu"].exists():
       raise RuntimeError("npu output is required but missing")
   for name, directory in groups.items():
       if not directory.exists():
           continue
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

   功能验证通过条件：输出 WAV 数量与输入有效样本一致、WAV 可读、时长大于 0、非全静音。

## OpenMOSS/TTSD-eval 测评

`OpenMOSS/TTSD-eval` 可以用于 MOSS-TTSD-v0.5 的公共客观测评。它要求输入 JSONL 至少包含：

- `text`：带 `[S1]` / `[S2]` 标签的 dialogue script；
- `output_audio`：待评测的生成音频；
- `prompt_audio_speaker1` / `prompt_audio_speaker2`：两位说话人的参考音频。

以下命令对中文和英文各 50 条生成 NPU 音频（必跑），并在本地具备 CUDA 时额外生成两组对照。TTSD-eval JSONL 中的 prompt 路径相对 `testset/`，而 v0.5 codec checkpoint 相对模型工作树；因此先把同一个 testset `audio/` 链接到两个工作树，再分别从工作树执行。NPU 输出与 CUDA 对照组输出互不覆盖：

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

  # 可选：未应用 patch 的原始 CUDA 对照组
  if [ -d "$MODEL_ROOT/.venv-cuda-original" ]; then
    source "$MODEL_ROOT/.venv-cuda-original/bin/activate"
    (
      cd "$MODEL_ROOT/upstream-original"
      HF_HOME="$MODEL_ROOT/hf-cache" HF_HUB_OFFLINE=1 \
        python inference.py \
          --jsonl "$MANIFEST" \
          --output_dir "$MODEL_ROOT/results/ttsd_eval/original_cuda_${LANG}" \
          --seed 42 \
          --use_normalize
    )
    deactivate
  fi

  # 可选：应用 patch 后的 CUDA 回归对照组
  if [ -d "$MODEL_ROOT/.venv-cuda-patched" ]; then
    source "$MODEL_ROOT/.venv-cuda-patched/bin/activate"
    (
      cd "$MODEL_ROOT/upstream-npu"
      python inference.py \
        --jsonl "$MANIFEST" \
        --output_dir "$MODEL_ROOT/results/ttsd_eval/patched_cuda_${LANG}" \
        --device cuda \
        --batch_size 1 \
        --seed 42 \
        --use_normalize
    )
    deactivate
  fi

  # 必跑：NPU 推理
  source "$MODEL_ROOT/.venv-npu/bin/activate"
  (
    cd "$MODEL_ROOT/upstream-npu"
    python inference.py \
      --jsonl "$MANIFEST" \
      --output_dir "$MODEL_ROOT/results/ttsd_eval/npu_${LANG}" \
      --device npu \
      --batch_size 1 \
      --seed 42 \
      --use_normalize
  )
  deactivate

  # 可选：CPU 对照组
  if [ -d "$MODEL_ROOT/.venv-cpu" ]; then
    source "$MODEL_ROOT/.venv-cpu/bin/activate"
    (
      cd "$MODEL_ROOT/upstream-npu"
      python inference.py \
        --jsonl "$MANIFEST" \
        --output_dir "$MODEL_ROOT/results/ttsd_eval/cpu_${LANG}" \
        --device cpu \
        --batch_size 1 \
        --seed 42 \
        --use_normalize
    )
    deactivate
  fi
done
```

推理完成为已生成的组生成 evaluator manifest（仅处理实际存在的输出目录）。工具会检查每个 `output_N.wav` 是否存在，并写入 manifest SHA256：

```bash
for LANG in zh en; do
  for GROUP in original_cuda patched_cuda cpu npu; do
    [ -d "results/ttsd_eval/${GROUP}_${LANG}" ] || continue
    python prepare_eval_data.py attach-output \
      --input_jsonl "third_party/TTSD-eval/testset/ttsd_eval_${LANG}.jsonl" \
      --output_jsonl "results/ttsd_eval/${GROUP}_${LANG}.jsonl" \
      --output_dir "results/ttsd_eval/${GROUP}_${LANG}" \
      --path_root third_party/TTSD-eval/testset
  done
done
```

### ACC/SIM/WER 逐份评测

TTSD-eval 的 prompt 路径相对 `testset/`，以下命令必须从该目录运行；输出路径使用绝对路径，避免切换目录后失效。NPU evaluator 为必跑项；CUDA、CPU evaluator 为可选对照，仅在本地具备对应环境且需要同口径对照时运行。所选 profile 须对所有已生成组结果保持一致，每组显式指定所有中间目录，避免 evaluator 默认目录互相污染。

#### NPU evaluator（必跑）

```bash
set -o pipefail
MODEL_ROOT="$PWD"
EVAL_ROOT="$MODEL_ROOT/third_party/TTSD-eval"
source "$MODEL_ROOT/.venv-npu/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ALIGN_DEVICE=npu
cd "$EVAL_ROOT/testset"

for LANG in zh en; do
  for GROUP in npu; do
    INPUT="$MODEL_ROOT/results/ttsd_eval/${GROUP}_${LANG}.jsonl"
    [ -f "$INPUT" ] || continue
    STEM="${GROUP}_${LANG}"
    RUN_ROOT="$MODEL_ROOT/results/ttsd_eval_metrics/$STEM"
    mkdir -p \
      "$RUN_ROOT/alignment_files" \
      "$RUN_ROOT/split_res" \
      "$RUN_ROOT/audio_segments"

    {
      python "$EVAL_ROOT/tools/align.py" \
        --input_jsonl "$INPUT" \
        --output_dir "$RUN_ROOT/alignment_files" \
        --output_jsonl "$RUN_ROOT/alignment.jsonl" \
        --cache_dir "$EVAL_ROOT/model" \
        --device "$ALIGN_DEVICE"
      test "$(wc -l < "$RUN_ROOT/alignment.jsonl")" -eq 50

      python "$EVAL_ROOT/tools/split.py" \
        --input_jsonl "$RUN_ROOT/alignment.jsonl" \
        --split_res_dir "$RUN_ROOT/split_res" \
        --segment_dir "$RUN_ROOT/audio_segments" \
        --output_jsonl "$RUN_ROOT/split.jsonl" \
        --num_workers 8
      test "$(wc -l < "$RUN_ROOT/split.jsonl")" -eq 50

      python "$EVAL_ROOT/tools/run_similarity.py" \
        --input_jsonl "$RUN_ROOT/split.jsonl" \
        --output_jsonl "$RUN_ROOT/sim.jsonl" \
        --metrics_txt "$RUN_ROOT/acc_sim.txt" \
        --model_dir "$EVAL_ROOT/model/voxblink2_samresnet100_ft" \
        --device npu
      test "$(wc -l < "$RUN_ROOT/sim.jsonl")" -eq 50
      test -s "$RUN_ROOT/acc_sim.txt"

      python "$EVAL_ROOT/wer/whisper_asr.py" \
        --input_jsonl "$INPUT" \
        --output_jsonl "$RUN_ROOT/asr.jsonl" \
        --model_id "$EVAL_ROOT/model/whisper-large-v3" \
        --device npu
      test "$(wc -l < "$RUN_ROOT/asr.jsonl")" -eq 50

      python "$EVAL_ROOT/wer/run_wer.py" \
        --lang "$LANG" \
        --input_jsonl "$RUN_ROOT/asr.jsonl" \
        --output_jsonl "$RUN_ROOT/wer.jsonl" \
        --metrics_txt "$RUN_ROOT/wer.txt"
      test "$(wc -l < "$RUN_ROOT/wer.jsonl")" -eq 50
      test -s "$RUN_ROOT/wer.txt"
    } 2>&1 | tee "$RUN_ROOT/evaluator.log"
  done
done
cd "$MODEL_ROOT"
deactivate
```

#### CUDA evaluator（可选对照）

```bash
set -o pipefail
MODEL_ROOT="$PWD"
EVAL_ROOT="$MODEL_ROOT/third_party/TTSD-eval"
source "$MODEL_ROOT/.venv-ttsd-eval/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ALIGN_DEVICE=cuda
SIM_NUM_GPUS=1
WHISPER_NUM_GPUS=1
cd "$EVAL_ROOT/testset"

for LANG in zh en; do
  for GROUP in original_cuda patched_cuda; do
    INPUT="$MODEL_ROOT/results/ttsd_eval/${GROUP}_${LANG}.jsonl"
    [ -f "$INPUT" ] || continue
    STEM="${GROUP}_${LANG}"
    RUN_ROOT="$MODEL_ROOT/results/ttsd_eval_metrics/$STEM"
    mkdir -p \
      "$RUN_ROOT/alignment_files" \
      "$RUN_ROOT/split_res" \
      "$RUN_ROOT/audio_segments"

    {
      python "$EVAL_ROOT/tools/align.py" \
        --input_jsonl "$INPUT" \
        --output_dir "$RUN_ROOT/alignment_files" \
        --output_jsonl "$RUN_ROOT/alignment.jsonl" \
        --cache_dir "$EVAL_ROOT/model" \
        --device "$ALIGN_DEVICE"
      test "$(wc -l < "$RUN_ROOT/alignment.jsonl")" -eq 50

      python "$EVAL_ROOT/tools/split.py" \
        --input_jsonl "$RUN_ROOT/alignment.jsonl" \
        --split_res_dir "$RUN_ROOT/split_res" \
        --segment_dir "$RUN_ROOT/audio_segments" \
        --output_jsonl "$RUN_ROOT/split.jsonl" \
        --num_workers 8
      test "$(wc -l < "$RUN_ROOT/split.jsonl")" -eq 50

      python "$EVAL_ROOT/tools/run_similarity.py" \
        --input_jsonl "$RUN_ROOT/split.jsonl" \
        --output_jsonl "$RUN_ROOT/sim.jsonl" \
        --metrics_txt "$RUN_ROOT/acc_sim.txt" \
        --model_dir "$EVAL_ROOT/model/voxblink2_samresnet100_ft" \
        --num_gpus "$SIM_NUM_GPUS"
      test "$(wc -l < "$RUN_ROOT/sim.jsonl")" -eq 50
      test -s "$RUN_ROOT/acc_sim.txt"

      python "$EVAL_ROOT/wer/whisper_asr.py" \
        --input_jsonl "$INPUT" \
        --output_jsonl "$RUN_ROOT/asr.jsonl" \
        --model_id "$EVAL_ROOT/model/whisper-large-v3" \
        --num_gpus "$WHISPER_NUM_GPUS"
      test "$(wc -l < "$RUN_ROOT/asr.jsonl")" -eq 50

      python "$EVAL_ROOT/wer/run_wer.py" \
        --lang "$LANG" \
        --input_jsonl "$RUN_ROOT/asr.jsonl" \
        --output_jsonl "$RUN_ROOT/wer.jsonl" \
        --metrics_txt "$RUN_ROOT/wer.txt"
      test "$(wc -l < "$RUN_ROOT/wer.jsonl")" -eq 50
      test -s "$RUN_ROOT/wer.txt"
    } 2>&1 | tee "$RUN_ROOT/evaluator.log"
  done
done
cd "$MODEL_ROOT"
deactivate
```

#### CPU evaluator（可选对照）

评测 CPU 推理生成的音频（`cpu` 组）。以下命令使用 `.venv-ttsd-eval`（CPU PyTorch + 评测依赖），仅评估 CPU 推理输出，与 NPU/CUDA evaluator 使用相同的评测工具和权重。

```bash
set -o pipefail
MODEL_ROOT="$PWD"
EVAL_ROOT="$MODEL_ROOT/third_party/TTSD-eval"
source "$MODEL_ROOT/.venv-ttsd-eval/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ALIGN_DEVICE=cpu
cd "$EVAL_ROOT/testset"

for LANG in zh en; do
  for GROUP in cpu; do
    INPUT="$MODEL_ROOT/results/ttsd_eval/${GROUP}_${LANG}.jsonl"
    [ -f "$INPUT" ] || continue
    STEM="${GROUP}_${LANG}"
    RUN_ROOT="$MODEL_ROOT/results/ttsd_eval_metrics/$STEM"
    mkdir -p \
      "$RUN_ROOT/alignment_files" \
      "$RUN_ROOT/split_res" \
      "$RUN_ROOT/audio_segments"

    {
      python "$EVAL_ROOT/tools/align.py" \
        --input_jsonl "$INPUT" \
        --output_dir "$RUN_ROOT/alignment_files" \
        --output_jsonl "$RUN_ROOT/alignment.jsonl" \
        --cache_dir "$EVAL_ROOT/model" \
        --device "$ALIGN_DEVICE"
      test "$(wc -l < "$RUN_ROOT/alignment.jsonl")" -eq 50

      python "$EVAL_ROOT/tools/split.py" \
        --input_jsonl "$RUN_ROOT/alignment.jsonl" \
        --split_res_dir "$RUN_ROOT/split_res" \
        --segment_dir "$RUN_ROOT/audio_segments" \
        --output_jsonl "$RUN_ROOT/split.jsonl" \
        --num_workers 8
      test "$(wc -l < "$RUN_ROOT/split.jsonl")" -eq 50

      python "$EVAL_ROOT/tools/run_similarity.py" \
        --input_jsonl "$RUN_ROOT/split.jsonl" \
        --output_jsonl "$RUN_ROOT/sim.jsonl" \
        --metrics_txt "$RUN_ROOT/acc_sim.txt" \
        --model_dir "$EVAL_ROOT/model/voxblink2_samresnet100_ft" \
        --device cpu
      test "$(wc -l < "$RUN_ROOT/sim.jsonl")" -eq 50
      test -s "$RUN_ROOT/acc_sim.txt"

      python "$EVAL_ROOT/wer/whisper_asr.py" \
        --input_jsonl "$INPUT" \
        --output_jsonl "$RUN_ROOT/asr.jsonl" \
        --model_id "$EVAL_ROOT/model/whisper-large-v3" \
        --device cpu
      test "$(wc -l < "$RUN_ROOT/asr.jsonl")" -eq 50

      python "$EVAL_ROOT/wer/run_wer.py" \
        --lang "$LANG" \
        --input_jsonl "$RUN_ROOT/asr.jsonl" \
        --output_jsonl "$RUN_ROOT/wer.jsonl" \
        --metrics_txt "$RUN_ROOT/wer.txt"
      test "$(wc -l < "$RUN_ROOT/wer.jsonl")" -eq 50
      test -s "$RUN_ROOT/wer.txt"
    } 2>&1 | tee "$RUN_ROOT/evaluator.log"
  done
done
cd "$MODEL_ROOT"
deactivate
```

TTSD-eval 的部分工具会记录 warning 后跳过失败样本并以 0 退出，因此每阶段的 50 行检查是正式门禁，不得删除。上述命令直接复用固定 commit 的官方评测组件 `tools/align.py`、`tools/split.py`、`tools/run_similarity.py`、`wer/whisper_asr.py` 和 `wer/run_wer.py`，不修改 evaluator，也不使用简化指标。

## 模型推理性能

MOSS-TTSD-v0.5 属自回归生成式 TTS/TTSD 模型，L2 性能以中英文全量生成音频总时长和端到端墙钟时间计算。以下展示 NPU 中文命令（必跑）；英文 split 使用相同参数和独立日志/输出目录。本地具备 CUDA 或 CPU 时可额外运行对照组，但不作为强制要求：

```bash
cd ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5
MODEL_ROOT="$PWD"
source .venv-npu/bin/activate
mkdir -p results/performance
cd upstream-npu
/usr/bin/time -v -o "$MODEL_ROOT/results/performance/npu_zh.time.txt" \
  python inference.py \
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

| 硬件 | 数据集 | 指标 | 得分 |
|---|---|---|---|
| Atlas 800I A2 | TTSD-eval 中文 50 条 | RTF | 待补充 |
| Atlas 800I A2 | TTSD-eval 英文 50 条 | RTF | 待补充 |

### 模型推理精度/质量

v0.5 官方未发布正式测试集和精度/质量指标。NPU 迁移结果以 `OpenMOSS/TTSD-eval` 公共评测的 ACC（说话人归因准确率）、SIM（说话人相似度）和 WER（词错误率）作为记录，与可取得的公开参考对比。详细精度/质量验收方案见 `ACCEPTANCE_PLAN.md`。

| 硬件 | 数据集 | 指标 | 得分 |
|---|---|---|---|
| Atlas 800I A2 | TTSD-eval 中文 50 条 | ACC / SIM / WER | 待补充 |
| Atlas 800I A2 | TTSD-eval 英文 50 条 | ACC / SIM / WER | 待补充 |

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
