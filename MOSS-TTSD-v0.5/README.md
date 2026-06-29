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
│   └── 0002-adapt-ttsd-eval-to-npu.patch       # TTSD-eval NPU 适配 patch（含 eval.sh / run_wer.sh 设备与输入路径透传、requirements.txt 去除 torch/torchaudio <=2.8.0 上限）
│   └── 0003-fix-s3prl-hub-resilient-imports.patch           # s3prl hub.py espnet_hubert/mos_prediction 导入容错（兼容 TorchAudio 2.9+ 及 NLTK cmudict 损坏/离线环境）
│   └── 0004-fix-wespeaker-float64-on-npu.patch              # wespeaker SimAM/pooling Python float 标量→float32 tensor（避免 NPU float64→float32 隐式转换导致 aclnnAddV3 崩溃）
│   └── 0005-fix-torchaudio-kaldi-rfft-abs-on-npu.patch      # torchaudio kaldi.py rfft().abs()→sqrt(real^2+imag^2)（NPU FFT backend 复数取模返回全零 → sim=0.0000）
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
        ├── requirements.txt                         # patch 去除 torch/torchaudio <=2.8.0 版本上限，允许 2.9+
    ├── eval.sh                                # 上游 ACC/SIM 评测入口（patch 增加 DEVICE/CACHE_DIR/MODEL_DIR/INPUT_JSONL_PATH 透传）
        ├── run_wer.sh                              # 上游 WER 评测入口（patch 增加 DEVICE/WHISPER_MODEL_ID/INPUT_JSONL_PATH/LANGUAGE 透传）
        ├── tools
        │   ├── align.py                            # patch 后支持 NPU
        │   ├── split.py
        │   └── run_similarity.py                   # patch 后支持 NPU
        └── wer
            ├── whisper_asr.py                      # patch 后支持 NPU
            └── run_wer.py
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

3. 安装 NPU 环境。非必要不使用 venv；当 CANN 基础镜像或系统 Python 已满足要求时直接使用。仅在同一机器需多套不兼容的 Python/PyTorch 版本时才使用 venv 隔离，并在文档中写明隔离原因。

   ```bash
   pip install --upgrade pip
   pip install torch==2.9.0 torch-npu==2.9.0 torchaudio==2.9.0 \
     -i https://mirrors.huaweicloud.com/repository/pypi/simple
   pip install "transformers==4.57.6"
   pip install -r upstream-npu/requirements.txt
   pip install -r upstream-npu/XY_Tokenizer/requirements.txt
   python - <<'PY'
   import torch
   import torch_npu
   print(torch.__version__, torch.randn(1).to("npu").device)
   PY
   ```

4. （可选）若同一机器需独立 CUDA 环境做对照，可使用 venv 隔离。本地不具备 CUDA 时可跳过本步，仅运行 NPU 组完成迁移验收：

   ```bash
   python3.11 -m venv .venv-cuda
   source .venv-cuda/bin/activate
   pip install --upgrade pip
   pip install torch==2.9.0 torchaudio==2.9.0
   pip install "transformers==4.57.6" flash-attn
   pip install -r upstream-npu/requirements.txt
   pip install -r upstream-npu/XY_Tokenizer/requirements.txt
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

TTSD-eval 支持多种评测 profile，NPU 为必跑项。NPU profile 复用推理环境（`torch/torchaudio==2.9.0 + torch-npu==2.9.0`），并需对 TTSD-eval 工作树应用 `patches/0002-adapt-ttsd-eval-to-npu.patch`。若需 CPU 精度对照，可复用同一环境，仅需将 `--device` 改为 `cpu`。以下命令均从 `ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5` 执行。

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
pip install --upgrade pip
git -C "$EVAL_ROOT" apply "$PWD/patches/0002-adapt-ttsd-eval-to-npu.patch"

# 分步安装评测依赖：
# 1) 先安装全部 PyPI 包（含 onnxruntime）。若整文件安装因 wespeaker
#    git 源不可达而失败，pip 会回退整个事务导致 onnxruntime 等关键依赖
#    缺失，因此将 wespeaker 与其余包分开安装。
grep -v '^wespeaker' requirements_eval.txt \
  | python -m pip install -r /dev/stdin

# 2) 从 GitHub 固定 commit 安装 wespeaker（需 GitHub 可达；
#    若不可达可先配置代理或使用国内镜像）。wespeaker 的 diar 模块在
#    包初始化时无条件导入 onnxruntime，因此 onnxruntime 必须在
#    import wespeaker 之前已安装（上一步已完成）。
python -m pip install \
  "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git@c92349a14d6b426808c4e09b8b12e076864dfc11"

# TorchAudio 2.9+ 移除 torchaudio.sox_effects，s3prl hub.py 全量导入
# mos_prediction/espnet_hubert 时会分别因 sox_effects 移除和 NLTK cmudict
# 损坏而失败；用 patch 包裹这两个 hubconf 导入为 try/except Exception
# NLTK 数据准备（g2p_en 依赖 cmudict 和 averaged_perceptron_tagger）
# 优先使用 curl/wget 下载，不依赖 nltk.download()——NPU 服务器代理可能
# 拦截 NLTK data server 公网请求并静默返回损坏文件（HTML 被保存为 .zip）
NLTK_DATA_DIR=$(python -c "import nltk; print(nltk.data.path[0])")
mkdir -p "$NLTK_DATA_DIR/corpora" "$NLTK_DATA_DIR/taggers"
curl -L --fail -o "$NLTK_DATA_DIR/corpora/cmudict.zip" \
  https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/cmudict.zip
curl -L --fail -o "$NLTK_DATA_DIR/taggers/averaged_perceptron_tagger.zip" \
  https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip
file "$NLTK_DATA_DIR/corpora/cmudict.zip"  # 应显示 Zip archive data
# 若 NLTK 数据目录无写入权限，改用用户目录：
#   mkdir -p ~/nltk_data/corpora ~/nltk_data/taggers
#   curl ... -o ~/nltk_data/corpora/cmudict.zip
#   curl ... -o ~/nltk_data/taggers/averaged_perceptron_tagger.zip
#   export NLTK_DATA=~/nltk_data  # 加入 eval.sh 运行前环境变量
# 若当前机器无法访问 GitHub，参考下方「NLTK 离线准备」章节在可联网机器下载后传输

# TorchAudio 2.9+ 移除 torchaudio.sox_effects，s3prl hub.py 全量导入
# mos_prediction/espnet_hubert 时会分别因 sox_effects 移除和 NLTK cmudict
# 损坏/缺失而失败；用 patch 包裹这两个 hubconf 导入为 try/except Exception
# 若上方 NLTK 数据已正确安装，patch 仅用于兜底 sox_effects 移除问题；
# 若 NLTK 数据无法安装，patch 也可兜底 cmudict 问题
S3PRL_DIR=$(python -c "import importlib.util; print(importlib.util.find_spec('s3prl').submodule_search_locations[0])") && \
cd "$S3PRL_DIR/../" && patch -p1 < "$MODEL_ROOT/patches/0003-fix-s3prl-hub-resilient-imports.patch"

# NPU 不支持 float64；wespeaker SimAM 和 pooling 层中 Python float 标量
# (1e-4, 0.5, 1e-7 等) 被 torch-npu 转为 float64 设备 tensor 后隐式
# 转 float32，触发 double->float 警告并可能导致 aclnnAddV3 崩溃
# (DDR address out of range, error code 507035)
WESPEAKER_DIR=$(python -c "import importlib.util; print(importlib.util.find_spec('wespeaker').submodule_search_locations[0])") && \
cd "$WESPEAKER_DIR/../" && patch -p1 < "$MODEL_ROOT/patches/0004-fix-wespeaker-float64-on-npu.patch"

# NPU 的 torch.fft.rfft().abs() 对复数张量返回全零，导致 WeSpeaker
# fbank 特征全零 → 嵌入全零 → sim=0.0000。手动用 sqrt(real^2+imag^2)
# 替代 .abs()，数学等价。
TORCHAUDIO_PATH=$(python -c "import torchaudio; import os; print(os.path.dirname(torchaudio.__file__))") && \
cd "${TORCHAUDIO_PATH}/../" && \
patch -p1 < "$MODEL_ROOT/patches/0005-fix-torchaudio-kaldi-rfft-abs-on-npu.patch"
python -m pip check
python -m pip freeze > results/ttsd_eval_setup/evaluator-pip-freeze.txt
python -c 'import torch, torch_npu; print(torch.__version__, torch.npu.is_available(), torch.npu.device_count())'
python -c 'import onnxruntime; print("onnxruntime", onnxruntime.__version__)'
python -c 'import wespeaker; print("wespeaker ok")'
python -c "import nltk; nltk.data.find('corpora/cmudict.zip'); print('cmudict ok')"
python -c "
import torchaudio.compliance.kaldi as k
import inspect
src = inspect.getsource(k.fbank)
assert 'fft.real' in src and 'fft.imag' in src, 'torchaudio kaldi fbank patch NOT applied'
print('torchaudio kaldi fbank patch verified')
"
deactivate
```

> **注意**：`wespeaker` 的 diar 模块在包初始化时无条件导入 `onnxruntime`（`wespeaker.__init__` → `cli.speaker` → `diar.extract_emb` → `onnxruntime`），即使相似度评测仅使用说话人嵌入模型也需要该依赖。若 `onnxruntime` 未安装，`import wespeaker` 或 `run_similarity.py` 将报 `ModuleNotFoundError: No module named 'onnxruntime'`。上面两条 `python -c` 命令用于确认两者均可用；若 `onnxruntime` 缺失，单独执行 `python -m pip install onnxruntime==1.23.2` 补装即可。
>
> **onnxruntime 在 NPU 评测中的角色**：`onnxruntime` 在本评测中 **仅用于满足 `import wespeaker` 的无条件依赖**，不参与任何实际推理计算。`run_similarity.py` 只调用 `Speaker.compute_similarity()` → `Speaker.extract_embedding()` → PyTorch 前向推理（在 NPU 上执行），**从未调用** `Speaker.diarize()` 或 `diar.extract_emb` 中的 ONNX 推理路径。因此安装标准 CPU 版 `onnxruntime` 即可（`pip install onnxruntime==1.23.2`），无需 `onnxruntime-cann`（CANN EP）或 CUDA EP。标准 CPU wheel 不依赖 GPU/NPU 驱动，与 `torch-npu` 无冲突。
>
> 安装命令已将 PyPI 包（含 `onnxruntime`）与 `wespeaker` git 源分开安装，避免 wespeaker git 源不可达时 pip 回退整个事务导致 `onnxruntime` 等依赖缺失。若 GitHub 不可达导致 wespeaker 安装失败，可配置代理后重试，或将 wespeaker 仓库镜像至可访问的 Git 服务后修改安装 URL；最终以 `pip check` 和上述 import 验证通过为准。

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
python -c 'import onnxruntime; print("onnxruntime", onnxruntime.__version__)'
python -c 'import wespeaker; print("wespeaker ok")'
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

2. 下载固定 S3 version ID 的 MMS-FA checkpoint。

   `align.py` 通过 `torch.hub.set_dir(cache_dir)` 将 hub 目录指向 `model/`，`load_state_dict_from_url` 在 `model/checkpoints/model.pt` 查找已缓存文件；文件必须完整且 SHA256 匹配，否则 `torch.load` 会报 `PytorchStreamReader failed reading zip archive` 错误。

   以下下载命令包含 SHA256 和文件大小校验；若校验失败，删除后重新下载：

   ```bash
   EVAL_ROOT="$PWD/third_party/TTSD-eval"
   mkdir -p "$EVAL_ROOT/model/checkpoints"
   MMS_PT="$EVAL_ROOT/model/checkpoints/model.pt"
   rm -f "$MMS_PT"
   curl -L --fail --retry 5 --retry-all-errors \
     -o "$MMS_PT" \
     "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt?versionId=dZWoHyjLHoCxDn.KL1FPSlVCD3CPRtOL"
   echo "20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680  $MMS_PT" \
     | sha256sum -c -
   test "$(stat -c %s "$MMS_PT")" = "1262047414"
   # 清理可能存在的错误位置同名文件（上游 README 的 wget 命令可能将 model.pt
   # 写到 model/ 根目录而非 model/checkpoints/，导致混淆）
   test -f "$EVAL_ROOT/model/model.pt" && rm -f "$EVAL_ROOT/model/model.pt"
   ```

3. 下载固定 revision 的 Whisper-large-v3，并写入 revision marker。正式评测只从本地目录离线加载。以下使用 NPU 环境（必跑）；CPU 精度对照将 `--device npu` 改为 `--device cpu`：

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
   import onnxruntime
   import wespeaker
   from torchaudio.pipelines import MMS_FA
   from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

   root = os.environ["EVAL_ROOT"]
   print("onnxruntime", onnxruntime.__version__)

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

迁移验收以 NPU 推理为必跑项；精度对比优先使用公开/官方指标，当无公开指标时使用 CPU 精度对照（`--device cpu`，复用同一环境）确认 NPU 适配正确性。本地不具备 CUDA 时可跳过 CUDA 对照组，仅执行 NPU 推理即可完成功能验证。CPU 精度对照为可选项。

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

4. 检查已生成组的输出 WAV。NPU 组必查；若运行了 CUDA 对照组，再比较原始与 patch 后 CUDA 的同设备回归，以及 patch 后 CUDA 与 NPU。

   ```bash
   python - <<'PY'
   from pathlib import Path
   import soundfile as sf

   groups = {
       "original_cuda": Path("upstream-original/outputs_original_cuda"),
       "patched_cuda": Path("upstream-npu/outputs_patched_cuda"),
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
done
```

推理完成为已生成的组生成 evaluator manifest（仅处理实际存在的输出目录）。工具会检查每个 `output_N.wav` 是否存在，并写入 manifest SHA256：

```bash
for LANG in zh en; do
  for GROUP in original_cuda patched_cuda npu; do
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

NPU 适配 patch 已为 `eval.sh` 和 `run_wer.sh` 增加环境变量 `DEVICE`、`CACHE_DIR`、`MODEL_DIR`、`WHISPER_MODEL_ID`、`INPUT_JSONL_PATH`、`LANGUAGE` 支持，使其可在 NPU 设备上直接 `bash` 执行原始评测脚本。`SCRIPT_DIR` 保持原项目 `"$0"` 不变——直接 `bash eval.sh` 时 `$0` 自然指向脚本自身，路径正确；不使用 `source` 是因为 `source` 时 `$0` 为父 shell 导致 `SCRIPT_DIR` 解析错误，且 `set -euo pipefail` / `trap` 会污染父进程。`INPUT_JSONL_PATH` 是标量环境变量（可 export），解决了 bash 数组不能跨进程传递的问题。以下命令从 `ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5` 执行。

#### NPU evaluator（必跑）

使用原项目 `eval.sh` 运行 ACC/SIM 评测，`run_wer.sh` 运行 WER 评测：

```bash
set -o pipefail
MODEL_ROOT="$PWD"
EVAL_ROOT="$MODEL_ROOT/third_party/TTSD-eval"
source "$MODEL_ROOT/.venv-npu/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- 预检：确认评测关键依赖可正常 import ----
# 若任一项失败，按「常见故障」中 Step 3 条目排查后再继续。
python - <<'PY'
import sys
ok = True
for mod in ("onnxruntime", "wespeaker", "torch", "torch_npu", "s3prl", "whisper"):
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "ok")
        print(f"  {mod}: {ver}")
    except Exception as exc:
        ok = False
        print(f"  {mod}: IMPORT FAILED -> {exc}", file=sys.stderr)
if not ok:
    sys.exit("\nFATAL: one or more evaluator dependencies failed to import. "
            "Fix before running eval.sh / run_wer.sh.")
PY
# ---- 预检结束 ----

for LANG in zh en; do
  for GROUP in npu; do
    INPUT="$MODEL_ROOT/results/ttsd_eval/${GROUP}_${LANG}.jsonl"
    [ -f "$INPUT" ] || continue
    STEM="${GROUP}_${LANG}"
    RUN_ROOT="$MODEL_ROOT/results/ttsd_eval_metrics/$STEM"
    mkdir -p "$RUN_ROOT"

    {
      # --- ACC/SIM via eval.sh ---
      # eval.sh 内部按 alignment → split → similarity 顺序调用，
      # 输出写到 TTSD-eval/output/ 并以时间戳命名。
      # 直接 bash 执行，$0 指向脚本自身，SCRIPT_DIR 正确。
      # 所有参数通过 export 的标量环境变量传递（INPUT_JSONL_PATH
      # 代替不可 export 的 bash 数组），无需 source。
      export DEVICE=npu
      export CACHE_DIR="$EVAL_ROOT/model"
      export MODEL_DIR="$EVAL_ROOT/model/voxblink2_samresnet100_ft"
      export INPUT_JSONL_PATH="$INPUT"
      bash "$EVAL_ROOT/eval.sh"

      # --- WER via run_wer.sh ---
      export DEVICE=npu
      export WHISPER_MODEL_ID="$EVAL_ROOT/model/whisper-large-v3"
      export INPUT_JSONL_PATH="$INPUT"
      export LANGUAGE="$LANG"
      bash "$EVAL_ROOT/run_wer.sh"
    } 2>&1 | tee "$RUN_ROOT/evaluator.log"
  done
done

deactivate
```

> **说明**：`eval.sh` 和 `run_wer.sh` 是 TTSD-eval 官方评测脚本，NPU 适配 patch 为其增加 `DEVICE` / `CACHE_DIR` / `MODEL_DIR` / `WHISPER_MODEL_ID` / `INPUT_JSONL_PATH` / `LANGUAGE` 环境变量透传，不改变脚本内部流程与默认输出路径。`INPUT_JSONL_PATH` 是标量环境变量（可 export），设置后优先于原脚本中的 `INPUT_JSONL` 数组默认值；`LANGUAGE` 同理优先于 `language` 默认值。原始脚本不设置这些环境变量时行为与上游完全一致（auto-detect CUDA/CPU，使用 `data/example/output.jsonl` 和 `language=zh`）。输出文件位于 `TTSD-eval/output/` 目录，以 `<input_stem>_<timestamp>` 命名。若需指定自定义输出路径，可直接调用底层 Python 工具并传显式参数。

> **常见故障**：
>
> - `Step 1: Alignment` 在 NPU 上报 `CAUTION: The operator 'torchaudio::forced_align' is not currently supported on the NPU backend and will fall back to run on the CPU`，表示 `torchaudio::forced_align` 自定义 C++ 算子无 NPU 后端实现。NPU 适配 patch 已在 `tools/align.py` 的 `_compute_alignments` 方法中显式将 emission 张量从 NPU 转移至 CPU 后再调用 aligner（`forced_align` 及后续 CTC 对齐运算均在 CPU 完成），wav2vec2 前向推理仍在 NPU 执行，不影响精度。若未应用最新 patch 或手动修改，该算子的 npu_cpu_fallback 机制可能导致设备不匹配或对齐结果异常。
>
> - `Step 1: Alignment` 报 `PytorchStreamReader failed reading zip archive: failed finding central directory`，表示 `CACHE_DIR/checkpoints/model.pt`（MMS-FA 权重）损坏或下载不完整。修复方法：删除后重新下载并校验 SHA256 和文件大小，参见「下载固定 S3 version ID 的 MMS-FA checkpoint」步骤。
>
> - `Step 3: Similarity` 报 `ModuleNotFoundError: No module named 'onnxruntime'`，表示 `onnxruntime` 未安装或其原生扩展库加载失败。`wespeaker` 的 diar 模块在包初始化时无条件导入 `onnxruntime`（导入链：`wespeaker.__init__` → `cli.speaker` → `diar.extract_emb` → `onnxruntime`），即使相似度评测仅使用说话人嵌入模型也需要该依赖。**在本评测中 `onnxruntime` 仅用于满足 import，不参与任何实际推理**——`run_similarity.py` 只调用 `Speaker.compute_similarity()` → PyTorch 前向推理（NPU），从未调用 `Speaker.diarize()` 或 ONNX 推理路径。因此安装标准 CPU 版 `onnxruntime` 即可，无需 `onnxruntime-cann`（CANN EP）或 CUDA EP。按以下步骤逐项排查：
>
>   1. **确认当前环境**：必须激活评测所用的 venv 再执行诊断，否则 `pip list` 和 `import` 可能指向不同的 Python 环境。执行 `which python && python -c "import sys; print(sys.executable, sys.prefix)"` 确认二进制路径和 venv 前缀一致。
>
>   2. **确认 onnxruntime 是否可 import**：`python -c "import onnxruntime; print(onnxruntime.__version__)"`。若此命令报错，即使 `pip list` 显示已安装，也说明原生 `.so` 无法 dlopen，继续下一步。
>
>   3. **查看详细加载错误**：`python -c "import importlib; importlib.import_module('onnxruntime')"`。此命令会显示完整的 `ImportError` 堆栈（包括缺少的 `.so` 依赖、GLIBC 版本不匹配或 CPU 架构 wheel 不一致等根因），而非 `wespeaker` 层面被截断的 `ModuleNotFoundError`。
>
>   4. **重装固定版本**：在激活的 venv 中执行 `python -m pip install --force-reinstall onnxruntime==1.23.2`。`--force-reinstall` 强制重新下载 wheel 并覆盖损坏的安装。若 1.23.2 wheel 与当前平台（aarch64 / x86_64）或 glibc 不兼容，可尝试 `python -m pip install --force-reinstall onnxruntime`（不固定版本）让 pip 自动选择兼容 wheel。
>
>   5. **验证**：`python -c "import onnxruntime; print(onnxruntime.__version__)"`。版本应与 `requirements_eval.txt` 固定值一致（NPU profile 为 1.23.2）；若因平台限制安装了其他版本，记录实际版本并在 `pip check` 通过后继续。
>
>   常见根因：① `pip install -r requirements_eval.txt` 因 wespeaker git 源不可达而整体回退，onnxruntime 未实际安装——安装命令已将 PyPI 包与 wespeaker 分开安装以避免此问题；② onnxruntime wheel 与平台 glibc/libstdc++ 不兼容（`importlib.import_module` 报 `OSError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.30 not found` 等错误），需升级系统 libstdc++ 或安装兼容版本的 onnxruntime；③ pip 安装在系统 Python 而评测运行于 venv（或反过来），二者 site-packages 不互通。
>
> - `Step 3: Similarity` 报 `zipfile.BadZipFile: File is not a zip file`（或 `import wespeaker` / `run_similarity.py` 报 `BadZipFile`），表示 NLTK `cmudict.zip` 数据文件损坏或缺失，导致 `g2p_en` 模块导入失败。完整导入链：`wespeaker → s3prl.frontend → s3prl.nn → s3prl.hub → espnet_hubert.hubconf → espnet2.tasks.hubert → espnet2.text.phoneme_tokenizer → g2p_en → nltk.data.find('corpora/cmudict.zip')`。`nltk.data.find()` 找到 zip 文件后会尝试打开并验证，若文件损坏则抛出 `BadZipFile`（`Exception` 子类）；`g2p_en` 自身的 `except LookupError` 无法捕获，导致整个 `import wespeaker` 链崩溃。TTSD-eval 相似度评测不使用 `espnet_hubert` 和 `g2p_en`。
>
>   **修复方法一（推荐，修复根因）**：按上方「创建评测环境」NPU profile 中的 `curl` 命令下载正确的 NLTK 数据文件并安装。NPU 服务器代理可能拦截 NLTK data server 公网请求并静默返回损坏文件，**不要使用 `nltk.download()` 下载**——必须使用 `curl`/`wget`。若当前机器无法访问 GitHub，参考下方「NLTK cmudict.zip 损坏或缺失（离线环境）」章节在可联网机器下载后传输。
>
>   **修复方法二（补丁兜底）**：若无法安装正确的 NLTK 数据，应用 `0003-fix-s3prl-hub-resilient-imports.patch`，使 `espnet_hubert` 导入失败时静默跳过，无需修复 `cmudict.zip` 即可让评测正常运行：
>
>   ```bash
>   S3PRL_DIR=$(python -c "import importlib.util; print(importlib.util.find_spec('s3prl').submodule_search_locations[0])") && \
>   cd "$S3PRL_DIR/../" && patch -p1 < "$MODEL_ROOT/patches/0003-fix-s3prl-hub-resilient-imports.patch"
>   python -c "import wespeaker; print('wespeaker import ok')"
>   ```
>
>   **修复方法二（补丁兜底，不修复根因）**：若无法安装正确的 NLTK 数据，应用 `0003-fix-s3prl-hub-resilient-imports.patch`，使 `espnet_hubert` 导入失败时静默跳过。详见下方「NLTK cmudict.zip 损坏或缺失（离线环境）」条目。
>
> - `import wespeaker` 或 `run_similarity.py` 报 `ModuleNotFoundError: No module named 'torchaudio.sox_effects'`，表示 TorchAudio 2.9+ 已移除 SoX 后端，而 `s3prl==0.4.18` 的 `upstream/mos_prediction/expert.py` 仍顶层导入 `from torchaudio.sox_effects import apply_effects_tensor`（该函数在 expert.py 中实际未被调用）。`s3prl/hub.py` 对全部 upstream 做星号导入，导致 import 链 `wespeaker → s3prl → hub → mos_prediction → torchaudio.sox_effects` 必定失败。**此问题仅出现在 NPU profile（TorchAudio 2.9.0），CPU/CUDA profile 使用 TorchAudio 2.8.0 不受影响。**
>
>   上述 `0003` 补丁已同时包裹 `espnet_hubert` 和 `mos_prediction` 两个 hubconf 导入，应用一次即可修复 `BadZipFile` 和 `torchaudio.sox_effects` 两个问题。若未应用该补丁，按上面「修复方法一」执行即可。若 `pip install` 重装 s3prl 导致修改被覆盖，重新应用补丁即可。
>
> - `Step 3: Similarity` 报 `aclnnAddV3 failed, error code is 507035` 及 `The DDR address of the MTE instruction is out of range`，前面伴有 `Device do not support double dtype now, dtype cast replace with float` 警告。根因：wespeaker 的 `SimAM` 注意力模块和 ASP 等 pooling 层在 tensor 运算中使用 Python float 标量（`1e-4`、`0.5`、`1e-7`、`1e-5` 等），CPython 中这些均为 float64；torch-npu 将其转为 float64 设备 tensor 后隐式转换为 float32，转换后的 tensor 内存布局异常导致 `aclnnAddV3` 算子访问越界崩溃。完整导入链：`SimAMBasicBlock.forward → SimAM → v + lambda_p(1e-4) → float64 设备 tensor → 隐式 cast → float32 → out += self.downsample(x) → aclnnAddV3 崩溃`。
>
>   **修复**：应用 `0004-fix-wespeaker-float64-on-npu.patch`，将所有 Python float 标量替换为显式 float32 设备 tensor：
>
>   ```bash
>   WESPEAKER_DIR=$(python -c "import importlib.util; print(importlib.util.find_spec('wespeaker').submodule_search_locations[0])") && \
>   cd "$WESPEAKER_DIR/../" && patch -p1 < "$MODEL_ROOT/patches/0004-fix-wespeaker-float64-on-npu.patch"
>   python -c "import wespeaker; from wespeaker.models.samresnet import SimAMBasicBlock; print('patch ok')"
>   ```
>
>   若 `pip install` 重装 wespeaker 导致修改被覆盖，重新执行上述命令即可。
>
> - `Step 3: Similarity` 所有 case 的 `sim=0.0000`（ACC 正常），表示 WeSpeaker 嵌入提取产出了全零向量。根因：NPU `torch.fft.rfft()` 返回复数张量后 `.abs()` 返回全零（NPU FFT backend 对复数取模 `.abs()` 支持不完整），导致 `torchaudio.compliance.kaldi.fbank()` 提取的 fbank 特征全零 → 嵌入向量全零 → 余弦相似度恒为 0.0000。ACC 不受影响（仅依赖文本对齐结果）。
>
>   **修复**：应用 `0005-fix-torchaudio-kaldi-rfft-abs-on-npu.patch`，将 `rfft().abs()` 替换为手动实部/虚部分解 `sqrt(real^2 + imag^2)`（数学等价）：
>
>   ```bash
>   TORCHAUDIO_PATH=$(python -c "import torchaudio; import os; print(os.path.dirname(torchaudio.__file__))") && \
>   cd "${TORCHAUDIO_PATH}/../" && \
>   patch -p1 < "$MODEL_ROOT/patches/0005-fix-torchaudio-kaldi-rfft-abs-on-npu.patch"
>   # 验证
>   python -c "import torchaudio.compliance.kaldi as k; import inspect; src = inspect.getsource(k.fbank); assert 'fft.real' in src; print('ok')"
>   ```
>
>   若 `patch` 命令不可用，也可手动修改 `${TORCHAUDIO_PATH}/compliance/kaldi.py`：
>   - 约 311 行：将 `fft.abs().pow(2.0)` 改为先计算 `fft_abs = torch.sqrt(fft.real.pow(2.0) + fft.imag.pow(2.0))` 再用 `fft_abs.pow(2.0)`
>   - 约 616 行：将 `spectrum = torch.fft.rfft(strided_input).abs()` 改为 `_rfft = torch.fft.rfft(strided_input); spectrum = torch.sqrt(_rfft.real.pow(2.0) + _rfft.imag.pow(2.0))`
>
>   此问题仅影响 NPU profile；CPU/CUDA 的 `.abs()` 正常工作。若 `pip install --force-reinstall torchaudio` 导致修改被覆盖，重新应用补丁即可。
>
> **NLTK cmudict.zip 损坏或缺失（离线环境）**
>
> 若环境需要 `g2p_en` 用于其他用途（非 TTSD-eval），或希望修复 `cmudict.zip` 根因，按以下离线步骤操作。TTSD-eval 用户只需应用 `0003` 补丁即可，无需修复 `cmudict.zip`。
>
> 1. **在可联网机器上下载 NLTK 数据包**（优先使用 curl，不使用 `nltk.download()`——NPU 服务器代理可能拦截 NLTK data server 公网请求并静默返回损坏文件）：
>
>    ```bash
>    # 从 GitHub 直接下载（优先方式，不依赖 NLTK data server）
>    mkdir -p ~/nltk_offline
>    curl -L --fail -o ~/nltk_offline/cmudict.zip \
>      https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/cmudict.zip
>    # 同样下载 POS tagger（g2p_en 也依赖）
>    curl -L --fail -o ~/nltk_offline/averaged_perceptron_tagger.zip \
>      https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip
>    ```
>
> 2. **计算校验值并打包传输**：
>
>    ```bash
>    cd ~/nltk_offline
>    sha256sum cmudict.zip averaged_perceptron_tagger.zip > checksums.txt
>    cat checksums.txt
>    tar czf nltk_data_offline.tar.gz cmudict.zip averaged_perceptron_tagger.zip checksums.txt
>    ```
>
> 3. **传输至目标机器**（scp / USB / 共享目录等）：
>
>    ```bash
>    scp nltk_data_offline.tar.gz user@npu-server:/tmp/
>    ```
>
> 4. **在目标机器上安装**（需激活评测 venv）：
>
>    ```bash
>    source .venv-npu/bin/activate
>
>    # 确认 NLTK 数据目录
>    NLTK_DATA_DIR=$(python -c "import nltk; print(nltk.data.path[0])")
>    echo "NLTK data dir: $NLTK_DATA_DIR"
>
>    # 解压离线包
>    cd /tmp && tar xzf nltk_data_offline.tar.gz
>    sha256sum -c checksums.txt
>
>    # 删除损坏文件并安装正确的数据
>    mkdir -p "$NLTK_DATA_DIR/corpora" "$NLTK_DATA_DIR/taggers"
>    rm -f "$NLTK_DATA_DIR/corpora/cmudict.zip" "$NLTK_DATA_DIR/taggers/averaged_perceptron_tagger.zip"
>    cp cmudict.zip "$NLTK_DATA_DIR/corpora/"
>    cp averaged_perceptron_tagger.zip "$NLTK_DATA_DIR/taggers/"
>
>    # 验证
>    python -c "import nltk; nltk.data.find('corpora/cmudict.zip'); print('cmudict ok')"
>    python -c "import g2p_en; print('g2p_en ok')"
>    deactivate
>    ```
>
>    若 NLTK 数据目录无写入权限（如 `/usr/local/python3.11.14/lib/python3.11/nltk_data`），使用用户目录并设置环境变量：
>
>    ```bash
>    mkdir -p ~/nltk_data/corpora ~/nltk_data/taggers
>    cp cmudict.zip ~/nltk_data/corpora/
>    cp averaged_perceptron_tagger.zip ~/nltk_data/taggers/
>    export NLTK_DATA=~/nltk_data
>    # 将 export NLTK_DATA=~/nltk_data 加入 eval.sh 运行前的环境变量中
>    ```
>
>    若 GitHub 也不可访问，可从国内镜像下载（如 gitee 镜像 `nltk_data` 仓库），或将可联网机器上已确认可用的 `~/nltk_data/corpora/` 和 `~/nltk_data/taggers/` 目录整个打包传输。注意：**不要使用 `nltk.download()` 在 NPU 服务器上下载**——代理拦截后返回的 HTML 错误页会被保存为 `.zip`，导致 `BadZipFile`。

#### CUDA / CPU evaluator（可选对照）

本地具备 CUDA 环境时，可用相同脚本做同口径对照，只需将 `DEVICE` 改为 `cuda` 或 `cpu`：

```bash
# CUDA 对照示例（需 .venv-ttsd-eval 环境与 CUDA 驱动）
source "$MODEL_ROOT/.venv-ttsd-eval/bin/activate"
export DEVICE=cuda
export CACHE_DIR="$EVAL_ROOT/model"
export MODEL_DIR="$EVAL_ROOT/model/voxblink2_samresnet100_ft"
export WHISPER_MODEL_ID="$EVAL_ROOT/model/whisper-large-v3"
export INPUT_JSONL_PATH="$INPUT"
bash "$EVAL_ROOT/eval.sh"
export LANGUAGE="$LANG"
bash "$EVAL_ROOT/run_wer.sh"
deactivate

# CPU 对照示例
source "$MODEL_ROOT/.venv-ttsd-eval/bin/activate"
export DEVICE=cpu
export CACHE_DIR="$EVAL_ROOT/model"
export MODEL_DIR="$EVAL_ROOT/model/voxblink2_samresnet100_ft"
export WHISPER_MODEL_ID="$EVAL_ROOT/model/whisper-large-v3"
export INPUT_JSONL_PATH="$INPUT"
bash "$EVAL_ROOT/eval.sh"
export LANGUAGE="$LANG"
bash "$EVAL_ROOT/run_wer.sh"
deactivate
```

TTSD-eval 的部分工具会记录 warning 后跳过失败样本并以 0 退出；如需对每个阶段做 50 行门禁检查，可直接调用底层 Python 工具并逐阶段校验，参见 patch 后 `tools/align.py`、`tools/split.py`、`tools/run_similarity.py`、`wer/whisper_asr.py`、`wer/run_wer.py` 的 `--help`。

## 模型推理性能

MOSS-TTSD-v0.5 属自回归生成式 TTS/TTSD 模型，L2 性能以中英文全量生成音频总时长和端到端墙钟时间计算。以下展示 NPU 中文命令（必跑）；英文 split 使用相同参数和独立日志/输出目录。本地具备 CUDA 时可额外运行原始 CUDA、patch 后 CUDA 对照组，但不作为强制要求：

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
