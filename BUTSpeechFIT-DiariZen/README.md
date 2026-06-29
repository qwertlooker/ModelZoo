# DiariZen 推理指导

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [文件目录](#文件目录)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备权重](#准备权重)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能](#模型推理性能)
- [公网地址说明](#公网地址说明)

## 概述

DiariZen 是 BUT-FIT 发布的说话人日志分轨模型，基于 WavLM 分割网络与 WeSpeaker embedding，输出 RTTM。本文档介绍该模型基于昇腾 NPU 的推理指导。

> 说明：本文档适配对象为 `BUT-FIT/diarizen-wavlm-large-s80-md` checkpoint，不包含 `large-s80-md-v2`、base 或 pruning checkpoint。

- 版本说明：

  ```text
  url=https://github.com/BUTSpeechFIT/DiariZen
  commit_id=a60b18151dbbe246e4199d8ef5cd2ece3872ea94
  model_name=DiariZen
  model=BUT-FIT/diarizen-wavlm-large-s80-md@a9b1b0e7974d96dcfd63af417e9da7ad8714040f
  embedding=pyannote/wespeaker-voxceleb-resnet34-LM@837717ddb9ff5507820346191109dc79c958d614
  dscore=nryant/dscore@e02f949ac6592279300a2c33d03daf9e0c12fd27
  reference=Ascend-SACT/BUTSpeechFIT-DiariZen@7961b5ab79b1232b9da367f14f8cd4f592694465
  ```

## 输入输出数据

- 输入数据

  支持一个或多个音频文件，或包含 `id`/`audio_path` 字段的 JSONL manifest。

- 输出数据

  输出为每个 session 一个 RTTM 文件，以及 `run.meta.json`。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

  | 配套 | 版本/要求 |
  |---|---|
  | 硬件 | 支持目标 CANN 的 Atlas 推理服务器 |
  | CANN、驱动、固件 | CANN 8.2.0 及其配套驱动/固件 |
  | Python | 3.10 |
  | PyTorch / torchaudio / torch-npu | 2.5.1 |
  | ONNX Runtime | CPU：`onnxruntime==1.22.1`；NPU：`onnxruntime-cann==1.22.1` |
  | NumPy | 1.26.4 |

  说明：Atlas 800I A2 推理卡请以 CANN 版本选择实际固件与驱动版本。

## 文件目录

```text
BUTSpeechFIT-DiariZen
├── infer.py                            # 推理脚本
├── prepare_eval_data.py                # 评测数据准备脚本
├── score_diarization.py                # DER 评测脚本
├── patches/0001-add-explicit-npu-pipeline-device.patch
├── requirements.txt
└── README.md                           # 推理指导文档
```

> 说明：
> - `prepare_eval_data.py`：将 Kaldi 风格 wav.scp + RTTM/UEM 转换为 `infer.py`
>   所需的 JSONL manifest，并校验音频可读性、session ID 一致性和文件 SHA256。
>   上游 DiariZen 社区通用 wav.scp 格式，本脚本是到 NPU 推理入口的桥接层。
> - `score_diarization.py`：封装固定版本 dscore（`nryant/dscore@e02f949`），
>   确保 `--ignore_overlaps` 作为 `store_true` 开关被正确处理，避免误写
>   `--ignore_overlaps false` 导致语义反转；同时输出评测配置 metadata。

## 快速上手

### 获取源码

1. 获取源码并应用适配补丁。

   ```bash
   git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git source
   git -C source checkout a60b18151dbbe246e4199d8ef5cd2ece3872ea94
   git -C source submodule update --init --recursive
   git -C source worktree add --detach ../upstream-original \
     a60b18151dbbe246e4199d8ef5cd2ece3872ea94
   git -C source worktree add --detach ../upstream-npu \
     a60b18151dbbe246e4199d8ef5cd2ece3872ea94
   git -C upstream-original submodule update --init --recursive
   git -C upstream-npu submodule update --init --recursive
   git -C upstream-npu apply --check \
     ../patches/0001-add-explicit-npu-pipeline-device.patch
   git -C upstream-npu apply \
     ../patches/0001-add-explicit-npu-pipeline-device.patch
   ```

2. 创建并安装 CPU 原始环境（需独立 venv，`onnxruntime` 与 `onnxruntime-cann` 不可共存，且原始/patched 的 editable 安装需要隔离）。

   ```bash
   python3.10 -m venv .venv-cpu-original
   source .venv-cpu-original/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.5.1 torchaudio==2.5.1 \
     --index-url https://download.pytorch.org/whl/cpu
   python -m pip install onnxruntime==1.22.1
   python -m pip install -r upstream-original/pyannote-audio/requirements.txt
   python -m pip install -r upstream-original/dscore/requirements.txt
   python -m pip install -r requirements.txt
   python -m pip install -e upstream-original/pyannote-audio --no-deps
   python -m pip install -e upstream-original --no-deps
   deactivate
   ```

3. 创建并安装 CPU patch 后环境。

   ```bash
   python3.10 -m venv .venv-cpu-patched
   source .venv-cpu-patched/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.5.1 torchaudio==2.5.1 \
     --index-url https://download.pytorch.org/whl/cpu
   python -m pip install onnxruntime==1.22.1
   python -m pip install -r upstream-npu/pyannote-audio/requirements.txt
   python -m pip install -r upstream-npu/dscore/requirements.txt
   python -m pip install -r requirements.txt
   python -m pip install -e upstream-npu/pyannote-audio --no-deps
   python -m pip install -e upstream-npu --no-deps
   deactivate
   ```

4. 创建并安装 NPU 环境。

   ```bash
   python3.10 -m venv .venv-npu
   source .venv-npu/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.5.1 torchaudio==2.5.1 torch-npu==2.5.1 \
     -i https://mirrors.huaweicloud.com/repository/pypi/simple
   python -m pip install onnxruntime-cann==1.22.1
   python -m pip install -r upstream-npu/pyannote-audio/requirements.txt
   python -m pip install -r upstream-npu/dscore/requirements.txt
   python -m pip install -r requirements.txt
   python -m pip install -e upstream-npu/pyannote-audio --no-deps
   python -m pip install -e upstream-npu --no-deps
   ```

5. 执行 NPU 导入门禁检查。

   ```bash
   python - <<'PY'
   import onnxruntime as ort
   import torch
   import torch_npu
   from diarizen.pipelines.inference import DiariZenPipeline
   assert "CANNExecutionProvider" in ort.get_available_providers()
   print(torch.__version__, ort.__version__, torch.randn(1).to("npu").device)
   PY
   ```

### 准备权重

1. 下载主模型和 embedding 权重。

   主模型地址：`https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md`
   embedding 地址：`https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM`

   **在线路径**：

   ```bash
   huggingface-cli download BUT-FIT/diarizen-wavlm-large-s80-md \
     --revision a9b1b0e7974d96dcfd63af417e9da7ad8714040f \
     --local-dir weights/diarizen-wavlm-large-s80-md
   huggingface-cli download pyannote/wespeaker-voxceleb-resnet34-LM \
     pytorch_model.bin \
     --revision 837717ddb9ff5507820346191109dc79c958d614 \
     --local-dir weights/wespeaker-voxceleb-resnet34-LM

   find weights -type f -print0 | sort -z | xargs -0 sha256sum
   ```

   **离线替代**（在可联网机器预下载后传输到 NPU 服务器）：

   ```bash
   mkdir -p weights/diarizen-wavlm-large-s80-md/plda
   DZ_BASE="https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md/resolve/a9b1b0e7974d96dcfd63af417e9da7ad8714040f"
   curl -L --fail -o weights/diarizen-wavlm-large-s80-md/config.toml "$DZ_BASE/config.toml"
   curl -L --fail -o weights/diarizen-wavlm-large-s80-md/pytorch_model.bin "$DZ_BASE/pytorch_model.bin"
   curl -L --fail -o weights/diarizen-wavlm-large-s80-md/plda/plda $DZ_BASE/plda/plda
   curl -L --fail -o weights/diarizen-wavlm-large-s80-md/plda/mean.vec "$DZ_BASE/plda/mean.vec"
   curl -L --fail -o weights/diarizen-wavlm-large-s80-md/plda/transform.mat "$DZ_BASE/plda/transform.mat"

   mkdir -p weights/wespeaker-voxceleb-resnet34-LM
   WS_BASE="https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/837717ddb9ff5507820346191109dc79c958d614"
   curl -L --fail -o weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin "$WS_BASE/pytorch_model.bin"
   curl -L --fail -o weights/wespeaker-voxceleb-resnet34-LM/config.yaml "$WS_BASE/config.yaml"

   find weights -type f -print0 | sort -z | xargs -0 sha256sum
   ```

   主模型目录必须包含 `config.toml`、`pytorch_model.bin` 和 `plda/`。

### 准备数据集

1. 准备功能验证样例。

   ```bash
   mkdir -p eval_data/functional
   printf 'EN2002a %s\n' "$PWD/source/example/EN2002a_30s.wav" \
     > eval_data/functional/wav.scp
   python prepare_eval_data.py \
     --wav_scp eval_data/functional/wav.scp \
     --output_manifest eval_data/functional/manifest.jsonl \
     --dataset upstream-example \
     --split functional
   ```

   参数说明：

   - `wav_scp`：每行一条 `session_id` 与 `audio_path` 的音频列表文件。
   - `output_manifest`：生成的 JSONL manifest 路径。
   - `dataset`：数据集名称标签。
   - `split`：数据集 split 标签。

2. 准备 AMI-SDM 评测数据。

   AMI 语料库需在 <https://groups.inf.ed.ac.uk/ami/corpus/> 注册后下载。
   下载后使用 Kaldi 工具或上游 DiariZen 数据处理脚本生成以下文件：

   ```text
   eval_data/ami/wav.scp          # 每行：<session_id> <absolute_audio_path>
   eval_data/ami/reference.rttm    # NIST RTTM 格式 reference
   eval_data/ami/all.uem           # 可选：UEM 文件
   ```

   期望的 wav.scp 格式（可直接从 DiariZen 上游提供的 AMI recipe 获得）：

   ```text
   IS1000a /path/to/ami/IS1000a.wav
   IS1000b /path/to/ami/IS1000b.wav
   ...
   ```

   reference RTTM 格式示例：

   ```text
   SPEAKER IS1000a 1 0.000 5.123 <NA> <NA> A <NA> <NA>
   SPEAKER IS1000a 1 5.500 3.200 <NA> <NA> B <NA> <NA>
   ...
   ```

   若 AMI 数据暂时不可取得，可先用上游示例音频完成功能验证（2.1 节），
   正式 DER 对齐留待数据就绪后补验。

3. 从上述文件生成正式评测 manifest。

   ```bash
   python prepare_eval_data.py \
     --wav_scp eval_data/ami/wav.scp \
     --reference_rttm eval_data/ami/reference.rttm \
     --uem eval_data/ami/all.uem \
     --output_manifest eval_data/ami/manifest.jsonl \
     --dataset AMI \
     --split SDM-eval
   ```

   参数说明：

   - `reference_rttm`：reference RTTM 文件路径。
   - `uem`：UEM 文件路径，可选。

### 模型推理

1. 执行未应用 patch 的原始 CPU baseline 推理。

   ```bash
   source .venv-cpu-original/bin/activate
   mkdir -p results/original_cpu
   python - <<'PY'
   from pathlib import Path
   from diarizen.pipelines.inference import DiariZenPipeline

   pipeline = DiariZenPipeline(
       diarizen_hub=Path("weights/diarizen-wavlm-large-s80-md"),
       embedding_model="weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin",
       rttm_out_dir="results/original_cpu",
   )
   pipeline("source/example/EN2002a_30s.wav", sess_name="EN2002a")
   PY
   deactivate
   ```

2. 执行应用 patch 后的 CPU 回归推理。

   ```bash
   source .venv-cpu-patched/bin/activate
   python infer.py \
     --manifest eval_data/functional/manifest.jsonl \
     --model_dir weights/diarizen-wavlm-large-s80-md \
     --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
     --device cpu \
     --output_dir results/patched_cpu
   deactivate
   ```

3. 执行 NPU 推理。

   ```bash
   source .venv-npu/bin/activate
   python infer.py \
     --manifest eval_data/functional/manifest.jsonl \
     --model_dir weights/diarizen-wavlm-large-s80-md \
     --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
     --device npu \
     --output_dir results/npu
   ```

   参数说明：

   - `manifest`：JSONL manifest 路径。
   - `model_dir`：主模型权重目录。
   - `embedding_model`：WeSpeaker embedding 权重文件路径。
   - `device`：推理设备，支持 `npu`、`cpu`。
   - `output_dir`：RTTM 和 `run.meta.json` 输出目录。

4. 执行正式 DER 评测。

   ```bash
   python score_diarization.py \
     --dscore_dir source/dscore \
     --reference_rttm eval_data/ami/reference.rttm \
     --system_rttm results/npu/*.rttm \
     --uem eval_data/ami/all.uem \
     --collar 0.0 \
     --output results/npu/der.txt
   ```

   参数说明：

   - `dscore_dir`：vendored dscore 目录路径。
   - `reference_rttm`：reference RTTM 文件路径。
   - `system_rttm`：待评测的 system RTTM 文件，支持 glob。
   - `uem`：UEM 文件路径。
   - `collar`：DER collar（秒），官方口径使用 `0.0`。
   - `ignore_overlaps`：`store_true` 开关。官方口径保留 overlap 时必须完全省略该参数，不能写成无效的 `--ignore_overlaps false`；需要忽略 overlap 的独立非官方模式才显式增加该参数。
   - `output`：DER 报告输出路径。

## 模型推理性能

NPU L2 性能测试示例：

```bash
mkdir -p results/npu
/usr/bin/time -v -o results/npu/l2.time.txt python infer.py \
  --manifest eval_data/ami/manifest.jsonl \
  --model_dir weights/diarizen-wavlm-large-s80-md \
  --embedding_model weights/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin \
  --device npu \
  --output_dir results/npu
```

`results/npu/run.meta.json` 提供 elapsed/RTF/provider。性能数据待正式验收。

## 适配与精度口径

### 设备边界与适配

upstream `DiariZenPipeline` 将设备写为 `cuda:0`（CUDA 可用时）否则 CPU，不能显式选择 NPU；pyannote 的 ONNX WeSpeaker wrapper 只识别 CPU/CUDA，其他设备会告警后回退 CPU。当前 patch：

1. pipeline 构造器和 `from_pretrained` 接收显式 device；
2. `infer.py` 默认 `--device npu`，仅 NPU 路径导入 `torch_npu`；
3. ONNX WeSpeaker 在 NPU 上显式使用 `CANNExecutionProvider`，provider 缺失时由 ONNX Runtime 直接失败；
4. Kaldi fbank 是 CPU 预处理，并将 numpy 特征送入 CANN ONNX session，不需要修改 site-packages；
5. CPU/CUDA 原路径不变。

设备分工：分割网络运行在 PyTorch NPU；WeSpeaker embedding 运行在 ONNX Runtime `CANNExecutionProvider`；Kaldi fbank 明确保留为 CPU 前处理，不是模型推理回退。`infer.py` 运行时读取 provider 并在 NPU 路径强制首 provider 为 `CANNExecutionProvider`，同时写出 `run.meta.json`。

环境隔离要求：原始和 patch 后 CPU baseline 使用独立环境，不要安装 upstream 根 `requirements.txt` 中的 `onnxruntime-gpu`；NPU 环境不得安装 CPU 索引 wheel；不要修改 site-packages；CPU/CUDA baseline 使用 CPU/CUDA ONNX Runtime，不能与 `onnxruntime-cann` 混装。

### 官方 DER 指标

upstream 对 `diarizen-wavlm-large-s80-md` 公布无 collar、保留 overlap、所有数据集共用 clustering 参数的 DER（%）：

| 数据集 | 官方 DER |
|---|---:|
| AMI-SDM | 14.0 |
| AISHELL-4 | 9.8 |
| AliMeeting far | 12.5 |
| NOTSOFAR-1 single-channel | 17.9 |
| MSDWild | 15.6 |
| DIHARD3 full | 14.5 |
| RAMC | 11.0 |
| VoxConverse | 9.2 |

特殊口径：AISHELL-4 先用 `sox in.wav -c 1 out.wav` 转单声道；NOTSOFAR-1 仅使用 single-channel 录音。后处理为固定 checkpoint `config.toml` 中的 segmentation、median filtering、speaker count、embedding 和 clustering 参数，输出 RTTM；metric 使用 dscore DER，`collar=0.0` 且不忽略 overlap。upstream README 未发布上述每项的精确 split revision、样本规模、wav.scp/RTTM/UEM SHA 和完整评测命令，因此这些字段明确记为"官方未发布"。

### 迁移对齐门禁

必须保留三组结果：未应用 patch 的固定 upstream CUDA 原始路径、应用 patch 后的 CUDA 回归路径、应用 patch 后的 NPU 路径，三组使用同一 manifest、config 和权重。通过条件：

- NPU DER 相对 CUDA 绝对劣化 `<= 0.2` 个百分点；
- miss/false alarm/confusion 分项都报告；
- 同输入 RTTM session、时间轴范围和 speaker 数约束一致；
- 不允许 embedding ONNX session 使用 `CPUExecutionProvider` 冒充 NPU。

`0.2` 个百分点是暂定迁移门禁，不是 upstream 官方容差。正式 L2 必须先测量原始 CUDA 与 patch 后 CUDA 的重复运行/聚类波动，再决定是否收紧或放宽。

### 权重许可

模型权重使用 CC BY-NC 4.0，正式部署前必须确认非商业使用及数据许可要求。数据许可和 split 必须由使用者根据官方 recipe 固定；不能自行猜测后宣称复现官方表。

### 性能评测方法

优先在 upstream 公布数据集的可取得全量 split 上记录总音频时长、RTF、分割/embedding 阶段耗时、batch 和峰值 HBM/RSS。`infer.py` 会写 `run.meta.json`；三组命令分别用 `/usr/bin/time -v -o` 保存独立资源日志，正式轮次至少重复 3 次并报告 RTF 中位数。官方 README 只发布 DER，未发布与当前 Atlas 路径可直接比较的硬件性能数值，因此报告 NPU/CUDA RTF 比值。

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 开源代码仓 | DiariZen 官方源码 | https://github.com/BUTSpeechFIT/DiariZen |
| 模型权重 | BUT-FIT/diarizen-wavlm-large-s80-md | https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md |
| 模型权重 | pyannote/wespeaker-voxceleb-resnet34-LM | https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM |
| 参考适配 | Ascend-SACT/BUTSpeechFIT-DiariZen | https://gitcode.com/Ascend-SACT/BUTSpeechFIT-DiariZen |
