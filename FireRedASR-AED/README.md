---
license: apache-2.0
hardware: NPU
---
# FireRedASR-AED NPU 适配

本目录提供 FireRedASR-AED 语音识别模型的 NPU 适配。适配方式是把上游 `.cuda()` 推理链路改为显式 `--device cpu|cuda|npu`，不依赖全局 monkey patch。

## 1. 硬件/软件约束

- 硬件：Atlas 800I A2 / 910B 单卡验证目标；实际卡号由 `ASCEND_RT_VISIBLE_DEVICES` 控制。
- Python：建议 3.10。
- NPU：昇腾驱动/固件 >= 25.0.RC1.1，CANN Toolkit/Kernel/NNAL >= 8.2.RC1。
- PyTorch / torch-npu：版本必须与 CANN 匹配，例如 `torch==2.5.1`、`torch-npu==2.5.1.post4`。

## 2. 环境搭建

```bash
python3 -m venv FireRedASR-AED/.venv
source FireRedASR-AED/.venv/bin/activate
pip install --upgrade pip

# CPU 验证最小依赖，参考上游 requirements.txt
pip install torch torchaudio kaldiio kaldi_native_fbank numpy sentencepiece cn2an transformers peft

# NPU 环境再安装与 CANN 匹配的 torch-npu
pip install torch-npu
```

`FireRedASR-AED/requirements.txt` 是历史环境导出，不是最小依赖清单。

## 3. 上游代码与 patch

```bash
git clone https://github.com/FireRedTeam/FireRedASR.git FireRedASR-AED/upstream
cd FireRedASR-AED/upstream
git apply ../patches/0001-add-npu-device-support.patch
cp ../infer.py infer.py
export PYTHONPATH=$PWD:$PYTHONPATH
```

基准 commit：`834635e4cf277ed8ca92049fc375b17c3dc20748`。如上游更新，先重新检查 `fireredasr/models/fireredasr.py` 和 `fireredasr/speech2text.py` 再套 patch。

## 4. 权重下载

官方权重：

- Hugging Face：`fireredteam/FireRedASR-AED-L`
- ModelScope：`FireRedTeam/FireRedASR-AED-L`

默认使用 Gitee HF endpoint：

```bash
./FireRedASR-AED/scripts/download_weights.sh \
  FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L
```

只检查仓库和文件 URL、但不下载大权重：

```bash
FIRERED_CHECK_ONLY=1 ./FireRedASR-AED/scripts/download_weights.sh \
  FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L
```

也可离线下载后放到：

```text
FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L/
```

推理时通过 `--model_dir` 指定该目录。

## 5. 测试数据下载/准备

```bash
./FireRedASR-AED/scripts/download_test_data.sh FireRedASR-AED/test_data
```

脚本会复制上游 `examples/wav/` 下的官方样例 wav、`wav.scp` 和 `text`。

### 5.1 正式评测数据准备建议

参考 Canary-1B 数据准备问题，FireRedASR-AED 的 AISHELL/LibriSpeech 等正式评测必须把“准备数据”和“评测”分开：

- L0 官方 wav 只验证链路，不能作为 CER/WER 结论。
- AISHELL-1、LibriSpeech 等数据准备命令必须显式指定 split，例如 `test` / `test-clean`，不要只依赖默认值。
- 准备脚本输出固定 `wav.scp` + `text` 或统一 JSONL manifest，字段建议包含 `uttid`、`audio_filepath`、`text`、`duration`、`language`、`split`。
- manifest 旁边生成 `*.meta.json`，记录 dataset/config/split、样本数、总时长、抽样 seed、下载源和文件大小。
- 评测脚本只读取本地 `wav.scp`/manifest，复用当前 `infer.py` 或上游解码机制，再用固定 normalizer 计算 CER/WER；CPU/CUDA/NPU 使用同一份 manifest 对比。
- 中英文数据建议分开准备、分开评测；不要用一个 `--task all` 让 LibriSpeech 下载失败阻塞 AISHELL，或反过来阻塞英文评测。

建议下载日志降噪：

```bash
export HF_HUB_VERBOSITY=error
export DATASETS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
```

## 6. CPU 验证

```bash
cd FireRedASR-AED/upstream
export PYTHONPATH=$PWD:$PYTHONPATH
python ../infer.py \
  --model_dir pretrained_models/FireRedASR-AED-L \
  --wav_path ../test_data/BAC009S0764W0121.wav \
  --uttid BAC009S0764W0121 \
  --device cpu
```

## 7. NPU 推理

```bash
cd FireRedASR-AED/upstream
export PYTHONPATH=$PWD:$PYTHONPATH
ASCEND_RT_VISIBLE_DEVICES=0 python ../infer.py \
  --model_dir pretrained_models/FireRedASR-AED-L \
  --wav_path ../test_data/BAC009S0764W0121.wav \
  --uttid BAC009S0764W0121 \
  --device npu
```

## 8. 文件说明

- `infer.py`：当前适配新增推理入口，默认 `--device npu`，CPU 验证用 `--device cpu`。
- `patches/0001-add-npu-device-support.patch`：上游设备显式化 patch。
- `scripts/download_weights.sh`：权重下载脚本。
- `scripts/download_test_data.sh`：测试数据准备脚本。
- `ANALYSIS.md` / `NPU_ADAPTATION.md` / `NPU_VALIDATION.md`：分析、适配和验证记录。
- `ACCEPTANCE_PLAN.md`：参考原始 FireRedASR-AED 功能/性能/精度的完整验收方案，包含数据集选择、CER/WER 验收、性能稳定性和报告模板。
