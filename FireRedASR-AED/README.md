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
