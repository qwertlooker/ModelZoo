# FireRedASR-AED NPU 适配指导

## 1. 基准信息

- 上游：<https://github.com/FireRedTeam/FireRedASR.git>
- 分支：`main`
- 本地基准 commit：`834635e4cf277ed8ca92049fc375b17c3dc20748`
- 本次复核日期：2026-05-25

> 远端最新 commit 需要在可联网环境执行 `git -C FireRedASR-AED/upstream ls-remote origin main` 复核；当前适配仍基于上述本地 upstream。

## 2. 环境搭建

```bash
python3 -m venv FireRedASR-AED/.venv
source FireRedASR-AED/.venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio kaldiio kaldi_native_fbank numpy sentencepiece cn2an transformers peft
# NPU 环境按 CANN 版本安装匹配 torch-npu
pip install torch-npu
```

`requirements.txt` 为历史整环境导出，不建议作为最小依赖。

## 3. 权重下载

官方权重：Hugging Face `fireredteam/FireRedASR-AED-L` / ModelScope `FireRedTeam/FireRedASR-AED-L`。

```bash
./FireRedASR-AED/scripts/download_weights.sh FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L
```

脚本默认使用：

```bash
HF_HOME=~/.cache/gitee-ai
HF_ENDPOINT=https://hf-api.gitee.com
```

离线环境将权重目录拷贝到 `FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L/`，推理时通过 `--model_dir` 指定。

## 4. 测试数据

```bash
./FireRedASR-AED/scripts/download_test_data.sh FireRedASR-AED/test_data
```

输出官方样例 wav，例如 `FireRedASR-AED/test_data/BAC009S0764W0121.wav`，以及 `sample_data.meta.json`。已有 `wav.scp`/`text` 会复用；离线环境可设置 `OFFLINE=1 ALLOW_DUMMY=0` 强制缺官方样例时报错。

正式 LibriSpeech `test-clean` 评测数据按指定目录准备：

```bash
./FireRedASR-AED/scripts/prepare_librispeech_test_clean.sh \
  FireRedASR-AED/eval_data/librispeech_raw \
  FireRedASR-AED/eval_data/librispeech_test-clean
OFFLINE=1 ./FireRedASR-AED/scripts/prepare_librispeech_test_clean.sh \
  FireRedASR-AED/eval_data/librispeech_raw \
  FireRedASR-AED/eval_data/librispeech_test-clean
```

离线模式要求 `test-clean.tar.gz` 或 `LibriSpeech/test-clean/` 已存在；输出 `wav.scp`、`text` 和 metadata，CPU/NPU 对比复用同一份清单。

## 5. 应用 patch

```bash
cd FireRedASR-AED/upstream
git apply ../patches/0001-add-npu-device-support.patch
export PYTHONPATH=$PWD:$PYTHONPATH
```

patch 内容：

- `fireredasr/models/fireredasr.py`：新增显式 device 解析，用 `.to(device)` 替代 `.cuda()`。
- `fireredasr/speech2text.py`：新增 `--device` 参数。

## 6. infer.py 参数

- `--model_dir`：AED 权重目录。
- `--wav_path` / `--uttid`：单条音频路径和 id。
- `--device`：`npu`、`cpu` 或 `cuda`，默认 `npu`。
- 其余参数为 AED 解码参数：`--beam_size`、`--nbest`、`--decode_max_len` 等。

## 7. CPU 推理

```bash
cd FireRedASR-AED/upstream
export PYTHONPATH=$PWD:$PYTHONPATH
python ../infer.py --model_dir pretrained_models/FireRedASR-AED-L --wav_path ../test_data/BAC009S0764W0121.wav --uttid BAC009S0764W0121 --device cpu
```

也可使用上游入口：

```bash
python fireredasr/speech2text.py --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L --wav_path ../test_data/BAC009S0764W0121.wav --device cpu --use_gpu 0
```

## 8. NPU 推理

```bash
cd FireRedASR-AED/upstream
export PYTHONPATH=$PWD:$PYTHONPATH
ASCEND_RT_VISIBLE_DEVICES=0 python ../infer.py --model_dir pretrained_models/FireRedASR-AED-L --wav_path ../test_data/BAC009S0764W0121.wav --uttid BAC009S0764W0121 --device npu
```

## 9. 常见问题

- `ModuleNotFoundError: torch_npu`：CPU 验证不需要 torch_npu；确认没有传 `--device npu`。
- 找不到 `model.pth.tar`：权重目录未放在 `pretrained_models/FireRedASR-AED-L/` 或 `--model_dir` 指错。
- 长音频异常：上游说明 AED 建议输入不超过 60 秒。

## 10. 完整验收

官方样例只用于链路 smoke test。正式验收需按 `ACCEPTANCE_PLAN.md` 覆盖中文普通话、英文和有条件时方言 ASR，计算 CER/WER，并记录 batch、RTF/RTFx、峰值 HBM 和稳定性。
