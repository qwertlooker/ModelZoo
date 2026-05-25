# FireRedASR-AED NPU 验证记录

验证日期：2026-05-25。当前机器无可用 NPU；NPU 命令保留为待目标环境补验。

## 1. upstream / commit

```bash
git -C FireRedASR-AED/upstream rev-parse HEAD
# 834635e4cf277ed8ca92049fc375b17c3dc20748
```

远端最新 commit 需在可联网环境执行：

```bash
git -C FireRedASR-AED/upstream ls-remote origin main
```

## 2. patch 检查

```bash
git -C FireRedASR-AED/upstream apply --check ../patches/0001-add-npu-device-support.patch
# 结果：通过
```

## 3. Python 语法检查

```bash
python3 -m py_compile FireRedASR-AED/infer.py
# 结果：通过
```

## 4. 权重与测试数据

- 权重路径：`FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L/`，当前环境未下载。
- 测试数据：`FireRedASR-AED/test_data/BAC009S0764W0121.wav`，来自上游官方 examples。

准备命令：

```bash
./FireRedASR-AED/scripts/download_weights.sh FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L
./FireRedASR-AED/scripts/download_test_data.sh FireRedASR-AED/test_data
```

测试数据脚本已执行成功。

## 5. 当前环境 CPU 验证

当前系统 `python3` 缺少 `torch`；使用已有 `Canary-1B/.venv-cpu/bin/python` 继续检查，阻塞于 FireRedASR 额外依赖：

```bash
PYTHONPATH=FireRedASR-AED/upstream Canary-1B/.venv-cpu/bin/python FireRedASR-AED/infer.py \
  --model_dir FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L \
  --wav_path FireRedASR-AED/test_data/BAC009S0764W0121.wav \
  --device cpu
```

结果：

```text
ModuleNotFoundError: No module named 'kaldiio'
```

结论：当前环境 CPU 完整推理未完成；阻塞原因是缺少 `kaldiio/kaldi_native_fbank/cn2an` 等 FireRedASR CPU 依赖，且权重尚未下载。安装依赖并下载权重后补验。

## 6. NPU 验证命令

```bash
cd FireRedASR-AED/upstream
# 若尚未应用 patch：git apply ../patches/0001-add-npu-device-support.patch
export PYTHONPATH=$PWD:$PYTHONPATH
ASCEND_RT_VISIBLE_DEVICES=0 python ../infer.py \
  --model_dir pretrained_models/FireRedASR-AED-L \
  --wav_path ../test_data/BAC009S0764W0121.wav \
  --uttid BAC009S0764W0121 \
  --device npu
```

预期：输出包含 `uttid`、`text`、`wav`、`rtf` 的识别结果，无 `.cuda()` 设备不匹配错误。

## 7. 批量/WER 补验

```bash
python fireredasr/speech2text.py --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L \
  --wav_scp ../test_data/wav.scp --output out/aed-npu.txt --batch_size 2 --device npu
python fireredasr/utils/wer.py --print_sentence_wer 1 --do_tn 0 --rm_special 0 \
  --ref ../test_data/text --hyp out/aed-npu.txt
```
