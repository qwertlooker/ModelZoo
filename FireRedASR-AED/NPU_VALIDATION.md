# FireRedASR-AED NPU 验证指导

## 1. 基础环境验证

```bash
python - <<'PY'
import torch
import torch_npu
print('torch:', torch.__version__)
print('npu available:', torch.npu.is_available())
print('npu count:', torch.npu.device_count())
PY
```

## 2. patch 可应用性验证

```bash
cd FireRedASR-AED/upstream
git apply --check ../patches/0001-add-npu-device-support.patch
```

## 3. 单条音频功能验证

```bash
cd FireRedASR-AED/upstream
git apply ../patches/0001-add-npu-device-support.patch
export PYTHONPATH=$PWD:$PYTHONPATH
python fireredasr/speech2text.py --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L --wav_path examples/wav/BAC009S0764W0121.wav --device npu
```

期望：输出包含 `uttid`、`text`、`rtf` 的识别结果。

## 4. 批量与 WER/CER 验证

```bash
python fireredasr/speech2text.py --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L --wav_scp examples/wav/wav.scp --output out/aed-npu.txt --batch_size 2 --beam_size 3 --nbest 1 --decode_max_len 0 --softmax_smoothing 1.25 --aed_length_penalty 0.6 --eos_penalty 1.0 --device npu
python fireredasr/utils/wer.py --print_sentence_wer 1 --do_tn 0 --rm_special 0 --ref examples/wav/text --hyp out/aed-npu.txt
```

## 5. CPU/NPU 对齐验证

```bash
python fireredasr/speech2text.py --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L --wav_path examples/wav/BAC009S0764W0121.wav --device cpu
python fireredasr/speech2text.py --asr_type aed --model_dir pretrained_models/FireRedASR-AED-L --wav_path examples/wav/BAC009S0764W0121.wav --device npu
```

对比识别文本、`rtf` 和是否存在设备不匹配错误。
