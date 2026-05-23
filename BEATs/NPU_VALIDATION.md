# BEATs NPU 验证指导

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
cd BEATs/upstream
git apply --check ../patches/0001-add-npu-fbank-device-support.patch
```

## 3. 单样例功能验证

```bash
cd BEATs/upstream
git apply ../patches/0001-add-npu-fbank-device-support.patch
cp ../infer.py beats/infer.py
cd beats
python infer.py --checkpoint /path/to/model.pt --wav /path/to/audio.wav --device npu --repeat 1 --warmup 0
```

期望：输出 top-k 标签和概率，无设备不匹配错误。

## 4. CPU/NPU 对齐验证

```bash
python infer.py --checkpoint /path/to/model.pt --wav /path/to/audio.wav --device cpu --repeat 1 --warmup 0
python infer.py --checkpoint /path/to/model.pt --wav /path/to/audio.wav --device npu --repeat 1 --warmup 0
```

对比 top-k label、输出概率误差和输出 shape。
