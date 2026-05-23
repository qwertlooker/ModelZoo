# FireRedASR-AED NPU 适配指导

## 1. 基准

- 上游：<https://github.com/FireRedTeam/FireRedASR.git>
- commit：`834635e4cf277ed8ca92049fc375b17c3dc20748`

## 2. 应用 patch

```bash
cd FireRedASR-AED/upstream
git apply ../patches/0001-add-npu-device-support.patch
```

patch 内容：

- 修改 `fireredasr/models/fireredasr.py`，将 `.cuda()` 改为显式 `.to(device)`。
- 修改 `fireredasr/speech2text.py`，新增 `--device` 参数。
- `FireRedASR-AED/infer.py` 不属于上游原项目文件，不进入 patch；直接作为当前适配脚本维护。

## 3. 环境依赖

```bash
pip install -r requirements.txt
pip install torch-npu==2.5.1.post4
```

注意 `torch`、`torch-npu`、CANN 版本必须匹配。

## 4. NPU 推理

```bash
cd FireRedASR-AED/upstream
git apply ../patches/0001-add-npu-device-support.patch
export PYTHONPATH=$PWD:$PYTHONPATH
python fireredasr/speech2text.py   --asr_type aed   --model_dir pretrained_models/FireRedASR-AED-L   --wav_path examples/wav/BAC009S0764W0121.wav   --device npu   --beam_size 3   --nbest 1   --decode_max_len 0   --softmax_smoothing 1.25   --aed_length_penalty 0.6   --eos_penalty 1.0
```

使用当前适配仓脚本：

```bash
# 在 FireRedASR-AED/upstream 下应用 patch 后
cp ../infer.py infer.py
python infer.py --model_dir pretrained_models/FireRedASR-AED-L --wav_path examples/wav/BAC009S0764W0121.wav --device npu
```

## 上游更新处理原则

正式适配或提交前必须重新检查远端默认分支最新 commit。若远端 commit 与本文档记录的基准 commit 不一致，不要直接套用旧 patch；应先在新 upstream 上执行 `git apply --check`，再审视相关源码节点是否发生语义变化，必要时重新生成 patch。

## 设备选择说明

代码中只指定设备类型，例如 `--device npu`、`--device cuda` 或 `--device cpu`，不在代码里绑定 0 号卡或多卡列表。实际使用哪张卡、可见哪些卡由环境变量控制，例如 `ASCEND_RT_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES`。
