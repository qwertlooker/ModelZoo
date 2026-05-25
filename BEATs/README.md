---
license: apache-2.0
language:
  - zh
hardware: NPU
---
# BEATs NPU 适配

本目录提供 Microsoft UniLM `beats/` 中 BEATs 音频分类模型的 NPU 适配。当前适配不改模型结构，只把 `torchaudio.compliance.kaldi.fbank` 前处理临时回退到 CPU，模型主体仍可在 NPU 执行。

## 1. 硬件/软件约束

- 硬件：Atlas 800I A2 / 910B 单卡验证目标；实际卡号由 `ASCEND_RT_VISIBLE_DEVICES` 控制。
- Python：建议 3.10。
- NPU：昇腾驱动/固件 >= 25.0.RC1.1，CANN Toolkit/Kernel/NNAL >= 8.2.RC1。
- PyTorch / torch-npu：版本必须与 CANN 匹配，例如 `torch==2.5.1`、`torch-npu==2.5.1.post4`。

## 2. 环境搭建

```bash
python3 -m venv BEATs/.venv
source BEATs/.venv/bin/activate
pip install --upgrade pip

# CPU 验证最小依赖
pip install torch torchaudio

# NPU 环境再安装与 CANN 匹配的 torch-npu
pip install torch-npu
```

`BEATs/requirements.txt` 是历史环境导出，不是最小依赖清单。

## 3. 上游代码与 patch

```bash
git clone https://github.com/microsoft/unilm.git BEATs/upstream
cd BEATs/upstream
git apply ../patches/0001-add-npu-fbank-device-support.patch
cp ../infer.py beats/infer.py
```

基准 commit：`833df7e7832e5064a281131ee64a481afa8e5b95`。如上游更新，先重新检查 `beats/BEATs.py::preprocess()` 再套 patch。

## 4. 权重下载

官方权重在上游 `beats/README.md` 的 OneDrive 链接中发布。请选择与任务匹配的 fine-tuned checkpoint。

```bash
./BEATs/scripts/download_weights.sh BEATs/weights
# 或提供可直接下载 URL
BEATS_WEIGHT_URL=<direct-url> ./BEATs/scripts/download_weights.sh BEATs/weights BEATs/weights/model.pt
```

只检查上游 README 中的官方 OneDrive 链接：

```bash
BEATS_CHECK_ONLY=1 ./BEATs/scripts/download_weights.sh BEATs/weights
```

注意：当前环境验证发现 OneDrive 非浏览器直连不稳定；如直接下载失败，请使用浏览器下载官方 checkpoint 或提供稳定直链给 `BEATS_WEIGHT_URL`。

离线部署时，把下载好的 `.pt` 放入 `BEATs/weights/`，推理时通过 `--checkpoint` 指定。

## 5. 测试数据下载/生成

```bash
./BEATs/scripts/download_test_data.sh BEATs/test_data
```

生成 `BEATs/test_data/dummy_1s_16k.wav`。该样例只用于验证加载、前处理和推理链路，不代表分类准确率。

## 6. CPU 验证

```bash
cd BEATs/upstream
cp ../infer.py beats/infer.py
cd beats
python infer.py \
  --checkpoint ../../weights/model.pt \
  --wav ../../test_data/dummy_1s_16k.wav \
  --device cpu --warmup 0 --repeat 1
```

## 7. NPU 推理

```bash
cd BEATs/upstream/beats
ASCEND_RT_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint ../../weights/model.pt \
  --wav ../../test_data/dummy_1s_16k.wav \
  --device npu --warmup 5 --repeat 20
```

## 8. 文件说明

- `infer.py`：当前适配新增推理入口，默认 `--device npu`，CPU 验证用 `--device cpu`。
- `patches/0001-add-npu-fbank-device-support.patch`：上游 `beats/BEATs.py` 的 NPU patch。
- `scripts/download_weights.sh`：权重下载/放置说明脚本。
- `scripts/download_test_data.sh`：生成最小 wav 测试样例。
- `ANALYSIS.md` / `NPU_ADAPTATION.md` / `NPU_VALIDATION.md`：分析、适配和验证记录。
