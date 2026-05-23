# BEATs NPU 适配指导

## 1. 基准

- 上游：<https://github.com/microsoft/unilm.git>
- 子目录：`beats/`
- commit：`833df7e7832e5064a281131ee64a481afa8e5b95`

## 2. 应用 patch

```bash
cd BEATs/upstream
git apply ../patches/0001-add-npu-fbank-device-support.patch
```

patch 内容：

- 修改 `beats/BEATs.py::preprocess()`，使 `ta_kaldi.fbank()` 在 CPU 上执行，再把 fbank 搬回原设备。
- `infer.py` 不属于上游原项目文件，不进入 patch；直接放在当前 `BEATs/` 目录作为适配脚本维护。

## 3. 环境依赖

建议最小依赖：

```bash
pip install torch==2.5.1 torchaudio==2.5.1 torch-npu==2.5.1.post4
```

并确保已安装匹配版本的 Ascend Driver、Firmware、CANN Toolkit/Kernel。实际版本需以目标机器上的 torch-npu 发布矩阵为准。

## 4. NPU 推理

```bash
cd BEATs/upstream
git apply ../patches/0001-add-npu-fbank-device-support.patch
cd beats
python infer.py   --checkpoint /path/to/BEATs_finetuned.pt   --wav /path/to/audio.wav   --device npu   --warmup 5   --repeat 20
```

## 5. 适配原则

- 不使用全局 monkey patch 替换 CUDA API。
- 不修改 BEATs 模型结构。
- 仅处理 NPU 不支持的 fbank 前处理节点。
- 保持 CPU/CUDA 路径可用。

## 上游更新处理原则

正式适配或提交前必须重新检查远端默认分支最新 commit。若远端 commit 与本文档记录的基准 commit 不一致，不要直接套用旧 patch；应先在新 upstream 上执行 `git apply --check`，再审视相关源码节点是否发生语义变化，必要时重新生成 patch。

## 设备选择说明

代码中只指定设备类型，例如 `--device npu`、`--device cuda` 或 `--device cpu`，不在代码里绑定 0 号卡或多卡列表。实际使用哪张卡、可见哪些卡由环境变量控制，例如 `ASCEND_RT_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES`。
