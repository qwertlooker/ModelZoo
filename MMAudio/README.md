# MMAudio 推理适配指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [应用适配补丁](#应用适配补丁)
  - [安装依赖](#安装依赖)
  - [下载模型权重](#下载模型权重)
  - [调用推理接口](#调用推理接口)
- [性能](#性能)
- [适配修改说明](#适配修改说明)


******


# 概述

&emsp;MMAudio 是一款 AI 音频生成模型，它能够根据文字描述和/或视频内容，自动生成同步的高质量音效与音频。它通过多模态联合训练，能根据文本或视频快速生成高契合度的音效与音频。该模型基于流匹配目标，凭借条件同步模块实现了帧级别的视音频精准对齐。本文档介绍该模型基于昇腾底座的推理指导。

- 版本说明：

  ```
  url=https://github.com/hkchengrex/MMAudio
  commit_id=974010a
  model_name=MMAudio
  ```

# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                | 版本    |
  | ------------------- | ------- |
  | 固件与驱动          | 25.5.1+ |
  | CANN                | 8.5.1   |
  | Python              | 3.11.14 |
  | PyTorch / torch_npu | 2.9.0   |
  | torchaudio          | 2.9.0   |
  | soundfile           | 0.13.1  |

  说明：Atlas 800I A2 推理卡请以 CANN 版本选择实际固件与驱动版本。

# 快速上手

## 获取源码

1. 获取MMAudio 源码

   ```bash
   git clone https://github.com/hkchengrex/MMAudio.git
   cd MMAudio
   git reset --hard 974010a

   ```

2. 获取本仓适配补丁

   ```bash
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cp ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/MMAudio/diff_mmuaudio.patch .
   cp ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/MMAudio/diff_torchaudio_kaldi.patch .
   cp ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/MMAudio/diff_ViT_model.patch .
   ```



## 应用适配补丁

### 1. 应用MMAudio 主仓补丁

```bash
git apply ./diff_mmuaudio.patch
```

### 2. 应用 torchaudio 补丁

NPU 不支持 `torch.fft.rfft()` 返回复数张量直接调用 `.abs()`，需要将复数取模运算手动拆分为实部虚部计算。

```bash
# 找到 torchaudio 安装路径
TORCHAUDIO_PATH=$(python3 -c "import torchaudio; import os; print(os.path.dirname(torchaudio.__file__))")
# 应用补丁
cd ${TORCHAUDIO_PATH}/../
patch -p1 < /path/to/diff_torchaudio_kaldi.patch
```

如 patch 命令不可用，也可手动修改 `${TORCHAUDIO_PATH}/compliance/kaldi.py` 第616行：

```python
# 原始代码：
spectrum = torch.fft.rfft(strided_input).abs()

# 修改为：
#spectrum = torch.fft.rfft(strided_input).abs()
spectrum = torch.fft.rfft(strided_input)
real_view = torch.view_as_real(spectrum)
spectrum = torch.sqrt(real_view[...,0].pow(2) + real_view[...,1].pow(2))
```

## 安装依赖

```bash
pip install -e .
```

## 获取权重数据

本案例中，从modelscope.cn下载DFN5B-CLIP-ViT-H-14-378。其他权重或其他下载方法，请自行适配。

```bash
modelscope download --model apple/DFN5B-CLIP-ViT-H-14-378  --local_dir DFN5B-CLIP-ViT-H-14-378
```
使用DFN5B-CLIP-ViT-H-14-378需要修改适配本地路径
```bash
git apply ./diff_ViT_model.patch
```


# 推理验证

```bash
python3 demo.py --duration=8 --prompt "your prompt" --video=<path to video>

```

## 启动参数说明

| 参数             | 说明                                                         | 默认值 |
| ---------------- | ------------------------------------------------------------ | ------ |
| **`--duration`** | **生成音频的持续时间（秒）。**                               | `8`    |
| **`--prompt`**   | **文本提示词（String）**。用于描述你期望生成的音频内容、环境音效或音乐风格。。 |        |
| **`--video`**    | **输入视频的文件路径**。用于“视频转音频”或“视频+文本转音频”任务。如果**省略此参数**，模型将直接进入**“纯文本转音频（Text-to-Audio）”**合成模式。 |     可选参数 |

# 性能
数据集下载
```bash
mkdir -p ../data/AudioCaps-test-audioldm-ver/
curl -L -o ../data/AudioCaps-test-audioldm-ver/data.csv  https://github.com/hkchengrex/MMAudio/releases/download/v0.1/AudioCaps_audioldm_data.csv

```

测试命令
```bash
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 batch_eval.py model=large_44k_v2   dataset
=audiocaps duration_s=8 batch_size=16 num_workers=4 compile=False amp=False exp_id=npu8_large44k_v2_audiocaps
```

RTF（Real-Time Factor，实时率）= 推理耗时 / 生成音频时长，衡量合成速度。RTF < 1 表示合成速度优于实时，值越小性能越好。

| 硬件    | 生成音频时长           | 推理耗时 | RTF    |
| ------- | ---------------------- | -------- | ------ |
| 800I A2 | 8s (Text-to-Audio模式) | 1.59s    | 0.1988 |



# 适配修改说明

本项目对原始 MMAudio做了以下 NPU 适配修改：

## 1) 关闭 Transformer/Nested Tensor 快速路径

**修改原因**  
aten::_transformer_encoder_layer_fwd在 Ascend NPU 上不支持，否则触发 CPU fallback 

**修改内容**
- `demo.py`、`gradio_demo.py`、`batch_eval.py`
  - 增加：`torch.backends.mha.set_fastpath_enabled(False)`
- `mmaudio/ext/synchformer/motionformer.py`
  - 在 `BaseEncoderLayer` 中设置：`self.enable_nested_tensor = False`

---

## 2) 将音频保存路径从 torchaudio.save 切到 soundfile.write

**修改原因**  
torchaudio 2.9 默认使用 torchcodec 保存音频，torchcodec 依赖 CUDA 库无法在 NPU 运行。 替换为 soundfile 实现：

**修改内容**
- `demo.py`、`gradio_demo.py`、`batch_eval.py`
  - 新增 `safe_torchaudio_save(...)`

---

## 3) 增加 NPU 设备分支与对应分布式后端

**修改原因**  
Ascend 多卡必须使用 NPU 设备选择与 HCCL 后端才能正确初始化分布式。

**修改内容**
- `demo.py`、`gradio_demo.py`
  - 设备优先级改为：`npu > cuda > mps > cpu`
- `batch_eval.py`
  - 增加 NPU 设备设置：`torch.npu.set_device(local_rank)`
  - 分布式后端按设备选择：`hccl (npu) / nccl (cuda) / gloo (cpu)`
- `demo.py`
  - NPU 路径显存统计改用：`torch.npu.max_memory_allocated()`

---

## 4) 去掉CUDA 专属配置

**修改原因**  
`allow_tf32` 是 CUDA 专属开关，NPU下关闭。

**修改内容**

- `demo.py`、`gradio_demo.py`
  - 关闭 CUDA TF32 相关设置
- `batch_eval.py`
  - 仅在 `torch.cuda.is_available()` 时启用 TF32

---

## 5) BigVGANv2 统一保持 FP32 推理

**修改原因**  
为避免 torch_npu 在 BigVGANv2 BF16 卷积链路上的数值不稳定。
**修改内容**

- `mmaudio/ext/bigvgan_v2/bigvgan.py`
  - 加载权重后通过 keep_fp32() 将 BigVGANv2 的参数和 buffer 统一保持为 float32
- `mmaudio/ext/autoencoder/autoencoder.py`
  - 在 vocoder 调用边界对 BigVGANv2 输入统一做一次 float32 对齐
