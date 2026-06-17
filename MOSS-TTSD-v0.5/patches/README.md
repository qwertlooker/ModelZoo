# MOSS-TTSD-v0.5 patches

本目录保存对原项目已有文件的适配补丁；不在模型目录新增独立推理代码文件。

当前补丁：

- `0001-adapt-v0.5-inference-to-npu.patch`
  - 基于 `OpenMOSS/MOSS-TTSD` tag `v0.5` / commit `0e078c62389922d3aa873ce182daf31142860b18`。
  - 修改原项目已有 `inference.py`、`generation_utils.py`、`modeling_asteroid.py`、`XY_Tokenizer/inference.py`、`XY_Tokenizer/utils/helpers.py`、`XY_Tokenizer/xy_tokenizer/model.py`、`XY_Tokenizer/xy_tokenizer/nn/quantizer.py`。
  - 增加显式 `--device npu/cpu/cuda`、`--dtype`、`--attn_implementation` 和权重/codec 路径参数；默认 `--device npu`。
  - NPU 默认 `--attn_implementation sdpa`，不依赖 CUDA/ROCm GPU 专用的 `flash-attn`；`flash_attention_2` 仅保留给显式 CUDA/ROCm 路径。
  - 将 prompt 音频文件读取从 `torchaudio.load` 改为原依赖中的 `soundfile`，规避 TorchAudio 2.9+ 需要 TorchCodec 的 `load_with_torchcodec` 路径。
  - 修正 XY Tokenizer encode/decode 默认 CUDA 设备假设，使其从输入 tensor 推断设备。
  - 将 quantizer 的 autocast device 从硬编码 `cuda` 改为当前 tensor device。
  - 修正自定义生成循环裁剪 shifted speech channels 后未同步 `cur_len` 的问题，避免 NPU SDPA 路径下 `aclnnFlashAttentionScore` 收到 `[B, 1, L+7, L]` 形状的 attention mask。

应用方式：

```bash
git -C MOSS-TTSD-v0.5/upstream checkout v0.5
git -C MOSS-TTSD-v0.5/upstream apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
```

校验方式：

```bash
git -C MOSS-TTSD-v0.5/upstream reset --hard v0.5
git -C MOSS-TTSD-v0.5/upstream apply --check ../patches/0001-adapt-v0.5-inference-to-npu.patch
```
