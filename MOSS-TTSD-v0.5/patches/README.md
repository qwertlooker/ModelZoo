# MOSS-TTSD-v0.5 patches

本目录保存对原项目已有文件的适配补丁；不在模型目录新增独立推理代码文件。

当前补丁：

- `0001-adapt-v0.5-inference-to-npu.patch`
  - 基于 `OpenMOSS/MOSS-TTSD` tag `v0.5` / commit `0e078c62389922d3aa873ce182daf31142860b18`。
  - 修改原项目已有 `inference.py`、`generation_utils.py`、`gradio_demo.py`、`podcast_generate.py`、`modeling_asteroid.py`、`requirements.txt`、`XY_Tokenizer/inference.py`、`XY_Tokenizer/utils/helpers.py`、`XY_Tokenizer/xy_tokenizer/model.py`、`XY_Tokenizer/xy_tokenizer/nn/quantizer.py`。
  - 推理入口只增加显式 `--device npu/cpu/cuda`，默认 NPU；模型与 codec 按固定目录读取，不增加 dtype、attention、batch 或权重路径参数。
  - 注册 NPU PFA/IFA attention backend，直接传递 query/KV head 数支持 GQA，避免 Transformers SDPA/eager 的 `repeat_kv` 实体展开；prefill 和单 token decode 分别走 `npu_prompt_flash_attention` / `npu_incre_flash_attention`。
  - NPU 设备内部固定选择 BF16 + PFA/IFA，CPU 使用 FP32 + SDPA，CUDA 保持 BF16 + `flash_attention_2`。
  - patch 直接从 `requirements.txt` 删除 CUDA/ROCm GPU 专用的 `flash-attn`；NPU 不依赖该包。
  - 将 prompt 音频文件读取从 `torchaudio.load` 改为原依赖中的 `soundfile`，将 WAV 写出从 `torchaudio.save` 改为 `soundfile.write`，规避 TorchAudio 2.9+ 需要 TorchCodec 的 `load_with_torchcodec` / `save_with_torchcodec` 路径。
  - 修正 XY Tokenizer encode/decode 默认 CUDA 设备假设，使其从输入 tensor 推断设备。
  - 将 quantizer 的 autocast device 从硬编码 `cuda` 改为当前 tensor device。
  - 修正自定义生成循环裁剪 shifted speech channels 后未同步 `cur_len` 的问题，避免 NPU SDPA 路径下 `aclnnFlashAttentionScore` 收到 `[B, 1, L+7, L]` 形状的 attention mask。

Patch SHA256：

```text
426303406d9289c0f981ca333604107af323a56a576c5129a844aacc83962056
```

应用方式（在 `ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5` 目录下执行）：

```bash
git clone https://github.com/OpenMOSS/MOSS-TTSD.git source
git -C source checkout 0e078c62389922d3aa873ce182daf31142860b18
git -C source worktree add --detach ../upstream-npu \
  0e078c62389922d3aa873ce182daf31142860b18
git -C upstream-npu apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
```

校验方式（在 `ACL_PyTorch/built-in/audio/MOSS-TTSD-v0.5` 目录下执行）：

```bash
git -C upstream-npu reset --hard \
  0e078c62389922d3aa873ce182daf31142860b18
git -C upstream-npu apply --check \
  ../patches/0001-adapt-v0.5-inference-to-npu.patch
```
