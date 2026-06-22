# MOSS-TTSD-v0.5 patches

本目录保存对原项目已有文件的适配补丁；不在模型目录新增独立推理代码文件。

当前补丁：

- `0001-adapt-v0.5-inference-to-npu.patch`
  - 基于 `OpenMOSS/MOSS-TTSD` tag `v0.5` / commit `0e078c62389922d3aa873ce182daf31142860b18`。
  - 修改原项目已有 `inference.py`、`generation_utils.py`、`gradio_demo.py`、`podcast_generate.py`、`modeling_asteroid.py`、`requirements.txt`、`XY_Tokenizer/inference.py`、`XY_Tokenizer/utils/helpers.py`、`XY_Tokenizer/xy_tokenizer/model.py`、`XY_Tokenizer/xy_tokenizer/nn/quantizer.py`。
  - 推理入口增加显式 `--device npu/cpu/cuda` 和 `--batch_size`；默认 NPU、
    batch size 1。TTSD-eval 按有界 batch 顺序生成并打印逐批进度。
    模型与 codec 按固定目录读取，不增加 dtype、attention 或权重路径参数。
  - 注册 NPU PFA/IFA attention backend，直接传递 query/KV head 数支持 GQA，避免 Transformers SDPA/eager 的 `repeat_kv` 实体展开；prefill 和单 token decode 分别走 `npu_prompt_flash_attention` / `npu_incre_flash_attention`。
  - NPU 设备内部固定选择 BF16 + PFA/IFA，CPU 使用 FP32 + SDPA，CUDA 保持 BF16 + `flash_attention_2`。
  - patch 直接从 `requirements.txt` 删除 CUDA/ROCm GPU 专用的 `flash-attn`；NPU 不依赖该包。
  - 将 prompt 音频文件读取从 `torchaudio.load` 改为原依赖中的 `soundfile`，将 WAV 写出从 `torchaudio.save` 改为 `soundfile.write`，规避 TorchAudio 2.9+ 需要 TorchCodec 的 `load_with_torchcodec` / `save_with_torchcodec` 路径。
  - 修正 XY Tokenizer encode/decode 默认 CUDA 设备假设，使其从输入 tensor 推断设备。
  - 将 quantizer 的 autocast device 从硬编码 `cuda` 改为当前 tensor device。
  - 修正自定义生成循环裁剪 shifted speech channels 后未同步 `cur_len` 的问题，避免 NPU SDPA 路径下 `aclnnFlashAttentionScore` 收到 `[B, 1, L+7, L]` 形状的 attention mask。

Patch SHA256：

```text
7d446e9c9c743b57ab41cb553422e428bf515b6d4e724d10450fa5b15b1a01ba
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

---

- `0002-adapt-ttsd-eval-to-npu.patch`
  - 基于 `OpenMOSS/TTSD-eval` commit `dea13b98529dc16dcfb5fe45779ad63ac9238337`。
  - 修改 `tools/align.py`、`tools/run_similarity.py`、`wer/whisper_asr.py` 三个评测器的设备路由，使其支持 NPU（`--device npu:0` / `--device npu`）。不修改任何指标计算逻辑。
  - `align.py`：默认设备自动检测增加 NPU 分支；`_worker_run_bucket` 增加 `device_type` 参数，NPU 路径用 `ASCEND_RT_VISIBLE_DEVICES` 隔离卡。
  - `run_similarity.py`：新增 `--device` 参数；`_worker_init` 增加 `device_type` 参数，NPU 路径用 `torch.npu.set_device`。
  - `whisper_asr.py`：`_init_worker` 增加 `device_type` 参数，NPU 路径构造 `npu:{id}` 设备字符串；新增 `--device` 参数覆盖 `--num_gpus` 自动检测。
  - 三个文件均增加 `_npu_available()` 辅助函数（条件导入 `torch_npu`），CPU/CUDA 原有路径行为不变。
  - 仅在 NPU 评测 profile 下应用；CPU/CUDA profile 保持上游原文件不变。

Patch SHA256：

```text
5dd9c5ab357d64e5d43543821ee3324f32b9c1210bb4ba63e9fe9dcaa7438607
```

应用方式（在 `third_party/TTSD-eval` 工作树根目录下执行）：

```bash
git -C third_party/TTSD-eval checkout dea13b98529dc16dcfb5fe45779ad63ac9238337
git -C third_party/TTSD-eval apply ../../patches/0002-adapt-ttsd-eval-to-npu.patch
```

校验方式：

```bash
git -C third_party/TTSD-eval checkout dea13b98529dc16dcfb5fe45779ad63ac9238337
git -C third_party/TTSD-eval apply --check ../../patches/0002-adapt-ttsd-eval-to-npu.patch
sha256sum third_party/TTSD-eval/tools/align.py \
         third_party/TTSD-eval/tools/run_similarity.py \
         third_party/TTSD-eval/wer/whisper_asr.py
```

patch 后文件 SHA256（`prepare_eval_data.py` 的 NPU profile 门禁按此校验）：

```text
722028e9a7adbc90dfad3eb74cb1ab307cd6919bbc2969134f52175c0c2c49f2  tools/align.py
65ccfb613a5248f9f40efe9a09fadaa2ebcf4aa05df6a9331bf7a619fdc6dc66  tools/run_similarity.py
b7a62bf6504ddf8a9fdf3c86f66ca4488608caf10ae5456901d4b275bf917194  wer/whisper_asr.py
```
