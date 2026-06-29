# Patch 说明

Patch 基于 DiariZen commit `a60b18151dbbe246e4199d8ef5cd2ece3872ea94`。

Patch SHA256：

```text
ee569c0e51d9805b9c054e80b1ee7d32dbf50d5c5a1fffd11b68f28a452a1542
```

- 为 pipeline 增加显式 `torch.device`，移除硬编码 `cuda:0`/自动 CPU 选择；
- NPU 上 speaker embedding ONNX 模型使用 `CANNExecutionProvider`；
- NPU 路径将 Kaldi fbank 预处理留在 CPU，避免修改 `torchaudio` site-packages；embedding 模型本身仍由 CANN provider 执行；
- CPU/CUDA 路径保持原行为。

```bash
git -C upstream apply --check \
  patches/0001-add-explicit-npu-pipeline-device.patch
git -C upstream apply \
  patches/0001-add-explicit-npu-pipeline-device.patch
```
