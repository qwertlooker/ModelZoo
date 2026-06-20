# speechscorer NPU 适配文档

## 1. 来源与边界

- 官方源码 commit：`bbe0be772b37f472994d5a97f809214fd67a2c8e`（2023-11-21）
- Ascend-SACT 参考 commit：`f1d6e3ee3d0f113c610a969e6fde4a29af3216d1`
- Whisper checkpoint：`openai/whisper-base.en`，HF HEAD `911407f4214e0e1d82085af863093ec0b66f9cd6`
- 检查日期：2026-06-20；上述源码远端 HEAD 未变化。
- 正式主路径：`whisper-clm`。HuBERT/WavLM 代码可使用显式设备参数，但未以 Whisper 的验收结果替代其各自验收。

## 2. 参考实现审查

Ascend-SACT 参考修改存在以下不满足正式标准的行为：

- `--use-npu/--use-gpu` 组合依赖运行时探测，并在 NPU 不可用时静默降级 CUDA/CPU；
- 用 `hasattr(torch, "npu")` 掩盖缺少 `torch_npu`；
- 包含宽泛兼容和大量调试日志；
- VCC2018 只有无标签语音，不是 upstream demo 使用的 SpeechOcean762 原始评测数据。

正式 patch 仅把设备选择改为 `--device npu/cpu/cuda`，默认 NPU；NPU 路径直接导入 `torch_npu`，缺依赖或设备不可用时暴露原始错误。模型和输入仍通过 upstream 原有 `.to(self.device)` 迁移，CPU/CUDA 算法不变。

## 3. 验证事实

2026-06-20 已完成：

- 固定 upstream、参考仓和 Whisper 权重 HEAD；
- 在干净的 upstream commit 上执行 `git apply --check` 通过；
- 实际应用 patch 后，`python -m compileall -q speechscorer` 通过；
- 检查 patch 后代码，不再存在 `use_gpu`、`--use-gpu`、
  `hasattr(torch, "npu")` 或 `torch.cuda.is_available()` 设备回退模式；
- 确认 upstream demo 使用 SpeechOcean762 `test` 和人工 `total` 分数；
- 确认 upstream 未发布数值相关性表，只发布散点图。

当前主机的系统 Python 不含 PyTorch、`torch_npu`、Transformers、模型权重或
NPU，因此未执行端到端数值验收。不得把参考仓截图或 VCC2018 smoke
作为正式精度结论。

安装和推理见 [README_INFERENCE.md](README_INFERENCE.md)，SpeechOcean762
对齐和相关性报告口径见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。
