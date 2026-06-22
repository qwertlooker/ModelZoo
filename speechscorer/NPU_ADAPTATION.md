# speechscorer NPU 适配文档

## 1. 来源与边界

- 官方源码 commit：`bbe0be772b37f472994d5a97f809214fd67a2c8e`（2023-11-21）
- Ascend-SACT 参考 commit：`f1d6e3ee3d0f113c610a969e6fde4a29af3216d1`
- 上游默认 smoke checkpoint：`openai/whisper-base.en`，HF HEAD
  `911407f4214e0e1d82085af863093ec0b66f9cd6`
- 原始公开图路径：`hubert_large_ll60k.pt` + `hubert-mlm` +
  `facebook/hubert-large-ls960-ft@ece5fabbf034c1073acae96d5401b25be96709d8`
- SpeechOcean762：`613968e3b0b789fc33936fb5eba1973176ba7d11`
- 检查日期：2026-06-20；上述源码远端 HEAD 未变化。
- 原始结果对齐主路径为 `hubert-mlm`。`whisper-clm` 只保留为上游默认 smoke；
  两条路径不得混写指标。

## 2. 参考实现审查

Ascend-SACT 参考修改存在以下不满足正式标准的行为：

- `--use-npu/--use-gpu` 组合依赖运行时探测，并在 NPU 不可用时静默降级 CUDA/CPU；
- 用 `hasattr(torch, "npu")` 掩盖缺少 `torch_npu`；
- 包含宽泛兼容和大量调试日志；
- VCC2018 只有无标签语音，不是 upstream demo 使用的 SpeechOcean762 原始评测数据。

正式 patch 仅把设备选择改为 `--device npu/cpu/cuda`，默认 NPU，并增加
`--output_csv` 防止不同设备结果互相覆盖；NPU 路径直接导入 `torch_npu`，
缺依赖或设备不可用时暴露原始错误。模型和输入仍通过 upstream 原有
`.to(self.device)` 迁移，CPU/CUDA 算法不变。

## 3. 验证事实

2026-06-20 已完成：

- 固定 upstream、参考仓和 Whisper 权重 HEAD；
- 在干净的 upstream commit 上执行 `git apply --check` 通过；
- 实际应用 patch 后，`python -m compileall -q speechscorer` 通过；
- 检查 patch 后代码，不再存在 `use_gpu`、`--use-gpu`、
  `hasattr(torch, "npu")` 或 `torch.cuda.is_available()` 设备回退模式；
- 确认 upstream 全量 demo 使用 SpeechOcean762 `test`、HuBERT-MLM 和人工
  `total` 分数；
- 确认 upstream 未发布数值相关性表，只发布散点图。

补充工具链验证：

- 更新后 patch SHA256 为
  `f2712ef70afee2176c6a34c0ca41383ef20233bfa3f96a24794f4d9e4c6e3ef1`；
- 在干净 upstream worktree 上重新执行 `git apply --check`、实际应用和
  `compileall` 均通过；
- 使用 2 条本地音频 fixture 验证 `prepare_eval_data.py` 可按 `wav.scp`
  复制音频、读取人工 `total` 并生成 manifest/meta；
- 使用 2 条合成 scorer CSV 验证 `evaluate_results.py` 可计算 Pearson/Spearman、
  notebook `groupby(age)` 汇总和 baseline 数值对齐。该 fixture 只验证工具，
  不是模型相关性结果。

当前主机的系统 Python 不含 PyTorch、`torch_npu`、Transformers、模型权重或
NPU，因此未执行端到端数值验收。不得把参考仓截图或 VCC2018 smoke
作为正式精度结论。

已补充 `prepare_eval_data.py` 和 `evaluate_results.py`，但在实际完成 HuBERT
checkpoint 下载、fairseq 导入和 CPU/NPU 全量运行前，交付状态仍是
**S1：静态适配完成；升级到 S2/S3 仍缺真实 HuBERT 功能验证和 SpeechOcean762
全量精度/性能对齐**。

独立重放必须使用未应用 patch 的 `upstream-original`、应用 patch 的
`upstream-npu` 和 NPU candidate 三组输出。NPU 环境不得复用 PyTorch CPU 索引
wheel；完整安装和比较命令见 `README.md`。

安装和推理见 [README.md](README.md)，SpeechOcean762
对齐和相关性报告口径见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

## 补充说明（来自 README.md）

### 两条评分路径

当前交付保留两条明确分离的路径：

- `whisper-clm`：上游默认入口，用于轻量功能 smoke；
- `hubert-mlm`：上游 README 图和 SpeechOcean762 notebook 实际使用的公开演示路径，是原始结果对齐主线。

### fairseq 依赖说明

`hubert-mlm` 的 fairseq 依赖较旧。必须在目标 Python/PyTorch 组合中实际完成导入和端到端验证；安装失败时不能改用 `whisper-clm` 冒充原始公开路径。

### 目录结构说明

执行时另外创建 `source/`、未应用 patch 的 `upstream-original/` 和应用 patch 的 `upstream-npu/`，避免覆盖原始 baseline。
