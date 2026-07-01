# 适配过程记录模板（不上库）

此文件是适配过程的内部工作日志模板，用于回溯决策、沉淀问题和反向改进 skill。不要提交到 `ACL_PyTorch/built-in`；只把用户需要知道的结论整理进 README 的 FAQ/已知问题。

建议路径：`<workdir>/.adaptation-notes/<model>-process.md` 或任务工作区外的等价位置。

## 基本信息

- 模型/上游链接：
- 固定 commit/revision：
- 使用者提供 checkpoint/权重目录：
- 目标芯片/镜像/CANN/torch_npu：
- 适配路线：ONNX-OM / torch_npu / TorchAir / vLLM-Ascend / 拆图混合
- 源仓 accuracy/performance 口径：

## 决策记录

| 时间 | 决策 | 依据 | 备选方案 | 影响 | 是否写入 README/FAQ |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

需要记录的决策包括：路线选择、是否拆图、是否保留 CPU fallback、是否替换推理后端、是否 patch 上游入口、是否更换依赖安装方式、是否使用使用者提供 checkpoint 以外的权重。

## 问题记录

| 时间 | 症状/报错 | 阻塞阶段 | 已尝试但无效 | 根因 | 最终修复 | 耗时/是否外部提示 | 是否应反向改进 skill |
|---|---|---|---|---|---|---|---|
|  |  | 环境/导出/转换/推理/精度/性能/文档 |  |  |  |  |  |

触发“必须记录”的条件：

- 一个问题排查超过约 30 分钟或反复出现。
- 需要用户、外部文档、reviewer 或其他人提示后才解决。
- 一开始选错路线，后来改为 ONNX-OM/TorchAir/vLLM/拆图/CPU fallback。
- accuracy 或 performance 口径与源仓不一致，需要解释。
- patch 出现只能执行一次、依赖冲突、环境隔离、TorchAir cache、ATC 长时间编译、动态 shape、custom op、音频 I/O、checkpoint 加载等问题。

## 验证记录

| 阶段 | 命令 | 关键输出/日志路径 | 结论 |
|---|---|---|---|
| 环境 |  |  |  |
| 单样例 |  |  |  |
| 精度 |  |  |  |
| 性能 |  |  |  |

## 反思与 skill 改进候选

每次解决阻塞后问：

1. 这个问题是否能通过 skill 的默认审计提前发现？
2. 是否应该新增搜索信号、patch 模式、README FAQ 模板或脚手架参数？
3. 是否已有 ModelZoo 样本或对应 PR 提供相同规律？
4. 该问题是模型特有，还是可复用为通用规则？
5. 是否需要更新 `adaptation-heuristics.md`、`patch-modification-patterns.md`、`output-contract.md` 或脚本？

只把可复用、非一次性的规律写回 skill；不要把本次模型私有路径、账号、临时日志、内部数据写入 skill。
