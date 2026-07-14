# 验证、证据与状态契约

## S0–S4 技术状态

| 状态 | 项目日志必需成功证据 | 允许的结论 |
|---|---|---|
| 未定级 | 仅初始化日志 | 已建项目，不能写“分析完成” |
| S0 分析完成 | `target_baseline`、`upstream_audit`；目标仓/上游/权重已冻结 | 完成适配分析 |
| S1 静态适配完成 | `static_check`；需要 patch 时另有 `patch_check` | 静态门禁通过 |
| S2 功能验证完成 | route/硬件已确认；`functional_inference`：CPU/CUDA/NPU 至少一条真实端到端输出 | 指定设备功能链路实测通过 |
| S3 L2 迁移对齐完成 | `npu_accuracy`、`npu_performance`、`comparison_report` | NPU L2 精度/质量和性能对齐通过 |
| S4 扩展验收通过 | `extended_validation` | 用户要求的长稳、L3 或业务扩展验收通过 |

`set-status` 只允许依次设置 S0→S4。失败证据必须保留真实非零退出码，不能解锁状态。dummy/单条样例最多支持 S2；文档、patch、compile 最多支持 S1。

## 上库候选就绪

技术状态与上库就绪分开：

- 至少达到 S3。
- `target_audit` 必须包含 `tools/audit_model_delivery.py <model_dir> --target-readiness --target-path ...` 或目标仓等价门禁。
- `clean_room_replay` 必须从独立候选目录按 README 重放最低正式路径，不读取内部日志、upstream、cache 或未声明环境变量。
- 满足后才运行 `project_log.py set-target-ready`。正式合入仍只能由目标仓 PR merged/commit 证明。

## 每条证据至少记录

- 类型、时间、完整命令、退出码和可验证 claim。
- 目标仓 commit、源码 commit/PR head、权重 revision/SHA、镜像 tag/digest。
- 对外硬件型号、CANN、Python、torch/torch_npu 和路线相关工具版本。
- 日志路径；权重、manifest、ONNX、OM、配置或结果文件 SHA256。
- 数据集版本/split、样本数、过滤规则、评测脚本 revision 和输出目录。

S0 前冻结的目标仓/上游/权重上下文不能在同一日志中静默改变；需要新 baseline 时创建新的项目日志。S2 后 route/硬件改变同样需要新日志，避免旧证据继续解锁状态。

## 验证层级

1. **S0 分析**：实时目标 master、拟合入路径、同名目录、近期实质样本、贡献门禁、upstream/权重/官方测试集与指标。
2. **S1 静态**：patch check、语法、lint（含编码规范：禁止函数体内 `sys.exit()`、分号多语句、算术操作符缺空格）、冲突标记、路径、下载 check、三类主文档和候选清单。交付脚本必须通过 `ruff check` 或等价静态检查零错误。
3. **S2 功能**：干净环境安装、权重/功能输入、真实端到端输出；注明实际 device/provider。
4. **S3 L2**：NPU 完整精度/质量、NPU 性能、自动 compare 和报告；官方公开基线不可比或 patch 改语义时补 CPU 精度对照。
5. **S4 扩展**：并发、长稳、错误请求、cache 冷热、L3/业务集等用户明确项。

## 性能计时

- NPU 操作异步。纯模型/设备计时前后 synchronize；端到端计时明确数据、排队、后处理、网络和异步完成边界。
- 参数拒绝 `loop <= 0`、`record <= 0`、空数据集和非法 batch。
- 首次编译/cache miss 与稳定阶段分开。
- 正式性能至少独立执行 3 次，报告中位数；同时保留每次结果、样本数，适用时补分位数和峰值 HBM。
- CPU 性能不用于推断 NPU 加速比。

## 精度结论

- 直接对齐官方指标必须保持 checkpoint、完整数据集/split、配置、随机种子、预后处理和 evaluator 一致。
- 替换权重、tokenizer、聚类、阈值或子模型后重新跑任务指标，并声明不可直接比较项。
- 公开基线不可比、patch 改变算法语义或需要隔离版本漂移时，补同环境 `--device cpu` 精度对照；CPU 对照不用于性能。
- 小样例、输出文件或截图不能冒充任务精度。

## 证据失效

源码/patch、权重/config、镜像核心栈、目标硬件、数据 split、evaluator、性能参数或计时实现变化会使对应证据失效。README、`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md` 和 PR 数字必须追溯到当前有效证据。
