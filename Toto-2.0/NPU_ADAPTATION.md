# Datadog Toto 2.0 22m 适配记录

## 版本边界

- 日期 2026-10-27；目标仓 master `933c1f91b60896f76719d9373e3aa7a9972b2eac`，本次最新变更为 GSNet。拟合入路径 `ACL_PyTorch/built-in/nlp/Toto-2.0`，本轮固定目标 commit 的 git ls-tree 未发现该路径，按新增准备。
- 参考目录：ACL_PyTorch/built-in/nlp/chronos-2，最后实质变更 058e04c7d4aa09bc535e30380bf9133b0ea25c60（2026-06-09），同为时序预测与在线 PyTorch；选择该目录是因为同领域或同在线 PyTorch 路线，提交前仍需按最新实质变更重新采样。
- 上游：https://github.com/DataDog/toto/tree/1d3e1c97ccf615016d38fe910602c905ae31fbb7；权重 revision `685e4ae3e2be8d8998025e53dd98e7fdcb296a89`，模型和 config SHA 见 README。许可证 Apache-2.0。

## 本轮目标仓与参考复核

- 检查日期：2026-10-27；目标仓 commit：`933c1f91b60896f76719d9373e3aa7a9972b2eac`，本轮重新 git ls-remote；五个目标路径均未发现同名目录，本次为新增候选准备。
- 近期参考目录：`ACL_PyTorch/built-in/nlp/chronos-2`，最后实质变更 `058e04c7d4aa09bc535e30380bf9133b0ea25c60`（2026-06-09）；同任务在线时序预测。其原始提交与当前目标目录 tree 均为 `0abeb7008df7adc3e3e064b94ffe6724e4246c83`；另核对 `nlp/Laya@e779117e8cfeba296a7105e26adac0741ab04768`（2026-10-12）作为更新的在线 PyTorch 环境/patch 补充，不套用其任务协议。
- 贡献门禁：规范来源为`Ascend/modelzoo@5eab9a4921c7f12edb555079836429a8f285cd1f`；参考许可证（License）已保留。正式在线推理自测、版权头归属、CLA、modelzoo_level.txt 与 PR CI 尚未满足或确认豁免，不生成虚假 OK 状态。

## 适配范围与收敛

固定 toto-2 2.0.0、dd-unit-scaling 0.1.0、Python >=3.12、torch >=2.14；固定源码与配置哈希仍强制校验。公开昇腾配套记录最高到 torch 2.12，尚无满足声明的已验证组合，属于阻塞项；不能以改包版本或替换 Toto 变体绕过。保留九分位数、中位数一致性与 float64 WQL 累积。

本轮仅整理已新增的适配入口，未修改 pip/site-packages 或上游源码，因此没有源码 patch，不生成空补丁。复用上游模型类；本地评测协议与公开指标的可比性仍按验收计划逐项确认。

- 推理、评测、比较、同步计时与三次性能汇总集中到 `model.py`，仅暴露本模型支持的任务/metric。
- 开发回归移到迁移仓 tools/tests 的两份模型测试，去掉其他模型的不可达模板分支；正式候选不依赖这些测试。
- 既有结果校验、失败非零、设备不静默回退保持不变；本轮不变更权重、训练配方、CANN 或原始模型计算。

## 上库文件清单与排除项

精确候选清单：`README.md`, `LICENSE`, `model.py`, `prepare_series.py`。这些文件可直接复制到拟合入路径，不需要迁移仓 tools、缓存或旧模块。

排除 NPU_ADAPTATION.md、ACCEPTANCE_PLAN.md、README_old.md、全部开发回归、worklog、日志、数据、权重、输出 sidecar、upstream/ 和 .codex-reference/。正式 NPU 自测仍是缺失门禁，不能用开发单测替代。

- LICENSE 来源：https://raw.githubusercontent.com/DataDog/toto/1d3e1c97ccf615016d38fe910602c905ae31fbb7/LICENSE；SHA256 `58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd`，保留上游许可原文；数据/权重授权仍按各来源核对。

## 本轮验证与状态

控制机为 Windows/Python 3.14.6/torch 2.13.0+cpu，无 torch_npu。回归包括迁移仓工具测试、CLI/负向路径、DINO 数据准备 fixture、性能报告边界，以及独立候选的无模型 CLI/轻量 fixture；精确命令、退出码和日志保存在 `.codex-reference/five-model-consolidation-20261027/checks.json`。未安装全局依赖或使用 NPU。未执行的检查及原因：控制机缺少 NPU 及目标运行栈，完整依赖安装、真实权重加载、CPU/NPU 功能、全量质量、三次 NPU 实测与完整 clean-room 均待验收。

本轮结果：63 项仓级回归和 57 项独立候选 fixture 重放通过；CLI、ruff、README/结构审计和 Bash 语法检查通过。独立候选 fixture 使用控制机现有依赖，未验证干净安装或真实模型。target-readiness 仍因 S3 和待验收项缺失而非零退出，保持未就绪。

本轮入口变化使旧代码 SHA 的功能/性能证据需重新核验；历史日志保留，不把历史 fixture 当成当前 NPU 证据。README 审计和结构审计结果与 target-readiness 分别记录。

当前状态: S0 分析和部分静态入口已准备；S1 尚待目标依赖链与完整静态门禁，S2/S3/S4 未完成，上库候选未就绪。
