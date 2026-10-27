# TimesFM 2.5 NPU 适配记录

## 版本边界

- 检查日期：2026-10-27；目标仓 master `933c1f91b60896f76719d9373e3aa7a9972b2eac`。
- 拟合入路径：`ACL_PyTorch/built-in/nlp/TimesFM-2.5`；固定目录清单及本次目标仓路径历史检查未发现同名目录，按新增准备。提交前仍需重新核对目标路径和 README。
- 参考模型：`ACL_PyTorch/built-in/nlp/chronos-2`，同为时序预测及在线 PyTorch；该目录最后实质变更 `058e04c7d4aa09bc535e30380bf9133b0ea25c60`，2026-06-09。适配路径参考其推理和评测分离。准备上库时需再次按最新实质变更重新采样。
- Transformers `v5.3.0` 提交 `aad13b87ed59f2afcfaebc985f403301887a35fc`；模型 revision `5a9806b9b291fad9233b5249d88263f1846304d3`；权重与 config 的 SHA 及来源见 README。
- ETDataset 提交 `1d16c8f4f943005d613b5bc962e9eeb06058cf07`，ETTh1 文件 SHA 已通过读取公网原文件核算；最终运行机仍需下载后重新校验。

## 本轮目标仓与参考复核

- 检查日期：2026-10-27；目标仓 commit：`933c1f91b60896f76719d9373e3aa7a9972b2eac`，本轮重新 git ls-remote；五个目标路径均未发现同名目录，本次为新增候选准备。
- 近期参考目录：`ACL_PyTorch/built-in/nlp/chronos-2`，最后实质变更 `058e04c7d4aa09bc535e30380bf9133b0ea25c60`（2026-06-09）；同任务在线时序预测。其原始提交与当前目标目录 tree 均为 `0abeb7008df7adc3e3e064b94ffe6724e4246c83`；另核对 `nlp/Laya@e779117e8cfeba296a7105e26adac0741ab04768`（2026-10-12）作为更新的在线 PyTorch 环境/patch 补充，不套用其任务协议。
- 贡献门禁：规范来源为`Ascend/modelzoo@5eab9a4921c7f12edb555079836429a8f285cd1f`；参考许可证（License）已保留。正式在线推理自测、版权头归属、CLA、modelzoo_level.txt 与 PR CI 尚未满足或确认豁免，不生成虚假 OK 状态。

## 适配范围与收敛

继续复用 Transformers 5.3.0 的 TimesFm2_5ModelForPrediction，固定 FP32、默认 SDPA，eager 仅显式选择；ETTh1 使用原始量纲 MSE/MAE，与原生 BOOM 协议不直接等价。

本轮仅整理已新增的适配入口，未修改 pip/site-packages 或上游源码，因此没有源码 patch，不生成空补丁。复用上游模型类；本地评测协议与公开指标的可比性仍按验收计划逐项确认。

- 推理、评测、比较、同步计时与三次性能汇总集中到 `timesfm_npu.py`，仅暴露本模型支持的任务/metric。
- 开发回归移到迁移仓 tools/tests 的两份模型测试，去掉其他模型的不可达模板分支；正式候选不依赖这些测试。
- 既有结果校验、失败非零、设备不静默回退保持不变；本轮不变更权重、训练配方、CANN 或原始模型计算。

## 上库文件清单与排除项

精确候选清单：`README.md`, `LICENSE`, `timesfm_npu.py`。这些文件可直接复制到拟合入路径，不需要迁移仓 tools、缓存或旧模块。

排除 NPU_ADAPTATION.md、ACCEPTANCE_PLAN.md、README_old.md、全部开发回归、worklog、日志、数据、权重、输出 sidecar、upstream/ 和 .codex-reference/。正式 NPU 自测仍是缺失门禁，不能用开发单测替代。

- LICENSE 来源：https://raw.githubusercontent.com/huggingface/transformers/aad13b87ed59f2afcfaebc985f403301887a35fc/LICENSE；SHA256 `77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049`，保留上游许可原文；数据/权重授权仍按各来源核对。

## 本轮验证与状态

控制机为 Windows/Python 3.14.6/torch 2.13.0+cpu，无 torch_npu。回归包括迁移仓工具测试、CLI/负向路径、DINO 数据准备 fixture、性能报告边界，以及独立候选的无模型 CLI/轻量 fixture；精确命令、退出码和日志保存在 `.codex-reference/five-model-consolidation-20261027/checks.json`。未安装全局依赖或使用 NPU。未执行的检查及原因：控制机缺少 NPU 及目标运行栈，完整依赖安装、真实权重加载、CPU/NPU 功能、全量质量、三次 NPU 实测与完整 clean-room 均待验收。

本轮结果：63 项仓级回归和 57 项独立候选 fixture 重放通过；CLI、ruff、README/结构审计和 Bash 语法检查通过。独立候选 fixture 使用控制机现有依赖，未验证干净安装或真实模型。target-readiness 仍因 S3 和待验收项缺失而非零退出，保持未就绪。

本轮入口变化使旧代码 SHA 的功能/性能证据需重新核验；历史日志保留，不把历史 fixture 当成当前 NPU 证据。README 审计和结构审计结果与 target-readiness 分别记录。

当前状态: S0 分析和部分静态入口已准备；S1 尚待目标依赖链与完整静态门禁，S2/S3/S4 未完成，上库候选未就绪。
