# TimesFM 2.5 NPU 适配记录

## 版本边界

- 检查日期：2026-10-27；目标仓 master `153c97646d940f0eefefe81cbc891693d7243eaa`。
- 拟合入路径：`ACL_PyTorch/built-in/nlp/TimesFM-2.5`；固定目录清单及本次目标仓路径历史检查未发现同名目录，按新增准备。提交前仍需重新核对目标路径和 README。
- 参考模型：`ACL_PyTorch/built-in/nlp/chronos-2`，同为时序预测及在线 PyTorch；该目录最后实质变更 `058e04c7d4aa09bc535e30380bf9133b0ea25c60`，2026-06-09。适配路径参考其推理和评测分离。准备上库时需再次按最新实质变更重新采样。
- Transformers `v5.3.0` 提交 `aad13b87ed59f2afcfaebc985f403301887a35fc`；模型 revision `5a9806b9b291fad9233b5249d88263f1846304d3`；权重与 config 的 SHA 及来源见 README。
- ETDataset 提交 `1d16c8f4f943005d613b5bc962e9eeb06058cf07`，ETTh1 文件 SHA 已通过读取公网原文件核算；最终运行机仍需下载后重新校验。

## 方案与上库文件清单

新增 `timesfm_npu.py`，复用上游 Transformers 模型，无上游源码补丁。NPU 显式设备、FP32、eager；保留显式 CPU/CUDA。动态输入、NPU 算子和软件配套尚未实测。ETTh1 原始量纲 MSE/MAE 是本地协议；BOOM 全量公开基准所需原生模型和预测配置另行对齐。

候选文件暂列 `README.md`、`timesfm_npu.py`、必要的许可证和目标仓贡献门禁文件；`test_timesfm_npu.py` 是否纳入按目标仓自测试要求确认。排除内部 `NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、权重、数据、日志、输出、缓存和临时目录。许可证与 PR 门禁、`modelzoo_level.txt` 的当前规则及适用性尚待重新核实，不能标记候选就绪。

## 验证和状态

2026-10-27：本地读取并校验固定 ETTh1 原文件（2,589,657 字节、17,420 行），实际数据准备生成 2,785 个窗口、19,495 条通道序列；运行 4 项离线检查数据、指标、失败传播和 NPU 不可用时不回退，结构审计及原报告 31 项回归通过。独立目录以 Python `-I` 执行 `--help`，只验证最低 CLI 路径；`ruff` 在 Windows 和 WSL 均未安装，未运行该 lint。Windows `curl` 返回 `Recv failure: Connection was reset`，随后以标准库 HTTP 读取原始文件、验 SHA 并短暂保存于项目内验证目录；验证后删除本次临时数据和 manifest。未执行真实权重加载、真实模型预测、NPU 算子、完整数据集质量、三次独立性能、完整 clean-room 或上库门禁。没有硬件时不得将静态代码或 fixture 记作 NPU 适配成功。

状态：S0 分析和部分静态入口已准备；S1 尚待完成完整静态审查及目标环境依赖核验，S2/S3/S4 未完成。
