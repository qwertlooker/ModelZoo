# TimesFM 2.5 NPU 适配记录

## 版本边界

- 检查日期：2026-10-27；目标仓 master `933c1f91b60896f76719d9373e3aa7a9972b2eac`。
- 拟合入路径：`ACL_PyTorch/built-in/nlp/TimesFM-2.5`；固定目录清单及本次目标仓路径历史检查未发现同名目录，按新增准备。提交前仍需重新核对目标路径和 README。
- 参考模型：`ACL_PyTorch/built-in/nlp/chronos-2`，同为时序预测及在线 PyTorch；该目录最后实质变更 `058e04c7d4aa09bc535e30380bf9133b0ea25c60`，2026-06-09。适配路径参考其推理和评测分离。准备上库时需再次按最新实质变更重新采样。
- Transformers `v5.3.0` 提交 `aad13b87ed59f2afcfaebc985f403301887a35fc`；模型 revision `5a9806b9b291fad9233b5249d88263f1846304d3`；权重与 config 的 SHA 及来源见 README。
- ETDataset 提交 `1d16c8f4f943005d613b5bc962e9eeb06058cf07`，ETTh1 文件 SHA 已通过读取公网原文件核算；最终运行机仍需下载后重新校验。

## 方案与上库文件清单

新增 `timesfm_npu.py`，复用上游 Transformers 模型，无上游源码补丁。NPU 显式设备、FP32、默认 SDPA，eager 仅为显式选项；保留显式 CPU/CUDA。动态输入、NPU 算子和软件配套尚未实测。ETTh1 原始量纲 MSE/MAE 是本地协议；BOOM 全量公开基准所需原生模型和预测配置另行对齐。

候选文件暂列 `README.md`、`timesfm_npu.py`、必要的许可证和目标仓贡献门禁文件；`test_timesfm_npu.py` 是否纳入按目标仓自测试要求确认。排除内部 `NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、权重、数据、日志、输出、缓存和临时目录。许可证与 PR 门禁、`modelzoo_level.txt` 的当前规则及适用性尚待重新核实，不能标记候选就绪。

## 验证和状态

2026-10-27：本地读取并校验固定 ETTh1 原文件（2,589,657 字节、17,420 行），实际数据准备生成 2,785 个窗口、19,495 条通道序列；运行 4 项离线检查数据、指标、失败传播和 NPU 不可用时不回退，结构审计及原报告 31 项回归通过。独立目录以 Python `-I` 执行 `--help`，只验证最低 CLI 路径；`ruff` 在 Windows 和 WSL 均未安装，未运行该 lint。Windows `curl` 返回 `Recv failure: Connection was reset`，随后以标准库 HTTP 读取原始文件、验 SHA 并短暂保存于项目内验证目录；验证后删除本次临时数据和 manifest。未执行真实权重加载、真实模型预测、NPU 算子、完整数据集质量、三次独立性能、完整 clean-room 或上库门禁。没有硬件时不得将静态代码或 fixture 记作 NPU 适配成功。

## 2026-10-27 本轮检视修正（以本节为当前结论）

增加 Transformers 5.3.0 实际版本检查、显式 FP32 加载、默认 SDPA（可显式选 eager）、manifest 行数与结果模型/上下文/horizon 校验。注意力路径改变后，任何旧推理/精度/性能证据均需重跑；当前未有可沿用的真实推理证据。

- 目标仓 commit：本轮重新 `git ls-remote` 确认目标 master 为 `933c1f91b60896f76719d9373e3aa7a9972b2eac`，固定该 commit 的 `git ls-tree` 未发现五个拟合入目录，本次仍按新增准备，无需替换目标现有文件。
- 最新参考目录：本轮按目标路径历史重新筛选样本：同任务 chronos-2 实质提交 `058e04c7d4aa09bc535e30380bf9133b0ea25c60`（2026-06-09），与当前目标的目录 tree 均为 `0abeb7008df7adc3e3e064b94ffe6724e4246c83`，内容未变；优先用于四个时序模型。另核对更新的 `nlp/Laya`（`e779117e8cfeba296a7105e26adac0741ab04768`，2026-10-12），作为在线 PyTorch 环境/patch 的补充，任务并非预测；DINOv2 采用 `cv/GSNet`（`933c1f91b60896f76719d9373e3aa7a9972b2eac`，2026-10-27）作同域近期参考，不照搬其分割指标。现有 checkout 为 shallow，已额外取得并检查 chronos-2 原始提交和 tree；GitHub commits API 本轮返回 403 rate limit，不声称取得完整 API 历史。
- 当前贡献规范重新查询为 `Ascend/modelzoo@5eab9a4921c7f12edb555079836429a8f285cd1f`。规范要求参考许可证、自测试和 built-in `modelzoo_level.txt`；已补真实参考 LICENSE 和离线回归。正式 NPU 自测未运行，不能生成 OK 状态文件；CLA、版权头归属和 PR CI 仍为上库阻塞项，不声明获豁免。
- 精确候选清单（覆盖上文暂定清单）：`LICENSE`, `README.md`, `performance.py`, `test_performance.py`, `test_timesfm_npu.py`, `timesfm_npu.py`。单元测试只验证工具，不能替代目标仓正式自测试。继续排除两份内部文档、worklog、权重、数据、日志、缓存、upstream/、.codex-reference/ 和性能 sidecar。
- LICENSE 来源：https://raw.githubusercontent.com/huggingface/transformers/aad13b87ed59f2afcfaebc985f403301887a35fc/LICENSE；SHA256 `77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049`，保留上游许可原文；数据/权重授权仍按各来源核对。
- 每次推理新增 `.performance.json`，持久化同步计时、吞吐、结果 SHA 和设备/软件元信息；`performance.py` 在至少三次不同运行 ID 且配置一致时报告中位数，拒绝 CPU 或混用输入/设备/参数。结果校验不等于真实性认证，CANN/驱动及物理卡映射须另留原始环境日志。
- 控制机实际为 Windows、Python 3.14.6、torch 2.13.0+cpu，未安装 torch_npu；本轮未安装/修改全局依赖或启动外部服务。已执行 8 项离线测试、ruff check、README 审计；所有离线数字均非真实模型精度或 NPU 性能。独立候选目录的 CLI 与 fixture 重放记录在本轮项目内 `.codex-reference/five-model-review-20261027/checks.json`；完整模型安装/加载与 NPU 重放仍未执行。

- 独立目录以 `python -E -s` 执行所有 CLI --help 均通过；TTM-R2/DINOv2/Toto/MOMENT 的隔离单测因禁用用户 site 后缺少 numpy 而失败，原始错误已保留；使用现有控制机依赖重放 fixture 仅验证候选边界，不代表干净环境安装成功。仓库结构审计器另有 6 项回归通过；未找到独立 Git Bash；使用 Windows 的 WSL bash -n，通过 UTF-8/LF stdin 对全部 README 命令块完成语法检查（不执行命令），未执行 Linux 安装。target-readiness 实际返回非零，S3/正式自测及验收仍未满足。

- 本轮固定 revision 的权重 URL HEAD 复查：config.json: HTTP 200, 914 字节, model.safetensors: HTTP 200, 925187448 字节；只检查元信息，没有下载权重或运行模型。完整记录保存在本轮项目内 `weight-head-checks.json`。

当前状态: S0 分析和部分静态入口已准备；S1 尚待完成完整静态审查及目标环境依赖核验，S2/S3/S4 未完成。
