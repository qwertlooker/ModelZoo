# Datadog Toto 2.0 22m 适配记录

## 版本边界

- 日期 2026-10-27；目标仓 master `933c1f91b60896f76719d9373e3aa7a9972b2eac`，本次最新变更为 GSNet。拟合入路径 `ACL_PyTorch/built-in/nlp/Toto-2.0`，固定提交 contents API 返回 404，当前按新增准备；提交上库前重查目标路径与 README。
- 参考目录：ACL_PyTorch/built-in/nlp/chronos-2，最后实质变更 058e04c7d4aa09bc535e30380bf9133b0ea25c60（2026-06-09），同为时序预测与在线 PyTorch；选择该目录是因为同领域或同在线 PyTorch 路线，提交前仍需按最新实质变更重新采样。
- 上游：https://github.com/DataDog/toto/tree/1d3e1c97ccf615016d38fe910602c905ae31fbb7；权重 revision `685e4ae3e2be8d8998025e53dd98e7fdcb296a89`，模型和 config SHA 见 README。许可证 Apache-2.0。

## 上库文件清单与贡献门禁

拟交付 `README.md`、`model.py`、`common.py`、`prepare_series.py`；`test_model.py` 是否随候选纳入以目标仓自测试规则为准。排除 `NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、权重、数据、缓存、日志、输出。贡献门禁、许可证文件及 `modelzoo_level.txt` 的当前适用要求仍待核实。源码未修改，上游使用固定版本；本目录为新增入口，不声称已上库。

## 方案与验证

输入 `[batch,variates,512]`，输出 `[batch,variates,horizon]` 中位数及九个分位数；本入口仅支持全部观测值有限的输入，不自动把缺失值置为已观测。 22m、4m 和 Toto 1.0 的权重与输出不同；ETTh1 未覆盖缺失、变长和多分块场景。WQL 是本输入集合的均值分位损失，分母为目标绝对值之和，不是 BOOM 聚合 CRPS。

2026-10-27 执行 7 项离线回归、结构门禁，以及独立目录 Python 隔离模式的 `--help`。ETTh1 真数据准备：测试集 stride 128 得到 22 个窗口，并核对 scaler 和输入布局。入口检查 Toto 模型和配置源码 SHA，拒绝忽略的字段或配置值变化。未执行真实权重加载、模型功能输出、NPU 精度/性能或完整 clean-room；CPU mock 不能作为真实运行证据。Windows/WSL 未安装 ruff，未运行该 lint；目标硬件和配套环境未确定。

## 2026-10-27 本轮检视修正（以本节为当前结论）

评测与比较强制中位预测等于第 5 个（0.5）分位数，WQL 使用 float64 累积，补充预测样本数和输入边界校验。

- 目标仓 commit：本轮重新 `git ls-remote` 确认目标 master 为 `933c1f91b60896f76719d9373e3aa7a9972b2eac`，固定该 commit 的 `git ls-tree` 未发现五个拟合入目录，本次仍按新增准备，无需替换目标现有文件。
- 最新参考目录：本轮按目标路径历史重新筛选样本：同任务 chronos-2 实质提交 `058e04c7d4aa09bc535e30380bf9133b0ea25c60`（2026-06-09），与当前目标的目录 tree 均为 `0abeb7008df7adc3e3e064b94ffe6724e4246c83`，内容未变；优先用于四个时序模型。另核对更新的 `nlp/Laya`（`e779117e8cfeba296a7105e26adac0741ab04768`，2026-10-12），作为在线 PyTorch 环境/patch 的补充，任务并非预测；DINOv2 采用 `cv/GSNet`（`933c1f91b60896f76719d9373e3aa7a9972b2eac`，2026-10-27）作同域近期参考，不照搬其分割指标。现有 checkout 为 shallow，已额外取得并检查 chronos-2 原始提交和 tree；GitHub commits API 本轮返回 403 rate limit，不声称取得完整 API 历史。
- 当前贡献规范重新查询为 `Ascend/modelzoo@5eab9a4921c7f12edb555079836429a8f285cd1f`。规范要求参考许可证、自测试和 built-in `modelzoo_level.txt`；已补真实参考 LICENSE 和离线回归。正式 NPU 自测未运行，不能生成 OK 状态文件；CLA、版权头归属和 PR CI 仍为上库阻塞项，不声明获豁免。
- 精确候选清单（覆盖上文暂定清单）：`LICENSE`, `README.md`, `common.py`, `model.py`, `performance.py`, `prepare_series.py`, `test_model.py`, `test_performance.py`。单元测试只验证工具，不能替代目标仓正式自测试。继续排除两份内部文档、worklog、权重、数据、日志、缓存、upstream/、.codex-reference/ 和性能 sidecar。
- LICENSE 来源：https://raw.githubusercontent.com/DataDog/toto/1d3e1c97ccf615016d38fe910602c905ae31fbb7/LICENSE；SHA256 `58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd`，保留上游许可原文；数据/权重授权仍按各来源核对。
- 每次推理新增 `.performance.json`，持久化同步计时、吞吐、结果 SHA 和设备/软件元信息；`performance.py` 在至少三次不同运行 ID 且配置一致时报告中位数，拒绝 CPU 或混用输入/设备/参数。结果校验不等于真实性认证，CANN/驱动及物理卡映射须另留原始环境日志。
- 控制机实际为 Windows、Python 3.14.6、torch 2.13.0+cpu，未安装 torch_npu；本轮未安装/修改全局依赖或启动外部服务。已执行 13 项离线测试、ruff check、README 审计；所有离线数字均非真实模型精度或 NPU 性能。独立候选目录的 CLI 与 fixture 重放记录在本轮项目内 `.codex-reference/five-model-review-20261027/checks.json`；完整模型安装/加载与 NPU 重放仍未执行。
- 本轮官方 Ascend/pytorch 配套表（源码 `94dd5bf251144e6227e2b5228ad9f878878c0bf4`）当前最高列 PyTorch/torch_npu 2.12.0，而固定 Toto 仍需 torch ≥2.14；未找到满足声明的公开配套，属于明确软件栈阻塞。不能降版本绕过或换成 Toto 1.0/4m。

- 独立目录以 `python -E -s` 执行所有 CLI --help 均通过；TTM-R2/DINOv2/Toto/MOMENT 的隔离单测因禁用用户 site 后缺少 numpy 而失败，原始错误已保留；使用现有控制机依赖重放 fixture 仅验证候选边界，不代表干净环境安装成功。仓库结构审计器另有 6 项回归通过；未找到独立 Git Bash；使用 Windows 的 WSL bash -n，通过 UTF-8/LF stdin 对全部 README 命令块完成语法检查（不执行命令），未执行 Linux 安装。target-readiness 实际返回非零，S3/正式自测及验收仍未满足。

- 本轮固定 revision 的权重 URL HEAD 复查：config.json: HTTP 200, 593 字节, model.safetensors: HTTP 200, 87669368 字节；只检查元信息，没有下载权重或运行模型。完整记录保存在本轮项目内 `weight-head-checks.json`。

当前状态: S0 分析和部分静态入口已准备；S1 尚待完整静态审查及依赖核验，S2/S3/S4 未完成。计划上库前需独立候选、完整 clean-room 及门禁。
