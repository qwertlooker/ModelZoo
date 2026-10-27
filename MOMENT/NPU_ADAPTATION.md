# AutonLab MOMENT-1-large，重建及训练预测头 适配记录

## 版本边界

- 日期 2026-10-27；目标仓 master `933c1f91b60896f76719d9373e3aa7a9972b2eac`，本次最新变更为 GSNet。拟合入路径 `ACL_PyTorch/built-in/nlp/MOMENT`，固定提交 contents API 返回 404，当前按新增准备；提交上库前重查目标路径与 README。
- 参考目录：ACL_PyTorch/built-in/nlp/chronos-2，最后实质变更 058e04c7d4aa09bc535e30380bf9133b0ea25c60（2026-06-09），同为时序预测与在线 PyTorch；选择该目录是因为同领域或同在线 PyTorch 路线，提交前仍需按最新实质变更重新采样。
- 上游：https://github.com/moment-timeseries-foundation-model/moment/tree/38f7310ad594100747ca2a8357e9c7ca7d323e0e；权重 revision `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`，模型和 config SHA 见 README。许可证 MIT。

## 上库文件清单与贡献门禁

拟交付 `README.md`、`model.py`、`common.py`、`prepare_series.py`、`train_head.py`；`test_model.py` 是否随候选纳入以目标仓自测试规则为准。排除 `NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、权重、数据、缓存、日志、输出。贡献门禁、许可证文件及 `modelzoo_level.txt` 的当前适用要求仍待核实。源码未修改，上游使用固定版本；本目录为新增入口，不声称已上库。

## 方案与验证

输入 `[batch,channels,512]`，预测输出 `[batch,channels,192]`；外部标准化只用训练分区。重建模式输出输入长度 512，默认全观测，不是缺失值补全验收。 官方预训练仅重建头；预测头必须先训练并保存 `head.safetensors` 及同名 `.json` 元数据，入口检查 SHA、训练步数、预测长度和 backbone。该元数据只验证记录一致性，仍需检查训练日志与数据真实性。

2026-10-27 执行 6 项离线回归、结构门禁，以及独立目录 Python 隔离模式的 `--help`。ETTh1 真数据准备：预测测试集 stride 128 得到 22 个窗口，训练子集 63 个窗口；重建测试子集 19 个窗口，核对完全位于测试区间。 未执行真实权重加载、模型功能输出、NPU 精度/性能或完整 clean-room；CPU mock 不能作为真实运行证据。Windows/WSL 未安装 ruff，未运行该 lint；目标硬件和配套环境未确定。

## 2026-10-27 本轮检视修正（以本节为当前结论）

训练前拒绝 NaN/Inf 学习率，补充输入与样本数边界校验和可保存的性能报告。训练配方、模型权重与 head 加载顺序保持不变。

- 目标仓 commit：本轮重新 `git ls-remote` 确认目标 master 为 `933c1f91b60896f76719d9373e3aa7a9972b2eac`，固定该 commit 的 `git ls-tree` 未发现五个拟合入目录，本次仍按新增准备，无需替换目标现有文件。
- 最新参考目录：本轮按目标路径历史重新筛选样本：同任务 chronos-2 实质提交 `058e04c7d4aa09bc535e30380bf9133b0ea25c60`（2026-06-09），与当前目标的目录 tree 均为 `0abeb7008df7adc3e3e064b94ffe6724e4246c83`，内容未变；优先用于四个时序模型。另核对更新的 `nlp/Laya`（`e779117e8cfeba296a7105e26adac0741ab04768`，2026-10-12），作为在线 PyTorch 环境/patch 的补充，任务并非预测；DINOv2 采用 `cv/GSNet`（`933c1f91b60896f76719d9373e3aa7a9972b2eac`，2026-10-27）作同域近期参考，不照搬其分割指标。现有 checkout 为 shallow，已额外取得并检查 chronos-2 原始提交和 tree；GitHub commits API 本轮返回 403 rate limit，不声称取得完整 API 历史。
- 当前贡献规范重新查询为 `Ascend/modelzoo@5eab9a4921c7f12edb555079836429a8f285cd1f`。规范要求参考许可证、自测试和 built-in `modelzoo_level.txt`；已补真实参考 LICENSE 和离线回归。正式 NPU 自测未运行，不能生成 OK 状态文件；CLA、版权头归属和 PR CI 仍为上库阻塞项，不声明获豁免。
- 精确候选清单（覆盖上文暂定清单）：`LICENSE`, `README.md`, `common.py`, `model.py`, `performance.py`, `prepare_series.py`, `test_model.py`, `test_performance.py`, `train_head.py`。单元测试只验证工具，不能替代目标仓正式自测试。继续排除两份内部文档、worklog、权重、数据、日志、缓存、upstream/、.codex-reference/ 和性能 sidecar。
- LICENSE 来源：https://raw.githubusercontent.com/moment-timeseries-foundation-model/moment/38f7310ad594100747ca2a8357e9c7ca7d323e0e/LICENSE；SHA256 `b91d51e5180bcbcbb06b2f13e124a71efc9df98049e6e9a188ad070ded2fa8c6`，保留上游许可原文；数据/权重授权仍按各来源核对。
- 每次推理新增 `.performance.json`，持久化同步计时、吞吐、结果 SHA 和设备/软件元信息；`performance.py` 在至少三次不同运行 ID 且配置一致时报告中位数，拒绝 CPU 或混用输入/设备/参数。结果校验不等于真实性认证，CANN/驱动及物理卡映射须另留原始环境日志。
- 控制机实际为 Windows、Python 3.14.6、torch 2.13.0+cpu，未安装 torch_npu；本轮未安装/修改全局依赖或启动外部服务。已执行 11 项离线测试、ruff check、README 审计；所有离线数字均非真实模型精度或 NPU 性能。独立候选目录的 CLI 与 fixture 重放记录在本轮项目内 `.codex-reference/five-model-review-20261027/checks.json`；完整模型安装/加载与 NPU 重放仍未执行。

- 独立目录以 `python -E -s` 执行所有 CLI --help 均通过；TTM-R2/DINOv2/Toto/MOMENT 的隔离单测因禁用用户 site 后缺少 numpy 而失败，原始错误已保留；使用现有控制机依赖重放 fixture 仅验证候选边界，不代表干净环境安装成功。仓库结构审计器另有 6 项回归通过；未找到独立 Git Bash；使用 Windows 的 WSL bash -n，通过 UTF-8/LF stdin 对全部 README 命令块完成语法检查（不执行命令），未执行 Linux 安装。target-readiness 实际返回非零，S3/正式自测及验收仍未满足。

- 本轮固定 revision 的权重 URL HEAD 复查：config.json: HTTP 200, 951 字节, model.safetensors: HTTP 200, 1385501256 字节；只检查元信息，没有下载权重或运行模型。完整记录保存在本轮项目内 `weight-head-checks.json`。

当前状态: S0 分析和部分静态入口已准备；S1 尚待完整静态审查及依赖核验，S2/S3/S4 未完成。计划上库前需独立候选、完整 clean-room 及门禁。
