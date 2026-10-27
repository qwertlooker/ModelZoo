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

状态：S0 分析和部分静态入口已准备；S1 尚待完整静态审查及依赖核验，S2/S3/S4 未完成。计划上库前需独立候选、完整 clean-room 及门禁。
