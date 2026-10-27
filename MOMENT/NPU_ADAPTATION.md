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

状态：S0 分析和部分静态入口已准备；S1 尚待完整静态审查及依赖核验，S2/S3/S4 未完成。计划上库前需独立候选、完整 clean-room 及门禁。
