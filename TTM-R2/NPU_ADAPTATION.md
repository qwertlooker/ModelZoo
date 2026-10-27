# IBM Granite TinyTimeMixer R2 512→96 适配记录

## 版本边界

- 日期 2026-10-27；目标仓 master `933c1f91b60896f76719d9373e3aa7a9972b2eac`，本次最新变更为 GSNet。拟合入路径 `ACL_PyTorch/built-in/nlp/TTM-R2`，固定提交 contents API 返回 404，当前按新增准备；提交上库前重查目标路径与 README。
- 参考目录：ACL_PyTorch/built-in/nlp/chronos-2，最后实质变更 058e04c7d4aa09bc535e30380bf9133b0ea25c60（2026-06-09），同为时序预测与在线 PyTorch；选择该目录是因为同领域或同在线 PyTorch 路线，提交前仍需按最新实质变更重新采样。
- 上游：https://github.com/ibm-granite/granite-tsfm/tree/9d0b839f997eb4918822a43036012ce8c277687c；权重 revision `d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f`，模型和 config SHA 见 README。许可证 Apache-2.0。

## 上库文件清单与贡献门禁

拟交付 `README.md`、`model.py`、`common.py`、`prepare_series.py`；`test_model.py` 是否随候选纳入以目标仓自测试规则为准。排除 `NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、权重、数据、缓存、日志、输出。贡献门禁、许可证文件及 `modelzoo_level.txt` 的当前适用要求仍待核实。源码未修改，上游使用固定版本；本目录为新增入口，不声称已上库。

## 方案与验证

PyPI `granite-tsfm==0.3.8` 元数据声明 torch ≥2.10、<2.11 和 Python ≥3.11、<3.14。社区报告使用 torch 2.9 并以 `--no-deps` 安装，不满足包声明；目标设备与官方配套范围需重新确认，不能以该案例宣布当前依赖链兼容。依赖来源：https://pypi.org/pypi/granite-tsfm/0.3.8/json 。

实际使用 PyPI 0.3.8 wheel，SHA256 `ec1c857c2b1807569ab661cfe54b5b266790b651e5aa3d35da4e2ce2e8d96d84`，其中模型源码 SHA256 `6c4cbcaa417ea09f643ee83918524e7bca6ac369b6b3500a5073c7ccc868dc14`，入口按后者校验；仓库 9d0b... 是公开资料审计快照，不宣称 wheel 来自该提交。

配置检查将旧字段 `torch_dtype` 对应到 Transformers 4.57.6 的 `dtype`，并将 `transformers_version` 视作序列化版本元数据；其余原始字段逐项核对。字段落地不证明每个字段都会被模型前向使用。

输入 `[batch,512,channels]`，标签/输出 `[batch,96,channels]`；使用训练分区 `[0,8640)` 逐通道均值和总体标准差，记录 scaler。评价 MSE/MAE 为标准化量纲。 需要复核 0.3.8 与固定 checkpoint 的配置字段实际使用；不同 R2 分支或 research-r2 不可替代。输入七通道是独立预测接口，不等于已验证跨通道混合。

2026-10-27 执行 7 项离线回归、结构门禁，以及独立目录 Python 隔离模式的 `--help`。ETTh1 真数据准备：测试集 stride 128 得到 22 个窗口，并核对 scaler 和输入布局。未执行真实权重加载、模型功能输出、NPU 精度/性能或完整 clean-room；CPU mock 不能作为真实运行证据。Windows/WSL 未安装 ruff，未运行该 lint；目标硬件和配套环境未确定。

状态：S0 分析和部分静态入口已准备；S1 尚待完整静态审查及依赖核验，S2/S3/S4 未完成。计划上库前需独立候选、完整 clean-room 及门禁。
