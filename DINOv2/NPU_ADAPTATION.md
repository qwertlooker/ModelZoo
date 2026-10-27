# DINOv2 ViT-L/14 无 register + ImageNet linear4 分类头 适配记录

## 版本边界

- 日期 2026-10-27；目标仓 master `933c1f91b60896f76719d9373e3aa7a9972b2eac`，本次最新变更为 GSNet。拟合入路径 `ACL_PyTorch/built-in/cv/DINOv2`，固定提交 contents API 返回 404，当前按新增准备；提交上库前重查目标路径与 README。
- 参考目录：ACL_PyTorch/built-in/cv/GSNet，最后实质变更 933c1f91b60896f76719d9373e3aa7a9972b2eac（2026-10-27），近期 CV 在线 torch_npu、DINO 骨干参考；任务为语义分割，与本模型分类头不同；选择该目录是因为同领域或同在线 PyTorch 路线，提交前仍需按最新实质变更重新采样。
- 上游：https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8；权重 revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`，模型和 config SHA 见 README。许可证 Apache-2.0。

## 上库文件清单与贡献门禁

拟交付 `README.md`、`model.py`、`common.py`、`make_manifest.py`、`split_manifest.py`、`prepare_images.py`、`aggregate_metrics.py`；`test_model.py` 是否随候选纳入以目标仓自测试规则为准。排除 `NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、权重、数据、缓存、日志、输出。贡献门禁、许可证文件及 `modelzoo_level.txt` 的当前适用要求仍待核实。源码未修改，上游使用固定版本；本目录为新增入口，不声称已上库。

## 方案与验证

输入图像/标签清单 CSV（列名 `path,label`；相对路径相对于清单目录，label 必须是 ImageNet 类别索引）；预处理为 `[N,3,224,224]`，输出 logits `[N,1000]`。仅当清单覆盖完整 ImageNet 验证集且标签映射正确时，top1/top5 才是全量结果。 分类头单独从 Meta 官方地址获取，尚未取得官方固定 SHA；运行前记录实际 SHA 只保证后续文件一致，不证明来源。HF backbone 的中间层归一化与 Meta 原始路径仍需逐层/数据集比较；图像批处理一次装入内存，大规模验证需分批及加权汇总。

2026-10-27 执行 7 项离线回归、结构门禁，以及独立目录 Python 隔离模式的 `--help`。无 ImageNet 图片，未运行真实图像预处理或分类。未执行真实权重加载、模型功能输出、NPU 精度/性能或完整 clean-room；CPU mock 不能作为真实运行证据。Windows/WSL 未安装 ruff，未运行该 lint；目标硬件和配套环境未确定。

状态：S0 分析和部分静态入口已准备；S1 尚待完整静态审查及依赖核验，S2/S3/S4 未完成。计划上库前需独立候选、完整 clean-room 及门禁。
