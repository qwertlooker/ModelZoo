# DINOv2 ViT-L/14 无 register + ImageNet linear4 分类头 验收计划

## 原始测试集与官方/公开指标

官方 ViT-L/14 无 register 的 ImageNet kNN 83.5%、linear 86.3%；复用匹配 head、预处理、评估范围及标签映射后才能比较。当前 HF backbone + Meta head 组合还未验证等价。 来源：https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md

输入图像/标签清单 CSV（列名 `path,label`；相对路径相对于清单目录，label 必须是 ImageNet 类别索引）；预处理为 `[N,3,224,224]`，输出 logits `[N,1000]`。仅当清单覆盖完整 ImageNet 验证集且标签映射正确时，top1/top5 才是全量结果。 分类头单独从 Meta 官方地址获取，尚未取得官方固定 SHA；运行前记录实际 SHA 只保证后续文件一致，不证明来源。HF backbone 的中间层归一化与 Meta 原始路径仍需逐层/数据集比较；图像批处理一次装入内存，大规模验证需分批及加权汇总。

## 功能验证

固定源码/依赖/权重/config/辅助 head 及 SHA；完成真实输入、设备、输出形状和缺失依赖失败验证。分别运行 NPU 和显式 CPU，保存同输入、同模型、同 head 的结果；阈值由真实验收要求决定。

## L2 精度与性能

完整数据集及正确 evaluator、预后处理/标签范围对齐后比较任务指标；没有可比公开指标时按相同数据和权重使用 CPU 结果辅助对齐。NPU 设备同步测量至少三次独立运行，取中位数，分别统计纯模型、端到端、峰值显存。禁止用 CPU 性能推算加速比或以 fixture 当任务精度。

## 最低正式验收与报告模板

记录目标卡型、驱动、CANN、Python、torch/torch_npu、上游和权重 SHA、数据/标签 SHA、命令、样本数、精度、数值误差、三次性能及不可比项。正式候选需通过目标仓贡献门禁及干净目录完整重放；当前 NPU 验证、完整数据精度、性能及 clean-room 均未执行。

## 入口与最低正式验收清单

- 数据准备：`prepare_data.py manifest/split/images`。
- 推理、评测和比较：同一模型入口的 `infer`、`evaluate`、`compare` 子命令；参数与 README 一致，失败必须非零退出。
- 性能：同一入口的 `performance --runs` 子命令，汇总至少三次独立 NPU 运行 sidecar；保留同步计时、输入/模型 SHA、设备和参数，CPU 报告不得用于 NPU 性能结论。
- 最低补验：确认硬件与配套环境、干净安装/import、权重/数据校验、真实 CPU/NPU 输出、完整范围精度对齐、有依据阈值比较、三次性能、端到端/首次编译/HBM、目标仓正式自测与候选完整重放。

开发回归集中在迁移仓 `tools/tests`，不进入正式候选。工具测试及 CLI/fixture 重放只能证明相应软件行为，不证明真实模型、NPU 精度或性能。入口和计时模块位置变化后，先前代码 SHA 对应的推理/性能证据不能直接沿用；当前仍无 S2/S3 实测证据。模型版本、权重、数据协议和主要计算逻辑保持原有边界。
