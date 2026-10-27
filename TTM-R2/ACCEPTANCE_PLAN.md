# IBM Granite TinyTimeMixer R2 512→96 验收计划

## 原始测试集与官方/公开指标

ETTh1 context 512/horizon 96，官方 Granite R2 原始结果表 zero-shot MSE 0.363；需逐项核对外部标准化、测试切分、模型分支及 evaluator 后才可比较。 来源：https://github.com/ibm-granite/granite-tsfm/blob/9d0b839f997eb4918822a43036012ce8c277687c/notebooks/hfdemo/tinytimemixer/full_benchmarking/combined_results-granite-r2.csv

输入 `[batch,512,channels]`，标签/输出 `[batch,96,channels]`；使用训练分区 `[0,8640)` 逐通道均值和总体标准差，记录 scaler。评价 MSE/MAE 为标准化量纲。 需要复核 0.3.8 与固定 checkpoint 的配置字段实际使用；不同 R2 分支或 research-r2 不可替代。输入七通道是独立预测接口，不等于已验证跨通道混合。

## 功能验证

固定源码/依赖/权重/config/辅助 head 及 SHA；完成真实输入、设备、输出形状和缺失依赖失败验证。分别运行 NPU 和显式 CPU，保存同输入、同模型、同 head 的结果；阈值由真实验收要求决定。

## L2 精度与性能

完整数据集及正确 evaluator、预后处理/标签范围对齐后比较任务指标；没有可比公开指标时按相同数据和权重使用 CPU 结果辅助对齐。NPU 设备同步测量至少三次独立运行，取中位数，分别统计纯模型、端到端、峰值显存。禁止用 CPU 性能推算加速比或以 fixture 当任务精度。

## 最低正式验收与报告模板

记录目标卡型、驱动、CANN、Python、torch/torch_npu、上游和权重 SHA、数据/标签 SHA、命令、样本数、精度、数值误差、三次性能及不可比项。正式候选需通过目标仓贡献门禁及干净目录完整重放；当前 NPU 验证、完整数据精度、性能及 clean-room 均未执行。
