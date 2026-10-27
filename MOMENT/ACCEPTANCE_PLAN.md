# AutonLab MOMENT-1-large，重建及训练预测头 验收计划

## 原始测试集与官方/公开指标

官方教程 large、ETTh1 horizon 192、训练 head 一轮后 MSE 0.422/MAE 0.432；当前 FP32 训练路径尚未证明与教程 AMP 配方及数据处理完全一致。 来源：https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/tutorials/forecasting.ipynb

输入 `[batch,channels,512]`，预测输出 `[batch,channels,192]`；外部标准化只用训练分区。重建模式输出输入长度 512，默认全观测，不是缺失值补全验收。 官方预训练仅重建头；预测头必须先训练并保存 `head.safetensors` 及同名 `.json` 元数据，入口检查 SHA、训练步数、预测长度和 backbone。该元数据只验证记录一致性，仍需检查训练日志与数据真实性。

## 功能验证

固定源码/依赖/权重/config/辅助 head 及 SHA；完成真实输入、设备、输出形状和缺失依赖失败验证。分别运行 NPU 和显式 CPU，保存同输入、同模型、同 head 的结果；阈值由真实验收要求决定。

## L2 精度与性能

完整数据集及正确 evaluator、预后处理/标签范围对齐后比较任务指标；没有可比公开指标时按相同数据和权重使用 CPU 结果辅助对齐。NPU 设备同步测量至少三次独立运行，取中位数，分别统计纯模型、端到端、峰值显存。禁止用 CPU 性能推算加速比或以 fixture 当任务精度。

## 最低正式验收与报告模板

记录目标卡型、驱动、CANN、Python、torch/torch_npu、上游和权重 SHA、数据/标签 SHA、命令、样本数、精度、数值误差、三次性能及不可比项。正式候选需通过目标仓贡献门禁及干净目录完整重放；当前 NPU 验证、完整数据精度、性能及 clean-room 均未执行。
