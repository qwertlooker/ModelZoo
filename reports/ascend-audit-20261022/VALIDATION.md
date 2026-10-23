# 修正版验证记录

日期：2026-10-23，Asia/Shanghai。

- 119项按原始编号逐项保留；代码、权重、数据、性能、精度分别记录证据类型和链接。
- 12项回归测试通过：覆盖顺序、字段结构、非认证边界、易混指标回归、原始资源边界、固定ModelZoo文件存在、CSV/Markdown完整性、链接清单覆盖、GPU重查结果及可重复构建。
- HTTP HEAD检查403个唯一链接：368个成功、35个未确认；其中32个Hugging Face限流429，证书、华为云567、ScienceDirect403各1个。不把限流/访问限制等同于链接不存在。
- 固定提交链接158个；动态或未版本化链接245个。数量按URL路径形式统计，不表示全部外部来源已冻结。
- 原始文件/JSON读取和HTTP结果均不等于独立模型复测。
- 未执行：NPU/CPU模型运行、依赖安装、资产下载及SHA、完整任务精度、三次独立NPU性能、clean-room和上库门禁。
- 详细边界、修正内容和命令见REVIEW_FIXES.md；逐URL状态见link-check.json。

## 提交边界

本次只暂存reports/ascend-audit-20261022内的修订，不触碰其他16个既有修改。远程origin属于当前ModelZoo项目，计划普通推送当前main到origin/main；已执行push --dry-run，失败后没有继续实际推送、强推或更改凭据。本地提交保留。

错误原文：`remote: Permission to qwertlooker/ModelZoo.git denied to pcb1990.`；`fatal: unable to access 'https://github.com/qwertlooker/ModelZoo.git/': The requested URL returned error: 403`。

首次diff检查发现Windows生成文件CRLF被仓库识别为尾空白；已修正生成器固定LF并重新生成，复查通过。

## GPU/NPU分列验证

CSV共119行、25列，保留原始编号与顺序。GPU与NPU分别包含代码、权重、数据、性能、精度及每项独立链接列；两类结论和注意事项分开；对照原文保存在JSON。新增设备字段/列完整性与明确GPU/NPU数值归属测试。GPU尚未专项核实的字段明确保留待核实，不以NPU支持代替。

## 阅读版修订

CSV删除每格重复免责声明、内部证据类型和长段对照原文；保留GPU/NPU五项说明及独立链接列。未查到、未核实分别表达无证据和资源使用未确认。新增阅读版回归测试，保留指标数字、batch和不确定性。链接集合不变，本次未重复网络检查。

## GPU重新查询（本轮）

119项GPU侧均已重新查询；105项读取原始资料并记录具体发现，14项查到发布/论文/CPU实现或名称仍不明确，未认定GPU交付。其中87项找到明确官方GPU实现，其余类型见GPU_RECHECK.md。GPU查询完成标记只表示进行了资料检索，不表示五项证据全部闭合。生成器已接入gpu_research.py，阅读版不再用“存在上游链接”覆盖实际GPU结论。

新增回归覆盖：GPU结论正确显示、N1.6版本及硬件数值、YOLOv8和RT-DETR数值、WALL-WM未发权重、Swin-Unet最终权重未分享、平台渲染不能冒充模型支持。`git diff --check`通过。ModelZoo master重新在线查询仍为dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202。
