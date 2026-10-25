# 修正版验证记录

日期：2026-10-23，Asia/Shanghai。

- 119项按原始编号逐项保留；代码、权重、数据、性能、精度分别记录证据类型和链接。
- 13项回归测试通过：覆盖顺序、字段结构、非认证边界、易混指标回归、原始资源边界、固定ModelZoo文件存在、CSV/Markdown完整性、链接清单覆盖、GPU重查、后续证据增补及可重复构建。
- 当前链接清单437个：407个有成功HEAD记录、30个未确认（27个Hugging Face限流429，证书、华为云567、ScienceDirect403各1个）。本轮只检查39个新增URL，全部成功；其余保留原检查日期，不冒充全部2026-10-25重查。不把限流/访问限制等同于链接不存在。
- 固定提交链接159个；动态或未版本化链接278个。数量按URL路径形式统计，不表示全部外部来源已冻结。
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

## 2026-10-25增补

- 保留119项、25列以及GPU/NPU五维结构。新增evidence_updates.json记录18项逐字段增补及17次GitHub正文读取的文件blob SHA；blob SHA不当作仓库commit使用。
- 补充AlphaFold3按tokens/A100/H100耗时，BIT/ChangeFormer切块与测试集指标，Swin-Unet与U-Net分开，Anomaly Transformer的point-adjust F1，InternVL2/3、LLaVA、Qwen2/2.5-VL的指定规模结果，Depth Anything度量/相对版本边界。
- 补充Cosmos-Predict2 GPU生成耗时及输出视频帧率区别、Reason1分阶段分口径评测、GigaBrain clean-only训练协议、LingBot基础/后训练/深度变体，撤回Pelican-VL未经确认的3B发布范围。
- 已记录原始资料冲突：LingBot论文与README的261 samples/s归属不一致；N1.6 LIBERO成功次数与部分百分比不一致；ChangeFormer论文写随机初始化而当前README脚本使用ADE预训练，DSIFN下载链接误指LEVIR。
- NPU字段未改动，设备检查日期分别保留；ModelZoo master重新查询仍为dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202。
- 13项回归通过，仅证明生成及已知归属规则；不证明全量分析完成。GPU仍有41项性能、14项精度保留字面“数值待核对”，还有其他具体待核说明。目标仍为全119项完整分析，不将本次18项增补当作完成。
