# ModelZoo 采样与离线兜底

## 当前快照

采样日期：2026-07-10。脚本从 GitCode `master` 分类页解析到 162 个可见目录：audio 29、cv 92、nlp 17、ocr 9、embedding 4、foundation_models 7、embodied_ai 4。

此数字和“最近更新时间”会快速变化，只用于说明离线快照范围；在线可用时必须重新运行 sampler，并记录 sparse clone 的 HEAD。

## 采样策略

- 先按目标 `--category` 过滤，再用任务/路线 `--keyword` 缩小范围；不要默认 clone 大量无关类别。
- 同任务/同路线样本优先于纯粹最新样本；必要时再补 1–2 个最新项目了解 README/CI 风格。
- `--clone-dir` 必须是不存在或空目录。脚本拒绝删除非空目录。
- 页面解析失败、过滤无结果或 clone 失败时明确报错；不能把历史参考声称为最新。
- 页面行中的 PR 只是候选信号，必须用 PR diff 验证对应 ModelZoo 路径。
- sparse clone 默认保留有限提交历史并输出每个目录最后路径变更 commit/date；若显示浅历史不可用，增加 `--history-depth` 或在完整 checkout 上重新查询。还要人工区分实质变更与仓级批量格式/链接更新。

## 2026-07-10 检视样本

本轮从 24 个近期目录得到 21 个唯一 PR，并用精确路径映射验证，覆盖：

- audio：Step_audio2_mini、DiariZen、YingMusic-SVC、Canary-1B。
- cv：LoFTR、RIFE、InstantID、PromptIR、F3Net、SAM2、SAM3、FocalFormer3D。
- embodied_ai：IsaacGR00T、vla/pi05、GraspNet。
- ocr：PaddleOCR-VL-1.5、PP-DocLayoutV2/V3。
- embedding、nlp、foundation_models：bge 系列、chronos-2、ProtBert、TabPFN、Chinese_CLIP。

评论样本暴露的新增通用问题已写入 `pr-review-heuristics.md`：子进程/patch helper 假成功、CLI 多值参数吞 flag、NPU 异步计时、下载到 HTML、生成配置来源、文件名拼写、数据生成边界和当前 head/历史 CI 混淆。

## 抓取边界

- GitCode HTML/API 可能变化，脚本是 best-effort 获取器，不是权威 PR API。
- `gitcode_pr_review_sampler.py --pr-path PR=PATH` 默认对 metadata、discussion、diff 和路径证据 fail-closed。
- `--allow-unverified` 只用于排障；其输出不能形成正式审查结论。
- 系统事件、push 记录和重复 CI 历史会被过滤/折叠；AI review 失败只表示没有获得意见。
- 正式结论仍需本地检查当前 PR head 和对应最新流水线。
