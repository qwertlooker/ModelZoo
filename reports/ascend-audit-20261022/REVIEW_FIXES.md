# 2026-10-23复核修正

- 119项恢复为原始顺序的一项一行；上一轮聊天链接表遗漏的32项全部保留，包括无充分证据或对象未唯一确认的条目。
- 原始ModelZoo审计保存在modelzoo-snapshot.json，仅用于固定快照历史输入，不能作为当前互联网结论；REPORT.md、inventory.json/csv由revised_report.py统一生成。
- 每个字段记录说明、具体链接、证据类型。源码文件、源码目录、指导文档、原始资源入口分别标注；README不能独立证明代码运行通过。
- 新增DrivingSDK、LifeScience、MindSPONGE、PaddleX及社区路线；MinerU3.3不再误判只有2.0，RT-DETR第一代不再误判只有v2；DINOv2及Depth Anything V2组件线索明确限制。
- FastWAM链接改为固定commit的原始性能/精度JSON；明确单窗口、继续训练120步、非CUDA迁移对照和非显著精度提升。
- π0.5保留global batch128/64差异；TimesNet不将GPU/CPU耗时归为NPU性能；训练loss、算子一致性不当作任务质量。
- ModelZoo、DrivingSDK、PaddleX远端分支本次重新查询；LifeScience、ThermoMPNN、FastWAM采用已读取Git对象的固定commit，未声称它们仍为远端最新HEAD。
- 对外部动态来源不虚构commit。未独立闭合的权重、数据文件、许可和SHA明确保留未验证，不将原始仓库入口写成文件可用性已确认。

## 重建与复查

```text
python reports/ascend-audit-20261022/build_report.py
python reports/ascend-audit-20261022/check_links.py
python reports/ascend-audit-20261022/test_report.py
```

build_report.py只重建报告；check_links.py只进行有界HTTP HEAD，不下载权重、不执行第三方源码。HTTP成功仅说明请求成功，不能证明页面不是登录页、资源完整或技术结论成立。

source-evidence.md、code-findings.md、repository-files.txt保留原ModelZoo固定快照的采集证据，不是新增互联网资料的全部正文档案。

## 剩余未完成

外部动态来源冻结、全部上游源码逐文件复核、权重/数据下载及SHA校验、官方任务完整评测、三次独立NPU性能测量、clean-room均未执行。不认证S3/S4或上库就绪。
