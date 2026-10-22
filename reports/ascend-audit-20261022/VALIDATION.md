# 执行记录

日期：2026-10-22，Asia/Shanghai。

- 实时查询目标仓master：`dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202`，独立浅克隆HEAD相同。
- 全文件树95,062项，README/清单3,402份；重点提取39个目录530份源码、补丁、依赖与说明文本。
- `python reports/ascend-audit-20261022/collect.py`：通过；仅采集Git对象，不运行第三方源码。
- `python reports/ascend-audit-20261022/build_report.py`：通过；119项按原顺序输出。
- `python reports/ascend-audit-20261022/test_report.py`：6/6通过，覆盖清单完整性、逐项字段、固定commit源码路径存在、CSV与JSON一致、相邻版本隔离、外链与组件结论。
- `git diff --check`：通过。
- 未执行：NPU/CPU模型推理、依赖安装、权重与数据下载校验、官方精度评测、三次独立NPU性能测量、clean-room和候选上库门禁。本次为证据核查，没有连接执行机或准备模型运行环境。
- Qwen旧外链重定向到新版教程索引，已继续打开新版Embedding/Reranker专项教程；外部latest文档未固定commit，其内容不等同主仓固定快照内代码。
- YOLOv8的GPL外链由主仓ACL README明确列出；本次外部正文读取超时，不补写未经读取的权重与指标。

## 提交与推送边界

原工作区16个已有修改不属于本任务，不暂存、不提交。本次只提交`reports/ascend-audit-20261022/`的核查资料和生成工具；下载的目标仓、原始全文及缓存不提交。

已核实当前分支main、配置远程origin指向`https://github.com/qwertlooker/ModelZoo.git`，不存在本次任务开始前的待推送本地提交。计划仅普通推送当前提交到origin/main，不强推。

`git push --dry-run origin HEAD:main` 返回：

```text
remote: Permission to qwertlooker/ModelZoo.git denied to pcb1990.
fatal: unable to access 'https://github.com/qwertlooker/ModelZoo.git/': The requested URL returned error: 403
```

预检未通过，因此不继续执行真实推送。本地提交保留，不更改凭据或切换外部账号。
