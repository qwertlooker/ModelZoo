---
name: ascend-model-adapter
description: 将 Hugging Face、GitHub、PyTorch、ONNX、Paddle、vLLM、TorchAir 或研究模型适配到华为昇腾 Ascend NPU，并生成或审查 Ascend ModelZoo-PyTorch ACL_PyTorch/built-in 上库材料。用于 GPU/上游实现到 NPU 的路线选择、镜像与依赖审计、源码 patch、ONNX/OM/ATC、torch_npu/TorchAir/vLLM-Ascend 推理、精度性能验证、S0-S4、README/NPU_ADAPTATION/ACCEPTANCE_PLAN、项目级证据日志、clean-room 和 PR dry run；不将 CPU-only 结果宣称为 NPU 验证。
---

# Ascend Model Adapter

把模型适配或 PR 审查当作**证据驱动的工程流程**，不要把模板完整或脚本可启动等同于适配完成。

## 先确定模式

- **adapt**：从上游模型生成新的 ModelZoo 适配工程。读取 `references/workflow-adaptation.md`。
- **review**：审查已有目录或 PR。读取 `references/workflow-pr-review.md` 和
  `references/pr-review-heuristics.md`。对交付目录中的 README.md 先运行
  `python3 "$SKILL_DIR/scripts/audit_readme.py"`，阻塞项未清零前不输出
  "可上库"或"验证通过"结论。
- **resume**：继续已有项目。先读取该项目自己的 `worklog.json`，从当前状态后的第一道门禁继续；不要重建或覆盖日志。

开始前把本 `SKILL.md` 所在目录的绝对路径记为 `SKILL_DIR`。所有 bundled script 均用 `python3 "$SKILL_DIR/scripts/<name>.py"` 调用；不要假设当前目录是 Skill 根目录，也不要使用可能不存在的 `python` 命令。

如果当前仓存在 `AGENTS.md` 和 `模型NPU 适配标准流程.md`，先重新读取；仓库硬约束优先于本 Skill 的通用默认值。每次模型适配、补验或上库复核都重新查询目标 `ModelZoo-PyTorch` 的 `master` HEAD、拟合入路径及近期实质变更样本，不复用历史快照。

## 不可违反的约束

1. 推理、评测和 benchmark 的默认设备为 NPU；物理卡用 `ASCEND_RT_VISIBLE_DEVICES` 选择，CPU 只能是显式 baseline/fallback。
2. CPU-only、静态检查、ONNX 导出或材料生成不能标记为 NPU 通过；未实测结果写 `待 NPU 验证`。
3. 固定上游 commit/revision、权重及配套 config/tokenizer/label map；用户提供的 artifact 优先，替换必须说明。
4. 精度优先对齐同 checkpoint、官方完整数据集/split、评测脚本和预后处理。没有可复现官方指标时才以 CPU/upstream 对齐替代。
5. 性能只报告可复现 NPU 结果；官方/GPU 数据只作来源明确的参考，不展示本地 CPU 性能。正式性能至少独立执行 3 次并报告中位数，区分纯模型、端到端、首次编译和 CPU fallback。
6. 不虚构 ATC、OM、NPU 精度、NPU 性能、CI 或 review 证据。评论抓取失败、PR 路径不匹配和历史流水线状态都不能当作当前 head 的结论。
7. 在容器内处理不可信仓库和权重；先审计 requirements/setup/pyproject 和下载脚本，再安装或执行。容器命令模板按用户要求保留在 `references/output-contract.md`，但缓存、凭据和宿主机目录只挂载任务实际需要的范围。
8. 私有 JSON 过程日志不上库；每个项目使用独立日志，不能复用固定文件覆盖另一个项目。它只辅助追踪，不替代工作区必须维护的 `README.md`、`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`。

## 项目日志与状态门禁

创建一次项目日志：

```bash
python3 "$SKILL_DIR/scripts/project_log.py" init \
  --workspace <工作区> --project-id <category>__<model> \
  --model-url <URL> --target-path ACL_PyTorch/built-in/<category>/<model> \
  --mode adapt --route undecided --hardware-model unconfirmed
```

日志位于 `<工作区>/.ascend-adaptation/<project-id>/worklog.json`，并在 Git 工作区中写入本地 `.git/info/exclude`。技术状态统一使用仓库的 S0–S4，不能另造“完成”状态：

`未定级 → S0 分析完成 → S1 静态适配完成 → S2 功能验证完成 → S3 L2 迁移对齐完成 → S4 扩展验收通过`

每次提升前用 `project_log.py add-evidence` 记录命令、退出码、claim、环境、日志路径和产物 SHA256。证据类型及门禁见 `references/validation-contract.md`；字段解释和问题/决策记录见 `references/adaptation-process-log.md`。
S0 前用 `project_log.py update-context` 固定目标仓 commit/目标路径状态、source revision/checkpoint；S2 前固定 route 和对外硬件型号。达到 S3 只说明技术验收完成；还必须以 `target_audit` 和 `clean_room_replay` 证据执行 `set-target-ready`，才能写“上库候选就绪”。正式合入只能由目标仓 PR merged/commit 证明。

## 选择路线与参考样本

先做上游审计和最小探针，再按 `references/route-selection.md` 选择 `onnx-om`、`torch-npu`、`torchair`、`vllm-ascend` 或 `hybrid`。不要让 `auto` 占位值进入交付物。

每次任务都先查询目标 master HEAD；再按类别/关键词采样同领域、同推理形态的近期实质变更目录：

```bash
python3 "$SKILL_DIR/scripts/modelzoo_sampler.py" --count 12 --category <category> \
  --keyword <route-or-task> --clone --clone-dir /tmp/modelzoo-<task> \
  --out /tmp/modelzoo-<task>.md
```

采样失败时读取 `references/modelzoo-sampling.md`，并把离线样本标为“历史参考”，不能声称最新。写 patch 时按需读取 `references/patch-modification-patterns.md`；领域信号见 `references/adaptation-heuristics.md`。

## PR 审查证据

PR 评论必须与变更目录精确绑定：

```bash
python3 "$SKILL_DIR/scripts/gitcode_pr_review_sampler.py" \
  --pr-path <PR号>=ACL_PyTorch/built-in/<category>/<model> \
  --out /tmp/pr-review.md --json-out /tmp/pr-review.json
```

`--pr-path` 默认严格校验 diff；不要用 `--allow-unverified` 生成正式审查结论。拉取 PR 当前 head/merge ref 后运行：

```bash
python3 "$SKILL_DIR/scripts/modelzoo_pr_quickcheck.py" <repo> \
  --target ACL_PyTorch/built-in/<category>/<model> --strict
```

随后执行 patch dry run、数据最小样例、脚本 `--help`、单样例推理、精度和性能核验。在含 `tools/audit_model_delivery.py` 的工作仓中同时运行基础审计；准备上库时追加 `--target-readiness --target-path ...`，并从独立候选目录 clean-room 重放 README 最低正式路径。历史 CI 只用于发现模式；是否通过以当前 head 对应的最新流水线为准。

采样结果必须确认 discussion 的 `fetched/total` 完整且按 `created_at` 查看最新意见；截断或缺页按证据缺口处理。每次 push 后重新确认 PR head 已更新，并等待该 head 之后启动的文档、CodeCheck、SCA 等门禁进入终态；旧失败标签被移除或流水线显示 running 都不等于通过。

## 资源导航

- 适配执行与阶段产物：`references/workflow-adaptation.md`
- PR 审查执行：`references/workflow-pr-review.md`
- 路线决策：`references/route-selection.md`
- 验证层级、证据和状态：`references/validation-contract.md`
- ModelZoo 交付目录、README、保留的容器模板：`references/output-contract.md`
- 依赖、数据、pipeline、patch 领域信号：`references/adaptation-heuristics.md`
- patch 具体模式：`references/patch-modification-patterns.md`
- PR 检视规则：`references/pr-review-heuristics.md`
- 项目级日志字段：`references/adaptation-process-log.md`
- 离线样本与采样边界：`references/modelzoo-sampling.md`

用户要求“完成适配和验证”时默认目标至少为 S3。若只准备材料，停在真实 S0/S1/S2 并列出升级证据；S3/S4 与“上库候选就绪”分别报告，不能混为一谈。
