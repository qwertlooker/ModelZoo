# 模型适配工作流

## 目录

1. 目标仓基线与项目初始化
2. 上游、权重和指标审计（S0）
3. 静态适配（S1）
4. 功能验证（S2）
5. NPU L2 与扩展验收（S3/S4）
6. 上库候选门禁

## 1. 目标仓基线与项目初始化

每次任务实时执行：

```bash
git ls-remote https://gitcode.com/Ascend/ModelZoo-PyTorch.git refs/heads/master
```

确认拟合入路径、目标路径是否已存在及本次新增/替换/增量范围。按最后实质变更时间选择同领域、同推理形态样本，记录样本目录 commit/date 和选择原因；批量格式/链接更新不能当作最新实质样本。

在工作仓中为模型维护：

```text
README.md
NPU_ADAPTATION.md
ACCEPTANCE_PLAN.md
```

若仓内有 `tools/init_model.py`，优先用它生成骨架；否则可用 `scaffold_adapter.py --layout workspace`。正式 ACL 候选另用明确的 `--layout target` 组装，不能把内部文档混入。私有 `worklog.json` 每项目一个，只辅助证据追踪，不替代这三份文档。`NPU_ADAPTATION.md` 记录目标仓检查日期/commit、拟合入路径、参考目录、贡献门禁、候选文件清单和排除项。

## 2. 上游、权重和指标审计（S0）

1. 固定 upstream commit、模型变体、权重 revision/SHA、config/tokenizer/辅助模型和 license。
2. 盘点官方/公开 dataset、split、样本规模、metric、normalizer/后处理、decode 参数、精度与性能来源。
3. 检查 requirements/setup/pyproject、子模块、vendor、在线下载和许可证，确认不会覆盖 CANN 配套框架栈。
4. 搜索 CUDA、custom extension、动态 shape、control flow、硬编码后端和 pipeline 组件。
5. 形成可执行闭环计划：数据准备 → 推理 → evaluator → compare → report；只写计划而没有入口不算交付。

项目日志记录 `target_baseline` 和 `upstream_audit`，固定 target/upstream/weight 上下文后才能设置 S0。

## 3. 静态适配（S1）

- 读取 `route-selection.md`，用最小探针决定路线；记录拒绝其他路线和 CPU fallback 的原因。
- 优先 patch 上游入口；新增入口直接维护。上游修改基于固定 commit 生成可重复 patch。
- 默认 `--device npu`，物理卡只用 `ASCEND_RT_VISIBLE_DEVICES`；保持 CPU/CUDA 原行为，不静默 fallback。
- 必需依赖前置 import；仅设备后端可条件导入。禁止宽泛捕获后继续、未验证 monkey patch 和伪造兼容层。
- 子进程、patch、下载、数据、compare 失败必须非零退出。
- 完成 `git apply --check`、语法、lint、`--help`、下载 check、三类文档和候选清单检查。

记录 `static_check`；有 patch 时另记 `patch_check`，再设置 S1。

## 4. 功能验证（S2）

- 在干净环境安装依赖，准备真实权重和功能输入，至少在 CPU/CUDA/NPU 一种设备上完成真实端到端输出。
- `--help` 不应因 torch_npu、权重或非必要推理依赖缺失而失败。
- dummy/单条样例只证明功能链路，不证明精度或性能。
- CPU 精度对照可复用 NPU 环境并显式 `--device cpu`；不用 CPU 性能推断 NPU 加速。
- 为输出写独立目录和 sidecar，记录命令、版本、权重/manifest SHA、device/provider。

记录 `functional_inference` 后设置 S2。没有 NPU 时最多到 S2，且只能声明实际验证的设备。

## 5. NPU L2 与扩展验收（S3/S4）

S3 必须同时具备：

1. NPU 完整精度/质量结果；与同 checkpoint、数据集/split、evaluator 和参数的官方/公开基线比较。
2. 公开基线不可比、patch 改算法语义或需隔离版本漂移时，补同环境 CPU 精度对照。
3. NPU 性能至少独立运行 3 次并报告中位数；区分纯模型、端到端、首次编译和 CPU fallback。
4. 可执行 compare/report，阈值失败返回非零；阈值来源写入 `ACCEPTANCE_PLAN.md`。

记录 `npu_accuracy`、`npu_performance`、`comparison_report` 后设置 S3。长稳、L3、业务集等仅在用户要求并有 `extended_validation` 时设置 S4。

## 6. 上库候选门禁

- `README.md` 自包含，只引用正式候选文件；`NPU_ADAPTATION.md` 和 `ACCEPTANCE_PLAN.md` 默认留在工作仓而不进入最终 ACL 候选。
- 运行 strict quickcheck、数据 fixture、patch dry run 和仓内 `tools/audit_model_delivery.py <model_dir>`。
- 准备上库时运行 `--target-readiness --target-path ACL_PyTorch/built-in/<领域>/<模型目录>`。
- 按 `NPU_ADAPTATION.md` 候选清单组装独立目录，从 README clean-room 重放最低正式路径，不借用 upstream/cache/内部日志。
- 记录 `target_audit` 和 `clean_room_replay`，且技术状态至少 S3，才能设置 `target_ready=true`。

“S3/S4”“上库候选就绪”“正式合入”是三个不同结论；正式合入必须由目标仓 PR merged/commit 证明。
