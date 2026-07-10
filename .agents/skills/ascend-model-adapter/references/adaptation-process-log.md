# 项目级适配日志

每个项目只使用自己的私有日志：

```text
<workspace>/.ascend-adaptation/<category>__<model>/worklog.json
```

不要把多个项目写入同一个固定文件，也不要把 `.ascend-adaptation` 提交到 ModelZoo。`scripts/project_log.py init` 会拒绝覆盖已有日志，并在 Git 工作区的 `.git/info/exclude` 中加入本地排除项。该 JSON 不替代模型工作区必须维护的 `README.md`、`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`。

## 基本字段

- `project_id`：稳定项目标识，推荐 `<category>__<model>`。
- `model_url`、`target_repo_commit`、`target_path`、`target_path_exists`、`mode`、`route`、`hardware_model`、`patch_required`。
- `source_revision`、`checkpoint`：审计后用 `update-context` 从 `unconfirmed` 改为固定值；不适用时显式写 `not-applicable`。
- `status`：未定级或 S0–S4；`target_ready` 与技术状态分开。
- `evidence`：命令、退出码、claim、环境、日志和产物 SHA256。
- `decisions`：路线、权重、fallback、依赖和口径决策。
- `issues`：症状、无效尝试、根因、修复和复用价值。

## 添加证据

```bash
python3 "$SKILL_DIR/scripts/project_log.py" add-evidence \
  --workspace <工作区> --project-id <category>__<model> \
  --type npu_accuracy --command '<完整命令>' --exit-code 0 \
  --claim '指定完整 split 的 NPU metric 已产生' \
  --environment 'Atlas 800I A2; CANN ...; torch_npu ...' \
  --log-path <日志> --artifact <输出文件>
```

失败命令同样记录真实非零退出码；它不会解锁状态。不要为通过门禁修改退出码或把静态检查记成 NPU 证据。

## 提升状态

前序状态和该级全部证据齐全后再执行，例如当前已是 S2 且 S3 三项证据已记录：

```bash
python3 "$SKILL_DIR/scripts/project_log.py" set-status \
  --workspace <工作区> --project-id <category>__<model> \
  --status S3
```

脚本只允许逐级提升。S0 后 target/upstream/weight baseline 冻结，S2 后 route/硬件冻结；需要改变时另建项目日志，避免旧证据解锁新上下文。达到 S3/S4 后仍需 `target_audit` 和 `clean_room_replay`，再运行 `set-target-ready`。

## 必须记录的问题

- 排查耗时较长、反复出现或依赖外部提示。
- 导致路线、权重、评测配置或 CPU fallback 变化。
- patch/安装/下载/helper 错误未传播，曾造成假成功。
- accuracy/performance 口径、数据 split 或计时边界发生变化。
- 动态 shape、custom op、cache、长时间编译、依赖冲突、数据生成或许可证问题。

完成后只把用户复现所需结论提炼到 README FAQ；账号、内部路径、临时日志和完整排障历史留在项目日志中。
