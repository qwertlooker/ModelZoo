# <MODEL_NAME> NPU 适配记录

## 版本边界

| 项目 | 取值 |
|---|---|
| 检查日期 | `<CHECK_DATE>` |
| upstream 源码 | `<UPSTREAM_REPO>` |
| upstream commit | `<UPSTREAM_COMMIT>` |
| 权重来源 | `<WEIGHT_SOURCE>` |
| 权重 revision / SHA | `<WEIGHT_REVISION_OR_SHA>` |
| 辅助模型 / evaluator | `<AUXILIARY_REPOS_AND_COMMITS>` |
| 非目标变体 | `<EXCLUDED_VARIANTS>` |

## 目标仓快照

| 项目 | 取值 |
|---|---|
| 目标仓 master commit | `<TARGET_REPO_COMMIT>` |
| 拟合入路径 | `ACL_PyTorch/built-in/<DOMAIN>/<MODEL_DIR>` |
| 目标路径状态 | `<NEW_REPLACE_OR_INCREMENTAL>` |
| 最新参考目录 | `<REFERENCE_DIR>` |
| 最后实质变更 commit/date | `<REFERENCE_COMMIT_AND_DATE>` |
| 选择原因 | `<REFERENCE_REASON>` |

## 源码分析

- 原始入口：`<UPSTREAM_ENTRYPOINT>`
- 设备相关假设：`<CUDA_OR_DEVICE_ASSUMPTIONS>`
- 第三方源码变更：`<PATCH_SCOPE>`
- 保持不变的 CPU/CUDA 行为：`<UNCHANGED_BEHAVIOR>`

## 适配实现

| 文件 | 类型 | 说明 |
|---|---|---|
| `infer.py` | 新增 | `<PURPOSE>` |
| `patches/adapt.patch` | patch | `<PATCH_PURPOSE>` |

## 依赖与环境

- CPU 环境：`<CPU_ENV>`
- NPU 环境：`<NPU_ENV>`
- CANN / torch-npu：`<CANN_AND_TORCH_NPU>`
- ONNX Runtime CANN EP：`<ORT_CANN_IF_USED>`

## 验证

| 阶段 | 命令 / 证据 | 结果 |
|---|---|---|
| patch 检查 | `<GIT_APPLY_CHECK>` | `<RESULT>` |
| Python 语法 | `<PY_COMPILE>` | `<RESULT>` |
| `--help` | `<HELP_COMMAND>` | `<RESULT>` |
| 功能验证 | `<FUNCTIONAL_COMMAND>` | `<RESULT>` |
| NPU L2 | `<NPU_L2_COMMAND>` | `<RESULT>` |

## 未执行

| 项目 | 原因 | 补验条件 |
|---|---|---|
| `<ITEM>` | `<BLOCKER>` | `<WHAT_IS_NEEDED>` |

## 上库文件清单

### 候选文件

- `README.md`
- `infer.py`
- `requirements.txt`

### 排除项

- `NPU_ADAPTATION.md`
- `ACCEPTANCE_PLAN.md`
- `README_old.md`
- `upstream/`
- `weights/`
- `eval_data/`
- `results/`

## 许可证与 PR 门禁

- 上游许可证：`<LICENSE>`
- 文件版权头：`<COPYRIGHT_STATUS>`
- `modelzoo_level.txt`：`<APPLICABILITY>`
- 自测试入口：`<SELF_TEST_STATUS>`
- 豁免或差异：`<WAIVERS_OR_DIFFERENCES>`

## 当前状态

当前状态: `<S0_S1_S2_S3_OR_S4>`

升级到下一状态仍缺少：

- `<MISSING_EVIDENCE>`
