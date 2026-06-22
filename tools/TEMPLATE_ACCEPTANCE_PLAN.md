# <MODEL_NAME> 验收方案与结果

## 原始测试集

| 数据集 | split | 样本规模 | 来源 | 版本 / commit |
|---|---|---:|---|---|
| `<DATASET>` | `<SPLIT>` | `<COUNT>` | `<URL>` | `<VERSION>` |

## 官方指标

| 来源 | checkpoint | metric | normalizer / 后处理 | decode / 推理参数 | 结果 |
|---|---|---|---|---|---:|
| `<PAPER_OR_MODEL_CARD>` | `<CHECKPOINT>` | `<METRIC>` | `<NORMALIZER>` | `<PARAMS>` | `<VALUE>` |

## CPU / CUDA 回归

| 条件 | 是否必须 | 命令 | 输出 |
|---|---|---|---|
| `<CONDITION>` | `<YES_OR_NO>` | `<COMMAND>` | `<OUTPUT_PATH>` |

## NPU 验收

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 infer.py \
  --device npu \
  --weights weights \
  --input eval_data/manifest.jsonl \
  --output results/npu
```

## 功能验证

| 输入 | 数量 | 命令 | 通过标准 | 结果 |
|---|---:|---|---|---|
| `<FUNCTIONAL_INPUT>` | `<COUNT>` | `<COMMAND>` | `<PASS_CRITERIA>` | `<RESULT>` |

## L2 精度 / 质量

| 数据集 | 样本数 | metric | baseline | NPU | 阈值 | 结论 |
|---|---:|---|---:|---:|---:|---|
| `<DATASET>` | `<COUNT>` | `<METRIC>` | `<BASELINE>` | `<NPU>` | `<THRESHOLD>` | `<PASS_OR_FAIL>` |

## 性能

| 输入规模 | 设备 | batch size | 指标 | baseline | NPU | 结论 |
|---|---|---:|---|---:|---:|---|
| `<INPUT_SIZE>` | `<DEVICE>` | `<BATCH_SIZE>` | `<METRIC>` | `<BASELINE>` | `<NPU>` | `<PASS_OR_FAIL>` |

## 最低正式验收清单

- [ ] 权重版本和完整性检查完成；
- [ ] 数据准备命令生成固定 manifest/meta；
- [ ] NPU 推理写入独立输出目录；
- [ ] evaluator 或官方等价 metric 命令完成；
- [ ] compare 脚本按阈值返回成功或失败；
- [ ] `README.md` 中最低路径已 clean-room 重放。

## 报告模板

| 项目 | 证据 | 状态 |
|---|---|---|
| 原始 baseline | `<SOURCE_AND_VALUE>` | `<DONE_OR_MISSING>` |
| patch 回归 | `<COMMAND_AND_OUTPUT>` | `<DONE_OR_NOT_APPLICABLE_OR_MISSING>` |
| NPU 精度 / 质量 | `<COMMAND_AND_OUTPUT>` | `<DONE_OR_MISSING>` |
| NPU 性能 | `<COMMAND_AND_OUTPUT>` | `<DONE_OR_MISSING>` |
| 上库候选 | `<CLEAN_ROOM_RECORD>` | `<DONE_OR_MISSING>` |
