# BEATs 完整验收方案

## 0. 验收目标与范围

本方案用于验收 `microsoft/unilm` 仓库 `beats/` 子目录中 BEATs 音频表示/音频分类模型在昇腾 NPU 上的适配结果。验收不能只依赖 1 秒 dummy wav，而应覆盖原始 BEATs 的两类主要能力：

- 预训练模型：从 16 kHz 音频提取帧级/片段级 representation；
- AudioSet fine-tuned 模型：输出多标签音频事件分类概率和 top-k 标签。

**当前适配边界**

- 源码：`microsoft/unilm`，`beats/` 子目录，当前记录 commit `833df7e7832e5064a281131ee64a481afa8e5b95`。
- 权重：官方 OneDrive 发布的 BEATs `.pt`；当前仓库尚未固定具体 checkpoint，正式验收必须明确 checkpoint 名称、来源链接和 SHA256。
- 当前 NPU patch 只处理 `BEATs.py::preprocess()` 中 `torchaudio.compliance.kaldi.fbank` 的 CPU 回退，模型主体仍在指定 device 上运行。

## 1. 原始模型能力与参考指标

### 1.1 功能能力

| 能力 | 原始代码支持 | 本适配验收要求 |
|---|---|---|
| tokenizer | `Tokenizers.extract_labels()` | 可选；仅当验收 tokenizer checkpoint 时执行 |
| representation | pre-trained BEATs `extract_features()` | 必须验证输出 tensor shape、dtype、device、数值有限 |
| AudioSet 分类 | fine-tuned BEATs `extract_features()` 返回多标签概率 | 必须验证 top-k 标签、概率范围、label_dict 映射 |
| batch | 示例中支持 batch 输入 | 验证 batch 1/2/4/8，记录最大可用 batch |
| 采样率 | 官方示例假设 16 kHz audio input | 验证 16 kHz mono；非 16 kHz 必须预处理或记录失败原因 |

### 1.2 公开精度参考

BEATs 论文/官方说明的代表性公开结果：

| 数据集 | 任务 | 公开参考 |
|---|---|---:|
| AudioSet-2M eval | 多标签音频事件分类 | mAP 50.6% |
| ESC-50 | 环境声分类 | Accuracy 98.1% |

说明：这些是论文级结果，依赖具体 BEATs iteration、fine-tuning 配置和评测脚本。NPU 适配验收的核心是同 checkpoint、同数据、同脚本下 NPU 相对 CPU/CUDA 不退化；只有在完整复现数据和配置时，才可宣称复现公开指标。

## 2. 数据集选择：大小、获取难度与用途

| 数据 | 规模/获取难度 | 覆盖能力 | 用途 | 建议 |
|---|---|---|---|---|
| `dummy_1s_16k.wav` | 1 秒，脚本生成，极易 | 加载、fbank、forward | L0 smoke | 必跑；不看准确率 |
| 自制小样本 | 10~50 条，5~15 分钟；需人工标注或只看 top-k 合理性 | top-k、batch、稳定性 | L1 功能回归 | 放 `test_data/eval_smoke/` 或内网存储 |
| ESC-50 | 2000 条、50 类、约 2.8 小时；公开易获取 | 单标签环境声分类 | L2 推荐精度 | 下载难度低，优先作为正式分类精度集 |
| AudioSet eval | 约 2 万条 10 秒 clip；YouTube 可用性波动，下载偏难 | 多标签音频事件分类 | L2/L3 原始主指标 | 可先用本地缓存子集，完整复现难度较高 |
| AudioSet-2M | 约 200 万条；下载和授权成本高 | 训练/大规模复现 | L3 研究级复现 | 不作为 NPU 适配最低验收要求 |

### 2.1 数据准备与评测解耦要求

吸取 Canary-1B/FLEURS 数据准备经验，BEATs 正式验收必须先生成本地固定 manifest，再运行评测：

| 阶段 | 要求 | 验收产物 |
|---|---|---|
| 准备 ESC-50 | 明确 split/fold；下载或解压后生成 `audio_filepath,label,duration,sample_id,split` manifest | `esc50_*.csv/jsonl` + `*.meta.json` |
| 准备 AudioSet 子集 | 明确 eval/balanced/unbalanced 与 shard/YouTube 列表；若抽样仍需完整 shard，记录原因和大小 | `audioset_*.csv/jsonl` + `*.meta.json` |
| 评测 | 只读取 manifest，复用 BEATs forward/infer 机制，不再触发下载 | `metrics.json`、逐样本预测 |

metadata 至少记录 dataset、split/fold、样本数、总时长、抽样 seed、下载源、文件大小和标签映射版本。正式 CPU/CUDA/NPU 对比必须使用同一份 manifest。

## 3. 分层验收

| 层级 | 数据量 | 必测内容 | 结论用途 |
|---|---:|---|---|
| L0 smoke | 1 条 dummy | checkpoint 加载、fbank CPU 回退、模型 forward、top-k/shape 输出 | 只证明链路可运行 |
| L1 功能回归 | 10~50 条 | pre-trained representation、fine-tuned classification、batch、重复运行 | 判断功能完整性 |
| L2 推荐验收 | ESC-50 全量或 AudioSet eval 子集 | accuracy/mAP、RTF/吞吐、NPU vs CPU/CUDA 一致性 | 正式交付验收 |
| L3 完整复现 | AudioSet eval 全量 + ESC-50 标准 split | 对齐论文/官方公开指标 | 发布级报告 |

## 4. 功能验收矩阵

| 用例 | checkpoint | 输入 | batch | 预期 |
|---|---|---|---:|---|
| 预训练 representation | BEATs pre-trained `.pt` | 16 kHz mono wav | 1 | 输出 feature tensor，shape 非空，数值有限 |
| AudioSet top-k | AudioSet fine-tuned `.pt` | 16 kHz mono wav | 1 | 输出 top-5 label/prob，prob 在 `[0,1]` 或符合模型输出定义 |
| batch 推理 | fine-tuned `.pt` | 多条等长/近似等长 wav | 2/4/8 | 输出条数等于输入条数，无 device mismatch |
| 重复运行 | fine-tuned `.pt` | 小样本集 | 1 | 连续 30 轮无崩溃、显存无持续增长 |
| 非 16 kHz 输入 | fine-tuned `.pt` | 8 kHz 或 44.1 kHz wav | 1 | 若不支持自动重采样，文档明确要求预处理为 16 kHz mono |

## 5. 精度验收

### 5.1 指标

| 任务 | 指标 | 工具 |
|---|---|---|
| 单标签分类 | top-1 accuracy、top-5 accuracy | sklearn / 自定义脚本 |
| 多标签分类 | mAP、AUC、d-prime 可选 | sklearn `average_precision_score` |
| 回归一致性 | top-k 一致率、logit 最大绝对/相对误差 | 自定义 CPU/CUDA/NPU 对比脚本 |

### 5.2 推荐通过条件

在同一 checkpoint、同一数据和同一评测脚本下：

- NPU 与 CPU/CUDA 的 top-1/top-5 预测一致率 ≥ 99%（允许极小浮点差异导致 tie-break 差异）；
- ESC-50 accuracy 相对 CPU/CUDA 下降 ≤ 0.2 个百分点；
- AudioSet 子集 mAP 相对 CPU/CUDA 下降 ≤ 0.2 个百分点；
- 若直接对公开结果，必须使用匹配 checkpoint 和官方评测配置，否则只能标注为“参考对照”。

## 6. 性能验收

| 指标 | 记录方式 | 最低要求 |
|---|---|---|
| 加载时间 | checkpoint load 到可推理 | 记录即可 |
| 单条延迟 | batch=1，排除/包含 fbank 各记录一份（如能拆分） | NPU 能稳定完成 |
| 吞吐 | batch=1/2/4/8/16，samples/s 或 audio seconds/s | batch 增大应有正向收益或记录瓶颈 |
| RTF/RTFx | 音频总时长 / wall time | 正式部署建议 RTFx > 1 |
| 峰值内存/HBM | `/usr/bin/time -v`、`npu-smi` | 记录最大可用 batch |
| 稳定性 | 30~100 轮循环 | 无崩溃、无持续显存泄漏 |

注意：当前 patch 将 fbank 前处理放在 CPU，因此性能报告应说明端到端耗时包含 CPU 前处理和 NPU forward，必要时分别度量。

## 7. 正式验收最低清单

- [ ] 固定一个预训练或 fine-tuned checkpoint，记录官方链接、文件名、SHA256。
- [ ] L0 dummy smoke test 通过。
- [ ] 至少 1 个 pre-trained representation 或 1 个 fine-tuned top-k 功能用例通过；若只验其中一种，需说明适配边界。
- [ ] batch=1/4/8 至少两档通过，或记录 OOM/瓶颈。
- [ ] ESC-50 全量或不少于 200 条有标签样本计算 accuracy；如果使用 AudioSet，则计算 mAP。
- [ ] 同数据 CPU/CUDA vs NPU 精度差异满足第 5.2 节。
- [ ] 记录性能：加载时间、batch=1/4 的吞吐、峰值内存/HBM。
- [ ] 连续 30 轮稳定性测试通过。

## 8. 报告模板

```markdown
# BEATs NPU 验收报告

## 环境
- 日期：
- NPU 型号/数量：
- CANN/驱动：
- Python/torch/torch-npu/torchaudio：
- upstream commit：
- checkpoint 名称/来源/SHA256：

## 功能
| 用例 | checkpoint | batch | 结果 | 备注 |
|---|---|---:|---|---|

## 精度
| 数据集 | 条数/时长 | 指标 | CPU/CUDA | NPU | 差异 | 是否通过 |
|---|---:|---|---:|---:|---:|---|

## 性能
| batch | 音频总时长 | wall time | samples/s | RTFx | 峰值 HBM | 是否通过 |
|---:|---:|---:|---:|---:|---:|---|

## 结论
- 通过/不通过：
- 风险：
- 后续补验：
```

## 9. 参考来源

- BEATs 官方代码：<https://github.com/microsoft/unilm/tree/master/beats>
- BEATs 论文：<https://arxiv.org/abs/2212.09058>
- AudioSet：<https://research.google.com/audioset/>
- ESC-50：<https://github.com/karolpiczak/ESC-50>
