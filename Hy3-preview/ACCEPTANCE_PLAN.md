# Hy3-preview 验收计划

## 0. 版本边界

- checkpoint：`tencent/Hy3-preview@549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a`
- 官方仓：`38ac237dc0bf4329f054d09054aaf22fdaf6f553`
- vLLM/vllm-ascend：`v0.18.0rc1`
- NPU patch SHA256：`2e59facbbb4428c83f97e974f931e2dadeda418a073248bf3d2744038ea71735`
- 不包含 `Hy3-preview-Base`。

## 1. 原始测试集与官方指标

官方 instruct 模型卡公开：

| Benchmark | 官方分数 |
|---|---:|
| SWE-bench Verified | 74.4% |
| Terminal-Bench 2.0 | 54.4% |
| BrowseComp | 67.1% |
| WideSearch | 70.2% |

公开 benchmark 规模和口径：

| Benchmark | split/规模 | metric | 评测/后处理状态 |
|---|---|---|---|
| SWE-bench Verified | Verified test，500 个 GitHub issue | resolved rate | 需要固定 SWE-bench harness、容器和 agent；Hy3 模型卡未发布精确 revision/agent 配置 |
| Terminal-Bench 2.0 | Hy3 模型卡未发布精确 dataset version 和样本数 | task success rate | 需要固定 Harbor/Terminal-Bench task manifest 和 verifier |
| BrowseComp | 完整集，1,266 题 | accuracy/success rate | 依赖搜索工具和 judge；Hy3 模型卡未发布工具、judge 和采样次数 |
| WideSearch | 完整集，200 题（英文 100、中文 100） | benchmark success score | 依赖大规模信息收集、结构化输出和官方 grader |

官方服务推荐参数为 `temperature=0.9`、`top_p=1.0`，复杂任务使用
`reasoning_effort=high`，直答使用 `no_think`。这些是使用建议，不足以证明四项
benchmark 的实际 decode/agent 参数；benchmark-specific 参数官方未发布。

官方 Base 模型另发布 MMLU、MMLU-Pro、MMLU-Redux、ARC-Challenge、DROP、PIQA、SuperGPQA、SimpleQA、MBPP-plus、CRUXEval、LiveCodeBench-v6、GSM8K、MATH、CMath、C-Eval、CMMLU、MMMLU、INCLUDE 等表，但 Base 与本适配 instruct checkpoint 不可混用。

来源：

- <https://github.com/Tencent-Hunyuan/Hy3-preview>
- <https://github.com/SWE-bench/SWE-bench>
- <https://github.com/laude-institute/terminal-bench>
- <https://openai.com/index/browsecomp/>
- <https://widesearch-seed.github.io/>

官方 README 没有发布上述 instruct 指标对应的精确 dataset revision/manifest SHA、agent harness commit、工具环境、采样次数和完整 decode 参数。必须标记这些字段“官方未发布”，获得作者 recipe 前不得宣称精确复现。

## 2. 迁移对齐主线

### 2.1 服务数值/功能对齐

使用同一 checkpoint、vLLM commit、prompt JSONL、chat template 和 sampling 参数，比较 CUDA vLLM 与 NPU vLLM-Ascend：

1. 关闭 MTP，`temperature=0`、`top_p=1`，覆盖短/长上下文、中文/英文、JSON、tool call、reasoning parser。
2. 记录逐 token top-1、可用时记录 logprob。
3. 再开启 MTP，比较相同任务集。

通过条件：

- 请求成功率 100%，无权重缺失、rank/device、parser 错误；
- greedy top-1 token agreement `>= 99.5%`；
- 结构化 JSON/tool call schema 有效率 100%；
- MTP 开关后的任务级正确率不低于关闭 MTP 超过 0.5 个百分点。

### 2.2 官方任务对齐

优先选择可固定环境的 SWE-bench Verified 和 Terminal-Bench 2.0；BrowseComp/WideSearch 依赖外部搜索和动态网页，必须冻结工具、网页快照或采用官方环境。

- CUDA 和 NPU 使用同 agent harness、容器、工具和运行次数；
- 报告 pass@1/成功率及置信区间；
- NPU 相对 CUDA 下降 `<= 1.0` 个百分点；
- 只有官方未公开字段补齐且配置一致时，才与官方表比较。

## 3. 分层验收

| 层级 | 范围 |
|---|---|
| L0 | 模型加载、`/v1/models`、单轮 chat |
| L1 | 100 条固定 prompt；tool/reasoning/streaming/JSON |
| L2 | 固定公开 benchmark 子集和 32K 长上下文 |
| L3 | 官方四项 benchmark 完整 recipe；256K context |

L0/L1 不构成官方质量指标验收。

## 4. 性能与稳定性

分别报告关闭/开启 MTP，输入/输出长度 1K/1K、1K/4K、10K/1K，concurrency 1/8/64：

- TTFT、TPOT、ITL、E2E P50/P90/P99；
- input/output/total tokens/s、QPS、tok/s/NPU；
- 权重加载时间、峰值 HBM/host memory；
- 连续 2 小时服务成功率和内存趋势。

## 5. 当前验收状态

- 已通过：patch SHA 固定；在 vLLM `v0.18.0rc1` 精确 commit 上
  `git apply --check`；新增/修改 Python 文件 `compileall`。
- 未执行：295B 权重下载、16 卡加载、API/parser、CUDA/NPU token 对齐、
  四项 benchmark 和性能稳定性。
- 当前结论：patch 静态门禁通过；模型 NPU 验收未完成。

## 6. 报告模板

```text
模型/vLLM/vllm-ascend/patch SHA:
16卡硬件、CANN、镜像:
启动参数、环境变量、MTP:
prompt/benchmark revision和规模:
CUDA/NPU token agreement与任务指标:
tool/reasoning/streaming矩阵:
TTFT/TPOT/throughput/HBM/稳定性:
官方未发布字段与未完成项:
结论:
```
