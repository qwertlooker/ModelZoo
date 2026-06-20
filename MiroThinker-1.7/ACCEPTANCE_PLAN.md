# MiroThinker-1.7 验收计划

## 0. 版本边界

- checkpoint：`miromind-ai/MiroThinker-1.7@1a42014ce72e1025fdbf3c48d54545715ab3eea8`
- agent framework：`MiroMindAI/MiroThinker@370f98361553ddf787bedc5745760e04114cb161`
- benchmark archive：`MiroFlow-Benchmarks@09900fb9f7297b853f56e1b785491494e93ac85d` 的
  `data_20251115_password_protected.zip`
- archive SHA256：`35816f69ba5f0d2baf45b248c68dd4a8e0f9b30cac6f41076f44099d5073f377`
- vLLM/vllm-ascend：`v0.17.0rc1`
- 不包含 `MiroThinker-1.7-mini`、v1.5/v1.0 或 proprietary H1。

## 1. 原始测试集与官方指标

官方对 235B `MiroThinker-1.7` 公布：

| Benchmark | 官方分数 |
|---|---:|
| BrowseComp | 74.0% |
| BrowseComp-ZH | 75.3% |
| GAIA-Val-165 | 82.7% |
| HLE-Text | 42.9% |

固定官方 archive 后的原始评测口径：

| Benchmark | 数据文件/规模 | metric/judge | 运行参数 |
|---|---|---|---|
| BrowseComp | `browsecomp/standardized_data.jsonl`，1,266 题 | pass@1；`gpt-4.1-2025-04-14` LLM judge | 3 runs，`mirothinker_1.7_keep5_max300` |
| BrowseComp-ZH | `browsecomp_zh/standardized_data.jsonl`，289 题 | pass@1；中文 BrowseComp LLM judge | 3 runs，`mirothinker_1.7_keep5_max300` |
| GAIA-Val-165 | `gaia-2023-validation/standardized_data.jsonl`，165 题 | pass@1；当前 framework 使用 GAIA text-style LLM judge | 8 runs，`mirothinker_1.7_keep5_max200` |
| HLE-Text | `hle-text-2158/standardized_data_original.jsonl`，2,158 题 | pass@1；HLE LLM judge | 3 runs，`mirothinker_1.7_keep5_max200` |

共同参数：`MAX_CONTEXT_LENGTH=262144`、`MAX_CONCURRENT=10`、
`PASS_AT_K=1`、`TEMPERATURE=1.0`。评测读取官方预切分 JSONL，不另做文本
normalizer；后处理和正确性判定必须使用固定 framework 的 evaluator。GAIA
多模态附件由官方文档指定的 GPT-4o 工具预处理成文本。

来源：

- <https://github.com/MiroMindAI/MiroThinker>
- <https://huggingface.co/datasets/miromind-ai/MiroFlow-Benchmarks>
- MiroThinker-1.7 technical report

这些是完整 deep-research agent 的结果，不是裸 LLM checkpoint 的离线问答分数。官方框架要求搜索、网页抽取和代码执行工具；部分工具需要 `SERPER_API_KEY`、`JINA_API_KEY`、`E2B_API_KEY`，并在评测中屏蔽可能泄漏答案的网站。

官方仓已发布数据 archive、脚本、agent set、采样次数和通用参数，但没有冻结动态
网页内容、搜索结果、外部 API 服务版本和全部原始运行轨迹。缺失字段必须标记
“官方未发布/未固定”，不能用模型服务单轮问答冒充。

## 2. 迁移对齐主线

### 2.1 模型服务

相同 checkpoint、vLLM commit、chat template、prompt JSONL 和 sampling 参数，比较 CUDA vLLM 与 NPU vLLM-Ascend：

- `temperature=0`、`top_p=1`；
- 覆盖短输入、8K、32K、工具调用格式、300 轮上限边界；
- top-1 token agreement `>= 99.5%`；
- JSON/tool schema 有效率 100%；
- 无 device/rank/stream/graph capture 错误。

### 2.2 Agent benchmark

固定 MiroThinker official commit、`.env` 所指工具服务版本、benchmark revision、网站访问策略、judge 模型和运行次数。CUDA 与 NPU 只替换模型 endpoint。

通过条件：

- 四项 benchmark 的 NPU pass@1/accuracy 相对 CUDA 下降 `<= 1.0` 个百分点；
- 报告每题轨迹成功、tool error、超时和 judge 结果；
- 只有所有官方条件一致时才比较官方表。

## 3. 分层验收

| 层级 | 范围 |
|---|---|
| L0 | 服务加载、单轮 chat |
| L1 | 100 条固定 prompt、tool schema、8K context |
| L2 | 官方 benchmark 固定小子集，完整 agent/tools |
| L3 | 四项完整 benchmark、256K 和长链 300 tools |

## 4. 性能与稳定性

复测参考场景 1K/4K 和 10K/1K、concurrency 64、requests 256，并补 concurrency 1/8。记录 TTFT、TPOT、ITL、E2E、output/total tok/s、QPS、tok/s/NPU、HBM、加载时间和 2 小时稳定性。

## 5. 当前验收状态

- 已通过：源码/model/vLLM/vllm-ascend 版本取证；固定 benchmark archive
  SHA256；解压并核对四项原始数据规模和官方评测脚本参数。
- 未执行：235B/16 卡服务、CUDA/NPU token 对齐、外部工具 agent 评测、
  性能和稳定性。
- 当前结论：验收输入和口径已固定；模型及 agent 的 NPU 验收未完成。

## 6. 报告模板

```text
模型/官方框架/vLLM/vllm-ascend SHA:
硬件/CANN/镜像/启动参数:
benchmark revision、工具、judge、网页策略:
CUDA/NPU token和agent指标:
tool error/超时/污染检查:
TTFT/TPOT/throughput/HBM/稳定性:
官方未发布字段:
结论:
```
