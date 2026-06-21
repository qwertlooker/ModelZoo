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

`99.5%` 是暂定数值诊断门禁，需根据固定 CUDA/NPU prompt baseline 校准；它不是
官方质量容差，也不能替代 agent 任务指标。

使用固定入口：

```bash
python ../tools/openai_service_eval.py \
  --base_url http://127.0.0.1:8002/v1 \
  --model MiroThinker-1.7 \
  --prompts test_data/service_prompts.jsonl \
  --request_logprobs \
  --output results/npu.jsonl
python ../tools/compare_openai_service_results.py \
  --baseline results/cuda.jsonl \
  --candidate results/npu.jsonl \
  --require_logprobs \
  --output results/cuda_vs_npu.json
```

逐 token 一致率是数值差异定位手段；首个 token 分叉会导致后续级联不同，因此
还必须报告首个分叉位置、结构化输出有效率和任务级正确率。

### 2.2 Agent benchmark

固定 MiroThinker official commit、`.env` 所指工具服务版本、benchmark revision、网站访问策略、judge 模型和运行次数。CUDA 与 NPU 只替换模型 endpoint。

服务的 `--served-model-name`、脚本的 `LLM_MODEL` 必须统一为
`MiroThinker-1.7`。官方 benchmark 的 `MAX_CONTEXT_LENGTH=262144` 要求模型服务
也以 `--max-model-len 262144` 启动；8K smoke 服务不得直接用于该评测。

通过条件：

- 四项 benchmark 的 NPU pass@1/accuracy 相对 CUDA 下降 `<= 1.0` 个百分点；
- 报告每题轨迹成功、tool error、超时和 judge 结果；
- 只有所有官方条件一致时才比较官方表。

`1.0` 个百分点同样是暂定非劣化线。正式报告需结合多次运行方差/置信区间判断，
不能用一次随机采样差异直接判定 NPU 退化。

## 3. 功能验证与 L2

| 层级 | 范围 |
|---|---|
| 功能验证 | 仓内 4 条固定 prompt；服务加载、chat、JSON、streaming |
| L2 | 优先执行四项官方 benchmark 全量和官方配置；外部状态无法冻结时使用固定公开子集 | agent 精度、服务性能和工具运行性能 |

## 4. 功能矩阵

| 维度 | 必测值 |
|---|---|
| 服务 | `/v1/models`、非流式、流式、JSON |
| 上下文 | 功能验证 8K；L2 按 benchmark 需要使用 256K 服务 |
| Agent | max200、BrowseComp/BrowseComp-ZH max300 |
| 工具 | search、scrape/summary、E2B code、judge |
| Benchmark | BrowseComp、BrowseComp-ZH、GAIA-Val-165、HLE-Text |
| 异常 | API key 缺失、工具配额、网页超时、judge 失败、模型超长输入 |

## 5. L2 精度与性能验证

精度优先执行固定 archive 中 BrowseComp、BrowseComp-ZH、GAIA-Val-165、
HLE-Text 全量，并保持官方 agent set、run 数、judge、context 和工具配置。动态网页
或 API 版本无法冻结时，固定公开子集和运行窗口，明确结果不是官方精确复现。

性能分两部分：

- 模型服务：用 `vllm bench serve --dataset-name random --seed 42` 的确定性 100 条
  固定长度请求，在 CUDA/NPU 执行相同配置，记录 TTFT、TPOT、ITL、E2E、
  request/output/total throughput
  和峰值 HBM；
- Agent：记录每项 benchmark wall time、题/小时、平均 tool calls、tool error、
  timeout 和 judge failure。

官方未发布同硬件性能表，因此报告相对比值，不编造官方 speedup 通过线。

## 6. 最低正式验收清单

- [ ] 模型、框架、benchmark archive、vLLM/vllm-ascend 版本和 SHA 已固定。
- [ ] NPU 镜像 digest 和仓内 4 条功能 prompt SHA 已记录，CUDA/NPU 共用同一文件。
- [ ] Agent `uv sync --frozen`、`.env` 必需工具和 judge 连通性检查通过。
- [ ] 8K 功能验证的 CUDA/NPU 服务对齐通过，结果写入独立目录。
- [ ] 256K 服务实际启动并通过长上下文请求；不能以配置声明替代。
- [ ] 至少一个固定 benchmark 子集完成相同工具环境的 CUDA/NPU 对齐。
- [ ] 四项完整 benchmark 的未执行项、动态网页和 API 版本风险已报告。
- [ ] L2 服务性能 JSON及 agent wall time、题/小时、工具错误/超时已归档。

## 7. 当前验收状态

- 已通过：源码/model/vLLM/vllm-ascend 版本取证；固定 benchmark archive
  SHA256；解压并核对四项原始数据规模和官方评测脚本参数。
- 未执行：235B/16 卡功能验证、CUDA/NPU token 对齐和 L2 agent 精度/性能。
- 当前结论：验收输入和口径已固定；模型及 agent 的 NPU 验收未完成。

## 8. 报告模板

```text
模型/官方框架/vLLM/vllm-ascend SHA:
硬件/CANN/镜像/启动参数:
benchmark revision、工具、judge、网页策略:
CUDA/NPU token和agent指标:
tool error/超时/污染检查:
服务TTFT/TPOT/throughput/HBM；agent wall time和题/小时:
官方未发布字段:
结论:
```
