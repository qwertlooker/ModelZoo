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

可执行入口：

```bash
python ../tools/openai_service_eval.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model hy3-preview \
  --prompts test_data/service_prompts.jsonl \
  --request_logprobs \
  --output results/npu.jsonl
python ../tools/compare_openai_service_results.py \
  --baseline results/cuda.jsonl \
  --candidate results/npu.jsonl \
  --require_logprobs \
  --require_exact_tool_calls \
  --output results/cuda_vs_npu.json
```

通过条件：

- 请求成功率 100%，无权重缺失、rank/device、parser 错误；
- greedy top-1 token agreement `>= 99.5%`；
- 结构化 JSON/tool call schema 有效率 100%；
- MTP 开关后的任务级正确率不低于关闭 MTP 超过 0.5 个百分点。

逐 token 一致率会在首个分叉后级联下降，因此它只能和首个分叉位置、logprob、
结构化输出有效率及任务正确率一起解释，不能单独作为生成质量结论。
这些阈值是迁移初始门禁，必须用固定 CUDA/NPU baseline 的实测分布校准，不是官方
发布容差。

### 2.2 官方任务对齐

优先选择可固定环境的 SWE-bench Verified 和 Terminal-Bench 2.0；BrowseComp/WideSearch 依赖外部搜索和动态网页，必须冻结工具、网页快照或采用官方环境。

- CUDA 和 NPU 使用同 agent harness、容器、工具和运行次数；
- 报告 pass@1/成功率及置信区间；
- NPU 相对 CUDA 下降 `<= 1.0` 个百分点；
- 只有官方未公开字段补齐且配置一致时，才与官方表比较。

## 3. 功能验证与 L2

| 层级 | 范围 |
|---|---|
| 功能验证 | 仓内 4 条固定 prompt；模型加载、`/v1/models`、chat、tool/reasoning/JSON，streaming 单独检查 |
| L2 | 优先按官方四项 benchmark 全量和已公开配置；recipe 不完整时固定公开子集，并使用确定性 100 条性能请求 | 任务精度、服务 TTFT/TPOT/吞吐和资源 |

功能验证不构成官方质量指标验收。

## 4. 功能矩阵

| 维度 | 必测值 |
|---|---|
| MTP | 关闭 baseline、开启 1 token |
| 输出模式 | 非流式、流式 |
| parser | no-think、high reasoning、单/多 tool call |
| 结构化输出 | JSON object、非法 schema 失败 |
| 上下文 | 短输入、8K；L2 使用 32K |
| 并发 | 功能验证使用 1；L2 按固定 benchmark 配置 |
| 异常 | 错模型名、超长输入、缺 shard、rank/HCCL 故障 |

## 5. L2 精度与性能验证

精度优先执行 SWE-bench Verified、Terminal-Bench 2.0、BrowseComp、WideSearch
全量官方数据和官方 harness。缺失 agent/tool/judge/decode 字段时，固定能取得的
公开子集、harness commit 和环境，并明确结果仅为迁移对齐。

性能使用 `vllm bench serve --dataset-name random --seed 42` 生成的确定性 100 条
固定长度请求，CUDA/NPU 分别执行相同配置，记录成功率、TTFT、TPOT、ITL、E2E、
request/output/total throughput 和峰值 HBM。关闭与开启 MTP 分别报告，结果写入
独立 JSON。官方未发布同硬件性能值，不伪造 speedup 线；必须报告 NPU/CUDA 比值和
是否达到项目另行给定的目标。

## 6. 最低正式验收清单

- [ ] 镜像 digest、vLLM/vllm-ascend/patch SHA 已记录。
- [ ] 仓内 4 条功能 prompt 的 SHA 已记录，CUDA/NPU 使用同一文件。
- [ ] 未应用 patch 的原始 vLLM 不支持 HyV3 的失败日志已归档；patched CUDA 作为数值 baseline。
- [ ] 全部权重 shard、config、tokenizer 文件清单和 SHA 已归档。
- [ ] 16 卡可见性、HCCL 和模型加载通过。
- [ ] 关闭和开启 MTP 的 CUDA/NPU 功能验证通过。
- [ ] tool/reasoning/streaming/JSON 功能矩阵通过。
- [ ] 至少一个固定公开 benchmark 子集完成同 harness 的 CUDA/NPU 对齐。
- [ ] L2 CUDA/NPU 的确定性 100 条服务性能 JSON、TTFT/TPOT/吞吐/HBM 和比值已归档。

## 7. 当前验收状态

- 已通过：patch SHA 固定；在 vLLM `v0.18.0rc1` 精确 commit 上
  `git apply --check`；新增/修改 Python 文件 `compileall`。
- 未执行：295B 权重下载、16 卡功能验证、CUDA/NPU token 对齐和 L2 精度/性能。
- 当前结论：patch 静态门禁通过；模型 NPU 验收未完成。

## 8. 报告模板

```text
模型/vLLM/vllm-ascend/patch SHA:
16卡硬件、CANN、镜像:
启动参数、环境变量、MTP:
prompt/benchmark revision和规模:
CUDA/NPU token agreement与任务指标:
tool/reasoning/streaming矩阵:
L2 TTFT/TPOT/ITL/throughput/HBM及比值:
官方未发布字段与未完成项:
结论:
```

## 9. 补充说明（来自 README.md）

以下验收警告与数据口径说明从推理指导文档迁移至此，避免终端用户在 README 中被过多限制条件干扰。

### 9.1 L2 降级路径与官方 recipe 缺失

Hy3 模型卡没有发布四项 instruct benchmark 的完整 agent/tool/judge/decode recipe，因此当前可从零执行的 L2 降级路径是：100 条固定服务精度回归 + 确定性 100 请求性能测试。它计算 token agreement、JSON/tool 有效率、TTFT、TPOT 和吞吐；报告必须明确“非官方 benchmark 复现”。拿到官方 recipe 后再优先替换为全量 benchmark。

### 9.2 内部固定集的指标边界

该 100 条内部固定集不能推导官方四项 benchmark 指标。功能验证与 L2 服务精度回归仅用于 CUDA/NPU 迁移对齐，不构成官方质量指标验收。

### 9.3 上下文长度与启动配置

`32K/bs8` 只是启动配置，不代表 256K 已验收。8K 服务结果不能用于宣称 256K benchmark 已验收。长上下文能力验收需单独使用接近 256K 的固定输入集并记录峰值 HBM。

### 9.4 benchmark 状态表

| 项目 | 官方值/当前状态 |
|---|---|
| SWE-bench Verified | 74.4%，精确 recipe 未完整发布 |
| Terminal-Bench 2.0 | 54.4%，精确 recipe 未完整发布 |
| BrowseComp / WideSearch | 67.1% / 70.2%，依赖外部工具 |
| A3 服务性能与 256K | 官方硬件性能未发布；当前未实测 |
