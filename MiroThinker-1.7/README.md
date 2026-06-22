# MiroThinker-1.7 推理指导

- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [文件目录](#文件目录)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备权重](#准备权重)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能](#模型推理性能)
- [公网地址说明](#公网地址说明)

## 概述

MiroThinker-1.7 是 MiroMind 发布的 235B MoE 推理模型，架构为 `Qwen3MoeForCausalLM`，声明支持 256K context 和最多 300 次工具交互。本文档介绍该模型基于昇腾 NPU 的推理指导。

> 说明：本文档适配对象为 `miromind-ai/MiroThinker-1.7` 235B 权重，不包含 30B `MiroThinker-1.7-mini`。

- 版本说明：

  ```text
  model=miromind-ai/MiroThinker-1.7@1a42014ce72e1025fdbf3c48d54545715ab3eea8
  official=MiroMindAI/MiroThinker@370f98361553ddf787bedc5745760e04114cb161
  reference=Ascend-SACT/MiroThinker-1.7@a4199f82dcadf88e81e296eb2d0e79bdb5805184
  vllm=v0.17.0rc1@b31e9326a7d9394aab8c767f8ebe225c65594b60
  vllm-ascend=v0.17.0rc1@e20f0b1a0d2fdb1d86a15d55d70fe60a7a1b5a45
  ```

## 输入输出数据

- 输入数据

  OpenAI-compatible chat messages/completions；功能验证使用 `test_data/service_prompts.jsonl`（仓内固定 4 条），L2 服务精度回归使用工具生成的 100 条 manifest。

- 输出数据

  模型生成的 chat/completions 文本；服务评测结果为 JSONL，CUDA/NPU 对比结果为 JSON。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

  | 配套 | 版本/要求 |
  |---|---|
  | 硬件 | Atlas A3，当前边界 TP16 |
  | 容器 | `quay.io/ascend/vllm-ascend:v0.17.0rc1-a3` |
  | vLLM | `b31e9326a7d9394aab8c767f8ebe225c65594b60` |
  | vllm-ascend | `e20f0b1a0d2fdb1d86a15d55d70fe60a7a1b5a45` |
  | Agent Python | 3.12（以固定 commit 的 `pyproject.toml` 为准） |
  | Agent 依赖 | `uv sync --frozen` |

## 文件目录

```text
MiroThinker-1.7
├── test_data/service_prompts.jsonl
└── README.md
tools
├── openai_service_eval.py
├── compare_openai_service_results.py
└── prepare_service_prompts.py
```

## 快速上手

### 获取源码

1. 获取官方 agent 框架源码。

   ```bash
   git clone https://github.com/MiroMindAI/MiroThinker.git upstream
   git -C upstream checkout 370f98361553ddf787bedc5745760e04114cb161
   ```

2. 启动推理容器。以下命令从 `MiroThinker-1.7` 目录执行：

   ```bash
   export MODELZOO_DIR=$(cd .. && pwd)
   export MODEL_DIR=/models/MiroThinker-1.7
   export IMAGE=quay.io/ascend/vllm-ascend:v0.17.0rc1-a3
   docker pull "$IMAGE"
   docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}'

   docker run --rm -it \
     --name mirothinker-1-7 \
     --network host \
     --shm-size 32g \
     --device /dev/davinci_manager \
     --device /dev/hisi_hdc \
     --device /dev/devmm_svm \
     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
     -v /usr/local/dcmi:/usr/local/dcmi:ro \
     -v "$MODELZOO_DIR":/workspace/ModelZoo \
     -v /models:/models \
     -e MODEL_DIR="$MODEL_DIR" \
     "$IMAGE" bash
   ```

3. 容器中核对版本。

   ```bash
   cd /workspace/ModelZoo/MiroThinker-1.7
   npu-smi info
   test "$(git -C /vllm-workspace/vllm rev-parse HEAD)" = \
     b31e9326a7d9394aab8c767f8ebe225c65594b60
   test "$(git -C /vllm-workspace/vllm-ascend rev-parse HEAD)" = \
     e20f0b1a0d2fdb1d86a15d55d70fe60a7a1b5a45
   ```

### 准备权重

1. 下载 `MiroThinker-1.7` 权重。

   ```bash
   huggingface-cli download miromind-ai/MiroThinker-1.7 \
     --revision 1a42014ce72e1025fdbf3c48d54545715ab3eea8 \
     --local-dir "$MODEL_DIR"

   test -f "$MODEL_DIR/config.json"
   test -f "$MODEL_DIR/model.safetensors.index.json"
   find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print | sort
   sha256sum "$MODEL_DIR"/config.json "$MODEL_DIR"/model.safetensors.index.json
   ```

### 准备数据集

1. 生成固定 100 条 L2 服务精度回归 prompt。

   ```bash
   python /workspace/ModelZoo/tools/prepare_service_prompts.py \
     --base /workspace/ModelZoo/MiroThinker-1.7/test_data/service_prompts.jsonl \
     --output /workspace/ModelZoo/MiroThinker-1.7/eval_data/service_prompts_l2.jsonl \
     --count 100
   test "$(wc -l < /workspace/ModelZoo/MiroThinker-1.7/eval_data/service_prompts_l2.jsonl)" = 100
   ```

2. 下载完整 agent benchmark archive 并安装 agent 依赖。

   ```bash
   wget -O /tmp/miroflow-benchmarks.zip \
     https://huggingface.co/datasets/miromind-ai/MiroFlow-Benchmarks/resolve/09900fb9f7297b853f56e1b785491494e93ac85d/data_20251115_password_protected.zip
   echo "35816f69ba5f0d2baf45b248c68dd4a8e0f9b30cac6f41076f44099d5073f377  /tmp/miroflow-benchmarks.zip" \
     | sha256sum -c -
   unzip -P 'pf4*' /tmp/miroflow-benchmarks.zip \
     -d /workspace/ModelZoo/MiroThinker-1.7/upstream

   cd /workspace/ModelZoo/MiroThinker-1.7/upstream/apps/miroflow-agent
   uv sync --frozen
   cp .env.example .env
   ```

   编辑 `.env`，至少配置 `SERPER_*`、`JINA_*`、`E2B_API_KEY`、summary LLM 和 judge 的 `OPENAI_*`。

### 模型推理

1. 启动 8K 服务用于功能验证和服务精度回归。

   ```bash
   vllm serve "$MODEL_DIR" \
     --served-model-name MiroThinker-1.7 \
     --tensor-parallel-size 16 \
     --host 0.0.0.0 \
     --port 8002 \
     --trust-remote-code \
     --max-model-len 8192 \
     --max-num-seqs 128 \
     --gpu-memory-utilization 0.88 \
     --compilation-config \
     '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}'
   ```

2. 保存 NPU 服务结果。

   ```bash
   mkdir -p /workspace/ModelZoo/MiroThinker-1.7/results
   python /workspace/ModelZoo/tools/openai_service_eval.py \
     --base_url http://127.0.0.1:8002/v1 \
     --model MiroThinker-1.7 \
     --prompts /workspace/ModelZoo/MiroThinker-1.7/test_data/service_prompts.jsonl \
     --output /workspace/ModelZoo/MiroThinker-1.7/results/functional_npu.jsonl

   python /workspace/ModelZoo/tools/openai_service_eval.py \
     --base_url http://127.0.0.1:8002/v1 \
     --model MiroThinker-1.7 \
     --prompts /workspace/ModelZoo/MiroThinker-1.7/eval_data/service_prompts_l2.jsonl \
     --request_logprobs \
     --output /workspace/ModelZoo/MiroThinker-1.7/results/npu.jsonl
   ```

3. 流式接口单独检查。

   ```bash
   curl -N -s http://127.0.0.1:8002/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"MiroThinker-1.7","messages":[{"role":"user","content":"Return exactly: ready"}],"temperature":0,"max_tokens":8,"stream":true}' \
     | tee /workspace/ModelZoo/MiroThinker-1.7/results/npu-stream.txt
   ```

4. 在 CUDA 环境中使用相同 checkpoint 和参数启动并保存结果。

   ```bash
   vllm serve "$MODEL_DIR" \
     --served-model-name MiroThinker-1.7 \
     --tensor-parallel-size 16 \
     --host 0.0.0.0 \
     --port 8002 \
     --trust-remote-code \
     --max-model-len 8192 \
     --max-num-seqs 128 \
     --gpu-memory-utilization 0.88 \
     --compilation-config \
     '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}'

   python /workspace/ModelZoo/tools/openai_service_eval.py \
     --base_url http://127.0.0.1:8002/v1 \
     --model MiroThinker-1.7 \
     --prompts /workspace/ModelZoo/MiroThinker-1.7/test_data/service_prompts.jsonl \
     --output /workspace/ModelZoo/MiroThinker-1.7/results/functional_cuda.jsonl

   python /workspace/ModelZoo/tools/openai_service_eval.py \
     --base_url http://127.0.0.1:8002/v1 \
     --model MiroThinker-1.7 \
     --prompts /workspace/ModelZoo/MiroThinker-1.7/eval_data/service_prompts_l2.jsonl \
     --request_logprobs \
     --output /workspace/ModelZoo/MiroThinker-1.7/results/cuda.jsonl
   ```

5. 比较 CUDA 与 NPU 服务结果。

   ```bash
   python /workspace/ModelZoo/tools/compare_openai_service_results.py \
     --baseline /workspace/ModelZoo/MiroThinker-1.7/results/functional_cuda.jsonl \
     --candidate /workspace/ModelZoo/MiroThinker-1.7/results/functional_npu.jsonl \
     --output /workspace/ModelZoo/MiroThinker-1.7/results/functional_cuda_vs_npu.json

   python /workspace/ModelZoo/tools/compare_openai_service_results.py \
     --baseline /workspace/ModelZoo/MiroThinker-1.7/results/cuda.jsonl \
     --candidate /workspace/ModelZoo/MiroThinker-1.7/results/npu.jsonl \
     --require_logprobs \
     --output /workspace/ModelZoo/MiroThinker-1.7/results/cuda_vs_npu.json
   ```

6. 启动 256K 服务用于官方 benchmark。

   ```bash
   vllm serve "$MODEL_DIR" \
     --served-model-name MiroThinker-1.7 \
     --tensor-parallel-size 16 \
     --host 0.0.0.0 \
     --port 8002 \
     --trust-remote-code \
     --max-model-len 262144 \
     --max-num-seqs 10 \
     --gpu-memory-utilization 0.88
   ```

7. 执行四项 benchmark，对 CUDA 和 NPU endpoint 各运行一套：

   ```bash
   cd /workspace/ModelZoo/MiroThinker-1.7/upstream/apps/miroflow-agent
   export LLM_PROVIDER=qwen
   export LLM_MODEL=MiroThinker-1.7
   export BASE_URL=http://127.0.0.1:8002/v1
   export API_KEY=EMPTY
   export MAX_CONTEXT_LENGTH=262144
   export MAX_CONCURRENT=10
   export PASS_AT_K=1
   export TEMPERATURE=1.0

   LLM_MODEL=MiroThinker-1.7 \
   NUM_RUNS=3 \
   AGENT_SET=mirothinker_1.7_keep5_max300 \
   bash scripts/run_evaluate_multiple_runs_browsecomp.sh

   NUM_RUNS=3 \
   AGENT_SET=mirothinker_1.7_keep5_max300 \
   bash scripts/run_evaluate_multiple_runs_browsecomp_zh.sh

   NUM_RUNS=8 \
   AGENT_SET=mirothinker_1.7_keep5_max200 \
   bash scripts/run_evaluate_multiple_runs_gaia-validation.sh

   NUM_RUNS=3 \
   AGENT_SET=mirothinker_1.7_keep5_max200 \
   bash scripts/run_evaluate_multiple_runs_hle-text-2158.sh
   ```

## 模型推理性能

模型服务性能使用 `vllm bench serve` 测试，CUDA 与 NPU 执行同参命令：

```bash
mkdir -p /workspace/ModelZoo/MiroThinker-1.7/results/performance
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8002 \
  --endpoint /v1/completions \
  --model MiroThinker-1.7 \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 100 \
  --max-concurrency 8 \
  --seed 42 \
  --save-result \
  --save-detailed \
  --result-dir /workspace/ModelZoo/MiroThinker-1.7/results/performance \
  --result-filename npu.json
```

## 适配与精度口径

### 适配结论

vLLM v0.17.0rc1 已原生支持 `Qwen3MoeForCausalLM`，vllm-ascend 提供 NPU 后端，不需要 patch。`--trust-remote-code` 仅针对固定本地权重目录，不是远程代码信任。服务名、`LLM_MODEL` 环境变量和请求 `model` 字段必须三者统一，否则 vLLM 会拒绝请求。

### 运行边界

- 推荐 TP16；必须显式设置 `compilation_config` 的 capture sizes，否则图捕获会失败。
- 8K smoke 服务（`--max-model-len 8192`、128 并发）只用于链路验证，不等同于 256K context 能力验收；长上下文验收需单独使用接近 256K 的固定输入集并记录峰值 HBM。
- benchmark 不能复用 8K 服务，必须重启 256K 服务后再跑；CUDA 与 NPU 服务不能同时占用 8002 端口，必须分别运行。
- agent benchmark 质量依赖外部工具：Serper/Jina 检索、E2B 代码沙箱、summary LLM 和 OpenAI judge API，这些 key 和额度由使用者自备，缺失会导致 benchmark 跳过对应样本。
- 8K smoke 与 256K benchmark 是两套独立配置，不得混用参数。

### 官方指标边界

官方在 MiroFlow-Benchmarks 上发布以下 agent 指标：

| Benchmark | 官方分数 |
|---|---:|
| BrowseComp | 74.0% |
| BrowseComp-ZH | 75.3% |
| GAIA-Val-165 | 82.7% |
| HLE-Text | 42.9% |

评测口径：固定数据集 revision、OpenAI 作为 judge、每题多 run 取多数、固定 agent tool set、统一采样参数。官方未发布上述每项的精确 manifest SHA、agent harness commit、工具环境快照和完整 decode 参数，因此这些字段明确记为"官方未发布"，获得作者 recipe 前不得宣称精确复现。

### 迁移对齐门禁

使用同一 checkpoint、vLLM commit、prompt JSONL 和 sampling 参数，比较 CUDA vLLM 与 NPU vLLM-Ascend：

- greedy top-1 token agreement `>= 99.5%`；
- 结构化 JSON/tool call schema 有效率 100%；
- NPU benchmark 相对 CUDA 下降 `<= 1.0` 个百分点。

这些阈值是迁移初始门禁，不是官方发布容差，必须用固定 CUDA/NPU baseline 的实测分布校准。

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 模型权重 | MiroThinker-1.7 Hugging Face 模型仓 | https://huggingface.co/miromind-ai/MiroThinker-1.7 |
| 开源代码仓 | MiroThinker 官方框架 | https://github.com/MiroMindAI/MiroThinker |
| benchmark | MiroFlow-Benchmarks 数据集 | https://huggingface.co/datasets/miromind-ai/MiroFlow-Benchmarks |
| 参考适配 | Ascend-SACT 参考实现 | https://gitcode.com/Ascend-SACT/MiroThinker-1.7 |
