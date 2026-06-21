# MiroThinker-1.7 推理指导

## 概述

本目录适配 235B `miromind-ai/MiroThinker-1.7`，架构为 `Qwen3MoeForCausalLM`，声明支持 256K context 和最多 300 次工具交互。当前 NPU 服务边界为 Atlas A3、vLLM-Ascend、TP16。

```text
model=miromind-ai/MiroThinker-1.7@1a42014ce72e1025fdbf3c48d54545715ab3eea8
official=MiroMindAI/MiroThinker@370f98361553ddf787bedc5745760e04114cb161
reference=Ascend-SACT/MiroThinker-1.7@a4199f82dcadf88e81e296eb2d0e79bdb5805184
vllm=v0.17.0rc1@b31e9326a7d9394aab8c767f8ebe225c65594b60
vllm-ascend=v0.17.0rc1@e20f0b1a0d2fdb1d86a15d55d70fe60a7a1b5a45
```

不包含 30B `MiroThinker-1.7-mini`。

## 输入输出数据

- 裸模型服务输入输出：OpenAI-compatible chat messages/completions。
- 功能验证输入：`test_data/service_prompts.jsonl`，仓内固定 4 条。
- 公共工具确定性生成 100 条 L2 服务精度回归 manifest；正式精度优先使用官方
  agent benchmark。
- 完整官方结果输入：MiroFlow benchmark JSONL、MiroThinker agent、搜索/抓取/代码工具和 LLM judge。
- 服务结果用公共 `tools/openai_service_eval.py` 和 `tools/compare_openai_service_results.py` 固定。

## 推理环境准备

| 配套 | 版本/要求 |
|---|---|
| 硬件 | Atlas A3，当前边界 TP16 |
| 容器 | `quay.io/ascend/vllm-ascend:v0.17.0rc1-a3` |
| vLLM | `b31e9326a7d9394aab8c767f8ebe225c65594b60` |
| vllm-ascend | `e20f0b1a0d2fdb1d86a15d55d70fe60a7a1b5a45` |
| Agent Python | 3.12（以固定 commit 的 `pyproject.toml` 为准） |
| Agent 依赖 | `uv sync --frozen` |

完整 benchmark 还需要 Serper、Jina、E2B、summary LLM 和 OpenAI judge API。动态搜索结果和网页内容不能被视为完全冻结的数据资产。

## 文件目录

```text
MiroThinker-1.7
├── test_data/service_prompts.jsonl
├── patches/README.md
├── README_INFERENCE.md
├── NPU_ADAPTATION.md
└── ACCEPTANCE_PLAN.md
tools
├── openai_service_eval.py
├── compare_openai_service_results.py
└── prepare_service_prompts.py
```

## 快速上手

### 获取源码和启动容器

以下命令从 `MiroThinker-1.7` 目录执行：

```bash
git clone https://github.com/MiroMindAI/MiroThinker.git upstream
git -C upstream checkout 370f98361553ddf787bedc5745760e04114cb161

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

容器中先核对：

```bash
cd /workspace/ModelZoo/MiroThinker-1.7
npu-smi info
test "$(git -C /vllm-workspace/vllm rev-parse HEAD)" = \
  b31e9326a7d9394aab8c767f8ebe225c65594b60
test "$(git -C /vllm-workspace/vllm-ascend rev-parse HEAD)" = \
  e20f0b1a0d2fdb1d86a15d55d70fe60a7a1b5a45
```

### 准备权重

```bash
huggingface-cli download miromind-ai/MiroThinker-1.7 \
  --revision 1a42014ce72e1025fdbf3c48d54545715ab3eea8 \
  --local-dir "$MODEL_DIR"

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print | sort
sha256sum "$MODEL_DIR"/config.json "$MODEL_DIR"/model.safetensors.index.json
```

正式报告记录全部 shard、config、tokenizer 和 chat template 的文件清单、大小及 SHA256。

### 准备数据集

仓内 4 条用于功能验证。生成固定 100 条 L2 服务精度回归 prompt：

```bash
python /workspace/ModelZoo/tools/prepare_service_prompts.py \
  --base /workspace/ModelZoo/MiroThinker-1.7/test_data/service_prompts.jsonl \
  --output /workspace/ModelZoo/MiroThinker-1.7/eval_data/service_prompts_l2.jsonl \
  --count 100
test "$(wc -l < /workspace/ModelZoo/MiroThinker-1.7/eval_data/service_prompts_l2.jsonl)" = 100
```

完整 agent benchmark：

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

功能验证和服务精度回归使用 8K 服务：

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

保存 NPU 服务结果：

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

流式接口单独检查：

```bash
curl -N -s http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiroThinker-1.7","messages":[{"role":"user","content":"Return exactly: ready"}],"temperature":0,"max_tokens":8,"stream":true}' \
  | tee /workspace/ModelZoo/MiroThinker-1.7/results/npu-stream.txt
```

在安装了同一 vLLM commit 的 CUDA 环境中使用相同 checkpoint 和参数启动：

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

CUDA 与 NPU 服务不能同时占用 8002 端口；分别运行并保存结果后比较：

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

官方 benchmark 不能使用上述 8K 服务。必须重启成 256K 服务，并降低并发以满足实际 HBM：

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

四项 benchmark 分别执行。所有命令必须在同一个固定 `.env`、网页策略和工具服务
状态下，对 CUDA 和 NPU endpoint 各运行一套：

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

切换 CUDA/NPU 时只允许替换 `BASE_URL`，其余环境变量、`.env`、数据 archive 和
运行次数保持一致。

## 模型推理性能

模型服务性能与完整 agent 成功率分别报告。CUDA 与 NPU 服务执行同参命令：

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

服务报告 TTFT、TPOT、ITL、E2E、token/s、QPS 和峰值 HBM；agent 报告每项
benchmark wall time、题/小时、工具错误、超时、judge 和动态网页状态。CUDA 写入
独立 JSON。8K 服务结果不能用于宣称 256K benchmark 已验收。

| Benchmark | 官方分数 | 当前 NPU 状态 |
|---|---:|---|
| BrowseComp | 74.0% | 待完整 agent 验收 |
| BrowseComp-ZH | 75.3% | 待完整 agent 验收 |
| GAIA-Val-165 | 82.7% | 待完整 agent 验收 |
| HLE-Text | 42.9% | 待完整 agent 验收 |

官方未发布与当前 A3 TP16 服务直接可比的硬件性能数值。

## 公网地址说明

- 模型：<https://huggingface.co/miromind-ai/MiroThinker-1.7>
- 官方框架：<https://github.com/MiroMindAI/MiroThinker>
- benchmark：<https://huggingface.co/datasets/miromind-ai/MiroFlow-Benchmarks>
- 参考适配：<https://gitcode.com/Ascend-SACT/MiroThinker-1.7>

适配边界见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
