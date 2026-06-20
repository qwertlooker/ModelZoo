# MiroThinker-1.7 推理指导

## 概述

本目录适配 235B 参数的 `miromind-ai/MiroThinker-1.7`，架构为 `Qwen3MoeForCausalLM`，支持 256K context 和最多 300 次工具交互。当前 NPU 服务边界为 A3、vLLM-Ascend、TP16。

```text
model=https://huggingface.co/miromind-ai/MiroThinker-1.7
model_commit=1a42014ce72e1025fdbf3c48d54545715ab3eea8
official=https://github.com/MiroMindAI/MiroThinker.git
official_commit=370f98361553ddf787bedc5745760e04114cb161
reference=https://gitcode.com/Ascend-SACT/MiroThinker-1.7
reference_commit=a4199f82dcadf88e81e296eb2d0e79bdb5805184
vllm=v0.17.0rc1@b31e9326a7d9394aab8c767f8ebe225c65594b60
vllm-ascend=v0.17.0rc1@e20f0b1a0d2fdb1d86a15d55d70fe60a7a1b5a45
```

不包含 30B 的 `MiroThinker-1.7-mini`。

## 环境与权重

使用 A3 镜像 `vllm-ascend:v0.17.0rc1-a3`，先核对镜像内两个仓库 commit。下载固定权重：

```bash
huggingface-cli download miromind-ai/MiroThinker-1.7 \
  --revision 1a42014ce72e1025fdbf3c48d54545715ab3eea8 \
  --local-dir /models/MiroThinker-1.7
```

记录全部权重、config、tokenizer 和 chat template SHA256。

准备完整 agent benchmark 数据：

```bash
git clone https://github.com/MiroMindAI/MiroThinker.git upstream
git -C upstream checkout 370f98361553ddf787bedc5745760e04114cb161
wget -O /tmp/miroflow-benchmarks.zip \
  https://huggingface.co/datasets/miromind-ai/MiroFlow-Benchmarks/resolve/09900fb9f7297b853f56e1b785491494e93ac85d/data_20251115_password_protected.zip
unzip -P 'pf4*' /tmp/miroflow-benchmarks.zip -d upstream
sha256sum /tmp/miroflow-benchmarks.zip
```

预期 archive SHA256：

```text
35816f69ba5f0d2baf45b248c68dd4a8e0f9b30cac6f41076f44099d5073f377
```

## 启动

```bash
vllm serve /models/MiroThinker-1.7 \
  --tensor-parallel-size 16 \
  --host 0.0.0.0 \
  --port 8002 \
  --served-model-name miro \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.88 \
  --compilation-config \
  '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}'
```

这里的 8192 是当前启动配置，不代表完整 256K 已验收。

```bash
curl -sf http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"miro","messages":[{"role":"user","content":"介绍一下你自己"}],"max_tokens":512,"temperature":0,"top_p":1}'
```

模型服务只是 MiroThinker agent 的 LLM 后端。要复现 BrowseComp/GAIA 等官方指标，还必须按官方 MiroThinker/MiroFlow 框架配置搜索、抓取和代码工具，见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

例如完整 BrowseComp 使用 1,266 题、3 次运行和 max300 agent：

```bash
cd upstream/apps/miroflow-agent
LLM_MODEL=MiroThinker-1.7 \
BASE_URL=http://127.0.0.1:8002/v1 \
NUM_RUNS=3 \
AGENT_SET=mirothinker_1.7_keep5_max300 \
bash scripts/run_evaluate_multiple_runs_browsecomp.sh
```

运行前按 upstream `.env.example` 配置搜索、抓取、代码工具和 judge API。
