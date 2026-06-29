#!/usr/bin/env bash
# MiroThinker-1.7 vLLM-Ascend 服务启动入口（8K 功能验证与服务精度回归配置）。
#
# 在 vllm-ascend v0.17.0rc1 容器中执行；MODEL_DIR 指向已下载的
# miromind-ai/MiroThinker-1.7 权重目录。本模型无 patch：vLLM v0.17.0rc1
# 原生支持 Qwen3MoeForCausalLM，由 vllm-ascend 提供 NPU attention/MoE 后端。
#
# 用法（容器内）：
#   export MODEL_DIR=/models/MiroThinker-1.7
#   bash serve_npu.sh
#
# 256K 官方 agent benchmark 配置见 README.md「模型推理」第 6 步；
# 不要用本 8K 功能配置冒充 256K 长上下文验收。
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR must point to the MiroThinker-1.7 weights directory}"

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
