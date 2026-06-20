# Hy3-preview 推理指导

## 概述

Hy3-preview 是 Tencent Hy 发布的 295B MoE 模型，激活参数 21B，另含 3.8B MTP layer，BF16，256K context。本交付使用 vLLM + vllm-ascend，当前推荐边界为 A3、TP16、EP、MTP。

```text
model=https://huggingface.co/tencent/Hy3-preview
model_commit=549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a
official=https://github.com/Tencent-Hunyuan/Hy3-preview.git
official_commit=38ac237dc0bf4329f054d09054aaf22fdaf6f553
reference=https://gitcode.com/Ascend-SACT/Hy3-preview
reference_commit=eb533c1dfd9a1fa7f373f9b980a9c0f973f1dad8
vllm=v0.18.0rc1@262ddd0d81a1e4687e209f988d6ea32616e736fa
vllm-ascend=v0.18.0rc1@99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f
```

不包含 `Hy3-preview-Base`。

## 环境、补丁和权重

使用镜像 `quay.io/ascend/vllm-ascend:v0.18.0rc1-a3`，确认镜像内 vLLM/vllm-ascend commit 与上面一致：

```bash
git -C /vllm-workspace/vllm rev-parse HEAD
git -C /vllm-workspace/vllm-ascend rev-parse HEAD
git -C /vllm-workspace/vllm apply --check \
  /workspace/Hy3-preview/patches/0001-add-hy3-preview-support.patch
git -C /vllm-workspace/vllm apply \
  /workspace/Hy3-preview/patches/0001-add-hy3-preview-support.patch
```

下载并固定模型：

```bash
huggingface-cli download tencent/Hy3-preview \
  --revision 549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a \
  --local-dir /models/Hy3-preview
```

BF16 权重静态量级约 590GB，不含 KV cache、运行时 workspace 和加载峰值。必须记录全部权重文件 SHA256。

## 启动

```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
HCCL_OP_EXPANSION_MODE=AIV \
vllm serve /models/Hy3-preview \
  --served-model-name hy3-preview \
  --tensor-parallel-size 16 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 1 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --max-model-len 32768 \
  --max-num-seqs 8
```

`32K/bs8` 是当前交付启动配置，不代表已验收完整 256K context。

## 快速检查

```bash
curl -sf http://127.0.0.1:8000/v1/models
curl -sf http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy3-preview","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16,"temperature":0,"top_p":1,"chat_template_kwargs":{"reasoning_effort":"no_think"}}'
```

tool/reasoning 和正式指标验收见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，补丁分析见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
