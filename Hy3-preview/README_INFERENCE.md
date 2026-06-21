# Hy3-preview 推理指导

## 概述

Hy3-preview 是 295B total / 21B active 的 BF16 MoE 模型，包含一层 3.8B MTP，声明支持 256K context。当前交付基于固定 vLLM + vllm-ascend，在 Atlas A3 上使用 TP16、EP、MTP 和 HyV3 tool/reasoning parser。

```text
model=tencent/Hy3-preview@549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a
official=Tencent-Hunyuan/Hy3-preview@38ac237dc0bf4329f054d09054aaf22fdaf6f553
reference=Ascend-SACT/Hy3-preview@eb533c1dfd9a1fa7f373f9b980a9c0f973f1dad8
vllm=v0.18.0rc1@262ddd0d81a1e4687e209f988d6ea32616e736fa
vllm-ascend=v0.18.0rc1@99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f
```

不包含 `Hy3-preview-Base`。

## 输入输出数据

- 输入：OpenAI-compatible chat messages，可包含 tool schema 和 `reasoning_effort`。
- 输出：chat completion、reasoning/tool parser 结构和 token usage。
- 功能验证输入：`test_data/service_prompts.jsonl`，仓内固定 4 条。
- `tools/prepare_service_prompts.py` 可确定性生成 100 条 L2 服务精度回归 manifest
  和 metadata；正式任务精度仍优先使用公开 benchmark。
- CPU/CUDA 与 NPU 服务输出分别由 `../tools/openai_service_eval.py` 保存，再由 `../tools/compare_openai_service_results.py` 比较。

## 推理环境准备

| 配套 | 版本/要求 |
|---|---|
| 硬件 | Atlas A3，推荐 16 卡；约 590 GB 静态 BF16 权重外还需 KV/workspace |
| 容器 | `quay.io/ascend/vllm-ascend:v0.18.0rc1-a3` |
| vLLM | `262ddd0d81a1e4687e209f988d6ea32616e736fa` |
| vllm-ascend | `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f` |
| CANN、驱动、固件 | 使用镜像要求的配套版本，宿主机驱动需正确挂载 |

多机部署还必须配置固定的 HCCL rank table、网卡、容器网络和主机间免密/端口；本交付示例是单机 16 卡。

## 文件目录

```text
Hy3-preview
├── patches/0001-add-hy3-preview-support.patch
├── test_data/service_prompts.jsonl
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

在 ModelZoo 根目录设置宿主机路径：

```bash
export MODELZOO_DIR=$PWD
export MODEL_DIR=/models/Hy3-preview
export IMAGE=quay.io/ascend/vllm-ascend:v0.18.0rc1-a3
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}'

docker run --rm -it \
  --name hy3-preview \
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

不同宿主机的设备节点可能不同；进入容器后必须先用 `npu-smi info` 确认 16 卡可见，不能仅凭容器启动成功判断设备可用。

在容器中核对并应用补丁：

```bash
cd /workspace/ModelZoo/Hy3-preview
test "$(git -C /vllm-workspace/vllm rev-parse HEAD)" = \
  262ddd0d81a1e4687e209f988d6ea32616e736fa
test "$(git -C /vllm-workspace/vllm-ascend rev-parse HEAD)" = \
  99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f

git -C /vllm-workspace/vllm apply --check \
  /workspace/ModelZoo/Hy3-preview/patches/0001-add-hy3-preview-support.patch
git -C /vllm-workspace/vllm apply \
  /workspace/ModelZoo/Hy3-preview/patches/0001-add-hy3-preview-support.patch
python -m compileall -q /vllm-workspace/vllm/vllm
```

### 准备权重

在有足够磁盘空间的环境下载：

```bash
huggingface-cli download tencent/Hy3-preview \
  --revision 549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a \
  --local-dir "$MODEL_DIR"

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print | sort
sha256sum "$MODEL_DIR"/config.json "$MODEL_DIR"/model.safetensors.index.json
```

正式报告保存镜像 digest、完整文件清单、总大小和所有 shard SHA256。

### 准备数据集

仓内 4 条用于功能验证。生成固定 100 条 L2 服务精度回归 prompt：

```bash
python /workspace/ModelZoo/tools/prepare_service_prompts.py \
  --base /workspace/ModelZoo/Hy3-preview/test_data/service_prompts.jsonl \
  --output /workspace/ModelZoo/Hy3-preview/eval_data/service_prompts_l2.jsonl \
  --count 100
test "$(wc -l < /workspace/ModelZoo/Hy3-preview/eval_data/service_prompts_l2.jsonl)" = 100
```

该 100 条内部固定集不能推导官方四项 benchmark 指标。

Hy3 模型卡没有发布四项 instruct benchmark 的完整 agent/tool/judge/decode recipe，
因此当前可从零执行的 L2 降级路径是：100 条固定服务精度回归 + 确定性 100 请求
性能测试。它计算 token agreement、JSON/tool 有效率、TTFT、TPOT 和吞吐；报告必须
明确“非官方 benchmark 复现”。拿到官方 recipe 后再优先替换为全量 benchmark。

### 模型推理

先关闭 MTP 建立迁移 baseline：

```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
HCCL_OP_EXPANSION_MODE=AIV \
vllm serve "$MODEL_DIR" \
  --served-model-name hy3-preview \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --max-model-len 32768 \
  --max-num-seqs 8
```

baseline 通过后停止服务，并用完整命令开启 MTP：

```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
HCCL_OP_EXPANSION_MODE=AIV \
vllm serve "$MODEL_DIR" \
  --served-model-name hy3-preview \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

固定 vLLM 版本将 `--speculative-config` 定义为单个 JSON 参数，不支持把内部字段
拆成点号形式的多个 CLI 参数。

服务健康检查：

```bash
curl -sf http://127.0.0.1:8000/v1/models
mkdir -p /workspace/ModelZoo/Hy3-preview/results
python /workspace/ModelZoo/tools/openai_service_eval.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model hy3-preview \
  --prompts /workspace/ModelZoo/Hy3-preview/test_data/service_prompts.jsonl \
  --output /workspace/ModelZoo/Hy3-preview/results/functional_npu.jsonl
```

保存 NPU L2 服务精度回归结果：

```bash
python /workspace/ModelZoo/tools/openai_service_eval.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model hy3-preview \
  --prompts /workspace/ModelZoo/Hy3-preview/eval_data/service_prompts_l2.jsonl \
  --request_logprobs \
  --output /workspace/ModelZoo/Hy3-preview/results/npu.jsonl
```

流式 parser 单独检查：

```bash
curl -N -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy3-preview","messages":[{"role":"user","content":"Return exactly: ready"}],"temperature":0,"max_tokens":8,"stream":true,"chat_template_kwargs":{"reasoning_effort":"no_think"}}' \
  | tee /workspace/ModelZoo/Hy3-preview/results/npu-stream.txt
```

原始 vLLM commit 不包含 HyV3 架构，未应用 patch 的原始 baseline 应保存模型注册/
加载失败日志；它不能作为数值 baseline。应用 patch 后 CUDA 回归 baseline 是数值
迁移基线，使用相同 patch、checkpoint、vLLM commit 和参数。在安装了同 commit 且
应用同 patch 的 CUDA vLLM 环境中执行：

```bash
vllm serve "$MODEL_DIR" \
  --served-model-name hy3-preview \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --max-model-len 32768 \
  --max-num-seqs 8

python /workspace/ModelZoo/tools/openai_service_eval.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model hy3-preview \
  --prompts /workspace/ModelZoo/Hy3-preview/test_data/service_prompts.jsonl \
  --output /workspace/ModelZoo/Hy3-preview/results/functional_cuda.jsonl

python /workspace/ModelZoo/tools/openai_service_eval.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model hy3-preview \
  --prompts /workspace/ModelZoo/Hy3-preview/eval_data/service_prompts_l2.jsonl \
  --request_logprobs \
  --output /workspace/ModelZoo/Hy3-preview/results/cuda.jsonl
```

CUDA 侧不得使用 NPU 专用 `--enable-ep-weight-filter`。生成 `cuda.jsonl` 后比较：

```bash
python /workspace/ModelZoo/tools/compare_openai_service_results.py \
  --baseline /workspace/ModelZoo/Hy3-preview/results/functional_cuda.jsonl \
  --candidate /workspace/ModelZoo/Hy3-preview/results/functional_npu.jsonl \
  --require_exact_tool_calls \
  --output /workspace/ModelZoo/Hy3-preview/results/functional_cuda_vs_npu.json

python /workspace/ModelZoo/tools/compare_openai_service_results.py \
  --baseline /workspace/ModelZoo/Hy3-preview/results/cuda.jsonl \
  --candidate /workspace/ModelZoo/Hy3-preview/results/npu.jsonl \
  --require_logprobs \
  --require_exact_tool_calls \
  --output /workspace/ModelZoo/Hy3-preview/results/cuda_vs_npu.json
```

## 模型推理性能

官方未发布与当前 A3/vLLM-Ascend 环境可直接对齐的性能数值。CUDA 与 NPU 服务
分别执行以下同参 L2 性能命令；关闭/开启 MTP 各跑一轮，只修改结果文件名：

```bash
mkdir -p /workspace/ModelZoo/Hy3-preview/results/performance
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model hy3-preview \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 100 \
  --max-concurrency 8 \
  --seed 42 \
  --save-result \
  --save-detailed \
  --result-dir /workspace/ModelZoo/Hy3-preview/results/performance \
  --result-filename npu_mtp_off.json
```

报告 Successful requests、TTFT、TPOT、ITL、E2E、request/output/total
throughput、加载时间和峰值 HBM。CUDA 使用相同命令连接 CUDA 服务并写独立 JSON。
`32K/bs8` 只是启动配置，不代表 256K 已验收。

| 项目 | 官方值/当前状态 |
|---|---|
| SWE-bench Verified | 74.4%，精确 recipe 未完整发布 |
| Terminal-Bench 2.0 | 54.4%，精确 recipe 未完整发布 |
| BrowseComp / WideSearch | 67.1% / 70.2%，依赖外部工具 |
| A3 服务性能与 256K | 官方硬件性能未发布；当前未实测 |

## 公网地址说明

- 模型：<https://huggingface.co/tencent/Hy3-preview>
- 官方代码：<https://github.com/Tencent-Hunyuan/Hy3-preview>
- 参考适配：<https://gitcode.com/Ascend-SACT/Hy3-preview>

官方指标边界和正式验收见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，补丁分析见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
