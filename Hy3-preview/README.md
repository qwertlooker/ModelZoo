# Hy3-preview 推理指导

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

Hy3-preview 是 295B total / 21B active 的 BF16 MoE 模型，包含一层 3.8B MTP，声明支持 256K context。当前交付基于固定 vLLM + vllm-ascend，在 Atlas A3 上使用 TP16、EP、MTP 和 HyV3 tool/reasoning parser。本文档介绍该模型基于昇腾 NPU 的推理指导。

> 说明：本文档适配对象为 `tencent/Hy3-preview` instruct checkpoint，不包含 `Hy3-preview-Base`。

- 版本说明：

  ```text
  model=tencent/Hy3-preview@549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a
  official=Tencent-Hunyuan/Hy3-preview@38ac237dc0bf4329f054d09054aaf22fdaf6f553
  reference=Ascend-SACT/Hy3-preview@eb533c1dfd9a1fa7f373f9b980a9c0f973f1dad8
  vllm=v0.18.0rc1@262ddd0d81a1e4687e209f988d6ea32616e736fa
  vllm-ascend=v0.18.0rc1@99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f
  ```

## 输入输出数据

- 输入数据

  OpenAI-compatible chat messages，可包含 tool schema 和 `reasoning_effort`。功能验证使用仓内固定 4 条 `test_data/service_prompts.jsonl`。

- 输出数据

  chat completion、reasoning/tool parser 结构和 token usage。CPU/CUDA 与 NPU 服务输出分别由 `../tools/openai_service_eval.py` 保存，再由 `../tools/compare_openai_service_results.py` 比较。

## 推理环境准备

- 该模型需要以下插件与驱动。

  **表 1** 版本配套表

  | 配套 | 版本/要求 |
  |---|---|
  | 硬件 | Atlas A3，推荐 16 卡；约 590 GB 静态 BF16 权重外还需 KV/workspace |
  | 容器 | `quay.io/ascend/vllm-ascend:v0.18.0rc1-a3` |
  | vLLM | `262ddd0d81a1e4687e209f988d6ea32616e736fa` |
  | vllm-ascend | `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f` |
  | CANN、驱动、固件 | 使用镜像要求的配套版本，宿主机驱动需正确挂载 |

说明：本交付示例是单机 16 卡；多机部署还必须配置固定的 HCCL rank table、网卡、容器网络和主机间免密/端口。进入容器后必须先用 `npu-smi info` 确认 16 卡可见，不能仅凭容器启动成功判断设备可用。

## 文件目录

```text
Hy3-preview
├── patches/0001-add-hy3-preview-support.patch
├── test_data/service_prompts.jsonl
└── README.md
tools
├── openai_service_eval.py
├── compare_openai_service_results.py
└── prepare_service_prompts.py
```

## 快速上手

### 获取源码

1. 在 ModelZoo 根目录设置宿主机路径并启动容器。

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

2. 在容器中核对并应用补丁。

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

1. 在有足够磁盘空间的环境下载权重，并核对文件完整性。

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

1. 生成固定 100 条 L2 服务精度回归 prompt。

   ```bash
   python /workspace/ModelZoo/tools/prepare_service_prompts.py \
     --base /workspace/ModelZoo/Hy3-preview/test_data/service_prompts.jsonl \
     --output /workspace/ModelZoo/Hy3-preview/eval_data/service_prompts_l2.jsonl \
     --count 100
   test "$(wc -l < /workspace/ModelZoo/Hy3-preview/eval_data/service_prompts_l2.jsonl)" = 100
   ```

   参数说明：

   - `base`：仓内固定 4 条功能验证 prompt 文件。
   - `output`：生成的 100 条 L2 manifest 路径。
   - `count`：生成 prompt 条数。

### 模型推理

1. 先关闭 MTP 建立迁移 baseline。

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

2. baseline 通过后停止服务，并用完整命令开启 MTP。

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

3. 服务健康检查与功能验证。

   ```bash
   curl -sf http://127.0.0.1:8000/v1/models
   mkdir -p /workspace/ModelZoo/Hy3-preview/results
   python /workspace/ModelZoo/tools/openai_service_eval.py \
     --base_url http://127.0.0.1:8000/v1 \
     --model hy3-preview \
     --prompts /workspace/ModelZoo/Hy3-preview/test_data/service_prompts.jsonl \
     --output /workspace/ModelZoo/Hy3-preview/results/functional_npu.jsonl
   ```

4. 保存 NPU L2 服务精度回归结果。

   ```bash
   python /workspace/ModelZoo/tools/openai_service_eval.py \
     --base_url http://127.0.0.1:8000/v1 \
     --model hy3-preview \
     --prompts /workspace/ModelZoo/Hy3-preview/eval_data/service_prompts_l2.jsonl \
     --request_logprobs \
     --output /workspace/ModelZoo/Hy3-preview/results/npu.jsonl
   ```

5. 流式 parser 单独检查。

   ```bash
   curl -N -s http://127.0.0.1:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"hy3-preview","messages":[{"role":"user","content":"Return exactly: ready"}],"temperature":0,"max_tokens":8,"stream":true,"chat_template_kwargs":{"reasoning_effort":"no_think"}}' \
     | tee /workspace/ModelZoo/Hy3-preview/results/npu-stream.txt
   ```

6. 在应用同 patch 的 CUDA vLLM 环境中执行 baseline 推理。

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

7. 生成 `cuda.jsonl` 后比较 CUDA/NPU 结果。

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

CUDA 与 NPU 服务分别执行以下同参 L2 性能命令；关闭/开启 MTP 各跑一轮，只修改结果文件名。

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

报告 Successful requests、TTFT、TPOT、ITL、E2E、request/output/total throughput、加载时间和峰值 HBM。CUDA 使用相同命令连接 CUDA 服务并写独立 JSON。官方未发布与当前 A3/vLLM-Ascend 环境可直接对齐的性能数值。

## 适配与精度口径

### 补丁分析

补丁只修改 vLLM，vllm-ascend 不需要源码修改，由现有 Ascend attention、MoE、EP/HCCL 后端承载：

- 注册 `hy_v3` config、`HyV3ForCausalLM` 和 `HYV3MTPModel`；
- 实现 dense/MoE layer、shared expert、expert weight mapping 和 MTP；
- 注册 `hy_v3` reasoning/tool parsers；
- 增加 speculative config 的 HyV3 MTP 映射。

补丁基于精确 vLLM commit，并已通过 `git apply --check`；不能对其他 vLLM commit 直接套用。

### 运行边界

- 推荐 TP16 + EP；TP8 能否满足加载和 KV 需求需单独验证。
- `--enable-ep-weight-filter` 降低每 rank 非本地 expert 加载压力；CUDA baseline 不得使用该 NPU 专用参数，应省略，其余参数与 NPU 保持一致。
- MTP 属于推测解码。精度对齐先关闭 MTP 建立 baseline，再开启并验证输出质量与性能。
- tool/reasoning parser 是服务接口的一部分，必须单独测试流式/非流式、多个工具参数和 `reasoning_effort`。
- 32K/bs8 只是当前可行配置，不等同于 256K context 能力验收；长上下文能力验收需单独使用接近 256K 的固定输入集并记录峰值 HBM。
- 固定 vLLM 版本将 `--speculative-config` 定义为单个 JSON 参数，不支持把内部字段拆成点号形式的多个 CLI 参数，开启 MTP 时必须以完整 JSON 字符串传入，例如 `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`。
- 未应用 patch 的 vLLM 不支持 HyV3，原始 baseline 是可审计的注册/加载失败；数值 baseline 使用应用相同 patch 的 CUDA vLLM，candidate 使用 NPU。

### 官方指标边界

官方 instruct 模型卡公开：

| Benchmark | 官方分数 |
|---|---:|
| SWE-bench Verified | 74.4% |
| Terminal-Bench 2.0 | 54.4% |
| BrowseComp | 67.1% |
| WideSearch | 70.2% |

官方服务推荐参数为 `temperature=0.9`、`top_p=1.0`，复杂任务使用 `reasoning_effort=high`，直答使用 `no_think`。官方 README 没有发布上述 instruct 指标对应的精确 dataset revision/manifest SHA、agent harness commit、工具环境、采样次数和完整 decode 参数，必须标记这些字段"官方未发布"，获得作者 recipe 前不得宣称精确复现。官方 Base 模型另发布 MMLU 等表，但 Base 与本适配 instruct checkpoint 不可混用。

### 迁移对齐门禁

使用同一 checkpoint、vLLM commit、prompt JSONL、chat template 和 sampling 参数，比较 CUDA vLLM 与 NPU vLLM-Ascend：

- 请求成功率 100%，无权重缺失、rank/device、parser 错误；
- greedy top-1 token agreement `>= 99.5%`；
- 结构化 JSON/tool call schema 有效率 100%；
- MTP 开关后的任务级正确率不低于关闭 MTP 超过 0.5 个百分点。

逐 token 一致率会在首个分叉后级联下降，因此它只能和首个分叉位置、logprob、结构化输出有效率及任务正确率一起解释，不能单独作为生成质量结论。这些阈值是迁移初始门禁，必须用固定 CUDA/NPU baseline 的实测分布校准，不是官方发布容差。

## 公网地址说明

| 类型 | 说明 | 公网地址 |
|---|---|---|
| 模型权重 | tencent/Hy3-preview Hugging Face 模型仓 | https://huggingface.co/tencent/Hy3-preview |
| 开源代码仓 | Tencent-Hunyuan/Hy3-preview 官方源码 | https://github.com/Tencent-Hunyuan/Hy3-preview |
| 参考适配 | Ascend-SACT/Hy3-preview 参考适配仓 | https://gitcode.com/Ascend-SACT/Hy3-preview |
