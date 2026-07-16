# MechVL-4B-RL 昇腾 NPU 推理指导

## 概述

本目录将机械工程图纸视觉问答模型
[`XiaofengAlg/MechVL-4B-RL`](https://huggingface.co/XiaofengAlg/MechVL-4B-RL)
迁移到昇腾 NPU。模型基于 Qwen3-VL-4B，使用官方支持 Qwen3-VL 4B 的
`vLLM Ascend 0.18.0` OpenAI-compatible 服务路线；没有修改上游模型源码或权重。

当前交付已经完成固定版本、静态检查和离线数据链路冒烟。由于构建机没有
Ascend NPU 和 Docker，真实 NPU 功能、精度及性能仍须按本文命令补验；文中
“官方基线”不是本机 NPU 结果。

```text
commit_id=c9d4e7dc8a951fb9365e5ebe42601b0101d34ba3
```

## 版本与输入输出

| 项目 | 固定值 |
|---|---|
| 模型 | `XiaofengAlg/MechVL-4B-RL` |
| 权重 revision | `2c6fda8a16e57d8a6fe1019412092d09a0363850` |
| 上游代码 | `https://github.com/xiaofengShi/MechVQA` |
| 上游 commit | `8841ee083c2704f2d8ccf426a8c0bb61ad911890` |
| 输入 | 一张或多张机械图纸图片和自然语言问题 |
| 输出 | `&lt;think&gt;…&lt;/think&gt;&lt;answer&gt;…&lt;/answer&gt;`；客户端同时保存提取后的 `answer` |
| 主要指标 | MechVQA 三裁判多数投票准确率 |
| 权重许可证 | Apache-2.0，详见 `LICENSE` |

`weights.sha256` 固定 15 个运行时文件，其中权重约 9.0 GB。服务名、评测配置和
客户端统一使用 `MechVL-4B-RL`，避免把 checkpoint 路径误当作模型标识。

## 推理环境

推荐直接使用官方镜像，避免手工混装 `torch`、`torch_npu`、`vllm` 和
`vllm-ascend`。

| 组件 | 版本 / 要求 |
|---|---|
| OS | Linux |
| 硬件 | Atlas 800I A2 / Atlas 800 A3；本文默认单卡 |
| 镜像（A2） | `quay.io/ascend/vllm-ascend:v0.18.0` |
| 镜像（A3） | `quay.io/ascend/vllm-ascend:v0.18.0-a3` |
| Python | `>=3.10,<3.12` |
| CANN / NNAL | `9.0.0` |
| PyTorch / torch-npu | `2.9.0` / `2.9.0.post2` |
| vLLM / vLLM Ascend | `0.18.0` / `0.18.0` |

官方安装说明：<https://docs.vllm.ai/projects/ascend/en/v0.18.0/installation.html>；
Qwen3-VL 支持矩阵：
<https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/support_matrix/supported_models.html>。

## 文件目录

```text
MechVL-4B-RL/
├── README.md                 # 最低可重放路径
├── download_weights.sh       # 固定 revision 下载并逐文件校验
├── weights.sha256
├── weight_manifest.json
├── serve.sh                  # NPU 自检与 vLLM Ascend 服务
├── infer.py                  # 单样例推理
├── prepare_eval_data.py      # 固定公开测试集
├── make_eval_config.py       # 生成三裁判官方 evaluator 配置
├── compare_accuracy.py       # 质量门禁
├── benchmark.py              # 三轮端到端性能测试
└── requirements.txt
```

`runtime/`、权重、评测输入及输出均是本地生成物，不提交到 Git。

## 快速上手

### 1. 在宿主机准备固定资产

以下命令在本目录执行。先做只读可达性检查；去掉 `MODEL_CHECK_ONLY=1` 才会下载
并校验全部文件。

```bash
mkdir -p runtime

MODEL_CHECK_ONLY=1 ./download_weights.sh runtime/weights/MechVL-4B-RL
./download_weights.sh runtime/weights/MechVL-4B-RL

git clone https://github.com/xiaofengShi/MechVQA.git runtime/MechVQA
git -C runtime/MechVQA checkout 8841ee083c2704f2d8ccf426a8c0bb61ad911890
test "$(git -C runtime/MechVQA rev-parse HEAD)" = \
  "8841ee083c2704f2d8ccf426a8c0bb61ad911890"

python3 prepare_eval_data.py \
  --upstream-dir runtime/MechVQA \
  --output-dir runtime/eval_data
```

数据准备会在写出清单前校验 Git commit、源 JSONL SHA256、1,185 条问答、562 张
唯一图片以及每个图片路径。评测阶段不再隐式联网。

### 2. 启动单卡 NPU 容器

先在宿主机确认驱动可见：

```bash
npu-smi info
export IMAGE=quay.io/ascend/vllm-ascend:v0.18.0
# Atlas 800 A3 改为 quay.io/ascend/vllm-ascend:v0.18.0-a3

docker pull "$IMAGE"
docker run --rm -it \
  --name mechvl-npu \
  --net=host \
  --shm-size=16g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$PWD":/workspace/MechVL-4B-RL \
  -w /workspace/MechVL-4B-RL \
  "$IMAGE" bash
```

容器内记录真实环境并启动服务。物理卡只由上面的 Docker `--device` 选择，脚本
内部不重绑卡号。

```bash
npu-smi info
python3 - <<'PY'
import torch
import torch_npu
import vllm
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", vllm.__version__)
print("npu_available", torch.npu.is_available())
print("probe", torch.ones(1).to("npu").device)
PY

export MAX_MODEL_LEN=16384
export MAX_NUM_BATCHED_TOKENS=16384
export MAX_NUM_SEQS=16
export TENSOR_PARALLEL_SIZE=1
./serve.sh runtime/weights/MechVL-4B-RL
```

`serve.sh` 会先复核全部权重 SHA256 和真实 NPU 张量，再执行 `vllm serve`。若单卡
HBM 不足，先降低 `MAX_MODEL_LEN`、`MAX_NUM_BATCHED_TOKENS` 和 `MAX_NUM_SEQS`；
不要静默回退到 CPU。

### 3. 功能验证

另开一个宿主机终端，进入同一容器：

```bash
docker exec -it mechvl-npu bash
cd /workspace/MechVL-4B-RL

curl --fail http://127.0.0.1:8000/v1/models

python3 infer.py \
  --device npu \
  --image runtime/MechVQA/benchmark_data/images/59/59692aadeeac223740bc2facade0bbf21c03fe5e.jpg \
  --question '根据主剖视图和安装步骤图，支撑板高度调节的机械原理是什么？' \
  --output runtime/eval_results/smoke.json
```

通过标准：服务模型 ID 是 `MechVL-4B-RL`；命令退出码为 0；输出 JSON 中
`device=npu`、`provider=vllm-ascend-openai-compatible`，`raw_response` 和提取后的
`answer` 非空；日志无 CPU fallback、CUDA-only 或设备不一致错误。

### 4. 精度验证

原项目的 evaluator 是两阶段 OpenAI-compatible 评测：生成配置显式覆盖其简化默认
suffix，使用与 RL 训练/推理一致的完整 `mech_r1` 格式；再由
GPT-OSS-120B、DeepSeek-V3.2 和 Kimi-k2 以 `temperature=0.1` 独立判分并多数投票。
三个 judge 服务及凭据必须由验收环境提供，密钥只放环境变量。

```bash
python3 -m pip install -r runtime/MechVQA/evaluation/requirements.txt

: "${GPT_OSS_BASE_URL:?set the GPT-OSS-120B OpenAI-compatible /v1 URL}"
: "${DEEPSEEK_BASE_URL:?set the DeepSeek-V3.2 OpenAI-compatible /v1 URL}"
: "${KIMI_BASE_URL:?set the Kimi-k2 OpenAI-compatible /v1 URL}"
: "${GPT_OSS_API_KEY:?set GPT_OSS_API_KEY}"
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"
: "${KIMI_API_KEY:?set KIMI_API_KEY}"
export VQA_TARGET_API_KEY=EMPTY

python3 make_eval_config.py \
  --manifest runtime/eval_data/mechvqa_public_test.jsonl \
  --image-root runtime/MechVQA/benchmark_data \
  --results-dir runtime/eval_results \
  --output runtime/eval_config.json \
  --gpt-oss-base-url "$GPT_OSS_BASE_URL" \
  --deepseek-base-url "$DEEPSEEK_BASE_URL" \
  --kimi-base-url "$KIMI_BASE_URL"

cd runtime/MechVQA/evaluation
bash scripts/run_all.sh /workspace/MechVL-4B-RL/runtime/eval_config.json
cd /workspace/MechVL-4B-RL

python3 compare_accuracy.py \
  --stats runtime/eval_results/stats_MechVL-4B-RL.json \
  --evaluated runtime/eval_results/evaluated_MechVL-4B-RL.jsonl \
  --output runtime/eval_results/accuracy_comparison.json
```

模型卡/论文报告的 `Total=84.85` 被用作公开基线，但没有发布一份与本目录固定的
1,185 条 manifest 一一绑定的重放结果。因此默认 `0.8485 - 0.01` 只是一条保守的
工程门槛，正式验收前必须由维护者在同数据、同 prompt、同三裁判版本上校准。
`compare_accuracy.py` 还要求 1,185 条全部完成、`record_idx` 无缺失或重复、目标回答
非空、每条恰好有三个指定 judge 且均无错误；不接受抽样或只看 stats 替代。

### 5. 性能验证

精度服务保持相同启动参数，对固定清单前 10 条做 1 次预热和 3 次独立测量：

```bash
python3 benchmark.py \
  --manifest runtime/eval_data/mechvqa_public_test.jsonl \
  --image-root runtime/MechVQA/benchmark_data \
  --record 10 \
  --warmup 1 \
  --runs 3 \
  --output runtime/eval_results/benchmark_npu.json
```

报告范围是端到端 HTTP、排队、图片预处理、模型生成和传输，输出每轮均值/P50/P95
时延、requests/s、completion tokens/s 及三轮中位数。必须同时保存镜像摘要、
`npu-smi info`、服务启动参数和 HBM 采样；不同范围的吞吐不能直接比较。

## 模型推理性能&精度

| 项目 | 结果 |
|---|---|
| Python 编译与 CLI `--help` | 通过 |
| 固定公开集完整性检查 | 通过（1,185 QA / 562 图片） |
| 3 条 payload 离线构造 | 通过；`performance_valid=false`，不是性能结果 |
| 权重 URL 可达性 | 通过（固定 revision 的 15 个文件） |
| 真实 NPU 功能 | 待 Ascend 环境补验 |
| NPU 三裁判精度 | 待 Ascend 与三个 judge 服务补验 |
| NPU 性能 | 待 Ascend 环境补验 |

当前可声明 S1 静态交付；没有 `modelzoo_level.txt`，也不声明目标仓上库就绪。

## 公网地址

| 资源 | 地址 |
|---|---|
| 模型与模型卡 | <https://huggingface.co/XiaofengAlg/MechVL-4B-RL> |
| 上游源码、公开集、evaluator | <https://github.com/xiaofengShi/MechVQA> |
| 论文 v2 | <https://arxiv.org/html/2605.30794v2> |
| vLLM Ascend 安装 | <https://docs.vllm.ai/projects/ascend/en/v0.18.0/installation.html> |
| Qwen3-VL 支持矩阵 | <https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/support_matrix/supported_models.html> |
| Qwen VL Dense 教程 | <https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen-VL-Dense.html> |
| 官方镜像仓库 | <https://quay.io/repository/ascend/vllm-ascend> |
| ModelZoo-PyTorch 目标仓 | <https://gitcode.com/Ascend/ModelZoo-PyTorch> |
