# MechVL-4B-RL NPU 适配记录

## 版本边界

| 项目 | 固定值 |
|---|---|
| 检查日期 | 2026-07-16 |
| upstream 源码 | `https://github.com/xiaofengShi/MechVQA` |
| upstream commit | `8841ee083c2704f2d8ccf426a8c0bb61ad911890` |
| 权重来源 | `https://huggingface.co/XiaofengAlg/MechVL-4B-RL` |
| 权重 revision | `2c6fda8a16e57d8a6fe1019412092d09a0363850` |
| 模型架构 | `Qwen3VLForConditionalGeneration`，4B，BF16 safetensors |
| evaluator | upstream `evaluation/`，与源码同 commit |
| 评测数据 | public test，1,185 QA / 562 images |
| 非目标变体 | MechVL-4B-SFT、训练/EasyR1、数据生成链路、Qwen3-VL 其他尺寸 |

`weights.sha256` 固定 15 个运行时文件；三个大权重 shard 的 SHA256 分别为
`cf62f4c…39332`、`5146487e…bc0a`、`8a1dc883…e68838`。完整哈希与字节数见
`weights.sha256` 和 `weight_manifest.json`。

## 目标仓快照

### 最新参考目录

| 项目 | 取值 |
|---|---|
| 目标仓 | `https://gitcode.com/Ascend/ModelZoo-PyTorch.git` |
| master commit | `c9d4e7dc8a951fb9365e5ebe42601b0101d34ba3` |
| 拟合入路径 | `ACL_PyTorch/built-in/foundation_models/MechVL-4B-RL` |
| 路径状态 | 新增；该 commit 下不存在同名目录 |
| 领域参考 | `ACL_PyTorch/built-in/foundation_models/PaddleOCR-VL-1.5` |
| 参考实质变更 | `fa2af89…`，2026-05-29，vLLM 多模态服务路线 |
| 同域近期参考 | TabPFN `3e3fed…`（2026-07-02）、SigLIP2 `dd6fe3…`（2026-04-09） |
| 选择原因 | MechVL 是视觉语言 foundation model；服务和评测都采用 OpenAI-compatible API |

采样工具在 Windows 上完成仓库克隆和报告输出，但其最终打印因 GBK 解码非 ASCII
Git 日志而报错；参考 commit 均用 `git log` 独立复核，未把该报错当作通过证据。

## 上游静态审计

上游主推理入口 `scripts/batch_infer.py` 直接使用 vLLM CUDA 路线，并通过
`CUDA_VISIBLE_DEVICES` 选卡。RL 模式使用 `prompts/mech_r1.jinja`，采样参数为
`temperature=0.6`、`top_p=0.95`、`top_k=20`、`max_tokens=4096`，上下文上限
16K。公开 evaluator 已去除私有端点和密钥，接收 OpenAI-compatible target/judge
服务。

审计只读取了固定 Git checkout，没有执行上游脚本。构建机没有 Docker，无法按
隔离策略动态执行不受信任上游代码；因此当前没有上游 CUDA/CPU 动态基线。

## 路线决策

选择 `vllm-ascend 0.18.0`，而不是自行给 Transformers eager 路径打
`torch_npu` patch：

1. 官方支持矩阵明确列出 Qwen3-VL 2B/4B/8B/32B，覆盖本模型的 Qwen3-VL-4B
   架构。
2. 官方提供 Atlas A2 与 A3 的 0.18.0 镜像，版本闭包为 CANN/NNAL 9.0.0、
   torch 2.9.0、torch-npu 2.9.0.post2、vLLM/vLLM Ascend 0.18.0。
3. 上游推理与 evaluator 本来就使用 vLLM/OpenAI-compatible 接口，适配可以只替换
   执行后端，不改变任务语义、RL prompt 或三裁判协议。
4. 没有修改第三方源码或权重，因此 patch 状态是 `no-patch`；交付内容是启动、校验、
   数据固定和验收脚本。

不选择 OM/ATC：Qwen3-VL 含动态视觉 token、长上下文生成和 KV cache，固定图导出
会扩大工程面且丢失上游服务语义。不选择 Paddle/TorchAir 路线：没有必要为已受
vLLM Ascend 官方支持的架构重写推理栈。

## 实现清单

| 文件 | 说明 |
|---|---|
| `download_weights.sh` | 固定 HF revision，支持只读 HEAD 检查，下载后逐文件 SHA256 校验 |
| `weights.sha256` / `weight_manifest.json` | 15 个运行时资产的哈希与字节数 |
| `serve.sh` | 先做权重和真实 NPU tensor 自检，再启动 BF16 vLLM 服务 |
| `infer.py` | 复现 RL prompt/采样参数，写入输入哈希、原始输出、答案、时延和 usage |
| `prepare_eval_data.py` | 固定 upstream commit、源 manifest 哈希、样本数、图片数和路径安全 |
| `make_eval_config.py` | 显式固定完整 `mech_r1` suffix 和三 judge 配置，密钥仅从环境变量读取 |
| `compare_accuracy.py` | 全量 1,185 条、目标/三 judge 零错误、索引完整、保守工程阈值，失败返回非零 |
| `benchmark.py` | 服务模型 ID 校验，至少三轮的端到端性能记录；dry-run 明确无效 |
| `.gitignore` | 排除权重、upstream、评测输出、密钥配置和缓存 |

适配没有加入自动 CPU fallback、CUDA alias、`try/except` 静默降级或硬编码物理卡。

## 已完成验证

| 层级 | 命令/证据 | 结果 |
|---|---|---|
| S0 | 固定 model/upstream/target 三个 commit；目标路径查重 | 通过 |
| S0 | 只读审计 upstream、HF metadata 和官方 vLLM Ascend 文档 | 通过 |
| S1 | 所有 Python 文件 `py_compile` | 通过 |
| S1 | 5 个 Python CLI 的 `--help` | 通过 |
| S1 | `bash -n download_weights.sh serve.sh` | 通过 |
| S1 | `prepare_eval_data.py --limit 3` | 通过；源集仍全量校验为 1,185/562 |
| S1 | `benchmark.py --dry-run --record 3` | 通过；仅验证 payload，明确 `performance_valid=false` |
| S1 | 固定 revision 的 15 个 HF 文件 HEAD | 通过 |

以上 S1 在 Windows 构建机的 Python 3.13 上完成，仅证明标准库脚本和静态数据链路，
不证明 vLLM、torch-npu 或模型可在该机运行。

## 未执行与补验条件

| 项目 | 原因 | 补验条件 |
|---|---|---|
| S2 单卡 NPU 冒烟 | 构建机无 Ascend 设备、驱动、CANN 和 Docker | Atlas A2/A3 + 官方 0.18.0 镜像，执行 README 第 2-3 节 |
| S3 全量精度 | 同上，且缺三个固定 judge 服务 | 目标 NPU 服务 + GPT-OSS-120B/DeepSeek-V3.2/Kimi-k2 OpenAI-compatible 服务 |
| S3 性能 | 无真实 NPU | 固定硬件、镜像摘要、服务参数，至少三轮 |
| S4 clean-room NPU 重放 | S2/S3 未闭环 | 全新目录/容器按 README 重放并归档日志 |
| CPU/CUDA 回归 | 安全策略下未动态执行上游；任务目标是 NPU 服务 | 可控 Linux CUDA 环境固定同 checkpoint/manifest 后补做 |

当前状态: S1。不能把 `--dry-run`、Python 编译、URL HEAD 或官方支持
矩阵写成 NPU 成功，也不能生成 `modelzoo_level.txt` 宣称完成验收。

## 上库文件清单与边界

目标目录候选：

- `.gitignore`
- `LICENSE`
- `README.md`
- `requirements.txt`
- `download_weights.sh`
- `weights.sha256`
- `weight_manifest.json`
- `serve.sh`
- `infer.py`
- `prepare_eval_data.py`
- `make_eval_config.py`
- `compare_accuracy.py`
- `benchmark.py`

在 S3/S4 完成前不生成 `modelzoo_level.txt`。以下内部过程材料不进入最终
ModelZoo-PyTorch 候选包：

- `NPU_ADAPTATION.md`
- `ACCEPTANCE_PLAN.md`
- `.codex-reference/`
- `upstream/`
- `runtime/`、`weights/`、`eval_data*/`、`eval_results*/`、`results*/`
- `.ascend-adaptation/` 项目日志

正式上库前仍需在目标 commit 上做冲突检查，按目标仓 `CONTRIBUTION.md` 复核目录名、
许可证、入口命令和 README 格式，再由目标仓维护者确认工程精度阈值。

## PR 门禁

只有 S3 精度/性能和 S4 clean-room 重放全部完成，候选文件审计无敏感信息、大文件、
内部过程材料和未固定依赖，且维护者批准精度阈值后，才可生成
`modelzoo_level.txt` 并提交目标仓 PR。当前不满足该门禁。
