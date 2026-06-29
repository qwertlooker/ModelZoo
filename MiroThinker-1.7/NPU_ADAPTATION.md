# MiroThinker-1.7 NPU 适配文档

## 1. 版本边界

- 模型：`miromind-ai/MiroThinker-1.7` 235B，HF commit `1a42014ce72e1025fdbf3c48d54545715ab3eea8`
- 官方 agent 框架：commit `370f98361553ddf787bedc5745760e04114cb161`
- Ascend-SACT：commit `a4199f82dcadf88e81e296eb2d0e79bdb5805184`
- vLLM/vllm-ascend：`v0.17.0rc1`
- MiroFlow-Benchmarks：commit `09900fb9f7297b853f56e1b785491494e93ac85d`，
  2025-11-15 archive SHA256
  `35816f69ba5f0d2baf45b248c68dd4a8e0f9b30cac6f41076f44099d5073f377`
- 检查日期：2026-06-29；上述远端 HEAD/tag 已核对。

模型配置：Qwen3 MoE、约 235B、94 层、64 attention heads、4 KV heads、128 experts top-8、hidden size 4096、256K context。不是 30B mini。

## 2. 适配结论

vLLM `v0.17.0rc1` 已有 `Qwen3MoeForCausalLM` 支持，vllm-ascend 提供 NPU attention/MoE/图编译后端。参考实现没有修改两个上游仓，因此正式交付不制造空泛 patch。

关键运行边界：

- TP16 以满足 235B 权重和 KV/workspace 需求；
- 显式 `compilation_config` capture sizes，避免默认档位与实际并发不匹配；
- `--trust-remote-code` 只针对固定、已校验的本地权重目录；
- 当前 8K/128 并发是参考服务配置，不等同于 256K 长上下文验收；
- agent 质量依赖外部 search/scrape/code tools，模型服务 smoke 不能替代官方 agent benchmark。
- 服务注册名、评测 `LLM_MODEL` 和请求 model 必须统一；8K smoke 与 256K
  benchmark 使用两套明确的启动配置。

## 3. 验证事实

2026-06-20 已完成官方/参考/model/vLLM 版本取证，核对官方公开指标和参考启动
参数；实际下载并解压固定 benchmark archive，确认 BrowseComp 1,266 条、
BrowseComp-ZH 289 条、GAIA-Val 165 条、HLE-Text-2158 2,158 条。

当前主机没有 A3/NPU、vLLM-Ascend 镜像和 235B 权重，未执行服务、CUDA/NPU 对齐、官方 agent benchmark 或性能测试。Ascend-SACT README 中的吞吐表是参考来源结果，不作为本次交付实测。

已补充容器设备挂载、Agent `uv sync --frozen`、`.env` 前置条件、固定服务
prompt 的 4 条功能集、100 条确定性 L2 服务回归、CUDA/NPU 比较入口、服务性能
命令和四项 benchmark 命令，并修正原文中服务名与评测模型名不一致以及 8K
服务直接用于 256K benchmark 的矛盾。当前状态仍是 **S1：部署和验收流程已固定；
升级到 S2/S3 仍缺 235B NPU 功能验证和同 endpoint 的 L2 agent 精度/性能对齐**。

公共服务评测工具已用本地 mock OpenAI endpoint 完成 JSONL 请求、原子输出、
metadata 和 token 对齐自测试；该结果只验证工具，不替代 235B 模型服务或 agent
benchmark。

用户部署见 [README.md](README.md)，官方 agent 验收参数和
报告模板见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

## 上库就绪与目标仓对齐

- 目标仓快照：`https://gitcode.com/Ascend/ModelZoo-PyTorch.git`，2026-06-29 重新查询
  `master` HEAD `7a02a6701c971b29df188a0f3241e1efe249d1df`（"modify document"）。
  2026-06-22 审阅快照为 `ec2a7b514973805f66b67c9178d2f5c9e97eee34`；本次不复用历史快照。
- 拟合入路径：`ACL_PyTorch/built-in/nlp/MiroThinker-1.7`。目标仓 `nlp/` 下不存在该目录，
  本次为新增，不涉及替换或增量更新。
- 最新参考目录：推理形态相同（优先）选 `ACL_PyTorch/built-in/audio/Index-TTS-vLLM-v2`
  （vLLM-Ascend 服务，最后实质变更 `6fecdfba7`，2026-06-18），提供同形态 vLLM-Ascend
  patch 交付参考；同领域选 `ACL_PyTorch/built-in/nlp/chronos-2`（nlp 在线 PyTorch，
  `6fecdfba7`，2026-06-18，`ascend_infer.py`/`eval_accuracy.py`/`eval_performance.py`），
  提供 nlp 精度/性能脚本参考。本模型无 patch（vLLM `v0.17.0rc1` 原生支持 `Qwen3MoeForCausalLM`），
  正式交付不制造空 patch，以 `serve_npu.sh` 作为可运行服务入口。
- 贡献规范与 PR 门禁：`Ascend/modelzoo` HEAD `5eab9a4921c7f12edb555079836429a8f285cd1f`
  的 CONTRIBUTING.md 要求源码、README、参考模型 License、测试用例；AASIST-L 另含
  `modelzoo_level.txt`，但 chronos-2、ProtBert 等目录未提供 LICENSE/modelzoo_level.txt，
  历史目录与当前 PR 门禁存在差异。按贡献规范提交，不跳过也不伪造。
- 上库文件清单（候选）：`README.md`、`serve_npu.sh`、`test_data/service_prompts.jsonl`；
  上库前补 `LICENSE`、`modelzoo_level.txt`。
- 排除项：`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、`patches/README.md`、`upstream/`、
  `weights/`、`eval_data/`、`results/`、`.codex-reference/`、日志与虚拟环境。
- 许可证：上游 `MiroMindAI/MiroThinker`、vLLM、vllm-ascend 各自 License 上库前核对；
  新增脚本按贡献规范追加华为 License 头部。`modelzoo_level.txt` 须在 NPU 实测后据实填写。

## 补充说明（来自 README.md）

### benchmark 外部依赖与数据冻结

完整 benchmark 还需要 Serper、Jina、E2B、summary LLM 和 OpenAI judge API。动态搜索结果和网页内容不能被视为完全冻结的数据资产。

### CUDA/NPU 服务端口与对齐

- CUDA 与 NPU 服务不能同时占用 8002 端口；分别运行并保存结果后比较。
- 切换 CUDA/NPU 时只允许替换 `BASE_URL`，其余环境变量、`.env`、数据 archive 和运行次数保持一致。

### 8K 与 256K 服务边界

官方 benchmark 不能使用上述 8K 服务。必须重启成 256K 服务，并降低并发以满足实际 HBM。
