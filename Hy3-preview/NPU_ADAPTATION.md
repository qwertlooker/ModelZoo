# Hy3-preview NPU 适配文档

## 1. 版本边界

- 官方模型代码：`38ac237dc0bf4329f054d09054aaf22fdaf6f553`
- instruct 权重：`tencent/Hy3-preview` commit `549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a`
- vLLM：`v0.18.0rc1` / `262ddd0d81a1e4687e209f988d6ea32616e736fa`
- vllm-ascend：`v0.18.0rc1` / `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f`
- Ascend-SACT：`eb533c1dfd9a1fa7f373f9b980a9c0f973f1dad8`
- 检查日期：2026-06-20；远端 HEAD/tag 与上述记录一致。

模型是 295B total / 21B active / 192 experts top-8 / 80 layers / GQA 64 heads、8 KV heads / BF16 / 256K context，含一层 MTP。不是 Base 变体。

## 2. 补丁分析

补丁只修改 vLLM：

- 注册 `hy_v3` config、`HyV3ForCausalLM` 和 `HYV3MTPModel`；
- 实现 dense/MoE layer、shared expert、expert weight mapping 和 MTP；
- 注册 `hy_v3` reasoning/tool parsers；
- 增加 speculative config 的 HyV3 MTP 映射。

vllm-ascend 不需要源码修改，由现有 Ascend attention、MoE、EP/HCCL 后端承载。补丁基于精确 vLLM commit，并已通过 `git apply --check`；不能对其他 vLLM commit 直接套用。

## 3. 运行边界和风险

- 推荐 TP16 + EP；TP8 能否满足加载和 KV 需求需单独验证。
- `--enable-ep-weight-filter` 降低每 rank 非本地 expert 加载压力。
- MTP 属于推测解码。精度对齐先关闭 MTP 建立 baseline，再开启并验证输出质量与性能。
- tool/reasoning parser 是服务接口的一部分，必须单独测试流式/非流式、多个工具参数和 `reasoning_effort`。
- 32K/bs8 只是当前可行配置，不等同于 256K 能力验收。
- CUDA baseline 和 NPU 功能验证必须使用同一 `test_data/service_prompts.jsonl`；
  L2 服务精度回归由 `tools/prepare_service_prompts.py` 生成同一 100 条 manifest，再由
  `tools/openai_service_eval.py` 写入不同结果文件；单条 curl 不构成迁移对齐。
- 未应用 patch 的 vLLM 不支持 HyV3，原始 baseline 是可审计的注册/加载失败；
  数值 baseline 使用应用相同 patch 的 CUDA vLLM，candidate 使用 NPU。

## 4. 验证事实

2026-06-20 已完成：

- 六个来源版本取证；
- 补丁 SHA256 固定；
- 在 vLLM `v0.18.0rc1` 精确 commit 上 `git apply --check` 通过；
- 官方模型卡指标和启动参数核对。

当前主机没有 A3/NPU、镜像和约 590GB 权重，未执行加载、服务、精度或性能测试。参考 README 中基于日志推导的 KV 容量不能作为本次实测结论。

已补充容器设备挂载、精确 commit 门禁、MTP 开关两阶段启动、固定功能/L2 prompt、
服务结果比较和 `vllm bench serve` 性能入口。当前状态仍是 **S1：patch 静态门禁
通过；升级到 S2/S3 仍缺 16 卡模型加载和 CUDA/NPU 功能、精度、性能对齐**。

公共服务评测工具已通过本地 mock OpenAI endpoint 验证：4 条 prompt 覆盖中英文、
JSON 和 tool call，JSON/tool schema 校验通过；相同结果比较的 content/tool/token
agreement 均为 `1.0`。这只证明评测工具链，不证明 Hy3 模型服务。

用户推理和补丁应用见 [README_INFERENCE.md](README_INFERENCE.md)，数据集与
CPU/CUDA/NPU 对齐要求见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。
