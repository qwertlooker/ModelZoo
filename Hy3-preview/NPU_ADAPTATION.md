# Hy3-preview NPU 适配文档

## 1. 版本边界

- 官方模型代码：`38ac237dc0bf4329f054d09054aaf22fdaf6f553`
- instruct 权重：`tencent/Hy3-preview` commit `549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a`
- vLLM：`v0.18.0rc1` / `262ddd0d81a1e4687e209f988d6ea32616e736fa`
- vllm-ascend：`v0.18.0rc1` / `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f`
- Ascend-SACT：`eb533c1dfd9a1fa7f373f9b980a9c0f973f1dad8`
- 检查日期：2026-06-29；远端 HEAD/tag 与上述记录一致。

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

用户推理和补丁应用见 [README.md](README.md)，数据集与
CPU/CUDA/NPU 对齐要求见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

权重下载优先使用 `huggingface-cli download`（在线路径，需 login），README 已补全等价 `curl` 带 token 离线替代命令（见 7.2 离线规则）。该模型为 gated 模型，离线下载前需先在可联网机器完成授权。

## 上库就绪与目标仓对齐

- 目标仓快照：`https://gitcode.com/Ascend/ModelZoo-PyTorch.git`，2026-06-29 重新查询
  `master` HEAD `7a02a6701c971b29df188a0f3241e1efe249d1df`（"modify document"）。
  2026-06-22 审阅快照为 `ec2a7b514973805f66b67c9178d2f5c9e97eee34`；本次不复用历史快照。
- 拟合入路径：`ACL_PyTorch/built-in/audio/Hy3-preview`。目标仓 `audio/` 下不存在该目录，
  本次为新增，不涉及替换或增量更新。
- 最新参考目录：同领域、同推理形态（audio / vLLM-Ascend 服务）选取
  `ACL_PyTorch/built-in/audio/Index-TTS-vLLM-v2`，其最后实质变更为 commit `6fecdfba7`
  （2026-06-18，`README.md` + `diff_index_tts_vllm_v2.patch` + `diff_torchaudio_kaldi.patch`）。
  选择原因是同属 vLLM-Ascend 服务形态且以 patch 交付，与本模型 patch 交付方式一致。
  更新的 `YingMusic-SVC_for_Pytorch`（`e98df562e`，2026-06-22）为 SVC 非 vLLM-Ascend 形态，
  仅作 PR 门禁参考。
- 贡献规范与 PR 门禁：`Ascend/modelzoo` HEAD `5eab9a4921c7f12edb555079836429a8f285cd1f`
  的 CONTRIBUTING.md 要求源码、README、参考模型 License、测试用例；AASIST-L 另含
  `modelzoo_level.txt`，但 Index-TTS-vLLM-v2、Canary-1B 等目录未提供 LICENSE/
  modelzoo_level.txt，历史目录与当前 PR 门禁存在差异。按贡献规范提交，不跳过也不伪造。
- 上库文件清单（候选）：`README.md`、`patches/0001-add-hy3-preview-support.patch`、
  `test_data/service_prompts.jsonl`；上库前补 `LICENSE`、`modelzoo_level.txt`。
- 排除项：`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、`patches/README.md`、`upstream/`、
  `weights/`、`eval_data/`、`results/`、`.codex-reference/`、日志与虚拟环境。
- 许可证：上游 `Tencent-Hunyuan/Hy3-preview`、vLLM、vllm-ascend 各自 License 上库前核对；
  新增 patch/脚本按贡献规范追加华为 License 头部。`modelzoo_level.txt` 须在 NPU 实测后据实填写。

## 5. 补充说明（来自 README.md）

以下适配决策与技术说明从推理指导文档迁移至此，便于终端用户保持 README 简洁。

### 5.1 容器设备可见性

不同宿主机的设备节点可能不同；进入容器后必须先用 `npu-smi info` 确认 16 卡可见，不能仅凭容器启动成功判断设备可用。

### 5.2 多机部署

多机部署还必须配置固定的 HCCL rank table、网卡、容器网络和主机间免密/端口；本交付示例是单机 16 卡。

### 5.3 speculative-config 参数形式

固定 vLLM 版本将 `--speculative-config` 定义为单个 JSON 参数，不支持把内部字段拆成点号形式的多个 CLI 参数。开启 MTP 时必须以完整 JSON 字符串传入，例如：

```text
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### 5.4 数值 baseline 来源

原始 vLLM commit 不包含 HyV3 架构，未应用 patch 的原始 baseline 应保存模型注册/加载失败日志；它不能作为数值 baseline。应用 patch 后 CUDA 回归 baseline 是数值迁移基线，使用相同 patch、checkpoint、vLLM commit 和参数。即：在安装了同 commit 且应用同 patch 的 CUDA vLLM 环境中执行 baseline 推理，candidate 使用 NPU。

### 5.5 CUDA 侧参数限制

CUDA 侧不得使用 NPU 专用 `--enable-ep-weight-filter`。CUDA baseline 的 `vllm serve` 命令应省略该参数，其余参数与 NPU 保持一致。
