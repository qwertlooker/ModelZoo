# Canary-1B NPU 适配分析

## 1. 上游信息

- 上游仓库：<https://github.com/NVIDIA-NeMo/NeMo.git>
- 分支：`main`
- 基准 commit：`44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`
- commit 信息：`ci: remove build-docs and build-test-publish-wheel workflows (#15685)`
- 检查日期：2026-05-23
- 模型权重：<https://huggingface.co/nvidia/canary-1b>
- 版本边界：当前适配的是原始 `nvidia/canary-1b` / `canary-1b.nemo`；不包含 `nvidia/canary-1b-flash` 或 `nvidia/canary-1b-v2`。已验证权重 SHA256：`b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a`。
- 本地上游副本：`Canary-1B/upstream/`（已通过 `git clone --depth 1` 获取）

## 2. 当前目录状态

当前 `Canary-1B/` 原有文件：

- `infer.py`：原始推理 demo，依赖 `torch_npu.contrib.transfer_to_npu`，并包含硬编码音频/缓存路径。
- `README.md`：NPU 运行说明，但要求手工移动脚本并修改路径。
- `requirements.txt`：当前环境导出的依赖，范围明显大于 Canary/NeMo ASR 推理最小依赖。

本次新增/调整：

- `infer.py`：改为当前适配目录维护的参数化 CPU/NPU 融合推理脚本，默认 `--device npu`。
- `patches/README.md`：说明本次没有上游源码 patch。
- `ANALYSIS.md`、`NPU_ADAPTATION.md`、`NPU_VALIDATION.md`：适配分析、迁移说明和验证记录。
- `.gitignore`：加入 `Canary-1B/upstream/`。

## 3. 与上游匹配情况

Canary-1B 通过 NeMo `EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')` 加载。上游 `nemo/collections/asr/models/aed_multitask_models.py` 的推理链路使用 `trcfg._internal.device`、`tensor.to(device)` 和模型自身 device 传递，未发现必须为 Canary 单独修改上游源码的 `.cuda()` 硬编码节点。

因此本次适配不修改 NeMo 上游已有文件，不生成 `.patch`；只交付当前模型目录新增的 `infer.py` 和文档。后续如发现某个 NeMo 版本在 Canary 推理链路中新增硬编码 CUDA/NCCL 节点，应先在 `Canary-1B/upstream/` 对应文件修改，再生成 patch。

## 4. 现有代码审视

| 文件 | 结论 | 说明 |
|---|---|---|
| `infer.py` | 已重写 | 默认 NPU，支持 `--device cpu` 验证；无 `auto/use_gpu`；不写死 `npu:0/cuda:0`；音频、任务、语言和模型路径参数化。 |
| `README.md` | 已更新 | 补充基准 commit、无需 patch、运行方式和验证方式。 |
| `requirements.txt` | 保留但不建议作为最小依赖 | 包含 CUDA/服务端/训练相关大量依赖，正式部署建议按 README 中最小依赖安装。 |
| `patches/` | 无上游 patch | 因未修改 NeMo 上游已有文件，仅保留 README 说明。 |
| `prepare_eval_data.py` / `eval_canary.py` | 已新增 | 提供评测数据准备和评测脚本。 |

## 5. 设备适配点

1. `infer.py::_resolve_device`：仅当 `--device npu` 时导入 `torch_npu` 注册后端；返回 `torch.device('npu')`，不绑定卡号。
2. `EncDecMultiTaskModel.from_pretrained(..., map_location=device)`：加载时按目标设备映射权重。
3. `model.to(device)`：显式迁移模型。
4. `model.transcribe(...)`：输入通过 manifest 显式传入 `taskname/source_lang/target_lang/pnc`，由 NeMo dataloader 和模型内部 device 机制处理 batch。

## 6. 风险与限制

- 当前未在本机真实 NPU 上执行端到端推理；已完成静态检查、`py_compile`、CPU 环境搭建和 CPU 推理启动验证。
- 已通过 HF 镜像下载 `canary-1b.nemo`，并完成当前环境 CPU smoke test，输出 `[0]  I'm a part of that.`。
- Canary-1B 约 1B 参数，NPU 显存、CANN/torch-npu/torch 版本需要匹配。
- NeMo 主分支持续变化；如果上游更新，应重新检查 `EncDecMultiTaskModel`、`ASRTranscriptionMixin` 和音频预处理链路。
- `requirements.txt` 非最小依赖，可能引入无关 CUDA 包；部署时优先安装与当前 CANN/torch-npu 匹配的 PyTorch、torch-npu 和 NeMo ASR 依赖。

## 7. 上游版本检查记录

- 2026-05-23：重新执行 `git clone --depth 1 https://github.com/NVIDIA-NeMo/NeMo.git Canary-1B/upstream` 成功。
- 2026-05-23：`git -C Canary-1B/upstream rev-parse HEAD` 输出 `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`。
- 2026-05-23：`git -C Canary-1B/upstream ls-remote origin refs/heads/main` 确认远端 `main` 同为 `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`。
- 2026-05-23：检查 `nemo/collections/asr/models/aed_multitask_models.py`，确认 Canary 主要推理代码使用 device 传递，无需 patch。
