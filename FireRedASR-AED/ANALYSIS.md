# FireRedASR-AED NPU 适配分析

## 1. 上游信息

- 上游仓库：<https://github.com/FireRedTeam/FireRedASR.git>
- 分支：`main`
- 基准 commit：`834635e4cf277ed8ca92049fc375b17c3dc20748`
- 本地上游副本：`FireRedASR-AED/upstream/`
- NPU patch：`FireRedASR-AED/patches/0001-add-npu-device-support.patch`

## 2. 当前目录状态

当前 `FireRedASR-AED/` 原有文件：

- `infer.py`：通过 `torch_npu.contrib.transfer_to_npu` 迁移上游 `.cuda()` 调用。
- `README.md`：NPU 运行说明。
- `requirements.txt`：当前环境导出的依赖，明显大于 FireRedASR-AED 实际最小依赖。

## 3. 与上游匹配情况

当前 `infer.py` 没有对应上游同名文件，是新增样例脚本。上游真实入口是 `fireredasr/speech2text.py`、`fireredasr/models/fireredasr.py` 和 `examples/inference_fireredasr_aed.sh`。

上游 `fireredasr/models/fireredasr.py` 中存在硬编码 CUDA 节点：`feats.cuda()`、`lengths.cuda()`、`self.model.cuda()`、`input_ids.cuda()`、`attention_mask.cuda()`。因此，如果不依赖 `transfer_to_npu` 的全局 monkey patch，原始上游代码需要做 device 显式化适配。

## 4. 现有代码审视

| 文件 | 结论 | 说明 |
|---|---|---|
| `infer.py` | 可跑通简单场景，但不够规范 | 依赖 `transfer_to_npu` 将 `.cuda()` 映射到 NPU；路径、音频列表、模型目录硬编码。 |
| `requirements.txt` | 不建议作为最小依赖 | 上游最小依赖只有 `cn2an/kaldiio/kaldi_native_fbank/numpy/peft/sentencepiece/torch/transformers` 等。 |
| `README.md` | 需增强 | 缺少基准 commit、patch 应用、CER/WER 验证和设备参数说明。 |

## 5. 需要修改上游代码的节点

1. `fireredasr/models/fireredasr.py`：增加 `import torch_npu` 和 `_resolve_device(args)`，用 `.to(device)` 替代 `.cuda()` / `.cpu()` 分支；保留 `use_gpu` 行为，优先 NPU，其次 CUDA。
2. `fireredasr/speech2text.py`：新增 `--device`，支持 `cpu`、`cuda`、`npu`，并传入 `model.transcribe()`。
3. `FireRedASR-AED/infer.py`：该脚本不是上游原项目文件，不放入 patch；作为当前适配仓的交付脚本直接维护。

## 6. 风险与限制

- FireRedASR-AED 的特征提取主要在 CPU 执行，模型前向在 NPU；端到端性能受音频 I/O 和 fbank 特征提取影响。
- patch 聚焦 PyTorch AED 推理链路，不适配 `runtime/triton_tensorrt/`。
- 原始适配只通过 `transfer_to_npu` 让 `.cuda()` 在运行时转到 NPU，能覆盖 demo，但遗漏了上游 `fireredasr/models/fireredasr.py` 中硬编码 `.cuda()` 的源码级适配点；因此这次补充为显式 `device` 机制。
- 上游 LLM 分支也会受益于 device 显式化，但本项目验证重点是 AED。

## 7. 上游版本检查记录

- 检查日期：2026-05-23。
- GitHub commits 页面显示 `FireRedTeam/FireRedASR` 的 `main` 最新提交为 `834635e`，日期为 2026-02-25，提交信息为 `Update README.md`。
- 当前适配基准 commit 为 `834635e4cf277ed8ca92049fc375b17c3dc20748`，与远端最新 `main` 一致。
- 后续如上游更新，必须先执行 `git fetch origin main`，再检查 `fireredasr/models/fireredasr.py`、`fireredasr/speech2text.py`、`examples/inference_fireredasr_aed.sh`、`requirements.txt` 是否变化，尤其确认 `.cuda()` 节点和 `--use_gpu` 参数是否已被上游重构。

## 8. 原始适配遗漏点说明

是的，原始适配主要依赖 `torch_npu.contrib.transfer_to_npu`，通过运行时 monkey patch 把上游 `.cuda()` 调用转到 NPU。这个方式对简单 demo 有效，但没有明确修改上游源码中的设备选择节点：`fireredasr/models/fireredasr.py::FireRedAsr.transcribe`。

因此严格来说，原始适配遗漏了源码级 device 抽象这一点。当前 patch 的目的不是新增业务逻辑，而是把隐式 monkey patch 改成显式、可审查、可回退的 `device` 参数机制，同时保留 CPU/CUDA/NPU 兼容。
