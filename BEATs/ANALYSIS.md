# BEATs NPU 适配分析

## 1. 上游信息

- 上游仓库：<https://github.com/microsoft/unilm.git>
- 上游子目录：`beats/`
- 分支：`master`
- 基准 commit：`833df7e7832e5064a281131ee64a481afa8e5b95`
- 本地上游副本：`BEATs/upstream/`
- NPU patch：`BEATs/patches/0001-add-npu-fbank-device-support.patch`

## 2. 当前目录状态

当前 `BEATs/` 目录原有文件：

- `BEATs.py`：基于上游 `beats/BEATs.py` 修改。
- `infer.py`：推理样例，显式 `--device`，默认 `npu`。
- `README.md`：NPU 运行说明。
- `requirements.txt`：当前环境导出的依赖，范围明显大于 BEATs 实际最小依赖。

## 3. 与上游匹配情况

`BEATs.py` 与上游 `beats/BEATs.py` 基本匹配，仅在 `BEATs.preprocess()` 中增加了 fbank 计算的 CPU 回退逻辑。该差异属于 NPU 适配，因为 `torchaudio.compliance.kaldi.fbank` 在常见 torch-npu 环境中不能直接处理 NPU Tensor。

适配策略：输入 waveform 原本在 NPU；fbank 前临时搬到 CPU；fbank 结果搬回原设备；后续 `Conv2d/Transformer` 仍在 NPU 执行。

## 4. 现有代码审视

| 文件 | 结论 | 说明 |
|---|---|---|
| `BEATs.py` | 可保留但建议用 patch 版本 | 当前改动点正确，但注释不足；patch 中补充了原因说明并保持上游兼容。 |
| `infer.py` | 已直接改为当前适配脚本 | 不属于上游原项目文件，不进 patch；显式指定 `--device`，默认 `npu`；不使用 `use_gpu`，CPU 验证时显式传 `--device cpu`。 |
| `requirements.txt` | 不建议作为最小依赖 | 包含大量无关 CUDA/NVIDIA/服务端依赖，应在正式指导中给出最小依赖。 |
| `README.md` | 需增强 | 当前说明要求替换 `BEATs.py`，但缺少基准 commit、patch 应用、CPU/NPU 一致性验证。 |

## 5. 需要修改上游代码的节点

1. `beats/BEATs.py::BEATs.preprocess`：上游直接对输入 tensor 调用 `ta_kaldi.fbank()`。当输入 tensor 已迁移到 NPU 时存在设备不兼容风险。patch 仅在 fbank 计算处临时 CPU 回退，不修改模型结构和数学逻辑。
2. `BEATs/infer.py`：该脚本不是上游原项目文件，不放入 patch；作为当前适配仓的交付脚本直接维护，并在文档中说明用法。

## 6. 风险与限制

- fbank 在 CPU 上执行，会引入一次 NPU↔CPU 数据搬运；对短音频影响较小，对大批量/长音频性能有影响。
- BEATs 官方 checkpoint 的 `label_dict` 与具体 fine-tuned 任务绑定，验证时必须使用匹配权重。
- 当前目录中的 `infer.py` 已改为参数化脚本；使用时需拷贝到应用 patch 后的上游 `beats/` 目录。

## 7. 上游版本检查记录

- 检查日期：2026-05-23。
- GitHub commits 页面显示 `microsoft/unilm` 的 `master` 最新提交为 `833df7e`，日期为 2026-01-23，提交信息为 `Merge pull request #1739 from Dod-o/patch-1`。
- 当前适配基准 commit 为 `833df7e7832e5064a281131ee64a481afa8e5b95`，与远端最新 `master` 一致。
- 后续如上游更新，必须先执行 `git fetch origin master`，再检查 `beats/BEATs.py`、`beats/README.md`、`beats/Tokenizers.py` 是否变化，尤其确认 `BEATs.preprocess()` 中 fbank 处理逻辑是否仍与 patch 上下文匹配。
