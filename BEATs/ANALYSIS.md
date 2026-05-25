# BEATs NPU 适配分析

## 1. 上游信息

- 上游仓库：<https://github.com/microsoft/unilm.git>
- 上游子目录：`beats/`
- 分支：`master`
- 本地 upstream commit：`833df7e7832e5064a281131ee64a481afa8e5b95`
- 本地上游副本：`BEATs/upstream/`
- NPU patch：`BEATs/patches/0001-add-npu-fbank-device-support.patch`
- 复核日期：2026-05-25

远端最新 commit 需在可联网环境通过 `git -C BEATs/upstream ls-remote origin master` 复核；本次未变更 upstream clone。

## 2. 当前目录文件分析

- `infer.py`：新增适配脚本，默认 `--device npu`，CPU 验证显式传 `--device cpu`；`torch_npu` 仅在选择 NPU 时条件导入。
- `patches/0001-add-npu-fbank-device-support.patch`：唯一上游源码改动 patch。
- `scripts/download_weights.sh`：权重下载/放置脚本。BEATs 官方权重为 OneDrive 链接，脚本支持用户提供直链。
- `scripts/download_test_data.sh`：生成 1 秒 16 kHz dummy wav。
- `requirements.txt`：历史整环境导出，不作为最小依赖。

## 3. 设备相关节点扫描结论

上游 `beats/BEATs.py::preprocess()` 对输入 waveform 调用 `torchaudio.compliance.kaldi.fbank()`。NPU 推理时 waveform 已在 NPU，常见 torch-npu 环境中 fbank 不能直接处理 NPU Tensor。

处理方式：记录输入设备，将 waveform 临时搬到 CPU 计算 fbank，再将 fbank 搬回输入设备。模型结构和后续前向不变。

## 4. 修改范围

- 修改上游已有文件：`beats/BEATs.py`，通过 patch 管理。
- 新增当前适配文件：`infer.py`、`scripts/download_weights.sh`、`scripts/download_test_data.sh`、文档文件，不进入 patch。

## 5. 当前验证状态

- `git apply --check`：通过。
- `python3 -m py_compile BEATs/infer.py`：通过。
- 测试数据生成：通过。
- CPU 当前环境验证：已尝试，因官方 checkpoint 未下载而阻塞。
- NPU 验证：当前环境无 NPU，待目标机器执行。

## 6. 风险与限制

- fbank CPU 回退存在 NPU/CPU 数据搬运开销。
- dummy wav 只验证链路，不验证准确率。
- fine-tuned checkpoint 的 `label_dict` 与任务绑定；预训练 checkpoint 可能无法输出可读分类标签。

## 7. 上游更新处理

上游更新时必须先检查 `beats/BEATs.py::preprocess()` 是否变化，执行：

```bash
git -C BEATs/upstream fetch origin master
git -C BEATs/upstream diff HEAD origin/master -- beats/BEATs.py
git -C BEATs/upstream apply --check ../patches/0001-add-npu-fbank-device-support.patch
```

如上下文变化，重新生成 patch。
