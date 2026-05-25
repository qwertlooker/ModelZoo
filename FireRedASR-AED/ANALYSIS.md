# FireRedASR-AED NPU 适配分析

## 1. 上游信息

- 上游仓库：<https://github.com/FireRedTeam/FireRedASR.git>
- 分支：`main`
- 本地 upstream commit：`834635e4cf277ed8ca92049fc375b17c3dc20748`
- 本地上游副本：`FireRedASR-AED/upstream/`
- NPU patch：`FireRedASR-AED/patches/0001-add-npu-device-support.patch`
- 复核日期：2026-05-25

远端最新 commit 需在可联网环境通过 `git -C FireRedASR-AED/upstream ls-remote origin main` 复核；本次未变更 upstream clone。

## 2. 当前目录文件分析

- `infer.py`：新增适配脚本，默认 `--device npu`，CPU 验证显式传 `--device cpu`；`torch_npu` 仅在选择 NPU 时条件导入。
- `patches/0001-add-npu-device-support.patch`：上游设备显式化 patch。
- `scripts/download_weights.sh`：默认通过 `huggingface_hub` + Gitee HF endpoint 下载 `fireredteam/FireRedASR-AED-L`。
- `scripts/download_test_data.sh`：复制上游官方 example wav/text/wav.scp。
- `requirements.txt`：历史整环境导出，不作为最小依赖。

## 3. 设备相关节点扫描结论

上游 `fireredasr/models/fireredasr.py` 存在硬编码 CUDA：

- `feats.cuda()` / `lengths.cuda()`
- `self.model.cuda()` / `self.model.cpu()`
- LLM 分支 `input_ids.cuda()` / `attention_mask.cuda()`

上游 `fireredasr/speech2text.py` 只有 `--use_gpu`，无法显式选择 CPU/CUDA/NPU。

## 4. 修改范围

- `fireredasr/models/fireredasr.py`：新增 `_resolve_device(args)`，使用 `.to(device)`，`--device npu` 时条件导入 `torch_npu`。
- `fireredasr/speech2text.py`：新增 `--device` 参数并传入 `transcribe()`。
- `infer.py` 等当前适配新增文件不进入 patch。

## 5. 当前验证状态

- `git apply --check`：通过。
- `python3 -m py_compile FireRedASR-AED/infer.py`：通过。
- 测试数据准备：通过。
- CPU 当前环境验证：已尝试，因缺少 `kaldiio` 等依赖且权重未下载而阻塞。
- NPU 验证：当前环境无 NPU，待目标机器执行。

## 6. 风险与限制

- 特征提取主要在 CPU，端到端耗时会受 I/O 和 fbank 影响。
- 本目录重点验证 AED；LLM 分支 device 显式化也覆盖了输入张量，但未做完整 LLM 权重验证。
- AED 官方建议音频不超过 60 秒。

## 7. 上游更新处理

上游更新时必须先检查相关文件：

```bash
git -C FireRedASR-AED/upstream fetch origin main
git -C FireRedASR-AED/upstream diff HEAD origin/main -- fireredasr/models/fireredasr.py fireredasr/speech2text.py
git -C FireRedASR-AED/upstream apply --check ../patches/0001-add-npu-device-support.patch
```

如 `.cuda()` 节点或 CLI 参数发生变化，重新生成 patch。
