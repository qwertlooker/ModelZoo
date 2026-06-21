# Patch 说明

`0001-add-explicit-device-selection.patch` 基于 upstream commit `bbe0be772b37f472994d5a97f809214fd67a2c8e` 生成。它将原有 `--use-gpu` 自动选择改为显式 `--device npu/cpu/cuda`，默认 NPU，仅在 NPU 路径导入 `torch_npu`，并增加 `--output_csv` 以隔离不同设备结果。

Patch SHA256：

```text
f2712ef70afee2176c6a34c0ca41383ef20233bfa3f96a24794f4d9e4c6e3ef1
```

应用与检查：

```bash
git -C upstream apply --check ../patches/0001-add-explicit-device-selection.patch
git -C upstream apply ../patches/0001-add-explicit-device-selection.patch
```
