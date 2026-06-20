# Patch 说明

`0001-add-explicit-device-selection.patch` 基于 upstream commit `bbe0be772b37f472994d5a97f809214fd67a2c8e` 生成。它将原有 `--use-gpu` 自动选择改为显式 `--device npu/cpu/cuda`，默认 NPU，并仅在 NPU 路径导入 `torch_npu`。

Patch SHA256：

```text
39a5c12c51b4b490b93697446c713963e12fb58e6d30f9e33d919fb464fbfefb
```

应用与检查：

```bash
git -C upstream apply --check ../patches/0001-add-explicit-device-selection.patch
git -C upstream apply ../patches/0001-add-explicit-device-selection.patch
```
