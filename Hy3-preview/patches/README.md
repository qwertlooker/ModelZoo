# Patch 说明

`0001-add-hy3-preview-support.patch` 原始来源为 Ascend-SACT commit `eb533c1dfd9a1fa7f373f9b980a9c0f973f1dad8` 中的 `hy3-delivery.patch`，SHA256：

```text
2e59facbbb4428c83f97e974f931e2dadeda418a073248bf3d2744038ea71735
```

目标版本：

```text
vLLM tag=v0.18.0rc1
vLLM commit=262ddd0d81a1e4687e209f988d6ea32616e736fa
vllm-ascend tag=v0.18.0rc1
vllm-ascend commit=99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f
```

2026-06-20 已执行：

```bash
git -C /tmp/vllm-v018-check apply --check \
  Hy3-preview/patches/0001-add-hy3-preview-support.patch
```

检查通过。补丁修改 vLLM 11 个文件，增加 HyV3/HYV3MTP 模型、配置、reasoning parser 和 tool parser；不修改 vllm-ascend。
