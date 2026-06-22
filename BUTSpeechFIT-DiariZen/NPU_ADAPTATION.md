# DiariZen NPU 适配文档

## 1. 版本边界

- DiariZen：`a60b18151dbbe246e4199d8ef5cd2ece3872ea94`
- vendored pyannote-audio：上述 commit 中 tree `b35947db4d409329452e51abc68b70b5f50c1324`
- dscore submodule：`e02f949ac6592279300a2c33d03daf9e0c12fd27`
- 主模型：`diarizen-wavlm-large-s80-md` commit `a9b1b0e7974d96dcfd63af417e9da7ad8714040f`
- embedding：`wespeaker-voxceleb-resnet34-LM` commit `837717ddb9ff5507820346191109dc79c958d614`
- Ascend-SACT 参考：`7961b5ab79b1232b9da367f14f8cd4f592694465`
- 检查日期：2026-06-20。
- 目标 NPU 组合：Python 3.10、CANN 8.2.0、PyTorch/torchaudio/torch-npu
  2.5.1、`onnxruntime-cann==1.22.1`；CPU baseline 使用独立环境中的
  `onnxruntime==1.22.1`。

## 2. 适配分析

upstream `DiariZenPipeline` 将设备写为 `cuda:0`（CUDA 可用时）否则 CPU，不能显式选择 NPU。更隐蔽的问题是 pyannote 的 ONNX WeSpeaker wrapper 只识别 CPU/CUDA，其他设备会告警后回退 CPU。

当前 patch：

1. pipeline 构造器和 `from_pretrained` 接收显式 device；
2. ModelZoo `infer.py` 默认 `--device npu`，仅 NPU 路径导入 `torch_npu`；
3. ONNX WeSpeaker 在 NPU 上显式使用 `CANNExecutionProvider`，provider 缺失时由 ONNX Runtime 直接失败；
4. Kaldi fbank 是 CPU 预处理，并将 numpy 特征送入 CANN ONNX session。这消除了参考实现要求修改 site-packages 中 `torch.fft.rfft(...).abs()` 的不可复现操作；
5. CPU/CUDA 原路径不变。

分割网络运行在 PyTorch NPU，speaker embedding 模型运行在 ONNX Runtime CANN。CPU fbank 只是前处理，不是模型推理回退。

`infer.py` 现在支持固定 JSONL manifest，运行时读取
`pipeline._embedding.session_.get_providers()` 并在 NPU 路径强制首 provider 为
`CANNExecutionProvider`，同时写出 `run.meta.json`。`prepare_eval_data.py`
固定 wav/RTTM/UEM，`score_diarization.py` 封装固定 dscore 参数，避免误写
`--ignore_overlaps false`。

## 3. 验证事实与限制

2026-06-20 已完成 upstream/reference/model revision 取证、CUDA/NPU 节点扫描、patch 静态检查和脚本语法检查。

当前主机没有 PyTorch、torch-npu、ONNX Runtime CANN、权重或 NPU，未执行：

- example RTTM CPU/NPU 对齐；
- 官方数据 DER；
- L2 性能：三组 RTF、峰值 RSS/HBM 和相对比值。

这些状态必须保留为“待验收”，不能使用参考 README 的运行描述代替本次实测。

当前状态是 **S1：源码适配和验收工具链已形成；升级到 S2/S3 仍缺模型功能 RTTM
实测及 L2 CPU/CUDA/NPU DER、RTF 和资源对齐**。

独立重放使用 `upstream-original`、`upstream-npu` 和 NPU 三组结果；原始与 patch
后的 editable 安装不能位于同一环境。

新增工具已做轻量 fixture 验证：

- `prepare_eval_data.py` 成功读取 30 秒上游样例、wav.scp、reference RTTM 和 UEM，
  生成 manifest/meta；
- `score_diarization.py` 使用固定 dscore commit、`collar=0`、保留 overlap，
  对相同 reference/system RTTM 得到 DER/JER `0.00`；
- 该测试证明此前错误的 `--ignore_overlaps false` 已被消除，但不包含模型推理。

安装和推理见 [README_INFERENCE.md](README_INFERENCE.md)，DER 数据口径和
CPU/CUDA/NPU 对齐标准见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

## 补充说明（来自 README_INFERENCE.md）

### 设备边界

分割网络运行在 PyTorch NPU；WeSpeaker embedding 运行在 ONNX Runtime
`CANNExecutionProvider`；Kaldi fbank 明确保留为 CPU 前处理。

### 目录结构

执行时创建 `source/`、未应用 patch 的 `upstream-original/` 和应用 patch 的
`upstream-npu/` 三个目录。

### 环境隔离要求

- 原始和 patch 后 CPU baseline 使用独立环境。不要安装 upstream 根
  `requirements.txt` 中的 `onnxruntime-gpu`。
- NPU 环境不得安装 CPU 索引 wheel。
- 不要修改 site-packages。执行 NPU 导入门禁。
- CPU/CUDA baseline 使用上述独立环境和 CPU/CUDA ONNX Runtime，不能与
  `onnxruntime-cann` 混装。
