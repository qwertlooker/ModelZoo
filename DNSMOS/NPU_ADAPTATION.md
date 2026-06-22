# DNSMOS NPU 适配文档

## 1. 版本与来源

- 官方源码：<https://github.com/microsoft/DNS-Challenge.git>
- 分支/commit：`master` / `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`
- Ascend-SACT 参考：<https://gitcode.com/Ascend-SACT/DNSMOS>，commit `d1e4c2c14df9cb935d61dc5f448e655772b12379`
- 检查日期：2026-06-20；两个远端 HEAD 与上述本地 commit 一致。
- 权重：官方仓 `DNSMOS/DNSMOS/model_v8.onnx`、`DNSMOS/DNSMOS/sig_bak_ovr.onnx`、`DNSMOS/pDNSMOS/sig_bak_ovr.onnx`。
- 变体边界：包含常规和 personalized DNSMOS P.835；不包含在线 DNSMOS API。
- 目标 NPU 运行组合：Python 3.10、CANN 8.2.0、
  `onnxruntime-cann==1.22.1`；CPU baseline 使用独立环境中的
  `onnxruntime==1.22.1`。

## 2. 代码分析与适配

官方 `dnsmos_local.py` 使用 ONNX Runtime 默认 provider。当前 `infer.py` 只增加显式设备边界：

- `--device npu` 默认选择 `CANNExecutionProvider`，provider 不存在时立即失败；
- `--device cpu` 显式选择 `CPUExecutionProvider`；
- 不调用 `npu-smi`、不修改设备性能模式、不写死卡号；
- 保留官方 16 kHz 重采样、短音频重复、9.01 秒窗口、1 秒 hop、所有窗口平均、P.808 mel 特征和多项式校正；
- 支持递归目录输入，并输出 provider、耗时和 RTF。
- 支持固定 JSONL manifest；每次运行写出模型 SHA、provider、环境和命令
  sidecar metadata；
- `prepare_eval_data.py` 负责验证 WAV 并固定 manifest，`compare_results.py`
  负责 CPU/CUDA 与 NPU 的逐字段门禁。

Ascend-SACT 脚本只评估中间窗口，长音频结果与官方逐窗平均不等价，因此未直接照搬。当前修改没有改动官方已有源码，不需要上游 patch。

## 3. 验证记录

2026-06-20 已完成：

- 固定官方与参考实现 commit；
- 对照官方 `dnsmos_local.py` 审查预处理、窗口、模型输入和校正公式；
- `python3 -m py_compile DNSMOS/infer.py` 通过；
- `prepare_eval_data.py` 和 `compare_results.py` 已加入正式交付入口；
- 权重文件在官方浅克隆中存在且不是 Git LFS 指针。

CPU 实测环境和结果：

- Python 3.12.3、ONNX Runtime 1.27.0 CPU、librosa 0.11.0、
  NumPy 2.4.6、soundfile 0.13.1；
- 输入：DiariZen `example/EN2002a_30s.wav`，30 秒，SHA256
  `55aa90540de2a01e6824ee2862d08763026d1752a3d7f6f8870015c74424e900`；
- 常规和 personalized 两条路径均使用 21 个窗口；
- 分别与同 commit 官方 `dnsmos_local.py` 比较
  `len_in_sec/sr/num_hops`、全部 raw/校正后 MOS 和 `P808_MOS`，
  最大/平均绝对误差均为 `0.0`；
- 当前脚本首次常规运行 `elapsed=9.834s, RTF=0.327785`；warm-up 后两次运行
  `elapsed=1.880~2.026s, RTF=0.062662~0.067521`。单条数据只用于 CPU
  链路和算法等价性，不作为性能结论。

补充闭环工具验证：

- `prepare_eval_data.py` 对同一 30 秒样例生成 1 条 manifest，记录采样率、声道、
  音频 SHA 和 manifest SHA；
- `infer.py --manifest ... --device cpu` 成功生成 CSV 和 `*.meta.json`，
  实际 provider 为 `CPUExecutionProvider`，本次 `RTF=0.076430`；
- `compare_results.py` 对该 CSV 自比较，七个字段最大/平均误差均为 `0.0`，
  单样本 Spearman 按完全相同处理为 `1.0`。

系统没有 `npu-smi`/CANN NPU 环境，因此未执行 NPU 数值或性能验收。正式验收必须按
[ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md) 生成 CPU/NPU CSV 并逐字段比较。
用户环境、权重和运行命令见 [README.md](README.md)。

当前交付状态：**S2，CPU 算法等价性已验证；升级到 S3 仍缺 NPU 同 manifest
精度和性能对齐**。不能仅凭
静态检查或本文档将状态标记为“适配验收完成”。

## 补充说明（来自 README.md）

### ONNX Runtime 与 CANN 版本配套

ONNX Runtime 官方 CANN EP 配套表将 1.22.1 对应到 CANN 8.2.0。CPU 和 NPU
环境必须分开创建，避免 CPU `onnxruntime` 覆盖 CANN 构建。

### 内部构建 CANN EP 的替换要求

如果目标基础镜像使用内部构建的 CANN EP，应以对应 wheel 替换上述 PyPI 包，并在
验收报告中记录 wheel 文件名和 SHA256；不得同时安装 `onnxruntime`。
