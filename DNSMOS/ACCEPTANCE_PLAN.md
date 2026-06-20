# DNSMOS 验收计划

## 0. 版本边界

- Microsoft DNS-Challenge：`master@591184a9fcb2cbdec02520fed81a32bbbf9d73ff`
- Ascend-SACT 参考：`d1e4c2c14df9cb935d61dc5f448e655772b12379`
- 权重：同一 commit 下的 `model_v8.onnx`、常规和 personalized
  `sig_bak_ovr.onnx`
- 不包含在线 DNSMOS API 或其他后续 DNSMOS 变体。

## 1. 原始测试集与官方指标

DNSMOS P.835 论文记录：

- 训练数据来自 DNS Challenge V3 和 Microsoft 内部降噪模型，共约 30,000 条带
  P.835 MOS 标签的音频，平均 9 秒、约 75 小时、约 40 个降噪模型；
- 原始 DNS Challenge V3 test set 为 600 条 noisy clips，经过约 40 个降噪模型处理；
- unseen real test set 为 850 条音频、17 个未见 Microsoft 内部降噪模型，覆盖
  emotional、英语、非英语（含/不含声调语言）、stationary noise 等类别；
- 论文没有公开该 unseen test set 的可下载 manifest、音频或 P.835 标签；
- 指标按两种粒度计算：每个降噪模型聚合后的 PCC/SRCC，以及 clip-level PCC/SRCC；
- 输出为 `SIG`、`BAK`、`OVRL`；本地脚本另输出 `P808_MOS`；
- 当前官方本地脚本口径为 16 kHz、9.01 秒窗口、1 秒 hop、窗口平均和多项式校正。

论文 Table 3 的 unseen test set 结果：

| 粒度/指标 | SIG | BAK | OVRL |
|---|---:|---:|---:|
| Model PCC | 0.94 | 0.98 | 0.98 |
| Model SRCC | 0.95 | 0.99 | 0.98 |
| Clip PCC | 0.71 | 0.83 | 0.82 |
| Clip SRCC | 0.72 | 0.82 | 0.81 |

来源：

- <https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS>
- <https://arxiv.org/abs/2110.01763>

论文测试音频及 P.835 主观标签没有公开，因此不得用 VCC2018 或随机音频冒充
官方相关性复现。VCC2018 只能作为独立、非官方的迁移回归集。

## 2. 迁移对齐主线

固定同一组三个 ONNX 文件、同一 WAV manifest、同一 `infer.py` 和参数，分别运行：

```bash
python3 infer.py --audio <manifest_paths...> --model_root weights \
  --device cpu --output_csv cpu.csv
ASCEND_RT_VISIBLE_DEVICES=0 python3 infer.py --audio <manifest_paths...> \
  --model_root weights --device npu --output_csv npu.csv
```

逐文件比较 `SIG_raw/BAK_raw/OVRL_raw`、校正后 `SIG/BAK/OVRL` 和 `P808_MOS`：

- 最大绝对误差 `<= 1e-4`；
- 平均绝对误差 `<= 1e-5`；
- 排序 Spearman `>= 0.9999`；
- 文件数、窗口数和输入时长完全一致。

若 CANN/ONNX Runtime 的算子精度导致阈值不满足，应报告真实差异，不得改用 CPU provider 生成 NPU 结果。

## 3. 分层验收

| 层级 | 数据 | 目标 |
|---|---|---|
| L0 | 1 条短于 9.01 秒 WAV | 短音频重复、两个 ONNX 会话、CSV 链路 |
| L1 | 至少 20 条，覆盖单/双声道、16 kHz/其他采样率、长短音频 | CPU/NPU 逐字段对齐 |
| L2 | 固定 VCC2018 子集或内部带人工 MOS 集，至少 1 小时 | 数值对齐、相关性、RTF、稳定性 |
| L3 | 获得论文原始测试音频和 P.835 标签后 | 按论文口径复现 Pearson/Spearman |

dummy 和 L0 只能证明链路可运行，不能作为精度结论。

## 4. 性能与稳定性

记录 CANN、驱动、ONNX Runtime、NPU 型号、ONNX SHA256、样本数、音频时长、首次/稳定运行耗时和 RTF。L2 连续运行 30 次，结果应稳定在数值阈值内且无内存持续增长。

## 5. 当前验收状态

- 已通过：固定权重 CPU 路径与官方 `dnsmos_local.py` 在同一 30 秒音频上，
  常规/personalized 全字段最大和平均绝对误差均为 `0.0`。
- 未执行：CANN provider NPU 推理、L1 多样音频矩阵、L2/L3 数据集相关性、
  NPU 性能和稳定性。
- 当前结论：CPU 算法等价性通过；NPU 迁移验收未完成。

## 6. 报告模板

```text
日期/硬件/CANN/ONNX Runtime:
三个权重 SHA256:
数据集、split、manifest SHA256、样本数、时长:
personalized:
CPU provider / NPU provider:
最大/平均绝对误差:
Spearman:
CPU/NPU elapsed、RTF:
30 轮稳定性:
结论与未完成项:
```
