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
python3 infer.py --manifest eval_data/vcc2018.jsonl --model_root weights \
  --device cpu --output_csv results/cpu.csv
python3 infer.py --manifest eval_data/vcc2018.jsonl \
  --model_root weights --device npu --output_csv results/npu.csv
python3 compare_results.py --baseline results/cpu.csv \
  --candidate results/npu.csv --output results/cpu_vs_npu.json
```

逐文件比较 `SIG_raw/BAK_raw/OVRL_raw`、校正后 `SIG/BAK/OVRL` 和 `P808_MOS`：

- 最大绝对误差 `<= 1e-4`；
- 平均绝对误差 `<= 1e-5`；
- 排序 Spearman `>= 0.9999`；
- 文件数、窗口数和输入时长完全一致。

若 CANN/ONNX Runtime 的算子精度导致阈值不满足，应报告真实差异，不得改用 CPU provider 生成 NPU 结果。
上述数值是初始工程门禁；首次获得不少于 20 条多样音频的原始 CPU/CANN 重复运行
分布后必须校准并记录依据，不能把暂定阈值描述为官方容差。

## 3. 功能验证与 L2

| 层级 | 数据 | 目标 |
|---|---|---|
| 功能验证 | 1 条短于 9.01 秒 WAV，并补单/双声道、16 kHz/其他采样率、长短音频 | 两个 ONNX 会话、CSV、重采样和失败用例 |
| L2 | 优先使用可取得的原始公开 benchmark 全量；否则固定 VCC2018 或内部带人工 MOS 集，至少 1 小时 | CPU/NPU 逐字段精度对齐、相关性和 RTF |

功能样例只能证明链路可运行，不能作为 L2 精度结论。

## 4. 功能矩阵

| 维度 | 必测值 |
|---|---|
| 模型 | 常规、personalized |
| 输入时长 | `<9.01s`、`=9.01s`、长音频 |
| 采样率 | 16 kHz、非 16 kHz |
| 声道 | mono、stereo |
| 输入方式 | 单文件、目录、固定 JSONL manifest |
| 异常 | 空 WAV、非 WAV、缺模型、NPU provider 缺失 |

## 5. L2 性能验证

CPU 和 NPU 使用同一 L2 manifest，分别保留 `*.csv.meta.json`。该 sidecar 已记录
音频总时长、elapsed 和 RTF。常规与 personalized 各执行至少 1 次冷启动和 3 次
稳定运行，报告中位数；同时用 `/usr/bin/time -v` 记录峰值 RSS，NPU 侧补峰值 HBM。

官方未发布可直接对齐的硬件性能值，因此不编造 speedup 通过线。最低性能结论为：
全量 L2 无 OOM/失败、输出数完整、RTF 可复现，并报告 NPU/CPU RTF 比值；若项目另有
目标 RTF，再按目标判定通过或不通过。

## 6. 最低正式验收清单

- [ ] 三个 ONNX SHA256 与固定版本一致。
- [ ] CPU/NPU 分属独立环境；版本分别为 `onnxruntime==1.22.1` 和
  `onnxruntime-cann==1.22.1`，NPU 实际 session 首 provider 为 CANN。
- [ ] 同一 manifest 完成 CPU 和 NPU 常规模型推理。
- [ ] 同一 manifest 完成 CPU 和 NPU personalized 推理。
- [ ] `compare_results.py` 两组比较均通过。
- [ ] 覆盖功能矩阵中的时长、采样率、声道及失败用例。
- [ ] L2 常规/personalized 的 CPU/NPU RTF、峰值 RSS/HBM 和相对比值已归档。
- [ ] 报告明确区分论文隐藏集指标与非官方 VCC2018 回归。

## 7. 当前验收状态

- 已通过：固定权重 CPU 路径与官方 `dnsmos_local.py` 在同一 30 秒音频上，
  常规/personalized 全字段最大和平均绝对误差均为 `0.0`。
- 未执行：CANN provider NPU 功能验证和 L2 精度/性能对齐。
- 当前结论：CPU 算法等价性通过；NPU 迁移验收未完成。

## 8. 报告模板

```text
日期/硬件/CANN/ONNX Runtime:
三个权重 SHA256:
数据集、split、manifest SHA256、样本数、时长:
personalized:
CPU provider / NPU provider:
最大/平均绝对误差:
Spearman:
CPU/NPU elapsed、RTF、峰值RSS/HBM、RTF比值:
结论与未完成项:
```

## 补充说明（来自 README_INFERENCE.md）

### VCC2018 数据集定位

VCC2018 只能作为非官方迁移回归集，不能冒充论文隐藏测试集。论文测试音频及
P.835 主观标签没有公开，因此不得用 VCC2018 或随机音频冒充官方相关性复现。

### 性能报告方法学

论文未发布可直接作为当前 NPU 通过线的硬件性能数值。正式报告至少记录样本数、
总音频时长、首次和稳定运行耗时、RTF、NPU 型号及 CANN/ONNX Runtime 版本。

### L2 性能测量方法

对 L2 同一 manifest 分别在两个环境执行，并保留独立资源日志：

```bash
mkdir -p results
/usr/bin/time -v -o results/cpu.time.txt python infer.py \
  --manifest eval_data/vcc2018.jsonl --model_root weights \
  --device cpu --output_csv results/cpu_perf.csv
/usr/bin/time -v -o results/npu.time.txt python infer.py \
  --manifest eval_data/vcc2018.jsonl --model_root weights \
  --device npu --output_csv results/npu_perf.csv
```

`*.csv.meta.json` 提供 elapsed/RTF/provider，`*.time.txt` 提供峰值 RSS；NPU 峰值 HBM
另由现场监控记录。常规和 personalized 均执行，正式轮次至少重复 3 次。
