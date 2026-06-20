# speechscorer 验收计划

## 0. 版本边界

- speechscorer：`main@bbe0be772b37f472994d5a97f809214fd67a2c8e`
- NPU patch：`patches/0001-add-explicit-device-selection.patch`
- 主验收路径：`whisper-clm`
- checkpoint：`openai/whisper-base.en@911407f4214e0e1d82085af863093ec0b66f9cd6`
- 不用 HuBERT/WavLM 或 multilingual `whisper-base` 的结果替代该主路径。

## 1. 原始测试集和指标

upstream 官方 demo 使用 [SpeechOcean762](https://github.com/jimbozhang/speechocean762) 的 `test` split：

- 语料总计 5,000 条英语句子；train/test 各 2,500 条，主验收使用 test 2,500 条；
- 音频：16 kHz，母语为普通话的儿童和成人英语学习者朗读语音；
- 人工标签：`accuracy`、`completeness`、`fluency`、`prosodic` 和 `total`；
- upstream scorer 输出：utterance 级 entropy 和 perplexity；
- 默认模型：`whisper-clm` / `openai/whisper-base.en`，greedy generation，`num_beams=1`、`do_sample=False`、`max_length=80`；
- 音频预处理由固定 checkpoint 的 `AutoProcessor` 完成；该无监督评分路径不做文本
  normalizer，后处理只对每步词表概率计算平均 entropy，再取 `exp(entropy)` 得到
  perplexity；
- upstream 展示 entropy 与人工 `total` 的散点图。

来源：

- <https://github.com/yaya-sy/speechscorer>
- <https://github.com/jimbozhang/speechocean762>

speechscorer upstream 未发布固定 checkpoint revision 下的 Pearson/Spearman 数值、
通过阈值或完整结果 CSV，因此必须明确写“官方数值指标未发布”。VCC2018
不是原始评测集，不能用于宣称复现官方评分能力。

## 2. 迁移对齐

固定：

- speechscorer commit 和 patch；
- Whisper checkpoint 文件及 SHA256；
- SpeechOcean762 `test/wav.scp` 顺序和音频；
- batch size、padding、max length、Transformers/PyTorch 版本。

分别以 `--device cpu`（资源允许时优先 CUDA）和 `--device npu` 生成 CSV。按 `utterance_id` 比较：

- 样本集合完全一致；
- entropy 最大绝对误差 `<= 1e-4`、平均绝对误差 `<= 1e-5`；
- perplexity 相对误差 `<= 1e-4`；
- CPU/CUDA 与 NPU entropy 排序 Spearman `>= 0.9999`。

然后将预测与 `test/all-info.json` 的人工 `total` 合并，分别报告 Pearson/Spearman。NPU 与 CPU/CUDA 的相关系数差绝对值应 `<= 0.001`。该相关性是当前复现实测值，不得标注为 upstream 官方发布值。

## 3. 分层验收

| 层级 | 数据 | 结论范围 |
|---|---|---|
| L0 | 1 条 16 kHz 英语 WAV | 链路 smoke，不验评分质量 |
| L1 | SpeechOcean762 test 固定 100 条 | CPU/NPU 数值和排序对齐 |
| L2 | SpeechOcean762 test 全量 | 迁移精度、人工分数相关性、性能 |
| L3 | 多 scorer、多 checkpoint 独立评测 | 仅在分别固定版本和指标后执行 |

## 4. 性能与稳定性

记录模型加载时间、总音频时长、推理 wall time、RTF、batch 1/4/8/16 吞吐和峰值 HBM。L2 连续三次结果必须满足数值阈值；30 轮 L1 无崩溃或持续内存增长。

## 5. 当前验收状态

- 已通过：patch 在固定 upstream commit 上 `git apply --check`；应用后全
  `speechscorer` 包 `compileall`；设备回退模式静态扫描。
- 未执行：Whisper 权重下载、SpeechOcean762 2,500 条推理、CPU/CUDA 与 NPU
  数值/相关性/性能验收。
- 当前结论：源码适配静态门禁通过；迁移精度未验收。

## 6. 报告模板

```text
源码/patch/checkpoint SHA:
环境、NPU、CANN、torch/torch-npu/transformers:
数据 split、样本数、manifest SHA:
decode/padding/batch:
CPU/CUDA vs NPU entropy/perplexity误差:
CPU/CUDA vs NPU排序Spearman:
各自与人工total的Pearson/Spearman:
性能和稳定性:
结论/未完成项:
```
