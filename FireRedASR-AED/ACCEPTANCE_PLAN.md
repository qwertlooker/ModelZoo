# FireRedASR-AED 完整验收方案

## 0. 验收目标与范围

本方案用于验收 FireRedASR-AED-L 在昇腾 NPU 上的适配结果。验收不只看官方单条样例是否跑通，而是对齐原始 FireRedASR-AED 的中文普通话、中文方言和英文 ASR 功能、性能与精度。

**当前适配边界**

- 源码：`FireRedTeam/FireRedASR`，当前记录 commit `834635e4cf277ed8ca92049fc375b17c3dc20748`。
- 权重：`FireRedTeam/FireRedASR-AED-L`，目录 `pretrained_models/FireRedASR-AED-L/`。
- 不包含：`FireRedASR-LLM-L`、`FireRedASR2S` / `FireRedASR2-AED` / VAD / LID / Punc / TensorRT 变体。
- 输入限制：官方 README 说明 AED 建议音频不超过 60 秒，超过 200 秒会触发位置编码错误。

## 1. 原始模型能力与公开参考指标

### 1.1 功能能力

| 能力 | 原始模型声明 | 本适配验收要求 |
|---|---|---|
| 中文普通话 ASR | AISHELL-1/2、WenetSpeech 等 | 至少普通话样例与公开测试子集 |
| 中文方言 ASR | KeSpeech | 至少方言样例；正式验收跑 KeSpeech 或内部方言集 |
| 英文 ASR | LibriSpeech test-clean/test-other | 至少英文样例；正式验收跑 LibriSpeech 子集 |
| 单条推理 | `--wav_path` | 必测 |
| 批量推理 | `--wav_scp`、`--batch_size` | 必测 batch 1/2/4，记录最大可用 batch |
| 解码参数 | beam、length penalty、softmax smoothing | 精度验收使用原始默认或官方推荐参数 |

### 1.2 公开精度参考

原始 README 中 FireRedASR-AED 的公开结果：

| 数据集 | 语言/类型 | 指标 | FireRedASR-AED |
|---|---|---|---:|
| AISHELL-1 | 普通话 | CER% | 0.55 |
| AISHELL-2 | 普通话 | CER% | 2.52 |
| WenetSpeech net | 普通话 | CER% | 4.88 |
| WenetSpeech meeting | 会议普通话 | CER% | 4.76 |
| Average-4 | 普通话平均 | CER% | 3.18 |
| KeSpeech | 中文方言 | CER% | 4.48 |
| LibriSpeech test-clean | 英文 | WER% | 1.93 |
| LibriSpeech test-other | 英文 | WER% | 4.44 |

说明：公开结果依赖原始权重、解码参数、文本归一化和评测脚本。NPU 适配正式验收首先要求同数据同脚本下 NPU 相对 CPU/CUDA 不退化；完整复现公开值需使用对应公开数据全量和一致 normalizer。

## 2. 数据集选择：大小、获取难度与用途

| 数据 | 规模/获取难度 | 覆盖 | 用途 | 建议 |
|---|---|---|---|---|
| 官方 `examples/wav` | 1~少量样例，已随 upstream 提供，极易 | 单条中文 ASR | L0 smoke | 必跑；不代表精度 |
| 自制小样本集 | 20~50 条，5~20 分钟；需参考文本 | 中文/英文/方言功能回归 | L1 | 每次交付必跑 |
| AISHELL-1 test | 公开、下载容易，约数小时 | 普通话 CER | L2 推荐 | 优先作为中文正式精度集 |
| AISHELL-2 test | 数据更大，获取中等 | 普通话 CER | L2/L3 | 资源允许时跑 |
| WenetSpeech test_net/test_meeting | 体量较大，下载/预处理较复杂 | 普通话/会议 | L3 | 发布级复现 |
| KeSpeech | 方言，获取和许可需确认 | 方言 CER | L2/L3 | 有数据授权时跑 |
| LibriSpeech test-clean/test-other | 每个约 5 小时，获取容易 | 英文 WER | L2 推荐 | 英文正式验收优先 |
| 内部业务集 | 视场景而定 | 真实场景鲁棒性 | 上线验收 | 需固定版本和标注规范 |

## 3. 分层验收

| 层级 | 数据量 | 必测内容 | 结论用途 |
|---|---:|---|---|
| L0 smoke | 1 条官方 wav | 权重加载、fbank、AED decode、NPU device | 只证明链路可运行 |
| L1 功能回归 | 20~50 条 | 中文/英文/方言、单条/批量、batch、60 秒边界 | 判断功能完整性 |
| L2 推荐验收 | AISHELL-1 test 或子集 + LibriSpeech 子集 | CER/WER、性能、NPU vs CPU/CUDA | 正式交付验收 |
| L3 完整复现 | AISHELL-1/2、WenetSpeech、KeSpeech、LibriSpeech 全量 | 对齐公开表格 | 发布级报告 |

## 4. 功能验收矩阵

| 用例 | 输入 | 参数 | 预期 |
|---|---|---|---|
| 单条中文 | 官方 `BAC009S0764W0121.wav` | `--device npu` | 输出 `uttid/text/wav/rtf` |
| 批量中文 | `wav.scp` 多条 | `--batch_size 2/4` | 输出条数正确，无 device mismatch |
| 英文样例 | LibriSpeech 或内部英文 wav | 默认 AED decode | 英文文本输出，不崩溃 |
| 方言样例 | KeSpeech 或内部方言 wav | 默认 AED decode | 中文文本输出，不崩溃 |
| 60 秒边界 | 55~60 秒音频 | 单条或 batch=1 | 可完成；>60 秒需记录官方限制 |
| 异常输入 | 非 16 kHz、空文件、超长音频 | 单条 | 错误清晰或文档要求先转 16 kHz mono |

## 5. 精度验收

### 5.1 指标与归一化

| 数据 | 指标 | 建议归一化 |
|---|---|---|
| 中文普通话/方言 | CER | 使用上游 `fireredasr/utils/wer.py`，明确 `--do_tn`、`--rm_special` 设置 |
| 英文 | WER | 统一大小写、标点处理；可使用上游脚本或 `jiwer`，需固定规则 |
| 回归一致性 | CER/WER 差异、文本完全一致率 | 同数据同参数 CPU/CUDA vs NPU |

### 5.2 推荐通过条件

- 同 checkpoint、同数据、同解码参数下，NPU CER/WER 相对 CPU/CUDA 绝对差异 ≤ 0.1 个百分点；
- 单条/小样本回归文本完全一致率建议 ≥ 99%，若存在浮点 tie-break 差异需给出样例和 CER/WER 差异；
- 若对公开值，AISHELL-1 / LibriSpeech 子集应接近公开指标；全量复现允许因数据版本、文本归一化和解码参数差异给出解释；
- 不允许出现系统性漏字、重复、乱码或语言错误。

## 6. 性能验收

| 指标 | 记录方式 | 要求 |
|---|---|---|
| 加载时间 | 从 `from_pretrained` 到可推理 | 记录 |
| 单条延迟 | 官方样例、10 秒、30 秒、60 秒 | 记录 P50/P90 或逐条耗时 |
| RTF/RTFx | `rtf` 输出或 wall time / 音频时长 | NPU 正式验收建议 RTFx > 1 |
| batch 吞吐 | batch 1/2/4/8 | 记录最大可用 batch 和 OOM 点 |
| 峰值 HBM/RSS | `npu-smi`、`/usr/bin/time -v` | 记录 |
| 稳定性 | 连续 30~100 轮或 1 小时音频 | 无崩溃、显存无持续增长 |

注意：FireRedASR 特征提取和 I/O 会影响端到端性能；报告中需说明是否包含音频读取、fbank、解码和首次编译/加载耗时。

## 7. 正式验收最低清单

- [ ] 权重目录完整，记录 `cmvn.ark/config.yaml/dict.txt/model.pth.tar/train_bpe1000.model` 来源和 SHA256 或 metadata。
- [ ] L0 官方样例 NPU smoke test 通过。
- [ ] 批量 `wav.scp` 推理通过，至少 batch=2。
- [ ] AISHELL-1 test 或不少于 30 分钟普通话有标注数据计算 CER。
- [ ] LibriSpeech test-clean 子集或不少于 30 分钟英文有标注数据计算 WER。
- [ ] 有条件时补充 KeSpeech 或内部方言集 CER。
- [ ] 同数据 CPU/CUDA vs NPU CER/WER 差异满足第 5.2 节。
- [ ] 性能记录 batch=1/2/4、RTF/RTFx、加载时间、峰值 HBM/RSS。
- [ ] 连续 30 轮稳定性测试通过。

## 8. 报告模板

```markdown
# FireRedASR-AED NPU 验收报告

## 环境
- 日期：
- NPU 型号/数量：
- CANN/驱动：
- Python/torch/torch-npu：
- upstream commit：
- 权重来源/SHA256：

## 功能
| 用例 | 数据 | batch | 结果 | 备注 |
|---|---|---:|---|---|

## 精度
| 数据集 | 语言 | 条数/时长 | 指标 | CPU/CUDA | NPU | 差异 | 是否通过 |
|---|---|---:|---|---:|---:|---:|---|

## 性能
| batch | 音频总时长 | wall time | RTF/RTFx | 峰值 HBM | 是否通过 |
|---:|---:|---:|---:|---:|---|

## 结论
- 通过/不通过：
- 阻塞项：
- 风险：
```

## 9. 参考来源

- FireRedASR 官方仓库：<https://github.com/FireRedTeam/FireRedASR>
- FireRedASR-AED-L 权重：<https://huggingface.co/fireredteam/FireRedASR-AED-L>
- FireRedASR 技术报告：<https://arxiv.org/pdf/2501.14350>
- AISHELL-1：<https://www.openslr.org/33/>
- LibriSpeech：<https://www.openslr.org/12/>
- WenetSpeech：<https://wenet.org.cn/WenetSpeech/>
