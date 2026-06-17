# MOSS-TTSD-v0.5 验收计划

本文件只保留适配迁移验收的重点：**先确认原始模型使用什么测试集、原始指标是多少，再用同一权重、同一输入、同一参数对齐原始模型结果**。性能、稳定性和扩展场景只作为补充，不能冲淡主线。

## 1. 验收核心结论先行

| 问题 | 当前结论 | 验收要求 |
|---|---|---|
| 原始模型官方测试集是什么？ | `OpenMOSS/MOSS-TTSD` v0.5 README 只提供 `examples/examples.jsonl` 作为本地推理示例，包含中文、英文各 1 条双说话人 dialogue；未在 v0.5 README 中发布正式 test split。 | L0 必跑官方 `examples/examples.jsonl`；若后续找到论文/模型卡指定 test set，优先替换为官方 test set。 |
| 原始模型官方指标是多少？ | v0.5 README 未发布 MOSS-TTSD 生成质量的正式 WER/CER、SIM、MOS、CMOS 或 DNSMOS 表；只给出 GPU 显存估算。 | 不编造“官方指标”。正式验收报告必须明确填写“官方指标：未发布”或引用后续官方来源。 |
| NPU 适配主要验收目标是什么？ | 证明 patch 后 NPU 推理结果与原始模型 CPU/CUDA 源路径一致或无明显退化。 | 同 checkpoint、同 JSONL、同 prompt audio、同 seed、同 dtype/attention 可解释配置下，对比原始 CPU/CUDA 与 NPU。 |
| 通过标准是什么？ | 不是“能生成 wav 就通过”。 | 官方示例必须跑通；固定对齐集上可懂度、音色、说话人切换、时长和人工听感相对原始 CPU/CUDA 无系统性退化。 |

> 记住：当前是 NPU 适配迁移，不是重新定义模型能力。验收文档和报告的第一优先级永远是“原始测试集 + 原始指标 + NPU 对齐原始结果”。

## 2. 原始模型基准盘点

### 2.1 版本边界

- 源码：`OpenMOSS/MOSS-TTSD` tag `v0.5` / commit `0e078c62389922d3aa873ce182daf31142860b18`。
- 模型：`fnlp/MOSS-TTSD-v0.5` 或同内容别名 `OpenMOSS-Team/MOSS-TTSD-v0.5`。
- Codec：原项目 `XY_Tokenizer` + `fnlp/XY_Tokenizer_TTSD_V0` 的 `xy_tokenizer.ckpt`。
- 适配 patch：`patches/0001-adapt-v0.5-inference-to-npu.patch`。
- 不包含：MOSS-TTSD v0.7、v1.0、SGLang 路径、服务化压测、重新训练或微调。

正式验收报告必须记录模型权重、codec checkpoint、patch SHA256；未记录 SHA256 时不能宣称可复现验收。

### 2.2 原始测试集与指标

| 类别 | 原始来源 | 当前状态 | 用途 |
|---|---|---|---|
| 官方示例集 | `upstream/examples/examples.jsonl` | 2 条：中文双说话人、英文双说话人；prompt audio 位于 `upstream/examples/`。 | L0 smoke，证明原始入口和 NPU 入口能跑通。 |
| 官方正式 test set | v0.5 README / 本仓库记录 | 未发布。 | 暂无；若后续官方发布，必须补入此表。 |
| 官方正式质量指标 | v0.5 README / 本仓库记录 | 未发布 MOSS-TTSD-v0.5 生成质量指标。 | 不得编造；报告中写“官方未发布”。 |
| 原始模型对齐基线 | 同权重、同 JSONL 在 CPU 或 CUDA 源路径运行结果 | 需验收时现场生成并归档。 | NPU 迁移的主要对照组。 |

补充说明：官方博客/技术资料中的 codec 指标或其他版本指标，不能直接当作 MOSS-TTSD-v0.5 生成模型的验收通过线；除非明确证明同版本、同测试集、同评测脚本。

## 3. 最小验收分层

| 层级 | 数据 | 目的 | 是否必跑 | 通过关注点 |
|---|---|---|---|---|
| L0 官方示例 | `examples/examples.jsonl` 全量 2 条 | 对齐原始示例入口 | 必跑 | CPU/CUDA 与 NPU 都能完成；输出 wav 可读、非空、无 NaN/Inf。 |
| L1 固定对齐集 | 10-30 条固定 JSONL，覆盖中/英、短/长文本、prompt 切换、说话人切换 | 迁移质量回归 | 正式交付必跑 | NPU 相对 CPU/CUDA 无明显可懂度、音色、切换错误退化。 |
| L2 官方/公开测试集 | 后续若官方发布正式 test set，则使用官方数据和脚本 | 对齐原始公开指标 | 有官方指标时必跑 | 指标接近官方结果；同时给出 CPU/CUDA vs NPU 差异。 |
| L3 扩展评测 | 内部业务集、人工 MOS/CMOS、长音频稳定性 | 上线风险评估 | 可选 | 只作为补充，不替代原始模型对齐。 |

## 4. NPU 对齐评测方法

### 4.1 固定输入与参数

所有对比必须固定：

- 同一 `examples/examples.jsonl` 或同一固定评测 JSONL；
- 同一 prompt audio 与 prompt text；
- 同一模型权重和 codec checkpoint；
- 同一 `--seed 42`；
- 同一 `--use_normalize` 设置；
- attention backend、dtype 和设备差异必须记录清楚。

NPU 推荐配置：`--device npu --dtype bfloat16 --attn_implementation sdpa`。若 `sdpa` 不可用，可显式改为 `eager` 并在报告中说明，不允许静默回退。

### 4.2 原始 CPU/CUDA 基线命令

```bash
cd MOSS-TTSD-v0.5/upstream
python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_cpu_baseline \
  --device cpu \
  --dtype float32 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

如有 CUDA 环境，可再跑 CUDA 作为更接近原始上游 GPU 路径的参考；没有 CUDA 时，CPU 基线必须保留。

### 4.3 NPU 命令

```bash
cd MOSS-TTSD-v0.5/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

## 5. 主要指标与通过线

由于 v0.5 未发布正式质量指标，以下指标用于 **NPU vs 原始 CPU/CUDA 对齐**，不是对外宣称的官方 SOTA 指标。

| 维度 | 指标 | 通过线 |
|---|---|---|
| 基础可用 | wav 可读、采样率/声道可记录、时长 > 0、非全零、无 NaN/Inf | CPU/CUDA 与 NPU 全部通过。 |
| 可懂度 | 固定 ASR 回识别 CER/WER | NPU 相对 CPU/CUDA 无系统性退化；建议绝对差 ≤ 1.0 或相对差 ≤ 10%，二者取宽。 |
| 文本覆盖 | 生成语音是否覆盖输入 dialogue 主要内容 | 人工抽检不得出现批量漏读、重复、提前结束。 |
| 说话人切换 | `[S1]` / `[S2]` turn-taking 正确率 | 固定集人工标注；NPU 错误数不得高于 CPU/CUDA。 |
| 音色保持 | speaker embedding cosine similarity 或人工 A/B | NPU 不低于 CPU/CUDA 明显水平；若用自动模型，需固定模型版本。 |
| 自然度 | MOS/CMOS/A-B 或 DNSMOS/UTMOS 辅助 | 人工听感不出现系统性噪声、断裂、机械感退化；自动指标只作辅助。 |
| 性能 | elapsed、RTF、峰值 HBM/RSS | 记录即可；除非用户另定性能目标，否则不作为质量对齐的替代条件。 |

严格失败原则：官方评测组件或选定 ASR/speaker embedding 模型不可用时，应直接报告失败/缺失，不要用正则、简化指标、第三方近似包或 CPU fallback 冒充正式结果。

## 6. 验收报告必须包含

正式报告建议保存到 `MOSS-TTSD-v0.5/validation_reports/YYYYMMDD_<device>.md`，至少包含：

```text
模型：MOSS-TTSD-v0.5
源码：OpenMOSS/MOSS-TTSD tag v0.5 / 0e078c62389922d3aa873ce182daf31142860b18
patch：patch 文件路径 + SHA256
模型权重：来源、revision、SHA256
codec：来源、revision、SHA256

原始测试集：
- 官方正式 test set：未发布 / 或填写官方名称、split、样本数
- 当前 L0：upstream/examples/examples.jsonl，2 条，中文/英文双说话人
- 当前 L1：固定 JSONL 路径、样本数、语言、总时长

原始官方指标：
- 未发布 / 或填写官方指标表和来源

对齐基线：
- CPU/CUDA 命令、日志、输出目录
- NPU 命令、日志、输出目录

质量指标：
- CER/WER：ASR 模型、normalizer、CPU/CUDA、NPU、差异
- 说话人切换：标注规则、CPU/CUDA 错误数、NPU 错误数
- 音色相似度：模型版本、CPU/CUDA、NPU、差异
- 人工听测：人数、样本数、MOS/CMOS/A-B、异常样例

性能记录：
- elapsed、RTF、dtype、attention backend、峰值 HBM/RSS

结论：
- 通过 / 不通过
- 若不通过，列出与原始 CPU/CUDA 相比退化的样例和指标
```

## 7. 最终准入标准

只有同时满足以下条件，才可说 MOSS-TTSD-v0.5 NPU 适配验收通过：

1. 已明确记录原始测试集状态：v0.5 官方正式 test set/质量指标未发布，当前以官方 `examples/examples.jsonl` + 固定对齐集验收；
2. 同一权重、同一输入、同一 seed 下，CPU/CUDA 基线和 NPU 输出均已归档；
3. 官方示例 2 条全部通过基础可用检查；
4. L1 固定对齐集上，NPU 相对 CPU/CUDA 的可懂度、说话人切换、音色和人工听感无系统性退化；
5. 所有依赖缺失、评测脚本缺失或官方字段缺失均直接暴露，不使用静默 fallback；
6. 性能、环境、命令、日志、权重 SHA256 和输出文件可复现。
