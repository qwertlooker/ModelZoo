# MOSS-TTSD-v0.5 完整验收方案

## 0. 验收目标与范围

本方案用于验收当前目录中 MOSS-TTSD-v0.5 在昇腾 NPU 上的适配结果。验收范围覆盖模型加载、双说话人 prompt、中文/英文对话生成、长上下文、性能和生成质量。

**模型边界**

- 适配对象：`OpenMOSS-Team/MOSS-TTSD-v0.5` / `fnlp/MOSS-TTSD-v0.5`。
- 辅助 codec：`OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf` / `fnlp/XY_Tokenizer_TTSD_V0_hf`。
- 不包含：MOSS-TTSD v0.7、v1.0、SGLang 服务化路径、SiliconFlow API、未固定版本的一键整合包。
- 官方能力边界：中文/英文双语、双说话人语音克隆、长对话生成，v0.5 模型卡声明单会话最长约 960 秒。

## 1. 验收分层

| 层级 | 目的 | 数据规模 | 必跑条件 | 结论用途 |
|---|---|---:|---|---|
| L0 smoke | 验证加载、设备迁移、生成、WAV 保存 | 1 条 JSONL，短文本 | 每次改动必跑 | 只证明链路可运行 |
| L1 功能回归 | 覆盖中文/英文、单/双 prompt、batch、normalize、长文本 | 10~30 条，约 5~20 分钟目标音频 | 每次交付必跑 | 判断功能完整性 |
| L2 推荐性能/质量 | 与 CPU/CUDA 源路径同 checkpoint 对齐 | 50~200 条，约 1~3 小时目标音频 | 正式验收必跑 | 判断 NPU 适配可接受性 |
| L3 完整复现/发布 | 官方/公开评测与人工听测 | 500+ 条，多场景，多听测人 | 有资源时跑 | 发布级报告 |

## 2. 功能能力与用例矩阵

| 能力 | 用例 | L1 最低要求 | 通过条件 |
|---|---|---:|---|
| 中文对话 | `[S1]` / `[S2]` 中文短句 | 2 条 | 可生成可播放音频，语种正确 |
| 英文对话 | 英文短句 | 2 条 | 可生成英文音频 |
| 中英混合 | 同一 JSONL 中中英文切换 | 2 条 | 不报错，输出语言跟随文本 |
| 双说话人 | `prompt_audio_speaker1/2` + `prompt_text_speaker1/2` | 4 条 | 说话人切换可听辨或相似度指标可区分 |
| 共享 prompt | `prompt_audio` + `prompt_text` | 2 条 | 官方 schema 可运行 |
| batch | `--batch_size 2/4` | 至少 4 条 | 输出数量与输入一致；OOM 需记录最大可用 batch |
| 文本归一化 | `--text_normalize` 开/关 | 各 2 条 | 开关均可运行；差异记录在报告 |
| 长文本 | 3~10 分钟目标音频 | 1~2 条 | 无截断、长静音、重复循环或设备错误 |

## 3. 数据集与获取难度

| 数据 | 覆盖 | 规模/难度 | 用途 |
|---|---|---|---|
| `prepare_test_data.py` 合成样本 | schema / 链路 | 极小，容易 | L0，仅冒烟 |
| 官方/上游示例 prompt | 真实 prompt 音频 | 小，需从 upstream/模型卡确认许可 | L1 功能 |
| 自建双语对话集 | 中文/英文/混合、多说话人 | 中，需人工整理 prompt 文本与音频 | L1/L2 |
| AISHELL-3 / CSMSC prompt 改造 | 中文音色相似度、ASR-CER | 中，公开数据可得 | L2/L3 |
| LibriTTS / VCTK prompt 改造 | 英文音色相似度、ASR-WER | 中，公开数据可得 | L2/L3 |
| TTSD-eval 或官方评测集 | 对话生成公开口径 | 中-高，需按官方脚本准备 | L3 |
| 人工 MOS/CMOS/A/B | 主观自然度和偏好 | 高，需听测平台 | L2/L3 |

## 4. 性能验收

### 4.1 指标

必须记录：

- `elapsed_seconds`；
- `generated_audio_seconds`；
- `RTF = elapsed_seconds / generated_audio_seconds`；
- `RTFx = generated_audio_seconds / elapsed_seconds`；
- 首条输出延迟；
- batch size、dtype、attention backend；
- NPU 峰值 HBM、CPU RSS；
- 首次编译/加载耗时与稳定推理耗时分开记录。

### 4.2 通过条件

- L1：所有功能用例退出码为 0，结构检查通过。
- L2：同 checkpoint、同 JSONL、同解码参数下，NPU 相对 CPU/CUDA 源路径不出现功能退化；RTF 和 HBM 满足业务目标，若没有业务目标，至少给出与 CPU/CUDA 的倍速/资源对比。
- L3：在公开/官方评测脚本下提供完整性能表和日志。

## 5. 质量与精度验收

TTS/TTSD 不能只看波形逐点一致。建议组合指标：

| 维度 | 指标 | 工具/说明 | 通过建议 |
|---|---|---|---|
| 可懂度 | ASR 回识别 CER/WER | 中文可用 Paraformer/Whisper，英文可用 Whisper；固定模型和 normalizer | NPU 与 CPU/CUDA 差异不显著 |
| 音色 | speaker embedding cosine / EER | 使用固定说话人模型，如 CAM++/Wespeaker 等 | NPU 不低于 CPU/CUDA 基线明显阈值 |
| 自然度 | DNSMOS / UTMOS | 作为客观参考，不替代人工听测 | NPU 相对基线无系统性下降 |
| 对话一致性 | 说话人切换准确率、串音率 | 人工或自动 VAD+speaker verification | 错误率不高于基线 |
| 主观 | MOS/CMOS/A-B preference | 至少 10~20 名听测人，随机盲测 | CMOS 不显著劣于基线 |

如使用官方 TTSD-eval，必须记录其 repo/commit、模型版本、指标配置和数据版本；不得用自制简化指标冒充官方指标。

## 6. L0/L1/L2/L3 命令模板

### L0

```bash
python MOSS-TTSD-v0.5/prepare_test_data.py --output_dir MOSS-TTSD-v0.5/test_data
ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-TTSD-v0.5/infer.py \
  --model_path MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5 \
  --codec_path MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf \
  --input_jsonl MOSS-TTSD-v0.5/test_data/smoke.jsonl \
  --output_dir MOSS-TTSD-v0.5/outputs_l0 \
  --device npu --dtype bfloat16 --attn_implementation sdpa --batch_size 1 --local_files_only
python MOSS-TTSD-v0.5/validate_outputs.py --manifest MOSS-TTSD-v0.5/outputs_l0/manifest.jsonl
```

### L1

准备 `MOSS-TTSD-v0.5/eval_data/l1_function.jsonl`，包含上表功能矩阵。运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-TTSD-v0.5/infer.py \
  --model_path MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5 \
  --codec_path MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf \
  --input_jsonl MOSS-TTSD-v0.5/eval_data/l1_function.jsonl \
  --output_dir MOSS-TTSD-v0.5/eval_results/l1_npu \
  --device npu --dtype bfloat16 --attn_implementation sdpa --batch_size 2 --text_normalize --local_files_only
```

### L2

同一份 JSONL 分别跑 CPU/CUDA 源路径和 NPU：

```bash
# CPU baseline
python MOSS-TTSD-v0.5/infer.py \
  --model_path MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5 \
  --codec_path MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf \
  --input_jsonl MOSS-TTSD-v0.5/eval_data/l2_quality.jsonl \
  --output_dir MOSS-TTSD-v0.5/eval_results/l2_cpu \
  --device cpu --dtype float32 --attn_implementation sdpa --batch_size 1 --local_files_only

# NPU
ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-TTSD-v0.5/infer.py \
  --model_path MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5 \
  --codec_path MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf \
  --input_jsonl MOSS-TTSD-v0.5/eval_data/l2_quality.jsonl \
  --output_dir MOSS-TTSD-v0.5/eval_results/l2_npu \
  --device npu --dtype bfloat16 --attn_implementation sdpa --batch_size 1 --local_files_only
```

随后使用固定 ASR、speaker embedding、DNSMOS/UTMOS 与人工听测流程生成报告。

## 7. 报告模板

```text
层级：L0/L1/L2/L3
模型/codec revision：
权重 SHA256：
数据集名称/版本/样本数：
硬件与软件环境：
命令：
输出目录：
结构检查：通过/失败
性能：elapsed / generated_seconds / RTF / RTFx / HBM
可懂度：CER/WER
音色：speaker similarity
自然度：DNSMOS/UTMOS/MOS
人工听测：MOS/CMOS/A-B
相对 CPU/CUDA 结论：
问题列表与日志：
最终结论：通过/不通过/需复测
```
