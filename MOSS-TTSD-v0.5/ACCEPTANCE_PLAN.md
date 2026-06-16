# MOSS-TTSD-v0.5 完整验收方案

## 0. 验收目标与范围

本方案用于验收 `OpenMOSS/MOSS-TTSD` tag `v0.5` 原项目代码在应用 `patches/0001-adapt-v0.5-inference-to-npu.patch` 后的昇腾 NPU 推理结果。

**模型边界**

- 源码：`OpenMOSS/MOSS-TTSD` tag `v0.5` / commit `0e078c62389922d3aa873ce182daf31142860b18`。
- 模型：`fnlp/MOSS-TTSD-v0.5`（<https://huggingface.co/fnlp/MOSS-TTSD-v0.5>）或同内容别名 `OpenMOSS-Team/MOSS-TTSD-v0.5`；本次记录 HEAD `8527b9136b6afefe2252ae597cecea2e80e7ebeb`。
- Codec：原项目 `XY_Tokenizer` + `fnlp/XY_Tokenizer_TTSD_V0` 的 `xy_tokenizer.ckpt`（<https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0>）；本次记录 HEAD `c83433728e698ed0698e88cb5096bc221fb8f8c5`。
- 不包含：MOSS-TTSD v0.7、v1.0、SGLang 路径、未固定版本的一键包改动。

## 0.1 权重下载命令

在 `MOSS-TTSD-v0.5/upstream/` 下执行：

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p weights/MOSS-TTSD-v0.5 XY_Tokenizer/weights

hf download fnlp/MOSS-TTSD-v0.5 \
  --revision 8527b9136b6afefe2252ae597cecea2e80e7ebeb \
  --local-dir weights/MOSS-TTSD-v0.5

hf download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt \
  --revision c83433728e698ed0698e88cb5096bc221fb8f8c5 \
  --local-dir XY_Tokenizer/weights
```

codec checkpoint 固定 URL：<https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/resolve/c83433728e698ed0698e88cb5096bc221fb8f8c5/xy_tokenizer.ckpt>。下载后必须把模型权重和 `xy_tokenizer.ckpt` 的 SHA256 写入验收报告。

## 1. 验收分层

| 层级 | 目的 | 数据规模 | 必跑条件 | 结论用途 |
|---|---|---:|---|---|
| L0 smoke | 验证 patch 后原项目 `inference.py` 可加载模型并输出 WAV | 1-2 条官方 examples | 每次改动必跑 | 只证明链路可运行 |
| L1 功能回归 | 覆盖中文/英文、双说话人、normalize、不同 prompt | 10-30 条 | 每次交付必跑 | 判断功能完整性 |
| L2 推荐性能/质量 | 与 CPU/CUDA 源路径同 checkpoint 对齐 | 50-200 条 | 正式验收必跑 | 判断 NPU 适配可接受性 |
| L3 完整复现/发布 | 官方/公开评测与人工听测 | 500+ 条，多场景 | 有资源时跑 | 发布级报告 |

## 2. 功能矩阵

| 能力 | L1 最低要求 | 通过条件 |
|---|---:|---|
| 官方 examples | 2 条 | 输出 WAV 数量正确且可播放 |
| 中文对话 | 2 条 | 输出语言与文本一致 |
| 英文对话 | 2 条 | 输出语言与文本一致 |
| 中英混合 | 2 条 | 不报错，语言切换可听辨 |
| 双说话人 | 4 条 | 说话人切换可听辨或相似度指标可区分 |
| `--use_normalize` | 开启默认路径 | 不报错，归一化结果可解释 |
| CPU/NPU 对比 | 同 JSONL | NPU 无设备错误和明显质量退化 |

## 3. 数据建议

| 数据 | 覆盖 | 难度 | 用途 |
|---|---|---|---|
| 原项目 `examples/examples.jsonl` | 基础链路 | 低 | L0 |
| 自建双语对话 JSONL | 语言/说话人/文本长度 | 中 | L1/L2 |
| AISHELL-3 / CSMSC prompt 改造 | 中文可懂度与音色 | 中 | L2/L3 |
| LibriTTS / VCTK prompt 改造 | 英文可懂度与音色 | 中 | L2/L3 |
| 人工 MOS/CMOS/A-B | 主观自然度 | 高 | L2/L3 |

## 4. 性能验收

必须记录：

- 样本数、输出 WAV 总时长；
- 端到端 elapsed seconds；
- `RTF = elapsed_seconds / generated_audio_seconds`；
- `RTFx = generated_audio_seconds / elapsed_seconds`；
- batch 策略、dtype、attention backend；
- NPU 峰值 HBM、CPU RSS；
- 首次加载/编译耗时与稳定推理耗时。

通过条件：同 checkpoint、同 JSONL、同参数下，NPU 相对 CPU/CUDA 源路径没有功能退化；性能满足业务目标或给出清晰瓶颈说明。

## 5. 质量验收

建议组合指标：

| 维度 | 指标 | 说明 |
|---|---|---|
| 可懂度 | ASR 回识别 CER/WER | 固定 ASR 模型和 normalizer |
| 音色 | speaker embedding cosine / EER | 固定 speaker 模型 |
| 自然度 | DNSMOS / UTMOS | 客观参考，不替代人工听测 |
| 对话一致性 | 说话人切换准确率、串音率 | 人工或自动 VAD+speaker verification |
| 主观 | MOS/CMOS/A-B preference | 盲测，记录人数和样本 |

不得用结构性 WAV 检查替代官方或人工质量验收。

## 6. 命令模板

L0 NPU：

```bash
cd MOSS-TTSD-v0.5/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_l0_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

L2 CPU/NPU 对比：复用同一 JSONL，分别指定 `--device cpu --dtype float32` 和 `--device npu --dtype bfloat16`，输出到不同目录后做性能和质量对比。

## 7. 报告模板

```text
层级：L0/L1/L2/L3
源码 tag/commit：
patch：
模型/codec 来源与 SHA256：
数据集名称/版本/样本数：
硬件与软件环境：
命令：
输出目录：
输出 WAV 数量/总时长：
性能：elapsed / RTF / RTFx / HBM
可懂度：CER/WER
音色：speaker similarity
自然度：DNSMOS/UTMOS/MOS
人工听测：MOS/CMOS/A-B
相对 CPU/CUDA 结论：
问题列表与日志：
最终结论：通过/不通过/需复测
```
