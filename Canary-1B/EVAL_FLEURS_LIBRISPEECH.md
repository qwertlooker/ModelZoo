# Canary-1B LibriSpeech / FLEURS 验证测试方案

本文给出针对当前 NPU 适配的 L2 精度验证方案：

- **ASR**：LibriSpeech `test-clean` 子集或全量。
- **AST**：FLEURS `test` 子集或全量，覆盖 `En→De/Es/Fr` 与 `De/Es/Fr→En`。

配套脚本：`Canary-1B/scripts/eval_canary.py`。

## 1. 前置依赖

在已有 Canary-1B NeMo 推理环境中补充：

```bash
pip install datasets soundfile librosa tqdm jiwer sacrebleu openai-whisper
```

说明：

- `datasets` 用于下载 LibriSpeech/FLEURS。
- `soundfile`/`librosa` 用于写出 16 kHz wav。
- `jiwer` 用于 ASR WER。
- `sacrebleu` 用于 AST BLEU。
- `openai-whisper` 提供英文 WER normalizer；若没有安装，脚本会退化为 lower/punctuation/space 简单归一化。

如果国内网络访问 Hugging Face 较慢，可按环境实际情况设置镜像，例如：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/hf_cache
```

## 2. ASR：LibriSpeech test-clean

### 2.1 最小 30 分钟子集

CPU/CUDA 基线和 NPU 需要使用同一命令参数，仅修改 `--device` 和可见设备。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --task asr \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --librispeech_minutes 30 \
  --librispeech_limit 0 \
  --asr_pnc no \
  --batch_size 1 \
  --beam_size 5 \
  --data_dir Canary-1B/eval_data \
  --output_dir Canary-1B/eval_results/npu_asr_librispeech_30min
```

### 2.2 test-clean 全量

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --task asr \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --librispeech_minutes 0 \
  --asr_pnc no \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_asr_librispeech_full
```

### 2.3 通过条件

- 主通过条件：同一数据、同一脚本、同一 `beam_size=5` 下，**NPU WER 相对 CPU/CUDA 不劣化**。
- 若直接对公开值，使用较宽阈值：`WER <= 公开值 + max(公开值 * 10%, 0.5)`。
- Canary-1B 在 Open ASR Leaderboard 的 LibriSpeech clean 参考值为 `1.48 WER`；公开值仅作参考，正式结论优先看同脚本 CPU/CUDA 对比。

## 3. AST：FLEURS En↔De/Es/Fr

### 3.1 每方向 50 条最小子集

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --task ast \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes \
  --batch_size 1 \
  --beam_size 5 \
  --data_dir Canary-1B/eval_data \
  --output_dir Canary-1B/eval_results/npu_ast_fleurs_50
```

### 3.2 FLEURS test 全量

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --task ast \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --fleurs_limit 0 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_ast_fleurs_full
```

### 3.3 通过条件

- 主通过条件：同一数据、同一脚本、同一 `beam_size=5` 下，**NPU BLEU 相对 CPU/CUDA 下降 ≤ 0.5**。
- 若直接对公开值，使用较宽阈值：`BLEU >= 公开值 - max(公开值 * 10%, 1.0)`。
- FLEURS 公开参考值：

| 方向 | BLEU |
|---|---:|
| En→De | 32.15 |
| En→Es | 22.66 |
| En→Fr | 40.76 |
| De→En | 33.98 |
| Es→En | 21.80 |
| Fr→En | 30.95 |

## 4. CPU/CUDA/NPU 对比流程

建议按以下顺序跑，保证数据与 manifest 固定：

1. 先用 NPU 或 CPU 执行一次 `--prepare_only` 生成 manifest 和 wav：

```bash
python Canary-1B/scripts/eval_canary.py \
  --task all \
  --model Canary-1B/weights/canary-1b.nemo \
  --device cpu \
  --librispeech_minutes 30 \
  --fleurs_limit 50 \
  --prepare_only
```

2. 分别跑 CPU/CUDA/NPU，输出到不同目录。
3. 对比各目录中的 `*.metrics.json`：
   - ASR 看 `wer_percent`。
   - AST 看 `bleu`。
   - 性能看 `elapsed_seconds`、`rtf`。

## 5. 输出文件

每个任务/方向会生成：

- `<tag>.tsv`：逐样本 `sample_id / audio_path / duration / reference / hypothesis`。
- `<tag>.metrics.json`：该任务指标。
- `summary.metrics.json`：全部任务指标汇总。
- `run_env.json`：Python、torch、NeMo、设备和命令行参数记录。

## 6. 注意事项

- 精度评测建议使用 `--beam_size 5`，与 Canary-1B 公开精度配置更接近。
- NPU 初次运行可能包含算子编译/缓存开销，性能统计建议单独预热后再跑正式轮次。
- `batch_size=1` 最稳；如要测吞吐，可逐步尝试 2/4/8，并记录 OOM 或 shape/device 异常。
- FLEURS 通过样本 `id` 对齐源语言音频和目标语言参考文本；脚本使用目标语言 `raw_transcription` 作为 AST BLEU reference。
- LibriSpeech ASR 使用 `pnc=no` 更适合 WER，对 hypothesis/reference 做英文归一化后计算。
