# Canary-1B LibriSpeech / FLEURS 验证测试方案

按要求将流程拆成两步：

1. **准备数据**：`scripts/prepare_eval_data.py` 只负责下载数据、转 16 kHz wav、写 JSONL manifest。
2. **评测**：`scripts/eval_canary.py` 只读取已准备好的 manifest，使用与 `infer.py` 相同的 NeMo `model.transcribe()` 机制做推理，再计算 WER/BLEU。

这样 CPU/CUDA/NPU 评测可以复用同一份 wav 和 manifest，避免每次评测重复下载或抽样不一致。

## 1. 前置依赖

```bash
pip install datasets soundfile librosa tqdm jiwer sacrebleu openai-whisper
```

- 数据准备需要：`datasets soundfile librosa tqdm`。
- 评测需要：`jiwer sacrebleu`，可选 `openai-whisper` 作为英文 WER normalizer。
- 如 Hugging Face 访问慢，可设置 `HF_ENDPOINT` / `HF_HOME`。

## 2. 准备数据

### 2.1 最小验收数据：LibriSpeech 30 分钟 + FLEURS 每方向 50 条

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task all \
  --data_dir Canary-1B/eval_data \
  --librispeech_minutes 30 \
  --asr_pnc no \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes
```

生成 manifest：

```text
Canary-1B/eval_data/librispeech_test_clean/manifest_asr_en.jsonl
Canary-1B/eval_data/fleurs/en-de/manifest_ast_en_de.jsonl
Canary-1B/eval_data/fleurs/en-es/manifest_ast_en_es.jsonl
Canary-1B/eval_data/fleurs/en-fr/manifest_ast_en_fr.jsonl
Canary-1B/eval_data/fleurs/de-en/manifest_ast_de_en.jsonl
Canary-1B/eval_data/fleurs/es-en/manifest_ast_es_en.jsonl
Canary-1B/eval_data/fleurs/fr-en/manifest_ast_fr_en.jsonl
```

### 2.2 只准备 ASR LibriSpeech test-clean 全量

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --librispeech_minutes 0 \
  --asr_pnc no
```

### 2.3 只准备 AST FLEURS test 全量

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task ast \
  --data_dir Canary-1B/eval_data \
  --fleurs_limit 0 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes
```

## 3. 评测

评测脚本默认读取第 2.1 节的标准 manifest 列表；也可以用 `--manifest` 显式指定一个或多个 manifest。

### 3.1 一次评测全部已准备任务

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_all
```

### 3.2 只评测 ASR

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --manifest Canary-1B/eval_data/librispeech_test_clean/manifest_asr_en.jsonl \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_asr_librispeech
```

### 3.3 只评测 FLEURS AST 六个方向

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --manifest \
    Canary-1B/eval_data/fleurs/en-de/manifest_ast_en_de.jsonl \
    Canary-1B/eval_data/fleurs/en-es/manifest_ast_en_es.jsonl \
    Canary-1B/eval_data/fleurs/en-fr/manifest_ast_en_fr.jsonl \
    Canary-1B/eval_data/fleurs/de-en/manifest_ast_de_en.jsonl \
    Canary-1B/eval_data/fleurs/es-en/manifest_ast_es_en.jsonl \
    Canary-1B/eval_data/fleurs/fr-en/manifest_ast_fr_en.jsonl \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_ast_fleurs
```

## 4. CPU/CUDA/NPU 对比

准备数据只跑一次。之后三种设备分别运行评测脚本，保持同一批 manifest：

```bash
# CPU 基线
python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device cpu \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/cpu_all

# NPU
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_all
```

对比：

- ASR：`*_metrics.json` 中的 `wer_percent`。
- AST：`*_metrics.json` 中的 `bleu`。
- 性能：`elapsed_seconds`、`rtf`。

## 5. 通过条件

### ASR：LibriSpeech test-clean

- 最小规模：30 分钟。
- 推荐规模：全量约 5 小时。
- 主通过条件：同一数据、同一脚本、同一 `beam_size=5` 下，NPU WER 相对 CPU/CUDA 不劣化。
- 若直接对公开值，`WER <= 公开值 + max(公开值 * 10%, 0.5)`。

### AST：FLEURS En↔De/Es/Fr

- 最小规模：每方向 50 条。
- 推荐规模：FLEURS test 全量。
- 主通过条件：同一数据、同一脚本、同一 `beam_size=5` 下，NPU BLEU 相对 CPU/CUDA 下降 ≤ 0.5。
- 若直接对公开值，`BLEU >= 公开值 - max(公开值 * 10%, 1.0)`。

FLEURS 公开参考：

| 方向 | BLEU |
|---|---:|
| En→De | 32.15 |
| En→Es | 22.66 |
| En→Fr | 40.76 |
| De→En | 33.98 |
| Es→En | 21.80 |
| Fr→En | 30.95 |

## 6. 输出文件

评测输出目录中包含：

- `<tag>.tsv`：逐样本 `sample_id / audio_path / duration / reference / hypothesis`。
- `<tag>.metrics.json`：单个 manifest 指标。
- `summary.metrics.json`：汇总指标。
- `run_env.json`：Python、torch、NeMo、设备和命令行参数记录。
