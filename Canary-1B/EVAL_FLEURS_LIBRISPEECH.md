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
- 如 Hugging Face 访问慢，可设置 `HF_ENDPOINT` / `HF_HOME`；但评测数据推荐使用下面的显式本地目录参数，便于离线迁移。

## 2. 准备数据

> 当前 `prepare_eval_data.py` 已支持在线/离线混合模式：FLEURS 使用 `--fleurs_parquet_dir` 指定 parquet 保存目录，LibriSpeech 使用 `--librispeech_dir` 指定 OpenSLR tar/解压目录；目标文件已存在时直接复用，缺失时在线下载到该目录，`--offline` 下缺失则直接报具体路径且不联网。
>
> FLEURS 不再依赖 `torchcodec` 自动解码：脚本将 HF `Audio` 列 cast 为 `decode=False`，再用 `soundfile` 读取 bytes/path 写 16 kHz wav。

### 2.0 推荐目录结构

```text
Canary-1B/eval_data/fleurs_parquet/
  en_us/test-00000-of-00001.parquet
  de_de/test-00000-of-00001.parquet
  es_419/test-00000-of-00001.parquet
  fr_fr/test-00000-of-00001.parquet
Canary-1B/eval_data/librispeech_raw/
  test-clean.tar.gz
  LibriSpeech/test-clean/
```

FLEURS 日志应看到 `loading local FLEURS parquet: ...` 或 `downloading FLEURS parquet to ...`。LibriSpeech 日志应看到 `using existing LibriSpeech directory/archive`、`downloading LibriSpeech test-clean to ...` 或 `extracting LibriSpeech archive ...`。

### 2.1 最小验收数据：LibriSpeech 30 分钟 + FLEURS 每方向 50 条

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task all \
  --data_dir Canary-1B/eval_data \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --librispeech_minutes 30 \
  --asr_pnc no \
  --fleurs_split test \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes
```

生成 manifest；同时生成同名 `.meta.json`，其中 `split` 应为 `test`：

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
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --librispeech_minutes 0 \
  --asr_pnc no
```

### 2.3 只准备 AST FLEURS test 全量

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task ast \
  --data_dir Canary-1B/eval_data \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --fleurs_split test \
  --fleurs_limit 0 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes
```

### 2.4 离线复用本地数据

当上述目录已经由在线脚本或手动命令准备好后，离线环境使用同一命令加 `--offline`：

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task all \
  --data_dir Canary-1B/eval_data \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --offline \
  --librispeech_minutes 30 \
  --fleurs_split test \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en
```

离线模式不会访问 Hugging Face/OpenSLR；缺失文件会直接报类似：

```text
Offline mode enabled and FLEURS parquet is missing: .../en_us/test-00000-of-00001.parquet
Offline mode enabled and LibriSpeech data is missing: .../LibriSpeech/test-clean or .../test-clean.tar.gz
```

### 2.5 手动命令行下载到脚本指定目录

FLEURS 四种语言 test parquet：

```bash
mkdir -p Canary-1B/eval_data/fleurs_parquet/{en_us,de_de,es_419,fr_fr}

curl -L -o Canary-1B/eval_data/fleurs_parquet/en_us/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/en_us/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/fleurs_parquet/de_de/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/de_de/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/fleurs_parquet/es_419/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/es_419/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/fleurs_parquet/fr_fr/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/fr_fr/test-00000-of-00001.parquet
```

LibriSpeech test-clean：

```bash
mkdir -p Canary-1B/eval_data/librispeech_raw
curl -L -o Canary-1B/eval_data/librispeech_raw/test-clean.tar.gz \
  https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf Canary-1B/eval_data/librispeech_raw/test-clean.tar.gz \
  -C Canary-1B/eval_data/librispeech_raw
```

手动下载后再运行第 2.4 节 `--offline` 命令，脚本会直接复用本地文件，不重复下载。

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
- `*.jsonl.meta.json`：数据准备元信息，包含 dataset/config/split/limit、本地数据目录和 `offline`，便于确认 FLEURS 使用的是 `test` split 且复用同一批本地文件。
