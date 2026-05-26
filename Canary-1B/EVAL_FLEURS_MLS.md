# Canary-1B MLS / LibriSpeech / FLEURS 验证测试方案

按要求将流程拆成两步：

1. **准备数据**：`scripts/prepare_eval_data.py` 只负责下载数据、转 16 kHz wav、写 JSONL manifest。
2. **评测**：`scripts/eval_canary.py` 只读取已准备好的 manifest，使用与 `infer.py` 相同的 NeMo `model.transcribe()` 机制做推理，再计算 WER/BLEU。

这样 CPU/CUDA/NPU 评测可以复用同一份 wav 和 manifest，避免每次评测重复下载或抽样不一致。

## 0. 官方参考指标

来源：

- NVIDIA Canary-1B model card：<https://huggingface.co/nvidia/canary-1b>
- Hugging Face Open ASR Leaderboard：<https://hf-audio-open-asr-leaderboard.hf.space/>
- Open ASR Leaderboard 代码/说明：<https://github.com/huggingface/open_asr_leaderboard>

### 0.1 官方精度数据

NVIDIA model card 说明 ASR/AST 公开结果使用 `beam width=5`、`length penalty=1.0`。ASR 使用 WER，并用 whisper-normalizer 归一化参考和预测文本；AST 使用 BLEU，并保留数据集原始标点和大小写。

| 任务 | 数据集 | 指标 | 官方参考 |
|---|---|---|---|
| ASR | MCV-16.1 test | WER | En 7.97 / De 4.61 / Es 3.99 / Fr 6.53 |
| ASR | MLS test | WER | En 3.06 / De 4.19 / Es 3.15 / Fr 4.12 |
| AST | FLEURS test | BLEU | En→De 32.15 / En→Es 22.66 / En→Fr 40.76 / De→En 33.98 / Es→En 21.80 / Fr→En 30.95 |
| AST | CoVoST-v2 test | BLEU | De→En 37.67 / Es→En 40.70 / Fr→En 40.42 |
| AST | mExpresso test | BLEU | En→De 23.84 / En→Es 35.74 / En→Fr 28.29 |

### 0.2 公开性能数据

原始 `nvidia/canary-1b` model card 没有单独发布硬件延迟/吞吐表。当前可引用的公开性能参考是 Hugging Face Open ASR Leaderboard 的 RTFx。该榜单说明开源模型评测在 NVIDIA A100-SXM4-80GB GPU、CUDA 12.6、PyTorch 2.4.0 下运行，batch size 尽量使用 64，显存不足时自适应降低。

截至 2026-05-26，`nvidia/canary-1b` 公开参考为：

| 指标 | 值 |
|---|---:|
| Average WER | 6.50 |
| RTFx | 235.34 |
| AMI WER | 13.90 |
| Earnings22 WER | 12.19 |
| GigaSpeech WER | 10.12 |
| LibriSpeech clean WER | 1.48 |
| LibriSpeech other WER | 2.93 |
| SPGISpeech WER | 2.06 |
| Tedlium WER | 3.56 |
| VoxPopuli WER | 5.79 |

上述 RTFx 只作为公开 GPU 量级参考，不是 NPU 通过线。本仓库评测输出的 `elapsed_seconds`、`rtf` 可换算 `RTFx = audio_seconds / elapsed_seconds`，并应和 `beam_size`、`batch_size`、设备、峰值内存一起记录。

## 1. 前置依赖

```bash
pip install datasets soundfile librosa tqdm jiwer sacrebleu openai-whisper
```

- 数据准备需要：`datasets soundfile librosa tqdm`。
- 评测需要：`jiwer sacrebleu openai-whisper`。ASR WER 固定走官方 Whisper `EnglishTextNormalizer` 路径（`from whisper.normalizers import EnglishTextNormalizer`）；依赖缺失或导入失败时会在脚本启动导入阶段直接抛出原始异常。仅安装 `whisper_normalizer` 不视为满足官方路径，且不使用本地 fallback normalizer。
- 如 Hugging Face 访问慢，可设置 `HF_ENDPOINT` / `HF_HOME`；但评测数据推荐使用下面的显式本地目录参数，便于离线迁移。

### 1.1 评测脚本 import / 依赖规范

项目级流程规范详见根目录《模型NPU 适配标准流程.md》的“项目级脚本严格失败原则”。`Canary-1B/scripts/eval_canary.py` 作为本模型评测入口必须遵守该项目级规范：

1. 除设备后端探测类 import（例如仅 `--device npu` 才需要的 `torch_npu`）外，评测依赖统一放在文件顶部导入，禁止在 metric 计算阶段临时 import 后再 fallback。
2. ASR WER 只能使用官方路径 `from whisper.normalizers import EnglishTextNormalizer`；不得改用 `whisper_normalizer` 包、regex/basic normalizer 或其他静默替代实现。
3. 任一必需依赖缺失时脚本应直接失败并暴露原始异常；不要用宽泛 `try/except` 包装成兼容路径，不要吞掉异常，不要继续推理后再给出不可对齐官方口径的指标。
4. 对 NeMo 配置、版本字段和解码配置使用当前官方预期字段；字段缺失表示环境或版本不匹配，应立即报错，不添加 `hasattr/getattr` 式静默兼容。

## 2. 准备数据

> 当前 `prepare_eval_data.py` 已支持在线/离线混合模式：ASR 精度使用 `--asr_parquet_dir` 指定 `facebook/multilingual_librispeech` parquet 保存目录，性能测试保留 LibriSpeech `test-clean`，使用 `--librispeech_dir` 指定 OpenSLR tar/解压目录，FLEURS 使用 `--fleurs_parquet_dir` 指定 parquet 保存目录；目标文件已存在时直接复用，缺失时在线下载到该目录，`--offline` 下缺失则直接报具体路径且不联网。
>
> MLS/LibriSpeech/FLEURS 都不再依赖 `torchcodec` 自动解码：脚本将 HF `Audio` 列 cast 为 `decode=False`，再用 `soundfile` 读取 bytes/path 写 16 kHz wav。

### 2.0 推荐目录结构

```text
Canary-1B/eval_data/mls_parquet/
  german/test-00000-of-00001.parquet
  spanish/test-00000-of-00001.parquet
  french/test-00000-of-00001.parquet
Canary-1B/eval_data/librispeech_raw/
  test-clean.tar.gz
  LibriSpeech/test-clean/
Canary-1B/eval_data/fleurs_parquet/
  en_us/test-00000-of-00001.parquet
  de_de/test-00000-of-00001.parquet
  es_419/test-00000-of-00001.parquet
  fr_fr/test-00000-of-00001.parquet
```

MLS 日志应看到 `loading local MLS parquet: ...` 或 `downloading MLS parquet to ...`；LibriSpeech 日志应看到 `using existing LibriSpeech directory/archive` 或 `downloading LibriSpeech test-clean to ...`；FLEURS 日志应看到 `loading local FLEURS parquet: ...` 或 `downloading FLEURS parquet to ...`。

### 2.1 最小验收数据：MLS 30 分钟 + LibriSpeech test-clean 30 分钟 + FLEURS 每方向 50 条

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task all \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --asr_split test \
  --asr_minutes 30 \
  --asr_pnc no \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --fleurs_split test \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes
```

生成 manifest；同时生成同名 `.meta.json`。MLS 的 `dataset` 应为 `facebook/multilingual_librispeech`，LibriSpeech 用于性能测试并记录 `purpose`：

```text
Canary-1B/eval_data/librispeech_test_clean/manifest_asr_en.jsonl
Canary-1B/eval_data/mls_test_german/manifest_asr_de.jsonl
Canary-1B/eval_data/mls_test_spanish/manifest_asr_es.jsonl
Canary-1B/eval_data/mls_test_french/manifest_asr_fr.jsonl
Canary-1B/eval_data/fleurs/en-de/manifest_ast_en_de.jsonl
Canary-1B/eval_data/fleurs/en-es/manifest_ast_en_es.jsonl
Canary-1B/eval_data/fleurs/en-fr/manifest_ast_en_fr.jsonl
Canary-1B/eval_data/fleurs/de-en/manifest_ast_de_en.jsonl
Canary-1B/eval_data/fleurs/es-en/manifest_ast_es_en.jsonl
Canary-1B/eval_data/fleurs/fr-en/manifest_ast_fr_en.jsonl
```

### 2.2 准备 ASR MLS test + LibriSpeech test-clean 全量

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --asr_split test \
  --asr_minutes 0 \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --asr_pnc no
```

### 2.3 只准备 MLS ASR test 全量（不含性能用 LibriSpeech）

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --asr_split test \
  --asr_minutes 0 \
  --no-include_librispeech_test_clean \
  --asr_pnc no
```

### 2.4 只准备性能测试用 LibriSpeech test-clean 全量

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --asr_configs "" \
  --include_librispeech_test_clean \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --librispeech_minutes 0 \
  --asr_pnc no
```

### 2.5 只准备 AST FLEURS test 全量

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

### 2.6 离线复用本地数据

当上述目录已经由在线脚本或手动命令准备好后，离线环境使用同一命令加 `--offline`：

```bash
python Canary-1B/scripts/prepare_eval_data.py \
  --task all \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --offline \
  --asr_configs german,spanish,french \
  --asr_minutes 30 \
  --fleurs_split test \
  --fleurs_limit 50 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en
```

离线模式不会访问 Hugging Face/OpenSLR；缺失文件会直接报类似：

```text
Offline mode enabled and MLS parquet is missing: .../german/test-00000-of-00001.parquet
Offline mode enabled and LibriSpeech data is missing: .../LibriSpeech/test-clean or .../test-clean.tar.gz
Offline mode enabled and FLEURS parquet is missing: .../en_us/test-00000-of-00001.parquet
```

### 2.7 手动命令行下载到脚本指定目录

MLS ASR 三种语言 test parquet：

```bash
mkdir -p Canary-1B/eval_data/mls_parquet/{german,spanish,french}

curl -L -o Canary-1B/eval_data/mls_parquet/german/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/german/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/mls_parquet/spanish/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/spanish/test-00000-of-00001.parquet
curl -L -o Canary-1B/eval_data/mls_parquet/french/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/french/test-00000-of-00001.parquet
```

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

手动下载后再运行第 2.6 节 `--offline` 命令，脚本会直接复用本地文件，不重复下载。

## 3. 评测

评测脚本默认读取第 2.1 节的标准 manifest 列表；也可以用 `--manifest` 显式指定一个或多个 manifest。

### 3.1 `beam_size` / `batch_size` 选择

- `beam_size` 是 Transformer decoder 的 beam search 宽度，不是 batch 大小：
  - `beam_size=1` 等价于 greedy decode，只保留 1 条候选，速度最快，适合 smoke test、吞吐测试和日常调试。
  - `beam_size=5` 每步保留 5 条候选，通常精度更好，但 decoder 计算量和显存占用都会增加。
- NVIDIA Canary-1B model card 的公开 ASR/AST 精度表使用 `beam width=5`、`length penalty=1.0`；因此正式精度对齐建议使用 `--beam_size 5`。
- NVIDIA model card 的普通 transcribe 示例使用 `batch_size=16`；本地 NPU/CUDA 性能评测应优先尝试 `--batch_size 16`，如显存不足再降到 `8/4/2/1`。
- `batch_size=1 + beam_size=5` 是最保守但很慢的组合，适合小规模 CPU/NPU 精度对齐，不适合完整吞吐评测。CPU 全量评测尤其慢，建议只做 smoke test 或小子集基线。

推荐参数：

| 场景 | 推荐参数 | 说明 |
|---|---|---|
| 精度对齐公开指标 | `--beam_size 5 --batch_size 16` | OOM 时将 batch 依次降到 `8/4/2/1` |
| NPU/CUDA 吞吐测试 | `--beam_size 1 --batch_size 16` | 对齐普通推理示例，优先看 RTF/RTFx |
| CPU 小子集基线 | `--beam_size 5 --batch_size 1` | 仅用于精度口径一致；全量会很慢 |
| 快速 smoke test | `--beam_size 1 --batch_size 1` | 只验证链路是否跑通 |

### 3.2 一次评测全部已准备任务（推荐：NPU 精度模式）

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_all_bs16_beam5
```

如出现 OOM，保持 `--beam_size 5` 不变，优先下调 `--batch_size 8/4/2/1`。

### 3.3 NPU 吞吐/速度模式

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 16 \
  --beam_size 1 \
  --output_dir Canary-1B/eval_results/npu_all_bs16_beam1
```

### 3.4 只评测 ASR

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --manifest \
    Canary-1B/eval_data/mls_test_german/manifest_asr_de.jsonl \
    Canary-1B/eval_data/mls_test_spanish/manifest_asr_es.jsonl \
    Canary-1B/eval_data/mls_test_french/manifest_asr_fr.jsonl \
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_asr_mls_bs16_beam5
```

### 3.5 只评测 FLEURS AST 六个方向

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
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_ast_fleurs_bs16_beam5
```

## 4. CPU/CUDA/NPU 对比

准备数据只跑一次。之后三种设备分别运行评测脚本，保持同一批 manifest 和同一解码参数。精度对齐时固定 `--beam_size 5`；性能对比时可另外跑 `--beam_size 1`。

```bash
# CPU 小子集/保守基线。全量会很慢，不建议作为吞吐路径。
python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device cpu \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/cpu_all

# NPU 精度模式。OOM 时只下调 batch_size，保持 beam_size=5。
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_all_bs16_beam5

# NPU 吞吐模式。
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 16 \
  --beam_size 1 \
  --output_dir Canary-1B/eval_results/npu_all_bs16_beam1
```

对比：

- ASR：`*_metrics.json` 中的 `wer_percent`。
- AST：`*_metrics.json` 中的 `bleu`。
- 性能：`elapsed_seconds`、`rtf`。

## 5. 通过条件

### ASR：MLS test

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
