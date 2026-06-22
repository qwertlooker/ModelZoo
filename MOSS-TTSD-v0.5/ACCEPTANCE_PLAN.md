# MOSS-TTSD-v0.5 验收计划

本文只定义可复现的迁移验收。v0.5 官方未发布正式 test split 和生成质量数值；
`OpenMOSS/TTSD-eval` 是公共评测集，不得把 v1.0 指标写成 v0.5 官方指标。

## 1. 原始测试集、官方指标和版本边界

| 项目 | 固定值 |
|---|---|
| 上游源码 | `OpenMOSS/MOSS-TTSD` tag `v0.5`，commit `0e078c62389922d3aa873ce182daf31142860b18` |
| 模型权重 | `fnlp/MOSS-TTSD-v0.5` revision `8527b9136b6afefe2252ae597cecea2e80e7ebeb` |
| Codec | `fnlp/XY_Tokenizer_TTSD_V0/xy_tokenizer.ckpt` revision `c83433728e698ed0698e88cb5096bc221fb8f8c5` |
| 官方示例 | `examples/examples.jsonl`，中文和英文各 1 条 |
| 官方正式 test split | 官方未发布 |
| 官方质量指标 | 官方未发布 |
| 公共评测 | `OpenMOSS/TTSD-eval` commit `dea13b98529dc16dcfb5fe45779ad63ac9238337`，中文/英文各 50 条 |
| TTSD-eval testset.zip SHA256 | `49ed8338f3e5323c5ffcff01f3480a9c245937256d9197d792c973cba5603e17` |
| TTSD-eval manifest SHA256 | 中文 `2c9cafed6eaea093e3dbdbc30dba0d3e87b91b4d1be9925ae97d7e8ce41a2dc4`；英文 `e779ed7c9ece3d0d0c0364bfd235fcbb591e17b543dd693b37e265f1cffb4d4d` |
| WeSpeaker 代码 | `wenet-e2e/wespeaker` commit `c92349a14d6b426808c4e09b8b12e076864dfc11` |
| WeSpeaker 权重 | `voxblink2_samresnet100_ft.zip`，SHA256 `ad0873d380acaa7f4256ff37d40217ee31e4955b26a45064a13a14998cc89d16` |
| MMS-FA checkpoint | S3 version ID `dZWoHyjLHoCxDn.KL1FPSlVCD3CPRtOL`，SHA256 `20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680` |
| Whisper | revision `06f233fe06e710322aca913c1bc4249a0d71fce1`；`model.safetensors` SHA256 `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` |
| evaluator 依赖文件 | `requirements_eval.txt`，SHA256 `e719dfc262acaa9486216ebb0e5b85afad4722f37a7f989ab0330985da6fb539` |
| 适配 patch | `patches/0001-adapt-v0.5-inference-to-npu.patch`，SHA256 `7d446e9c9c743b57ab41cb553422e428bf515b6d4e724d10450fa5b15b1a01ba` |

TTSD-eval 的 ACC、SIM、WER 用于同 checkpoint 的迁移对齐。正式报告必须同时写出
TTSD-eval commit、语言、样本数、MMS-FA、WeSpeaker 和 Whisper 版本，不得只写一个
汇总数。

## 2. 功能验证与 L2

| 层级 | 数据 | 样本规模 | 用途 |
|---|---|---:|---|
| 功能验证 | v0.5 `examples/examples.jsonl` | 2 | 三组入口、权重、codec、WAV 输出和失败用例 |
| L2 | `ttsd_eval_zh.jsonl`、`ttsd_eval_en.jsonl` | 50 + 50 | 全量 ACC、SIM、WER、RTF/RTFx 和资源对齐 |

功能验证只能证明链路可运行。功能验证和 L2 都必须保留三组结果：

1. `original_cuda`：未应用 patch 的原始 CUDA；
2. `patched_cuda`：应用 patch 后的同设备 CUDA；
3. `npu`：应用 patch 后的 NPU。

三组使用相同权重、manifest、seed 和 normalize 设置，写入不同目录。

## 3. TTSD-eval 工程准备硬门禁

完整准备命令统一维护在 `README_INFERENCE.md` 的“准备 TTSD-eval 工程”，包括：

1. evaluator 固定 commit，且受版本控制文件未修改；
2. 直接下载固定 Git LFS 对象并校验 50+50 manifest、200 个 prompt WAV；
3. 独立 Python 3.11 + PyTorch/TorchAudio 2.8.0 CPU/CUDA 环境；
4. 固定直接依赖和 WeSpeaker commit 的 `requirements_eval.txt`；
5. WeSpeaker、MMS-FA、Whisper-large-v3 的固定对象、SHA256、大小和目标路径；
6. 离线 import、五个 evaluator CLI `--help` 和三类模型加载预检。

不得修改 TTSD-eval 的 `eval.sh` / `run_wer.sh` 硬编码数组作为正式入口；正式命令
直接调用固定提交中的五个 Python 工具。进入 L2 前必须存在并归档：

```text
results/ttsd_eval_setup/source_data.json
results/ttsd_eval_setup/full.json
results/ttsd_eval_setup/evaluator-pip-freeze.txt
```

并重新执行完整门禁：

以下示例为 CUDA evaluator；CPU profile 必须改用 `--expected_device cpu`。

```bash
source .venv-ttsd-eval/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python prepare_eval_data.py verify-ttsd-eval \
    --eval_root third_party/TTSD-eval \
    --scope full \
    --expected_device cuda \
    --report results/ttsd_eval_setup/full.json
deactivate
```

任何 source、数据、环境、权重、hash、import 或模型加载失败都必须保留原始错误，并
保持验收未完成；不能换用名称相近的第三方模型或在线浮动 revision。

## 4. 三组推理

环境、源码工作树和权重按 `README_INFERENCE.md` 执行。功能验证使用官方示例：

```bash
MODEL_ROOT="$PWD"
MANIFEST="$MODEL_ROOT/upstream-original/examples/examples.jsonl"
mkdir -p results/functional

source .venv-cuda-original/bin/activate
(
  cd upstream-original
  HF_HOME="$MODEL_ROOT/hf-cache" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    python inference.py \
      --jsonl "$MANIFEST" \
      --output_dir "$MODEL_ROOT/results/functional/original_cuda" \
      --seed 42 \
      --use_normalize
)
deactivate

source .venv-cuda-patched/bin/activate
(
  cd upstream-npu
  CUDA_VISIBLE_DEVICES=0 python inference.py \
    --jsonl "$MANIFEST" \
    --output_dir "$MODEL_ROOT/results/functional/patched_cuda" \
    --device cuda \
    --batch_size 2 \
    --seed 42 \
    --use_normalize
)
deactivate

source .venv-npu/bin/activate
(
  cd upstream-npu
  ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
    --jsonl "$MANIFEST" \
    --output_dir "$MODEL_ROOT/results/functional/npu" \
    --device npu \
    --batch_size 2 \
    --seed 42 \
    --use_normalize
)
deactivate
```

原始入口没有 `--device` 参数，依赖 CUDA 和 `flash_attention_2`；不能给原始组添加
patch 后参数。若缺少 CUDA，S3 三组迁移验收即未完成，不能用 CPU 改写原始模型行为。
官方示例共 2 条，因此 patch 后 CUDA/NPU 都固定 `--batch_size 2`，与原始入口的
完整 manifest batch 一致。TTSD-eval 长清单的 patch 后 CUDA/NPU 固定
`--batch_size 1`；原始入口不支持该参数，报告必须明确记录其原生完整 batch 与
候选路径的差异，不得声称这是严格相同运行参数的逐样本数值对齐。

## 5. 输出检查和 evaluator manifest

检查采样率、时长、峰值、NaN/Inf 和全静音：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import soundfile as sf

for group in ("original_cuda", "patched_cuda", "npu"):
    paths = sorted(Path("results/functional", group).glob("output_*.wav"))
    if len(paths) != 2:
        raise RuntimeError(f"{group}: expected 2 outputs, got {len(paths)}")
    for path in paths:
        audio, sample_rate = sf.read(path, always_2d=True)
        if sample_rate <= 0 or len(audio) == 0:
            raise RuntimeError(f"empty audio: {path}")
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
            raise RuntimeError(f"invalid audio: {path}")
PY
```

波形逐点一致不是生成式 TTS 的通过条件。必须先比较
`original_cuda` 与 `patched_cuda`，确认 patch 后同设备指标无系统性变化；再比较
`patched_cuda` 与 `npu`。

## 6. L2 OpenMOSS/TTSD-eval

对 `ttsd_eval_zh.jsonl` 和 `ttsd_eval_en.jsonl` 重复第 4 节三组生成，输出到：

```text
results/ttsd_eval/original_cuda_zh
results/ttsd_eval/patched_cuda_zh
results/ttsd_eval/npu_zh
results/ttsd_eval/original_cuda_en
results/ttsd_eval/patched_cuda_en
results/ttsd_eval/npu_en
```

再为六组生成 manifest：

```bash
for LANG in zh en; do
  for GROUP in original_cuda patched_cuda npu; do
    python prepare_eval_data.py attach-output \
      --input_jsonl "third_party/TTSD-eval/testset/ttsd_eval_${LANG}.jsonl" \
      --output_jsonl "results/ttsd_eval/${GROUP}_${LANG}.jsonl" \
      --output_dir "results/ttsd_eval/${GROUP}_${LANG}" \
      --path_root third_party/TTSD-eval/testset
  done
done
```

TTSD-eval 的 prompt 路径相对 `testset/`。以下命令必须从该目录运行；输出路径使用
绝对路径，避免切换目录后失效。以下固定为单卡 CUDA evaluator；CPU profile 必须
设置 `CUDA_VISIBLE_DEVICES=""`、`ALIGN_DEVICE=cpu` 和 `WHISPER_NUM_GPUS=0`，
并对六组保持一致。每组显式指定所有中间目录，避免 evaluator 默认目录互相污染：

```bash
set -o pipefail
MODEL_ROOT="$PWD"
EVAL_ROOT="$MODEL_ROOT/third_party/TTSD-eval"
source "$MODEL_ROOT/.venv-ttsd-eval/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
ALIGN_DEVICE=cuda:0
SIM_NUM_GPUS=1
WHISPER_NUM_GPUS=1
cd "$EVAL_ROOT/testset"

for LANG in zh en; do
  for GROUP in original_cuda patched_cuda npu; do
    INPUT="$MODEL_ROOT/results/ttsd_eval/${GROUP}_${LANG}.jsonl"
    STEM="${GROUP}_${LANG}"
    RUN_ROOT="$MODEL_ROOT/results/ttsd_eval_metrics/$STEM"
    mkdir -p \
      "$RUN_ROOT/alignment_files" \
      "$RUN_ROOT/split_res" \
      "$RUN_ROOT/audio_segments"

    {
      python "$EVAL_ROOT/tools/align.py" \
        --input_jsonl "$INPUT" \
        --output_dir "$RUN_ROOT/alignment_files" \
        --output_jsonl "$RUN_ROOT/alignment.jsonl" \
        --cache_dir "$EVAL_ROOT/model" \
        --device "$ALIGN_DEVICE"
      test "$(wc -l < "$RUN_ROOT/alignment.jsonl")" -eq 50

      python "$EVAL_ROOT/tools/split.py" \
        --input_jsonl "$RUN_ROOT/alignment.jsonl" \
        --split_res_dir "$RUN_ROOT/split_res" \
        --segment_dir "$RUN_ROOT/audio_segments" \
        --output_jsonl "$RUN_ROOT/split.jsonl" \
        --num_workers 8
      test "$(wc -l < "$RUN_ROOT/split.jsonl")" -eq 50

      python "$EVAL_ROOT/tools/run_similarity.py" \
        --input_jsonl "$RUN_ROOT/split.jsonl" \
        --output_jsonl "$RUN_ROOT/sim.jsonl" \
        --metrics_txt "$RUN_ROOT/acc_sim.txt" \
        --model_dir "$EVAL_ROOT/model/voxblink2_samresnet100_ft" \
        --num_gpus "$SIM_NUM_GPUS"
      test "$(wc -l < "$RUN_ROOT/sim.jsonl")" -eq 50
      test -s "$RUN_ROOT/acc_sim.txt"

      python "$EVAL_ROOT/wer/whisper_asr.py" \
        --input_jsonl "$INPUT" \
        --output_jsonl "$RUN_ROOT/asr.jsonl" \
        --model_id "$EVAL_ROOT/model/whisper-large-v3" \
        --num_gpus "$WHISPER_NUM_GPUS"
      test "$(wc -l < "$RUN_ROOT/asr.jsonl")" -eq 50

      python "$EVAL_ROOT/wer/run_wer.py" \
        --lang "$LANG" \
        --input_jsonl "$RUN_ROOT/asr.jsonl" \
        --output_jsonl "$RUN_ROOT/wer.jsonl" \
        --metrics_txt "$RUN_ROOT/wer.txt"
      test "$(wc -l < "$RUN_ROOT/wer.jsonl")" -eq 50
      test -s "$RUN_ROOT/wer.txt"
    } 2>&1 | tee "$RUN_ROOT/evaluator.log"
  done
done
cd "$MODEL_ROOT"
deactivate
```

TTSD-eval 的部分工具会记录 warning 后跳过失败样本并以 0 退出，因此每阶段的 50 行
检查是正式门禁，不得删除。上述命令直接复用固定 commit 的官方评测组件，不修改
evaluator，也不使用简化指标。

## 7. L2 精度与性能标准

由于 v0.5 没有官方数值，通过线只定义相对迁移退化：

| 指标 | 判定 |
|---|---|
| 基础输出 | 每组输出数与 manifest 一致；全部 WAV 可读、非空、非全静音、无 NaN/Inf |
| patch 同设备回归 | `patched_cuda` 相对 `original_cuda` 的 ACC/SIM/WER 无系统性退化；差异和失败样例必须归档 |
| NPU 迁移 | `npu` 相对 `patched_cuda` 的 ACC/SIM/WER 无系统性退化 |
| WER 建议阈值 | 绝对差不超过 1.0 个百分点或相对差不超过 10%，取较宽者；超限必须人工复核 |
| ACC/SIM | 不预设伪造的官方阈值；报告绝对差、相对差、样本级异常和人工听感 |
| 性能 | 三组使用同一 L2 manifest，记录 elapsed、生成音频总时长、RTF/RTFx、峰值 HBM/RSS；至少重复 3 次报告中位数 |

TTSD-eval 是当前可取得的 OpenMOSS 公共全量 benchmark，因此 L2 使用中英文各 50 条
全量。v0.5 没有公开硬件性能值，不编造 speedup 线；最低性能结论是全量无失败、
RTF/RTFx 可复现，并报告 NPU 相对 patch 后 CUDA 的比值。项目另有性能目标时按该
目标判定。

若项目需要固定更严格阈值，应基于首轮三组真实结果评审后版本化写入，不能在无数据时
宣称某阈值来自官方。

## 8. 最低正式验收清单

- [ ] 源码、模型、codec、patch 和 testset revision/SHA256 已记录。
- [ ] TTSD-eval `source_data.json`、`full.json`、pip freeze 和模型加载预检已归档。
- [ ] evaluator CPU/CUDA profile 固定，六组使用同一设备、dtype 和依赖环境。
- [ ] 原始 CUDA、patch 后 CUDA、NPU 使用相同 manifest 和参数，输出互不覆盖。
- [ ] 功能验证 2 条、L2 中英文各 50 条均记录实际执行结果。
- [ ] 六份 L2 manifest 和 metadata 已归档。
- [ ] ACC/SIM/WER 使用固定 TTSD-eval 原始脚本完成。
- [ ] 报告同时给出原始→patch 回归差和 patch→NPU 迁移差。
- [ ] 三组 L2 elapsed、RTF/RTFx、峰值 RSS/HBM 和相对比值已归档。
- [ ] 日志包含 Python、CANN、torch、torch-npu、transformers、硬件和权重 SHA256。
- [ ] 任何未执行项、依赖缺失、OOM 或指标超限均明确标为阻塞/失败。

在全部完成前，当前交付状态最多为 S1 静态适配完成；不能写“迁移验收完成”或
“正式验收通过”。

## 9. 验收报告模板

报告保存到 `validation_reports/YYYYMMDD_<device>.md`，至少包含：

```text
状态：S1/S2/S3/S4
源码/模型/codec/patch/testset：revision + SHA256
环境：OS、Python、CANN、driver、torch、torch-npu、transformers、GPU/NPU
Evaluator：TTSD-eval/WeSpeaker commit、CPU/CUDA profile、pip freeze、三类权重 SHA256
数据：功能验证/L2 文件、样本数、manifest SHA256

三组命令和输出：
- original_cuda：
- patched_cuda：
- npu：

结果：
- 基础 WAV 检查：
- original_cuda -> patched_cuda：ACC/SIM/WER 差异、失败样例
- patched_cuda -> npu：ACC/SIM/WER 差异、失败样例
- 人工听测：
- elapsed、音频总时长、RTF/RTFx、峰值内存：

未执行/阻塞：
结论：通过/不通过；不得用功能样例替代 L2 精度或性能结论
```
