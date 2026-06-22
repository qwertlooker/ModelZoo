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
| WeSpeaker 代码 | `wenet-e2e/wespeaker` commit `c92349a14d6b426808c4e09b8b12e076864dfc11` |
| WeSpeaker 权重 | `voxblink2_samresnet100_ft.zip`，SHA256 `ad0873d380acaa7f4256ff37d40217ee31e4955b26a45064a13a14998cc89d16` |
| MMS-FA checkpoint | S3 version ID `dZWoHyjLHoCxDn.KL1FPSlVCD3CPRtOL`，固定大小 `1262047414` bytes |
| Whisper | `openai/whisper-large-v3` revision `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| 适配 patch | `patches/0001-adapt-v0.5-inference-to-npu.patch`，SHA256 `426303406d9289c0f981ca333604107af323a56a576c5129a844aacc83962056` |

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

## 3. 数据准备

以下命令从模型目录执行：

```bash
git clone https://github.com/OpenMOSS/TTSD-eval.git third_party/TTSD-eval
git -C third_party/TTSD-eval checkout \
  dea13b98529dc16dcfb5fe45779ad63ac9238337
curl -L --fail \
  -o third_party/TTSD-eval/testset.zip \
  https://media.githubusercontent.com/media/OpenMOSS/TTSD-eval/dea13b98529dc16dcfb5fe45779ad63ac9238337/testset.zip
echo "49ed8338f3e5323c5ffcff01f3480a9c245937256d9197d792c973cba5603e17  third_party/TTSD-eval/testset.zip" \
  | sha256sum -c -
unzip -oq third_party/TTSD-eval/testset.zip -d third_party/TTSD-eval

wc -l \
  third_party/TTSD-eval/testset/ttsd_eval_zh.jsonl \
  third_party/TTSD-eval/testset/ttsd_eval_en.jsonl
```

预期分别为 50、50。

评测环境和三类评测权重的完整下载、路径、revision、大小及 SHA256 校验命令见
`README_INFERENCE.md` 的“准备 TTSD-eval 评测环境与权重”。WeSpeaker 模型必须放到
`third_party/TTSD-eval/model/voxblink2_samresnet100_ft`；MMS-FA checkpoint 必须
放到 `third_party/TTSD-eval/model/checkpoints/model.pt`；Whisper 必须从本地
`third_party/TTSD-eval/model/whisper-large-v3` 加载。无法取得任一官方组件时应
记录原始错误并保持验收未完成，不能替换成名称相近的第三方实现。

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
    --seed 42 \
    --use_normalize
)
deactivate
```

原始入口没有 `--device` 参数，依赖 CUDA 和 `flash_attention_2`；不能给原始组添加
patch 后参数。若缺少 CUDA，S3 三组迁移验收即未完成，不能用 CPU 改写原始模型行为。

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
绝对路径，避免切换目录后失效。对六个 manifest 逐一执行：

```bash
MODEL_ROOT="$PWD"
EVAL_ROOT="$MODEL_ROOT/third_party/TTSD-eval"
source "$MODEL_ROOT/.venv-ttsd-eval/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$EVAL_ROOT/testset"

for LANG in zh en; do
  for GROUP in original_cuda patched_cuda npu; do
    INPUT="$MODEL_ROOT/results/ttsd_eval/${GROUP}_${LANG}.jsonl"
    STEM="${GROUP}_${LANG}"
    mkdir -p "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM"

    python "$EVAL_ROOT/tools/align.py" \
      --input_jsonl "$INPUT" \
      --output_jsonl "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/alignment.jsonl"
    python "$EVAL_ROOT/tools/split.py" \
      --input_jsonl "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/alignment.jsonl" \
      --output_jsonl "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/split.jsonl"
    python "$EVAL_ROOT/tools/run_similarity.py" \
      --input_jsonl "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/split.jsonl" \
      --output_jsonl "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/sim.jsonl" \
      --metrics_txt "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/acc_sim.txt"
    python "$EVAL_ROOT/wer/whisper_asr.py" \
      --input_jsonl "$INPUT" \
      --output_jsonl "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/asr.jsonl" \
      --model_id "$EVAL_ROOT/model/whisper-large-v3"
    python "$EVAL_ROOT/wer/run_wer.py" \
      --lang "$LANG" \
      --input_jsonl "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/asr.jsonl" \
      --output_jsonl "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/wer.jsonl" \
      --metrics_txt "$MODEL_ROOT/results/ttsd_eval_metrics/$STEM/wer.txt"
  done
done
cd "$MODEL_ROOT"
deactivate
```

这些命令直接复用固定 commit 的官方评测组件，不修改 evaluator，也不使用简化指标。

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
