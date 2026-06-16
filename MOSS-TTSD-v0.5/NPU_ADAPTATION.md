# MOSS-TTSD-v0.5 NPU 适配说明

## 1. 适配目标

将 MOSS-TTSD-v0.5 的旧截图式部署说明整理为可执行、可验证、可复用的 NPU 推理适配：

- 默认 `--device npu`；
- CPU 验证必须显式 `--device cpu`；
- 不使用 `auto/use_gpu/device_map="auto"` 作为设备选择；
- 不在代码中写死 `npu:0` / `cuda:0`；
- NPU 卡号由环境变量控制，例如 `ASCEND_RT_VISIBLE_DEVICES=0`；
- 缺少依赖、缺少官方字段、模型/codec revision 不匹配时直接失败并暴露原始错误。

## 2. 上游与 patch 策略

- GitHub upstream：<https://github.com/OpenMOSS/MOSS-TTSD>
- 基准 commit：`20dbb4fc44819435fee894d644a0402a0fee736a`
- 本次没有修改 `MOSS-TTSD-v0.5/upstream/` 中的上游已有文件，因此没有 `.patch`。
- `infer.py`、`download_weights.py`、`prepare_test_data.py`、`validate_outputs.py` 是当前适配新增文件，不进入 patch。

后续若必须修改上游已有文件，应在 `MOSS-TTSD-v0.5/upstream/` 中生成 patch：

```bash
git -C MOSS-TTSD-v0.5/upstream diff -- <upstream_existing_file> > MOSS-TTSD-v0.5/patches/0001-xxx.patch
git -C MOSS-TTSD-v0.5/upstream apply --check ../patches/0001-xxx.patch
```

## 3. 环境准备

### 3.1 NPU 环境

原 README 给出的 Ascend 约束可作为目标环境边界：

| 依赖 | 建议版本 |
|---|---|
| 昇腾 NPU 驱动/固件 | `>=25.0.RC1.1` |
| CANN Toolkit / Kernel / NNAL | `>=8.2.RC1` |
| Python | `>=3.10`，NPU 容器建议 `>=3.11` |
| PyTorch | 与 torch-npu/CANN 匹配，原说明为 `>=2.6.0` |
| torch-npu | 与 PyTorch/CANN 匹配，原说明为 `>=2.6.0` |

安装方式：

```bash
pip install torch torch-npu
pip install -r MOSS-TTSD-v0.5/requirements.txt
```

### 3.2 CPU 验证环境

CPU 验证只用于同 checkpoint、同 JSONL 的功能/质量/性能基线，不代表最终 NPU 性能：

```bash
python -m venv MOSS-TTSD-v0.5/.venv-cpu
source MOSS-TTSD-v0.5/.venv-cpu/bin/activate
pip install torch torchaudio
pip install -r MOSS-TTSD-v0.5/requirements.txt
```

当前执行环境未安装 `torch` / `transformers`，所以本次提交未做 CPU 实推；见 `NPU_VALIDATION.md`。

## 4. 权重与版本边界

默认固定：

- 模型：`OpenMOSS-Team/MOSS-TTSD-v0.5`，revision `8527b9136b6afefe2252ae597cecea2e80e7ebeb`。
- Codec：`OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf`，revision `c884072fd69ed00b72cd0d43355c06341c4f51a6`。

下载：

```bash
python MOSS-TTSD-v0.5/download_weights.py \
  --output_dir MOSS-TTSD-v0.5/weights
```

下载后请记录校验：

```bash
sha256sum MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5/model.safetensors
sha256sum MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf/pytorch_model.bin
```

## 5. 输入 JSONL

支持官方 v0.5 processor 的两类输入。

单共享 prompt：

```json
{"base_path":"/path/to/audio","text":"[S1]... [S2]...","prompt_audio":"shared.wav","prompt_text":"[S1]... [S2]..."}
```

双 speaker prompt：

```json
{"base_path":"/path/to/audio","text":"[S1]... [S2]...","prompt_audio_speaker1":"s1.wav","prompt_text_speaker1":"...","prompt_audio_speaker2":"s2.wav","prompt_text_speaker2":"..."}
```

`prepare_test_data.py` 生成的是第二种 schema。

## 6. 推理脚本

NPU：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-TTSD-v0.5/infer.py \
  --model_path MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5 \
  --codec_path MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf \
  --input_jsonl MOSS-TTSD-v0.5/test_data/smoke.jsonl \
  --output_dir MOSS-TTSD-v0.5/outputs \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --batch_size 1 \
  --text_normalize \
  --local_files_only
```

CPU：

```bash
python MOSS-TTSD-v0.5/infer.py \
  --model_path MOSS-TTSD-v0.5/weights/MOSS-TTSD-v0.5 \
  --codec_path MOSS-TTSD-v0.5/weights/XY_Tokenizer_TTSD_V0_hf \
  --input_jsonl MOSS-TTSD-v0.5/test_data/smoke.jsonl \
  --output_dir MOSS-TTSD-v0.5/outputs_cpu \
  --device cpu \
  --dtype float32 \
  --attn_implementation sdpa \
  --batch_size 1 \
  --local_files_only
```

输出结构：

- `sample_XXXX_YY.wav`：生成音频片段；
- `manifest.jsonl`：每条输入的输出音频、文本、模型/codec revision；
- `run_report.json`：耗时、生成音频时长、RTF、RTFx、设备和 dtype。

## 7. 与旧 README 手工修改的关系

旧 README 要用户手工修改多处 `cuda` 字符串和一键整合包文件。该路径存在两个问题：

1. 改动没有 patch，无法复现；
2. 一键包内部代码与当前官方 v0.5 HF remote-code snapshot 的版本边界不清晰。

本次改为固定官方模型/codec snapshot，并用新增 `infer.py` 控制设备迁移。若实际业务必须继续使用 ModelScope 一键整合包，应先记录包内源码/权重 commit 或 SHA256，再把必要源码修改整理成 patch，而不是继续依赖截图和人工编辑。
