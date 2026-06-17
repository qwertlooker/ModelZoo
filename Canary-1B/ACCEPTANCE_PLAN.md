# Canary-1B 完整验收方案

## 0. 验收目标与范围

本方案用于验收当前目录中 `nvidia/canary-1b` 原始模型在昇腾 NPU 上的适配结果。验收不只看 1 条 dummy 音频是否跑通，而是覆盖原始模型声明的功能、性能和精度边界。

**模型边界**

- 适配对象：Hugging Face `nvidia/canary-1b` / `canary-1b.nemo`。
- 不包含：`canary-1b-flash`、`canary-1b-v2`、Riva/NIM 服务化镜像。
- 上游加载方式：NeMo `EncDecMultiTaskModel`。
- 支持输入：16 kHz 单声道音频；也可由预处理链路重采样为 16 kHz 单声道后输入。
- 支持任务：
  - ASR：英语、德语、西班牙语、法语语音识别。
  - AST：英语 ↔ 德语/西班牙语/法语语音翻译。
  - PnC：输出可带或不带标点和大小写。

**验收分层**

| 层级 | 目的 | 数据规模 | 必跑条件 | 结论用途 |
|---|---|---:|---|---|
| L0 smoke | 验证模型加载、设备迁移、单条推理链路 | 1 条，约 1 秒 | 每次改动必跑 | 只证明可运行，不证明准确率 |
| L1 功能回归 | 覆盖 ASR/AST/语言/PnC/batch/manifest | 20~40 条，约 5~15 分钟 | 每次交付必跑 | 判断功能完整性 |
| L2 推荐精度/性能 | 与 CPU/CUDA 或公开指标做可解释对齐 | 每语种 30~300 分钟，建议总量 2~8 小时 | 正式验收必跑 | 判断 NPU 适配是否可接受 |
| L3 完整复现 | 尽量复现原始模型公开榜单和论文/模型卡指标 | 公开测试集全量，几十小时级 | 有资源时跑 | 用于严谨对标和发布报告 |

## 1. 原始模型公开能力与参考指标

公开模型卡给出的能力和参考结果如下，验收时应以这些能力作为覆盖范围，而不是只跑英文 ASR smoke test。

### 1.1 功能能力

| 能力 | 原始模型声明 | 本适配验收要求 |
|---|---|---|
| ASR | `en/de/es/fr` 4 种语言 | 4 种语言各至少 1 个样本；正式验收各跑公开测试子集 |
| AST | `en -> de/es/fr` 与 `de/es/fr -> en` | 6 个方向各至少 1 个样本；正式验收跑 FLEURS 或 CoVoST-v2 子集 |
| PnC | `yes/no` 两种输出 | 至少在英文 ASR 和 1 个 AST 方向验证 `--pnc yes/no` 均可运行 |
| 输入方式 | 音频路径列表或 JSONL manifest | 验证 manifest；可选验证音频路径列表兼容性 |
| batch | 模型卡示例 batch size 16；适配脚本支持 `--batch_size` | NPU/CUDA 优先验证 `batch_size=16`，OOM 时降到 `8/4/2/1` 并记录最大可用 batch；CPU 只建议小子集 |
| 解码 | 模型卡精度使用 beam size 5、length penalty 1.0；普通示例可用 greedy | 精度验收使用 `--beam_size 5`；吞吐/速度模式使用 `--beam_size 1`；性能同时记录 greedy/beam5 |

### 1.2 官方精度评测数据

来源：NVIDIA Hugging Face model card <https://huggingface.co/nvidia/canary-1b>。公开 ASR/AST 结果使用 `beam width=5`、`length penalty=1.0`。

> 注意：公开指标通常在 NVIDIA GPU、特定 NeMo 版本、特定 normalizer 和 decode 配置下得到。NPU 适配验收不要求逐项完全复现，但要求同数据、同脚本下 NPU 相对 CPU/CUDA 不出现明显退化。只有完整对齐数据集、normalizer、解码参数和评测脚本时，才可宣称复现官方指标。

**ASR WER（不带 PnC、使用 whisper-normalizer）**

| 数据集 | En | De | Es | Fr |
|---|---:|---:|---:|---:|
| MCV-16.1 test | 7.97 | 4.61 | 3.99 | 6.53 |
| MLS test | 3.06 | 4.19 | 3.15 | 4.12 |

**AST BLEU（使用原始标点和大小写）**

| 数据集 | 方向 | BLEU |
|---|---|---:|
| FLEURS | En→De | 32.15 |
| FLEURS | En→Es | 22.66 |
| FLEURS | En→Fr | 40.76 |
| FLEURS | De→En | 33.98 |
| FLEURS | Es→En | 21.80 |
| FLEURS | Fr→En | 30.95 |
| CoVoST-v2 | De→En | 37.67 |
| CoVoST-v2 | Es→En | 40.70 |
| CoVoST-v2 | Fr→En | 40.42 |
| mExpresso | En→De | 23.84 |
| mExpresso | En→Es | 35.74 |
| mExpresso | En→Fr | 28.29 |

### 1.3 官方/公开性能评测数据

来源：`nvidia/canary-1b` model card 链接的 Hugging Face Open ASR Leaderboard <https://hf-audio-open-asr-leaderboard.hf.space/>，以及 Open ASR Leaderboard 代码/说明 <https://github.com/huggingface/open_asr_leaderboard>。

原始 `nvidia/canary-1b` model card 没有发布单独的硬件延迟/吞吐表。当前可作为公开性能参考的是 Open ASR Leaderboard 的 RTFx；该榜单说明开源模型评测在 NVIDIA A100-SXM4-80GB GPU、CUDA 12.6、PyTorch 2.4.0 下运行，batch size 尽量使用 64，显存不足时自适应降低。RTFx 是跨模型相同条件下的公开 GPU 参考，不是 NPU 验收通过线。

截至 2026-05-26，公开参考如下：

| 指标 | 公开值 |
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

本适配性能验收必须另行记录 NPU 本机数据：`elapsed_seconds`、`rtf`、`RTFx=audio_seconds/elapsed_seconds`、最大可用 `batch_size`、`beam_size`、峰值 HBM/RSS、首次编译/加载耗时和稳定推理耗时。NPU 结论以同 checkpoint、同数据、同脚本、同 decode 参数下相对 CPU/CUDA 不退化为主；Open ASR Leaderboard 只用于公开 GPU 量级参考。

## 2. 数据集选择：规模、获取难度与建议用途

### 2.1 推荐组合

| 数据 | 覆盖 | 规模/难度 | 用途 | 建议 |
|---|---|---|---|---|
| 本地 dummy wav | 设备链路 | 1 秒，已生成，极易 | L0 smoke | 必跑，但不能作为精度依据 |
| 自制小样本集 | ASR/AST/PnC/batch | 20~40 条、5~15 分钟；需要准备参考文本 | L1 功能回归 | 必跑；放在内网对象存储或 `test_data/eval_smoke/` |
| MLS `test`（german/spanish/french） | 多语种 ASR | 每语种约 10~14 小时，parquet 获取较大 | L2/L3 多语种精度 | 优先每语种抽样 30~60 分钟，再跑全量 |
| Mozilla Common Voice / MCV test | 多语种 ASR | 多语种，下载和版本管理中等 | L2/L3 多语种 ASR | 用固定版本；记录 locale 与版本 |
| MLS test | 多语种 ASR | 多语种，体量较大，获取中等 | L2/L3 多语种 ASR | 可先抽样 30~60 分钟/语种 |
| FLEURS | ASR/AST，多语种 | 每语种测试集数百条，获取相对容易 | L2 AST 推荐 | 优先用于 6 个 AST 方向 |
| CoVoST-v2 | AST X→En | 体量较大，依赖 Common Voice，获取中等偏难 | L3 AST 复现 | 资源允许时跑全量 |
| mExpresso | AST En→X | 获取和预处理相对更复杂 | L3 AST 复现 | 可作为补充，不作为最低准入 |
| Open ASR Leaderboard 套件 | 英文 ASR 榜单 | 多数据集、脚本复杂、耗时较高 | L3 公开榜单复现 | 发布级报告再跑 |

### 2.2 分层数据量建议

| 层级 | 建议数据量 | 选择原则 |
|---|---:|---|
| L0 | 1 条 dummy | 只验证端到端调用，不验准确率 |
| L1 | 每个 ASR 语言 2 条；每个 AST 方向 2 条；PnC yes/no 各 1 轮 | 小而全，优先覆盖所有任务开关 |
| L2-min | ASR：每语种 30 分钟；AST：每方向 50~100 条 | 用于资源有限的正式验收 |
| L2-full | ASR：每语种 1~2 小时；AST：FLEURS test 全量 | 推荐正式验收目标 |
| L3 | 公开模型卡/Leaderboard 对应数据全量 | 用于对外声明“复现原始模型指标” |

## 3. 验收环境与前置检查

### 3.1 环境记录

验收报告必须记录：

```bash
python -V
pip freeze | grep -E 'torch|torch-npu|nemo|torchaudio|librosa|soundfile|jiwer|sacrebleu|whisper'
npu-smi info || true
uname -a
sha256sum Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo
```

最低需记录：CANN 版本、驱动版本、NPU 型号、NPU 数量、torch/torch-npu/NeMo 版本、权重 SHA256、NeMo 上游 commit、运行日期。

### 3.2 权重验收

```bash
sha256sum Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo
```

通过条件：SHA256 与当前记录一致：

```text
b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a
```

如果权重来源、镜像或文件名改变，必须重新记录来源 URL、下载时间、SHA256，并说明是否仍为 `nvidia/canary-1b` 原始 `.nemo`。

## 4. 功能验收

### 4.1 L0 smoke test

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/infer.py \
  --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
  --audio Canary-1B/test_data/dummy_1s_16k.wav \
  --device npu \
  --task asr \
  --source_lang en \
  --target_lang en \
  --pnc yes \
  --batch_size 1
```

通过条件：

- 退出码为 0；
- 无设备不一致错误，例如 `Expected all tensors to be on the same device`；
- 无 CUDA-only / NCCL-only / `.cuda()` 硬编码导致的错误；
- 能打印一条文本结果。

### 4.2 L1 全功能矩阵

准备一个 JSONL 或清单，至少包含：

| 用例 | task | source_lang | target_lang | pnc | 样本要求 |
|---|---|---|---|---|---|
| EN ASR | asr | en | en | yes/no | 英文语音 + 英文参考文本 |
| DE ASR | asr | de | de | yes | 德文语音 + 德文参考文本 |
| ES ASR | asr | es | es | yes | 西语语音 + 西语参考文本 |
| FR ASR | asr | fr | fr | yes | 法语语音 + 法语参考文本 |
| EN→DE AST | ast 或 s2t_translation | en | de | yes | 英文语音 + 德文参考翻译 |
| EN→ES AST | ast 或 s2t_translation | en | es | yes | 英文语音 + 西语参考翻译 |
| EN→FR AST | ast 或 s2t_translation | en | fr | yes | 英文语音 + 法语参考翻译 |
| DE→EN AST | ast 或 s2t_translation | de | en | yes | 德文语音 + 英文参考翻译 |
| ES→EN AST | ast 或 s2t_translation | es | en | yes | 西语语音 + 英文参考翻译 |
| FR→EN AST | ast 或 s2t_translation | fr | en | yes | 法语语音 + 英文参考翻译 |
| Batch | asr | en | en | yes | 至少 4 条音频，`batch_size=4` |
| 长音频 | asr | en | en | yes | 1 条 3~10 分钟音频；若失败需记录是否需要 chunked inference |

通过条件：所有用例退出码为 0；输出语言与目标语言一致；`pnc=yes/no` 能观察到大小写/标点开关差异或至少不报错；`batch_size>1` 不发生 shape/device/OOM 以外异常。

## 5. 精度验收

### 5.1 评测指标与归一化

| 任务 | 指标 | 归一化建议 | 工具 |
|---|---|---|---|
| ASR | WER | 与模型卡一致：使用官方 Whisper `EnglishTextNormalizer`；不带 PnC 的 ASR 结果用于主指标 | `jiwer` + `openai-whisper` |
| AST | BLEU | 保留原始标点和大小写；tokenization 固定 | `sacrebleu` 或 NeMo/Lightning BLEU |
| 回归一致性 | 文本完全一致率、WER/BLEU 差异 | 同模型同数据 CPU/CUDA vs NPU | 自定义脚本 |

项目级脚本规范见根目录《模型NPU 适配标准流程.md》的“项目级脚本严格失败原则”。Canary 的 `eval_canary.py` / `infer.py` 必须按该项目级规范执行：除按设备条件加载的后端模块（如 `torch_npu`）外，必需依赖统一前置 import；`whisper.normalizers.EnglishTextNormalizer`、`jiwer`、`sacrebleu`、NeMo/Torch 任一缺失均应在启动阶段暴露原始错误。禁止使用 `whisper_normalizer`、regex/basic normalizer 或其他 fallback 替代官方 ASR WER 路径；禁止为 NeMo 版本字段、解码配置等官方预期字段添加 `try/except`、`hasattr/getattr` 静默兼容，字段缺失应作为环境/版本不匹配直接失败。

### 5.2 推荐 L2 精度验收

**ASR**

| 数据 | 语言 | 最小规模 | 推荐规模 | 通过条件 |
|---|---|---:|---:|---|
| MLS test 子集 | en | 30 分钟 | 全量约 5 小时 | NPU WER 相对 CPU/CUDA 不劣化；若对公开值，WER ≤ 公开值 + 10% 相对或 +0.5 绝对二者取宽 |
| MCV 或 MLS 子集 | de/es/fr | 每语种 30 分钟 | 每语种 1~2 小时或全量 test | 同上 |

**AST**

| 数据 | 方向 | 最小规模 | 推荐规模 | 通过条件 |
|---|---|---:|---:|---|
| FLEURS 子集 | En→De/Es/Fr | 每方向 50 条 | FLEURS test 全量 | NPU BLEU 相对 CPU/CUDA 下降 ≤ 0.5；若对公开值，BLEU ≥ 公开值 - 10% 相对或 -1.0 绝对二者取宽 |
| FLEURS 子集 | De/Es/Fr→En | 每方向 50 条 | FLEURS test 全量 | 同上 |

### 5.3 L3 完整复现

完整复现才可宣称“达到原始模型公开精度”。建议：

1. 固定 NeMo 版本、decode 参数 `beam_size=5`、length penalty 1.0。
2. 固定数据集版本和 split：MCV-16.1、MLS、FLEURS、CoVoST-v2、mExpresso。
3. 固定文本 normalizer 和 BLEU 实现。
4. 分别输出每个数据集、每种语言、每个方向的指标。
5. 对比第 1.2 节公开指标，记录差异原因：数据版本、采样率处理、normalizer、解码参数、硬件精度、NeMo 版本。

## 6. 性能验收

### 6.1 指标

| 指标 | 含义 | 记录方式 |
|---|---|---|
| RTF | 推理耗时 / 音频时长，越低越好 | `wall_time / sum_audio_duration` |
| RTFx | 音频时长 / 推理耗时，越高越好 | `sum_audio_duration / wall_time` |
| 首次加载时间 | restore/from_pretrained 到模型可推理耗时 | 单独记录 |
| 峰值内存/显存 | CPU RSS、NPU HBM | `/usr/bin/time -v`、`npu-smi` |
| 最大 batch | 不 OOM 的最大 batch size | batch 1/2/4/8/16 逐级测试 |
| 稳定性 | 连续推理 N 轮失败率 | 至少 30 轮短音频或 1 小时音频 |

### 6.2 性能测试命令模板

```bash
/usr/bin/time -f 'elapsed=%E maxrss=%MKB' \
  env ASCEND_RT_VISIBLE_DEVICES=0 \
  python Canary-1B/infer.py \
    --model Canary-1B/weights/canary-1b-hfmirror/canary-1b.nemo \
    --audio <audio_1.wav> <audio_2.wav> ... \
    --device npu \
    --task asr \
    --source_lang en \
    --target_lang en \
    --pnc no \
    --beam_size 1 \
    --batch_size 4
```

同一数据分别跑：

| 模式 | 目的 |
|---|---|
| `beam_size=1, batch_size=1` | 低延迟基线 |
| `beam_size=1, batch_size=4/8/16` | 吞吐基线 |
| `beam_size=5, batch_size=16` | 推荐精度模式性能；OOM 时降到 `8/4/2/1` |
| CPU 同参数 | 小子集精度基线；全量 CPU 很慢，不作为吞吐路径 |
| CUDA 同参数（如有） | 与原始 GPU 路径对齐 |

### 6.3 通过条件

最低准入：

- NPU 推理 RTFx > 1，即快于实时；若低于实时，必须说明原因并标记为性能不通过。
- NPU 相比 CPU 同参数至少有明显加速；建议加速比 ≥ 5x，正式上线建议 ≥ 10x。
- 连续 30 轮短音频推理无崩溃、无显存持续增长。
- batch 提升时吞吐应有正向收益；若 batch>1 无收益或 OOM，需要记录最大可用 batch 和瓶颈。

增强准入：

- 在同等 decode 参数和同等精度下，与可用 CUDA/A100 基线对比，NPU RTFx 不低于 CUDA 基线的 50%~70%；若无法获得 CUDA 基线，则以 CPU 加速比和 NPU 稳定性作为上线依据。
- 公开 Open ASR Leaderboard 的 RTFx=235.34 仅作为 GPU/榜单环境参考，不作为单卡 NPU 强制门槛。

## 7. 稳定性和异常场景验收

| 场景 | 测试方法 | 通过条件 |
|---|---|---|
| 连续运行 | 同一批短音频循环 30~100 次 | 无崩溃，显存无持续增长 |
| 多 batch | batch 1/2/4/8/16 | 记录最大可用 batch；OOM 时错误可解释，不影响小 batch |
| 长音频 | 3~10 分钟音频 | 可完成或明确提示需 chunked inference |
| 非 16 kHz 音频 | 8 kHz/44.1 kHz 单声道样本 | 若 NeMo 自动处理失败，应在文档中要求预处理为 16 kHz mono |
| 多语种错误配置 | source/target 非支持语言 | 参数校验报错清晰 |
| NPU 选择 | 切换 `ASCEND_RT_VISIBLE_DEVICES` | 不写死 `npu:0`，可按环境变量选卡 |

## 8. 验收报告模板

正式验收输出建议保存为 `Canary-1B/validation_reports/YYYYMMDD_<device>.md`，包含：

```markdown
# Canary-1B NPU 验收报告

## 环境
- 日期：
- NPU 型号/数量：
- CANN/驱动：
- Python/torch/torch-npu/NeMo：
- 权重 SHA256：
- NeMo commit：

## 功能验收
| 用例 | 命令/manifest | 结果 | 备注 |
|---|---|---|---|

## 精度验收
| 任务 | 数据集 | 语言/方向 | 条数/时长 | 指标 | 公开/CPU/CUDA 基线 | 是否通过 |
|---|---|---|---:|---:|---:|---|

## 性能验收
| 模式 | batch | beam | 音频总时长 | wall time | RTFx | 峰值内存/HBM | 是否通过 |
|---|---:|---:|---:|---:|---:|---:|---|

## 稳定性
| 场景 | 次数/时长 | 结果 | 是否通过 |
|---|---:|---|---|

## 结论
- 通过/不通过：
- 阻塞项：
- 风险：
- 后续建议：
```

## 9. MLS / LibriSpeech / FLEURS 验证测试方案

本节保留具体数据准备、在线/离线/手动下载、评测命令、CPU/CUDA/NPU 对比、通过条件和输出文件说明。官方参考指标不在此处重复，见第 1 节。

按要求将流程拆成两步：

1. **准备数据**：`prepare_eval_data.py` 只负责下载数据、转 16 kHz wav、写 JSONL manifest。
2. **评测**：`eval_canary.py` 只读取已准备好的 manifest，使用与 `infer.py` 相同的 NeMo `model.transcribe()` 机制做推理，再计算 WER/BLEU。

这样 CPU/CUDA/NPU 评测可以复用同一份 wav 和 manifest，避免每次评测重复下载或抽样不一致。

### 9.1 前置依赖

```bash
pip install datasets soundfile librosa tqdm jiwer sacrebleu openai-whisper
```

- 数据准备需要：`datasets soundfile librosa tqdm`。
- 评测需要：`jiwer sacrebleu openai-whisper`。ASR WER 固定走官方 Whisper `EnglishTextNormalizer` 路径（`from whisper.normalizers import EnglishTextNormalizer`）；依赖缺失或导入失败时会在脚本启动导入阶段直接抛出原始异常。仅安装 `whisper_normalizer` 不视为满足官方路径，且不使用本地 fallback normalizer。
- 如 Hugging Face 访问慢，可设置 `HF_ENDPOINT` / `HF_HOME`；但评测数据推荐使用下面的显式本地目录参数，便于离线迁移。

#### 9.1.1 评测脚本 import / 依赖规范

项目级流程规范详见根目录《模型NPU 适配标准流程.md》的“项目级脚本严格失败原则”。`Canary-1B/eval_canary.py` 作为本模型评测入口必须遵守该项目级规范：

1. 除设备后端探测类 import（例如仅 `--device npu` 才需要的 `torch_npu`）外，评测依赖统一放在文件顶部导入，禁止在 metric 计算阶段临时 import 后再 fallback。
2. ASR WER 只能使用官方路径 `from whisper.normalizers import EnglishTextNormalizer`；不得改用 `whisper_normalizer` 包、regex/basic normalizer 或其他静默替代实现。
3. 任一必需依赖缺失时脚本应直接失败并暴露原始异常；不要用宽泛 `try/except` 包装成兼容路径，不要吞掉异常，不要继续推理后再给出不可对齐官方口径的指标。
4. 对 NeMo 配置、版本字段和解码配置使用当前官方预期字段；字段缺失表示环境或版本不匹配，应立即报错，不添加 `hasattr/getattr` 式静默兼容。

### 9.2 准备数据

> 当前 `prepare_eval_data.py` 已支持在线/离线混合模式：ASR 精度使用 `--asr_parquet_dir` 指定 `facebook/multilingual_librispeech` parquet 保存目录，性能测试保留 LibriSpeech `test-clean`，使用 `--librispeech_dir` 指定 OpenSLR tar/解压目录，FLEURS 使用 `--fleurs_parquet_dir` 指定 parquet 保存目录；目标文件已存在时直接复用，缺失时在线下载到该目录，`--offline` 下缺失则直接报具体路径且不联网。
>
> MLS/LibriSpeech/FLEURS 都不再依赖 `torchcodec` 自动解码：脚本将 HF `Audio` 列 cast 为 `decode=False`，再用 `soundfile` 读取 bytes/path 写 16 kHz wav。

#### 9.2.0 推荐目录结构

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

#### 9.2.1 最小验收数据：MLS 30 分钟 + LibriSpeech test-clean 30 分钟 + FLEURS 每方向 50 条

```bash
python Canary-1B/prepare_eval_data.py \
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

#### 9.2.2 准备 ASR MLS test + LibriSpeech test-clean 全量

```bash
python Canary-1B/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --asr_split test \
  --asr_minutes 0 \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --asr_pnc no
```

#### 9.2.3 只准备 MLS ASR test 全量（不含性能用 LibriSpeech）

```bash
python Canary-1B/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --asr_parquet_dir Canary-1B/eval_data/mls_parquet \
  --asr_configs german,spanish,french \
  --asr_split test \
  --asr_minutes 0 \
  --no-include_librispeech_test_clean \
  --asr_pnc no
```

#### 9.2.4 只准备性能测试用 LibriSpeech test-clean 全量

```bash
python Canary-1B/prepare_eval_data.py \
  --task asr \
  --data_dir Canary-1B/eval_data \
  --asr_configs "" \
  --include_librispeech_test_clean \
  --librispeech_dir Canary-1B/eval_data/librispeech_raw \
  --librispeech_minutes 0 \
  --asr_pnc no
```

#### 9.2.5 只准备 AST FLEURS test 全量

```bash
python Canary-1B/prepare_eval_data.py \
  --task ast \
  --data_dir Canary-1B/eval_data \
  --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
  --fleurs_split test \
  --fleurs_limit 0 \
  --ast_directions en-de,en-es,en-fr,de-en,es-en,fr-en \
  --ast_pnc yes
```

#### 9.2.6 离线复用本地数据

当上述目录已经由在线脚本或手动命令准备好后，离线环境使用同一命令加 `--offline`：

```bash
python Canary-1B/prepare_eval_data.py \
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

#### 9.2.7 手动命令行下载到脚本指定目录

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

### 9.3 评测

评测脚本默认读取第 2.1 节的标准 manifest 列表；也可以用 `--manifest` 显式指定一个或多个 manifest。

#### 9.3.1 `beam_size` / `batch_size` 选择

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
| NPU/CUDA 吞吐测试 | `--performance_mode --beam_size 1 --batch_size 64/128` | 对齐 Open ASR Leaderboard 计时口径：duration 降序、warmup、audio list、bf16 |
| CPU 小子集基线 | `--beam_size 5 --batch_size 1` | 仅用于精度口径一致；全量会很慢 |
| 快速 smoke test | `--beam_size 1 --batch_size 1` | 只验证链路是否跑通 |

#### 9.3.2 一次评测全部已准备任务（推荐：NPU 精度模式）

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_all_bs16_beam5
```

如出现 OOM，保持 `--beam_size 5` 不变，优先下调 `--batch_size 8/4/2/1`。

#### 9.3.3 NPU 吞吐/速度模式

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --manifest Canary-1B/eval_data/librispeech_test_clean/manifest_asr_en.jsonl \
  --performance_mode \
  --batch_size 64 \
  --beam_size 1 \
  --output_dir Canary-1B/eval_results/npu_librispeech_test_clean_perf_bs64_beam1
```

`--performance_mode` 是专门的性能计时路径，用于尽量贴近 Hugging Face Open ASR Leaderboard 的 NeMo 评测方式：按音频时长降序排序，先 warmup `--warmup_batches` 个 batch，正式计时只统计完整音频列表转写，默认在 NPU/CUDA 上使用 `bfloat16`，并向 Canary audio-list `transcribe()` 传入 `pnc=nopnc`、`num_workers=1`。该模式会额外输出 `rtfx = audio_seconds / elapsed_seconds`。如需强制精度类型，可显式传 `--compute_dtype float32|float16|bfloat16`。

性能模式下 `--beam_size 1` 默认使用 NeMo AED `greedy_batch` 解码策略，避免仍走 beam decoder 的额外开销。如需强制使用 beam decoder 做完全可比实验，可加 `--decoding_strategy beam`。

#### 9.3.4 只评测 ASR

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
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

#### 9.3.5 只评测 FLEURS AST 六个方向

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
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

### 9.4 CPU/CUDA/NPU 对比

准备数据只跑一次。之后三种设备分别运行评测脚本，保持同一批 manifest 和同一解码参数。精度对齐时固定 `--beam_size 5`；性能对比时可另外跑 `--beam_size 1`。

```bash
#### 9.4.1 CPU 小子集/保守基线。全量会很慢，不建议作为吞吐路径。
python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device cpu \
  --batch_size 1 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/cpu_all

#### 9.4.2 NPU 精度模式。OOM 时只下调 batch_size，保持 beam_size=5。
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
  --model Canary-1B/weights/canary-1b.nemo \
  --device npu \
  --batch_size 16 \
  --beam_size 5 \
  --output_dir Canary-1B/eval_results/npu_all_bs16_beam5

#### 9.4.3 NPU 吞吐模式。
ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/eval_canary.py \
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

### 9.5 通过条件

#### 9.5.1 ASR：MLS test

- 最小规模：30 分钟。
- 推荐规模：全量约 5 小时。
- 主通过条件：同一数据、同一脚本、同一 `beam_size=5` 下，NPU WER 相对 CPU/CUDA 不劣化。
- 若直接对公开值，`WER <= 公开值 + max(公开值 * 10%, 0.5)`。

#### 9.5.2 AST：FLEURS En↔De/Es/Fr

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

### 9.6 输出文件

评测输出目录中包含：

- `<tag>.tsv`：逐样本 `sample_id / audio_path / duration / reference / hypothesis`。
- `<tag>.metrics.json`：单个 manifest 指标。
- `summary.metrics.json`：汇总指标。
- `run_env.json`：Python、torch、NeMo、设备和命令行参数记录。
- `*.jsonl.meta.json`：数据准备元信息，包含 dataset/config/split/limit、本地数据目录和 `offline`，便于确认 FLEURS 使用的是 `test` split 且复用同一批本地文件。

## 10. 最低正式验收清单

如果时间和数据受限，最低正式验收不能低于以下清单：

- [ ] 权重 SHA256 校验通过。
- [ ] L0 NPU smoke test 通过。
- [ ] L1 覆盖 4 种 ASR 语言、6 个 AST 方向、PnC yes/no、batch>1。
- [ ] L2 英文 ASR 至少 30 分钟公开数据，计算 WER。
- [ ] L2 至少 1 个非英语 ASR 语种 30 分钟公开数据，计算 WER。
- [ ] L2 至少 2 个 AST 方向各 50 条公开数据，计算 BLEU。
- [ ] 性能记录 `beam=1/5`、`batch=1/4` 的 RTFx、加载时间和峰值内存。
- [ ] 稳定性连续 30 轮短音频无崩溃。
- [ ] 生成验收报告并说明与原始模型公开指标的差异。

## 11. 参考来源

- NVIDIA Canary-1B Hugging Face 模型卡：<https://huggingface.co/nvidia/canary-1b>
- NVIDIA NeMo 仓库：<https://github.com/NVIDIA/NeMo>
- Hugging Face Open ASR Leaderboard：<https://huggingface.co/spaces/hf-audio/open_asr_leaderboard>
- Common Voice：<https://commonvoice.mozilla.org/>
- MLS：<https://huggingface.co/datasets/facebook/multilingual_librispeech>；Canary-1B 使用 `prepare_eval_data.py --asr_parquet_dir <dir>` 下载/复用 `<dir>/{german,spanish,french}/test-00000-of-00001.parquet`。
- LibriSpeech：<https://www.openslr.org/12>；Canary-1B 保留 `test-clean` 作为 Hugging Face Open ASR Leaderboard 口径的性能测试集，使用 `prepare_eval_data.py --librispeech_dir <dir>` 下载/复用 `<dir>/test-clean.tar.gz` 或 `<dir>/LibriSpeech/test-clean/`。
- FLEURS：<https://huggingface.co/datasets/google/fleurs>；Canary-1B 使用 `prepare_eval_data.py --fleurs_parquet_dir <dir>` 下载/复用 `<dir>/<config>/test-00000-of-00001.parquet`，并用 `--offline` 禁止联网。
- CoVoST-v2：<https://github.com/facebookresearch/covost>

## 12. 已完成适配结果与经验补充

以下内容用于在正式验收时保留已完成 NPU 结果、公开参考值和复用经验。

### 适配结果

#### 性能

硬件：Atlas 800I A2

| 数据集 | 指标 | NPU 结果 | 公开 GPU 参考 |
|---|---|---:|---:|
| LibriSpeech test-clean | RTF | 0.005652242997176402 | 0.0042491714115747425 |

#### 精度

硬件：Atlas 800I A2

| 任务类型 | 语言/方向 | 数据集 | 指标 | NPU 结果 | 公开参考 |
|---|---|---|---|---:|---:|
| ASR | de | Multilingual LibriSpeech | WER(%) | 3.83 | 4.19 |
| ASR | es | Multilingual LibriSpeech | WER(%) | 2.30 | 3.15 |
| ASR | fr | Multilingual LibriSpeech | WER(%) | 3.69 | 4.12 |
| AST | en-de | FLEURS | BLEU | 31.41 | 32.15 |
| AST | en-es | FLEURS | BLEU | 22.69 | 22.66 |
| AST | en-fr | FLEURS | BLEU | 39.84 | 40.76 |
| AST | de-en | FLEURS | BLEU | 33.50 | 33.98 |
| AST | es-en | FLEURS | BLEU | 21.78 | 21.80 |
| AST | fr-en | FLEURS | BLEU | 30.29 | 30.95 |

结果表明，在保持官方权重、官方解码和公开评测口径基本一致的情况下，Canary-1B 可以在昇腾 NPU 上完成 ASR 与 AST 推理适配。多语种 ASR 指标达到或优于公开参考，AST 指标与公开参考整体接近，性能结果也处于可对照范围。

### 经验总结

1. Canary-1B 的适配重点不是模型结构修改，而是保证 NeMo 官方恢复、prompt、tokenizer、解码和评测链路在 NPU 上完整跑通。
2. 语音模型评测对数据和文本后处理非常敏感，WER/BLEU 必须明确 normalizer、标点大小写、beam size 和 length penalty。
3. NPU 性能测试需要区分模型计算耗时和数据准备耗时，正式计时前应 warmup，并尽量使用按时长排序后的批量输入。
4. `beam_size=1` 和 `beam_size=5` 服务于不同目标：前者适合吞吐评估，后者适合官方精度对齐，不能混用后直接比较指标。
5. 离线部署时应提前准备 `.nemo` 权重、parquet/音频数据和 manifest，推理阶段不应依赖远程下载。
6. 对 NPU 适配问题应显式失败并暴露原始错误，避免用 CPU fallback、简化 normalizer 或替代指标掩盖真实兼容性问题。
