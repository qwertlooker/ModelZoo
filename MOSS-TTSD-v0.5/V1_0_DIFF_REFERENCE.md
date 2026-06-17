# MOSS-TTSD v1.0 与当前 v0.5 适配差异参考

> 目的：记录 `OpenMOSS/MOSS-TTSD` 当前 v1.0 路径相对本仓已适配的 v0.5 路径的主要变化，方便后续做 v1.0 NPU 适配时确定边界、权重、patch 范围和验收重点。本文档只做分析参考，不修改原始 `README.md`，也不引入代码文件。
>
> 复查日期：2026-06-16。后续正式适配前需按《模型NPU 适配标准流程.md》重新确认上游 commit、权重 revision、依赖版本、SHA256 和评测数据可用性。

## 1. 版本边界

| 项 | 当前 v0.5 适配 | v1.0 参考边界 |
|---|---|---|
| 源码仓库 | `https://github.com/OpenMOSS/MOSS-TTSD` tag `v0.5` | `https://github.com/OpenMOSS/MOSS-TTSD` 默认分支 `main` |
| 源码 commit | `0e078c62389922d3aa873ce182daf31142860b18` | `20dbb4fc44819435fee894d644a0402a0fee736a` |
| 是否有 tag | 有 `v0.5` tag | 本次复查未见 `v1.0` tag；需固定 `main` commit |
| 旧版本位置 | 顶层即 v0.5 代码 | v0.7 已迁入 `legacy/v0.7/`；顶层为 v1.0 路径 |
| 当前本仓适配方式 | 对 tag `v0.5` 原项目代码打 `patches/0001-adapt-v0.5-inference-to-npu.patch` | 后续应另建 v1.0 适配边界，不能复用 v0.5 patch |

## 2. 权重与 codec 变化

| 项 | v0.5 | v1.0 |
|---|---|---|
| 主模型 HF | `fnlp/MOSS-TTSD-v0.5` / `OpenMOSS-Team/MOSS-TTSD-v0.5` | `OpenMOSS-Team/MOSS-TTSD-v1.0` |
| 主模型 HF HEAD | `8527b9136b6afefe2252ae597cecea2e80e7ebeb` | `c7cd852d87aff71cab5bd2b9b05509cedc0ef1ba` |
| 主模型 ModelScope | `openmoss/MOSS-TTSD-v0.5`，本次记录 HEAD `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59` | `openmoss/MOSS-TTSD-v1.0`，本次记录 HEAD `64fd6fb06a6d7c4211a1c9477c6038aff538970a` |
| 主权重形态 | 单个 `model.safetensors` | `model-00001-of-00004.safetensors` 到 `model-00004-of-00004.safetensors` + `model.safetensors.index.json` |
| Codec | 原项目 `XY_Tokenizer/` 代码 + `fnlp/XY_Tokenizer_TTSD_V0/xy_tokenizer.ckpt` | 独立 HF 模型 `OpenMOSS-Team/MOSS-Audio-Tokenizer` |
| Codec HEAD | HF `c83433728e698ed0698e88cb5096bc221fb8f8c5`；ModelScope `79082154409f5e883d9487c4d4b4be363323b039` | HF `3cd226ba2947efa357ef453bcad111b6eafba782`；ModelScope `d8ec39a98954fde962b16a2f0ec22666c04094d0` |
| Codec 权重形态 | 单个 `xy_tokenizer.ckpt` | `model-00001-of-00002.safetensors`、`model-00002-of-00002.safetensors` + index |
| Codec 代码位置 | 源码仓库内 `XY_Tokenizer/` | `MOSS-Audio-Tokenizer` 模型仓库的 `modeling_moss_audio_tokenizer.py` 等 remote code |
| 采样率 | v0.5 依赖 XY Tokenizer 配置 | v1.0 主模型和 codec 配置均记录 `sampling_rate` / `sample_rate` 为 `24000` |

v1.0 下载命令参考：

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p weights/MOSS-TTSD-v1.0 weights/MOSS-Audio-Tokenizer

hf download OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --revision c7cd852d87aff71cab5bd2b9b05509cedc0ef1ba \
  --local-dir weights/MOSS-TTSD-v1.0

hf download OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --revision 3cd226ba2947efa357ef453bcad111b6eafba782 \
  --local-dir weights/MOSS-Audio-Tokenizer
```

正式适配时必须记录全部 safetensors shard、index 文件、processor/config 文件的 SHA256；不要只记录模型仓库名。

## 3. 模型与处理器接口变化

| 项 | v0.5 | v1.0 |
|---|---|---|
| 主模型类型 | `model_type: moss_ttsd`，`MossTTSDForCausalLM` | `model_type: moss_tts_delay`，`MossTTSDelayModel` |
| Processor | `processing_moss_ttsd.MossTTSDProcessor` | `MossTTSDelayProcessor` |
| AutoProcessor 加载 | v0.5 适配显式加载模型和 `XY_Tokenizer` | `AutoProcessor.from_pretrained(..., codec_path=<MOSS-Audio-Tokenizer>)` |
| 音频 token 配置 | `channels: 8`，`speech_vocab_size: 1025` | `n_vq: 16`，`audio_vocab_size: 1024`，新增 audio slot token IDs |
| 推理输出解码 | v0.5 通过 `process_batch` 和 `spt` 解码 | `processor.decode(outputs)` 返回带 `audio_codes_list` 的 message，再写 wav |
| remote code 依赖 | 主要在 v0.5 主模型仓和源码仓内 | 主模型仓与 codec 仓均包含 remote code，NPU 适配需同时审查两边 |

适配含义：v1.0 不是简单替换 `MODEL_PATH`。它把 codec 从本地 `XY_Tokenizer` checkpoint 换成可通过 `AutoProcessor` 挂载的独立模型，主模型和 codec 的 remote code 都可能成为 NPU patch 对象。

## 4. 源码文件结构变化

| 项 | v0.5 tag | v1.0 main |
|---|---|---|
| 顶层 `XY_Tokenizer/` | 存在 | 不存在；旧实现位于 `legacy/v0.7/XY_Tokenizer/` |
| 顶层 `examples/` | 存在 `examples/examples.jsonl` 和示例音频 | 顶层未保留 v0.5 `examples/`；参考音频在 `asset/reference_02_s1.wav`、`asset/reference_02_s2.wav` |
| 顶层 `inference.py` | 简单单进程脚本 | 批处理、多 GPU `torch.multiprocessing.spawn`、多模式推理 |
| 顶层 `generation_utils.py` | `load_model`、`process_batch` 等 v0.5 逻辑 | JSONL streaming、采样参数解析、prompt 音频编码、输出 JSONL 合并 |
| Gradio | v0.5 简单 demo | v1.0 预加载 backend，显式 `--device` 但仍默认 CUDA 逻辑 |
| SGLang | v0.5 当前适配未覆盖 | v1.0 README 给出 fuse + `sglang serve --delay-pattern` 路径 |
| scripts | v0.5 顶层无 v1.0 fuse/request 脚本 | 新增 `scripts/fuse_moss_tts_delay_with_codec.py`、`scripts/processing_moss_tts_delay_with_codec.py`、`scripts/request_sglang_generation.py` |

## 5. CLI 与输入格式变化

### 5.1 v0.5 当前适配入口

```bash
python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

### 5.2 v1.0 上游入口

```bash
python inference.py \
  --model_path OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --codec_model_path OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --input_jsonl /path/to/input.jsonl \
  --save_dir outputs \
  --mode voice_clone_and_continuation \
  --batch_size 1 \
  --text_normalize
```

v1.0 上游参数新增或变化：

- `--codec_model_path` 替代 v0.5 的 `--spt_config_path` / `--spt_checkpoint_path`。
- `--input_jsonl` / `--save_dir` 替代 `--jsonl` / `--output_dir`。
- `--mode` 支持 `generation`、`continuation`、`voice_clone`、`voice_clone_and_continuation`。
- 新增采样参数：`--max_new_tokens`、`--temperature`、`--top_p`、`--top_k`、`--repetition_penalty`。
- 新增 `--sample_rate_normalize`。
- 上游 `inference.py` 没有 `--device`、`--device_id`、`--dtype`、`--attn_implementation` 参数；它自动选择 CUDA 或 CPU。

v1.0 JSONL 支持 1 到 5 个说话人：

```json
{
  "base_path": "/path/to/audio/files",
  "text": "[S1]... [S2]... [S3]... [S4]... [S5]...",
  "prompt_audio_speaker1": "speaker1.wav",
  "prompt_text_speaker1": "reference text 1",
  "prompt_audio_speaker2": "speaker2.wav",
  "prompt_text_speaker2": "reference text 2"
}
```

后续 v1.0 NPU 适配应保留 v1.0 官方字段，不要把 v1.0 强行退化成 v0.5 的双说话人格式。

## 6. 依赖变化

| 项 | v0.5 | v1.0 |
|---|---|---|
| PyTorch | `torch>=2.0.0` | `torch==2.9.1+cu128` |
| torchaudio | `torchaudio>=2.0.0` | `torchaudio==2.9.1+cu128` |
| Transformers | `>=4.30.0` | `==5.0.0` |
| numpy | `>=1.21.0` | `==2.1.0` |
| Gradio | `>=4.0.0` | `==6.5.1` |
| 其它 | `accelerate`、`PyPDF2`、`beautifulsoup4`、`openai` 等 | `safetensors`、`orjson`、`tiktoken`、flash-attn build deps 等 |

适配含义：v1.0 的 `requirements.txt` 明确绑定 CUDA 版 torch/torchaudio，并包含 CUDA/ROCm GPU 专用的 `flash-attn` 构建依赖，NPU 环境不能照抄安装。正式文档中应改为 Ascend 兼容 torch/torch-npu 组合，音频 I/O、重采样和 mel/Hz 转换不依赖 `torchaudio`，attention 后端默认使用 `sdpa/eager`，并把 CUDA wheel 与 `flash-attn` 约束作为上游 CUDA/ROCm 参考，不作为 NPU 安装命令。

## 7. 设备与 attention 路径差异

v1.0 上游 CUDA 假设更强：

- `inference.py` 使用 `torch.cuda.is_available()` 决定 CUDA/CPU，且多卡时 `torch.cuda.set_device(rank)` + `torch.cuda.device_count()`。
- CUDA 路径优先 `flash_attention_2`，失败后 warning 并回退 `sdpa`；CPU 路径用 `sdpa`。
- `gradio_demo.py` 默认 `--device cuda:0`，并使用 `torch.backends.cuda.*` 开关和 CUDA capability 判断。
- `MOSS-Audio-Tokenizer` remote code 中 `RingKVCache` 默认参数为 `device=torch.device("cuda")`，虽然多数调用会传入实际 device，但该默认值需要在 NPU 适配时审查。

NPU 适配建议：

1. 在 v1.0 `inference.py` 增加显式 `--device {cpu,cuda,npu}`、`--device_id`、`--dtype`、`--attn_implementation`，禁止自动悄悄落到 CPU。
2. 仅当 `--device npu` 时条件导入 `torch_npu` 并设置 NPU device。
3. 多卡先做单卡 NPU 闭环；多 NPU 需要单独验证 `torch.multiprocessing.spawn`、device rank 映射和输出合并。
4. attention 后端应由参数显式控制。缺少 NPU 支持时直接暴露错误或在文档中明确切到 `eager/sdpa`，不要新增静默 fallback。
5. `gradio_demo.py` 与 SGLang 路径不要作为第一阶段必改对象，除非明确要交付 WebUI 或服务化推理。

## 8. SGLang 路径差异

v1.0 README 新增端到端 SGLang 服务路径：

- SGLang 分支：`https://github.com/OpenMOSS/sglang` branch `moss-ttsd-v1.0-with-cat`，本次记录 HEAD `a0bc33136ee03e424c3f1e9ac513d3c7a4597351`。
- 先下载 `MOSS-TTSD-v1.0` 和 `MOSS-Audio-Tokenizer`。
- 通过 `scripts/fuse_moss_tts_delay_with_codec.py` 生成 fused model。
- 通过 `sglang serve --delay-pattern --trust-remote-code` 启动服务。
- `scripts/request_sglang_generation.py` 发送请求并保存 base64 WAV。

适配建议：第一阶段不要把 SGLang 作为 v1.0 NPU 适配主线。除非已经确认 Ascend/SGLang 对该 OpenMOSS 分支、`--delay-pattern` 和音频 remote code 的支持，否则应先适配原生 HF `AutoModel` / `AutoProcessor` 推理路径。

## 9. 功能与验收差异

| 项 | v0.5 | v1.0 |
|---|---|---|
| 说话人数 | 主要两说话人 `[S1]` / `[S2]` | 官方输入格式支持 `[S1]` 到 `[S5]` |
| 语言能力 | 中英双语 | README 声明支持 20 种语言 |
| 长上下文 | v0.5 长语音生成 | README 声明 v1.0 支持单次 60 分钟上下文 |
| 推荐模式 | v0.5 原脚本单一路径 | v1.0 推荐 `voice_clone_and_continuation` |
| 输出 | wav 文件 | wav 文件 + `output.jsonl`，多卡时先写 rank jsonl 后合并 |
| 评测 | 本仓要求 CPU/NPU 同 checkpoint、同数据、同脚本对比 | README 指向官方 `TTSD-eval`，指标包括 SIM、ACC、WER |

v1.0 验收至少应覆盖：

- `generation`、`continuation`、`voice_clone`、`voice_clone_and_continuation` 四种 mode 的最小冒烟。
- 1/2/3/5 说话人 JSONL，尤其多说话人 prompt_audio/prompt_text 配对检查。
- 中英和若干非中英语言样本，不能只沿用 v0.5 的中英双语集。
- CPU/CUDA 上游路径与 NPU 路径在同 checkpoint、同 JSONL、同采样参数下对比。
- TTSD-eval 或固定 ASR+speaker verification+人工听测，不能只以“生成 wav”作为质量结论。

## 10. 后续 v1.0 适配落地建议

1. 新建独立 v1.0 适配边界，例如 `MOSS-TTSD-v1.0/`，不要把 v1.0 patch 混入当前 v0.5 patch。
2. 固定三类 revision：
   - `OpenMOSS/MOSS-TTSD` main commit；
   - `OpenMOSS-Team/MOSS-TTSD-v1.0` 权重 revision；
   - `OpenMOSS-Team/MOSS-Audio-Tokenizer` codec revision。
3. 优先 patch 原项目已有文件：`inference.py`、`generation_utils.py`；如 remote code 存在 NPU 不兼容，再分别给主模型仓和 codec 仓准备独立 patch。
4. 第一阶段只交付原生 HF 推理 NPU 路径；SGLang、Gradio、服务化接口作为后续专项。
5. 依赖文档中明确区分“上游 CUDA requirements”和“NPU requirements”，不要让用户直接安装 `torch==2.9.1+cu128`。
6. 按项目约束保留必需依赖顶层导入，`torch_npu` 仅在 NPU 路径条件导入；缺少依赖、权重字段或官方评估组件时快速失败。
7. 适配完成后记录所有权重 shard 的 SHA256、CPU/CUDA/NPU 命令、日志、输出样本和质量报告。

## 11. v1.0 参考命令草案

以下命令是后续适配目标形态草案，当前上游未提供 `--device npu`，不能直接执行：

```bash
cd MOSS-TTSD-v1.0/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --model_path weights/MOSS-TTSD-v1.0 \
  --codec_model_path weights/MOSS-Audio-Tokenizer \
  --input_jsonl examples_v1/smoke.jsonl \
  --save_dir outputs_npu \
  --mode voice_clone_and_continuation \
  --batch_size 1 \
  --device npu \
  --device_id 0 \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --text_normalize \
  --sample_rate_normalize
```

如果正式适配时发现 NPU 不支持某个 attention 或 codec op，应在 patch/文档中明确失败点和替代官方路径；不要加入 CPU fallback、第三方 codec 或非官方评测替代。
