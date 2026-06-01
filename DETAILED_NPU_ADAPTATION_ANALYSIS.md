# Ascend-SACT 语音/音频模型 NPU 适配仓库详细评估报告

分析日期：2026-05-22  
分析范围：`/home/pei/ModelZoo` 下已克隆的 12 个 GitCode 仓库。  
分析方式：静态代码/文档分析，未在 Ascend NPU 上实际执行。判断依据来自各仓库 `README.md`、随仓脚本、`requirements.txt`、大文件/LFS 状态、上游工程说明以及是否已经形成类似 `Canary-1B/README.md`、`Canary-1B/README_INFERENCE.md` 的可交付推理文档。

## 0. 统一评价口径

### 0.1 复杂度标签

| 标签 | 含义 |
|---|---|
| 低 | 已有可直接执行脚本或清晰命令；主要工作是下载权重、替换路径、小样本冒烟和补文档。通常 0.5-2 人日可完成功能闭环。 |
| 中 | 需要补 CLI、批处理、评测脚本、数据准备脚本、上游集成或依赖版本修正。通常 2-5 人日。 |
| 高 | 需要多仓联调、多处源码补丁、复杂依赖/编译、大模型或主观评价方案。通常 5-10 人日。 |
| 很高 | 存在重大阻塞，如大文件未拉取、多卡/专用容器、上游代码大量 CUDA 假设、评价体系主观且需人工听测。通常超过 10 人日或需专项资源。 |

### 0.2 每个仓库交付件要求

后续每个仓库统一按 6 个子标题描述：

1. **仓库观察与判断依据**：说明当前仓库是否有脚本、patch、README、权重/数据说明、上游依赖和适配风险。
2. **后续适配**：说明在当前适配基础上还要做什么，是否需要补 CLI、patch、数据准备、评测脚本和推理指导文档。
3. **功能验证**：说明是否已有功能验证脚本，验证数据从哪里来、如何获取，权重如何获取，验收看哪些输出。
4. **性能验证**：参考源仓库/论文/官方模型卡或同 checkpoint 官方实现，至少选一个公开数据集与原仓库 CPU/CUDA 路径做对比；说明是否已有性能脚本，数据来源和生成方式。
5. **精度验证**：参考源仓库/论文/官方模型卡指标，至少选一个公开数据集与原仓库 CPU/CUDA 路径做对比；说明是否已有精度脚本，数据来源和生成方式。
6. **数据集获取**：汇总功能、性能、精度所需数据 URL、下载工具、可离线复用要求和注意事项。

### 0.3 统一工程标准

- 推理入口建议统一命名为 `infer.py` 或 `infer_npu.py`；评测入口建议统一命名为 `eval_*.py` 或 `benchmark.py`；数据准备入口建议统一命名为 `prepare_eval_data.py`。
- 数据准备必须支持“指定目录、存在即复用、缺失才下载、`--offline` 下不联网”。
- 权重下载必须写明官方 URL/ModelScope/Hugging Face 模型名、目标目录和完整性校验方式。
- 精度/性能对比不能只写“跑通”，必须写明对比对象：同 checkpoint、同数据、同评测脚本下的 CPU/CUDA/源仓结果。
- 不添加未验证的 CPU fallback、远程下载 fallback、非官方指标替代官方指标；缺失依赖或缺失官方字段应快速失败并暴露原始错误。

---
## 1. DNSMOS

### 1.1 仓库观察与判断依据

- 随仓包含 `DNSMOS/infer.py`，README 中也粘贴了完整推理脚本，当前功能链路最完整。
- 适配后端是 `onnxruntime` 的 `CANNExecutionProvider`，不是简单 `cuda -> npu` 字符串替换。
- 脚本已有 NPU 环境检查、`npu-smi info`、性能模式切换、批量处理和 CSV 输出。
- README 已给出 VCC2018 converted speech 下载命令：`https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz`。
- 权重要求来自 DNSMOS 官方 ONNX 文件：`DNSMOS/model_v8.onnx`、`DNSMOS/sig_bak_ovr.onnx`，可选 `pDNSMOS/sig_bak_ovr.onnx`。

### 1.2 后续适配

- 复杂度：低。
- 在当前脚本基础上补 `--device_id`、`--disable_performance_mode`、权重完整性检查和更清晰的错误信息。
- 固化最小依赖：`onnxruntime-cann`、CANN、`numpy`、`librosa`、`soundfile`、`pandas`，不要把完整开发环境冻结成部署依赖。
- 增加 CPU ONNXRuntime baseline 入口或 `--provider CPUExecutionProvider`，用于 NPU/CANN 输出一致性对比。
- 形成 Canary-1B 风格的 `README_INFERENCE.md`：环境表、权重下载、数据准备、单条/批量推理、性能/精度命令。

### 1.3 功能验证

- 已有验证脚本：有，`DNSMOS/infer.py` 可直接做功能验证。
- 验证数据来源：README 推荐 VCC2018 converted speech；冒烟可用任意 16 kHz/48 kHz WAV 或自录 WAV。
- 数据获取：`wget` 下载 VCC2018 压缩包后解压到 `DNSMOS/dataset/vcc2018`。
- 权重获取：从 DNSMOS 官方仓库或当前 README 指定的 DNSMOS 目录准备 ONNX 权重，放到 `--model_root` 下的 `DNSMOS/`、可选 `pDNSMOS/`。
- 验证命令：`python infer.py -t ./dataset/vcc2018 -o ./csv/vcc2018.csv --model_root . --batch_size 4`。
- 验收：CSV 包含 `filename,len_in_sec,MOS_SIG,MOS_BAK,MOS_OVRL,P808_MOS`，MOS 在 0-5 合理范围，短音频 padding、长音频分段和批量目录均不报错。

### 1.4 性能验证

- 已有性能脚本：部分已有；`infer.py` 有批处理和耗时统计基础，但还需明确端到端/纯 ONNX 推理分段计时。
- 对比对象：同一 DNSMOS ONNX 权重在源仓 CPU ONNXRuntime 与 NPU CANNExecutionProvider 的吞吐/延迟。
- 对比数据集：至少使用 VCC2018 converted speech 中固定 100/1000 条 WAV；如需要更稳定，可额外构造 9 秒、30 秒、60 秒三档音频清单。
- 数据生成：从 VCC2018 解压目录扫描 WAV，生成固定 manifest；不要每次随机抽样。
- 指标：端到端耗时、纯推理耗时、音频秒/秒、平均单文件耗时、P95、HBM/RSS、batch size 1/2/4/8。
- 参考源仓：DNSMOS 官方实现通常以 CPU/ONNX 推理为参考；NPU 通过线应是同权重同数据下输出一致且吞吐优于或不低于 CPU baseline。

### 1.5 精度验证

- 已有精度脚本：无独立精度脚本，需要新增 `eval_dnsmos.py` 或在 `infer.py` 增加 baseline 对齐模式。
- 对比对象：官方 DNSMOS ONNXRuntime CPU 输出；如果有主观 MOS 标注，再对比 DNSMOS 论文/官方报告的 MOS 相关性口径。
- 对比数据集：最低要求用 VCC2018 固定子集做 NPU vs CPU 数值一致性；正式相关性需带人工 MOS 标签的数据，不能用无标注 WAV 直接宣称 MOS 精度。
- 指标：`MOS_SIG/MOS_BAK/MOS_OVRL/P808_MOS` 的 MAE、最大绝对误差、Pearson/Spearman；建议 MAE < 1e-3，CANN 浮点差异较大时可放宽到 1e-2 并说明。
- 验收：NPU 与 CPU 排序相关 > 0.99，无 NaN、无空 CSV、同一输入重复运行结果稳定。

### 1.6 数据集获取

- VCC2018 converted speech：`https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz`。
- 冒烟数据：任意公开 WAV 或本地生成 1/9/30 秒正弦/语音 WAV。
- 权重：DNSMOS 官方 ONNX 文件，目标结构为 `DNSMOS/model_v8.onnx`、`DNSMOS/sig_bak_ovr.onnx`、可选 `pDNSMOS/sig_bak_ovr.onnx`。
- 建议规模：冒烟 10 条；性能 1000 条；精度一致性 500-1000 条。

---
## 2. Index-TTS-2

### 2.1 仓库观察与判断依据

- 随仓有 `infer_v2.py` 和 `run_web.sh`，README 指向上游 `https://github.com/triomino/index-tts.git`。
- 环境要求较高：CANN 8.3.RC1、Python 3.11.13、torch 2.8.0、torch_npu v2.8.0-7.2.0，并要求编译 GCC 13.3.0、CMake 3.31.0、torch_npu。
- README 要下载 IndexTTS-2、w2v-bert-2.0、MaskGCT、campplus、BigVGAN 多组权重。
- README 说明需改 `infer_v2.py`、新增/修改 `infer_v2_npu.py`、修改 `webui.py`，说明当前还不是完整可复现 patch 交付。
- README 已给出 warmup 后 RTF 约 0.6 的性能描述，但缺少标准测试集和精度/音质指标。

### 2.2 后续适配

- 复杂度：高。
- 把 README 中所有手工修改固化为 patch，覆盖上游 index-tts、WebUI、vocoder、speaker/emotion encoder 相关文件。
- 新增非 WebUI CLI：`--text --speaker_audio --emotion_audio --model_dir --output --device --seed --duration`。
- 增加 `check_assets.py`，统一检查 5 组权重和配置文件是否存在。
- 整理最小依赖和编译步骤；对无法避免的 GCC/CMake/torch_npu 编译写版本矩阵。
- 建议交付 `README_INFERENCE.md`，结构参考 Canary-1B，明确权重下载、示例数据、单条推理、批量性能和精度命令。

### 2.3 功能验证

- 已有验证脚本：有初始 `infer_v2.py`/`run_web.sh`，但缺少标准化 CLI 验证脚本。
- 验证数据来源：上游 examples、README 中示例文本、少量自备中文参考音频；音色/情感验证需准备 speaker prompt 和 emotion prompt。
- 权重获取：按 README 使用 ModelScope：`IndexTeam/IndexTTS-2`、`facebook/w2v-bert-2.0`、`amphion/MaskGCT`、`iic/speech_campplus_sv_zh-cn_16k-common`、`nv-community/bigvgan_v2_22khz_80band_256x`。
- 验证内容：自由生成、精确时长控制、零样本音色、情感控制、WebUI 上传/下载。
- 验收：输出 WAV 可播放、采样率/时长正确、非全零/非 NaN，文本短中长均可生成，WebUI 和 CLI 路径一致。

### 2.4 性能验证

- 已有性能脚本：没有独立 benchmark，需要新增 `benchmark_tts.py`。
- 对比对象：上游 index-tts 在同 checkpoint、同 prompt、同采样参数下的 CUDA/CPU 或官方报告；当前 README 的 warmup 后 RTF≈0.6 只能作为本仓 NPU 复现目标，不是官方基线。
- 对比数据集：至少用 AISHELL-3 或 CSMSC 抽取 50 条文本/参考音频；也可先用固定 20 条中文短中长文本和 3 个 speaker prompt 建立小基准。
- 数据生成：固定文本清单、speaker prompt 路径、seed、采样参数，生成 manifest。
- 指标：cold/warm RTF、首音频延迟、总生成耗时、P50/P95、显存、失败率；文本长度和 prompt 长度分桶统计。

### 2.5 精度验证

- 已有精度脚本：无，需要新增 `eval_tts.py`。
- 对比对象：源仓 CUDA/官方输出；生成式 TTS 不要求波形逐点一致，但要比较同文本同 prompt 下的可懂度、音色和自然度。
- 对比数据集：至少使用 CSMSC 或 AISHELL-3 子集；若验证情感控制，需补带情感标签的公开语音或人工标注 prompt。
- 指标：ASR 回识别 CER/WER、speaker embedding cosine similarity、目标时长误差、DNSMOS/UTMOS 或人工 MOS/CMOS；必要时做 A/B 偏好测试。
- 验收：NPU 相比源仓 CUDA 在 CER、音色相似度、时长误差上无显著退化；人工抽听无系统性杂音、断句、漏字。

### 2.6 数据集获取

- 上游代码：`https://github.com/triomino/index-tts.git`。
- 权重：README 已列 ModelScope 命令，目标目录分别为 `checkpoints/`、`models/facebook/w2v-bert-2.0/`、`models/amphion/MaskGCT/`、`models/iic/speech_campplus_sv_zh-cn_16k-common/`、`models/nv-community/bigvgan_v2_22khz_80band_256x/`。
- 功能数据：上游 examples 和自备 speaker/emotion prompt。
- 精度/性能数据：CSMSC、AISHELL-3；情感评估另需带情感标签的数据或人工标注。
- 建议规模：冒烟 10 条；性能 50-100 条；精度 100-500 条并人工抽检。

---
## 3. MMAudio

### 3.1 仓库观察与判断依据

- 仓库主要是 README 和截图，`mmaudio.tar.gz` 是 Git LFS 指针，真实内容约 10GB，当前未拉取。
- README 指向上游 `https://github.com/hkchengrex/MMAudio`，可选 Gitee 镜像 `https://gitee.com/MufcLiuKai/MMAudio`。
- 适配涉及多处手工修改：`cuda` 改 `npu`、CLIP 本地模型、VAE、BigVGAN filter/resample/bigvgan dtype 等。
- 需要 apple CLIP、nvidia BigVGAN 等额外模型，README 还要求 G8600/910B2C 2 卡。
- 任务是视频/文本到音频生成，功能和精度验证都依赖主观/客观生成质量评估。

### 3.2 后续适配

- 复杂度：高。
- 首先拉取 Git LFS 大包或改为明确 patch，不应依赖截图交付。
- 将 README 修改整理成统一 patch，覆盖 `demo.py`、`features_utils.py`、`vae.py`、BigVGAN alias-free activation 等。
- 增加 `--device npu:0`、`--dtype float32/bf16`、`--clip_dir`、`--bigvgan_dir`、`--output` 等参数。
- 明确单卡是否可运行；如果必须 2 卡，需要说明模块切分和资源要求。
- 建立 op/dtype 不支持清单，避免静默 CPU fallback。

### 3.3 功能验证

- 已有验证脚本：README 指向上游 demo，但当前仓库缺少完整可执行代码包和本地 patch。
- 验证数据来源：手写文本 prompt、短视频样例；上游 MMAudio demo assets 如可用应优先使用。
- 权重获取：apple/DFN5B-CLIP-ViT-H-14-378 可按 README 从 GitCode/ModelScope 获取；MMAudio/BigVGAN 权重按上游 README 或 ModelScope/HF 下载。
- 验证内容：文生音频、视频生音频、空 prompt/长 prompt/短视频/长视频异常输入。
- 验收：输出 WAV/视频音轨存在，采样率和时长符合参数，无全零、爆音、NaN，日志无严重 NPU op 报错。

### 3.4 性能验证

- 已有性能脚本：无，需要新增 `benchmark_mmaudio.py`。
- 对比对象：源仓 MMAudio CUDA 路径或论文/官方 demo 在同模型同 seed 下的耗时；若源仓没有硬件表，则以本地 CUDA/CPU 同数据为基线。
- 对比数据集：至少选 AudioCaps 或 VGGSound/Clotho 子集中的 50 条 prompt/视频；小规模可先构造 10 个固定 5s/10s/30s prompt/视频。
- 数据生成：统一视频分辨率、音频目标时长、seed、采样步数和模型配置，生成 manifest。
- 指标：RTF、端到端延迟、CLIP/BigVGAN/VAE 分段耗时、2 卡利用率、显存峰值；记录 float32 替换 bf16 后的性能损失。

### 3.5 精度验证

- 已有精度脚本：无，需要新增 `eval_mmaudio.py`。
- 对比对象：源仓 CUDA 输出和论文推荐指标；生成式音频不能以逐点波形作为唯一标准。
- 对比数据集：至少使用 AudioCaps 或 Clotho 做文本-音频一致性；视频任务可用 VGGSound 子集。
- 指标：CLAPScore、FAD、mel L1/频谱差异、人工 MOS/A-B 偏好；视频输入还需评估音画一致性。
- 验收：NPU 相比源仓 CUDA 在 CLAPScore/FAD 和人工抽听上无明显退化；dtype 改动不引入系统性杂音。

### 3.6 数据集获取

- 上游代码：`https://github.com/hkchengrex/MMAudio`；备选镜像 `https://gitee.com/MufcLiuKai/MMAudio`。
- CLIP：`apple/DFN5B-CLIP-ViT-H-14-378`，README 给出 GitCode 和 ModelScope 下载方式。
- 精度/性能数据：AudioCaps、Clotho、VGGSound；注意版权、下载体积和 YouTube 链接失效问题。
- 建议规模：冒烟 10 条；性能 50 条；精度 200-1000 条或人工抽检 50 条。

---
## 4. MOSS-Speech

### 4.1 仓库观察与判断依据

- 随仓有 `infer.py` 和超长 `requirements.txt`。
- README 指向三套资产：ModelScope `openmoss/MOSS-Speech`、ModelScope `AI-ModelScope/MOSS-Speech-Codec`、Hugging Face Space `OpenMOSS-Team/MOSS-Speech`。
- README 要修改 `diffusers`、`transformers` 源码，以及 Space 内 Matcha-TTS、CosyVoice HiFiGAN 文件。
- `infer.py` 中仍可见 `device="cuda"`、`device_map="cuda"`、`.to("cuda")`，依赖 `transfer_to_npu` 自动迁移，适配显式性不足。
- 任务是语音对话大模型，输出可能包含音频和文本，链路跨 LLM、codec、TTS、Whisper 特征和 vocoder。

### 4.2 后续适配

- 复杂度：高。
- 将 `infer.py` 设备参数显式化为 `--device npu/cpu/cuda`，移除硬编码 CUDA 设备字符串。
- 把 diffusers、transformers、Matcha-TTS、HiFiGAN 修改做成 patch，并绑定可复现版本。
- 增加 `--model_dir --codec_dir --space_dir --prompt_audio --text --output_dir` 等 CLI 参数。
- 梳理 README 中 CPU 执行的 Whisper 特征和 `istft`，明确这是否是必要路径，不能静默 fallback。
- 精简 `requirements.txt`，拆分最小推理依赖和完整开发环境。

### 4.3 功能验证

- 已有验证脚本：有 `infer.py`，但当前仍需大量外部源码补丁和权重路径准备后才能验证。
- 验证数据来源：官方示例 prompt、少量中文对话文本、自备 prompt audio。
- 权重获取：ModelScope `openmoss/MOSS-Speech`、`AI-ModelScope/MOSS-Speech-Codec`，Hugging Face Space `OpenMOSS-Team/MOSS-Speech` 代码/资源。
- 验证内容：文本输入生成文本、文本输入生成音频、带 prompt audio 的音色延续、多轮上下文、缺权重/错音频格式异常。
- 验收：文本非空，音频可播放且采样率正确，无明显截断/全零/NaN，错误路径暴露原始异常。

### 4.4 性能验证

- 已有性能脚本：无，需要新增 `benchmark_moss_speech.py`。
- 对比对象：源仓/Space 官方 CUDA 路径；如没有公开硬件报告，则用同 checkpoint 同输入的本地 CPU/CUDA 作为基线。
- 对比数据集：自建 20 条标准对话请求，覆盖短/长文本、是否生成音频、是否带 prompt audio；正式可加入语音对话 benchmark 子集。
- 数据生成：固定 prompt 文本、prompt audio、随机种子和生成参数。
- 指标：首 token、首音频、端到端延迟、RTF、LLM/codec/vocoder/CPU 特征各阶段耗时、显存和 RSS。

### 4.5 精度验证

- 已有精度脚本：无，需要分文本和音频两类新增评测。
- 对比对象：源仓 CUDA 输出；语音对话没有单一标准答案，不能只看数值一致。
- 对比数据集：文本任务可用自建固定问答集加人工验收；音频任务可用 TTS 公开语料抽样构造 prompt+文本。
- 指标：文本任务成功率/人工相关性，音频 ASR 回识别 CER/WER、speaker embedding 相似度、DNSMOS/UTMOS、人工 MOS/CMOS。
- 验收：NPU 输出相对源仓 CUDA 不出现系统性漏字、音色崩坏、明显噪声或延迟异常。

### 4.6 数据集获取

- 权重：`https://modelscope.cn/models/openmoss/MOSS-Speech`、`https://modelscope.cn/models/AI-ModelScope/MOSS-Speech-Codec`。
- Space 代码：`https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech/tree/main`。
- 功能数据：官方示例 prompt、自建中文对话、prompt audio。
- 精度数据：可从 CSMSC/AISHELL-3 抽 prompt+文本，正式对话能力需要人工评价协议。
- 建议规模：冒烟 5 条；功能 50 条；精度人工 50-100 条。

---
## 5. Canary-1B

### 5.1 仓库观察与判断依据

- 当前已形成较完整交付件：`README.md`、`README_INFERENCE.md`、`infer.py`、`eval_canary.py`、`prepare_eval_data.py`。
- README 明确上游 NeMo commit `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`，模型权重为 Hugging Face `nvidia/canary-1b` 的 `canary-1b.nemo`，并记录 SHA256。
- 当前适配不修改 NeMo 上游文件，没有 `.patch`；推理脚本默认 `--device npu`，CPU 验证使用 `--device cpu`。
- 已明确 `ASCEND_RT_VISIBLE_DEVICES` 控制卡号，不写死 `npu:0`。
- README 已包含官方精度表、Open ASR Leaderboard 性能参考、MLS/LibriSpeech/FLEURS 数据准备和在线/离线复用方案，是其他仓库的参考模板。

### 5.2 后续适配

- 复杂度：中，当前已基本完成工程化。
- 后续主要是把实际 NPU 环境运行结果补回 README：CANN/驱动/torch_npu 版本、batch size、beam size、RTF/RTFx、WER/BLEU。
- 补充 NPU 失败样例和常见错误排查，例如 NeMo 版本、`.nemo` 权重路径、`torchcodec`/音频解码依赖。
- 如需发布到统一 ModelZoo 目录，保持 `README_INFERENCE.md` 的路径命令不依赖本地绝对路径。

### 5.3 功能验证

- 已有验证脚本：有，`infer.py` 支持单条/多条音频；`eval_canary.py` 支持 manifest。
- 验证数据来源：单条 smoke 使用 PyTorch torchaudio tutorial WAV；正式数据使用 LibriSpeech、MLS、FLEURS。
- 权重获取：`https://huggingface.co/nvidia/canary-1b/resolve/main/canary-1b.nemo` 或 `huggingface_hub.snapshot_download("nvidia/canary-1b")`。
- 验证命令：README_INFERENCE 中已给出 ASR/AST 单条命令，参数包括 `--task --source_lang --target_lang --pnc --batch_size --beam_size`。
- 验收：ASR/AST 输出文本非空，语言/任务参数生效，manifest 批量可运行，CPU/NPU 都能加载同一 `.nemo` 权重。

### 5.4 性能验证

- 已有性能脚本：有，`eval_canary.py --performance_mode`。
- 对比对象：Hugging Face Open ASR Leaderboard 的 `nvidia/canary-1b` A100 公开 RTFx，以及同 checkpoint 本地 CPU/CUDA/NeMo 路径。
- 对比数据集：至少使用 LibriSpeech `test-clean`；该数据在 README_INFERENCE 中通过 `prepare_eval_data.py --task librispeech` 准备。
- 指标：`elapsed_seconds`、`rtf`、`RTFx=audio_seconds/elapsed_seconds`、batch size、beam size、峰值 HBM/RSS。
- 注意：Open ASR Leaderboard 的 A100 RTFx 只能作为公开 GPU 量级参考，不应直接作为 NPU 通过线。

### 5.5 精度验证

- 已有精度脚本：有，`eval_canary.py`。
- 对比对象：NVIDIA model card 官方 ASR WER/AST BLEU 表、Open ASR Leaderboard WER，以及本地 CPU/CUDA 同 checkpoint 结果。
- 对比数据集：ASR 使用 MLS/LibriSpeech；AST 使用 FLEURS，必要时 CoVoST-v2。
- 数据生成：`prepare_eval_data.py` 支持在线下载、本地目录复用和 `--offline` 禁止联网。
- 指标：ASR WER/CER，AST BLEU/chrF，NPU vs CPU 文本一致率；正式精度建议 `beam_size=5`、`length_penalty=1.0` 对齐官方口径。

### 5.6 数据集获取

- 权重：Hugging Face `nvidia/canary-1b`，文件 `canary-1b.nemo`。
- Smoke WAV：`https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav`。
- LibriSpeech：`https://www.openslr.org/12`。
- MLS：`https://huggingface.co/datasets/facebook/multilingual_librispeech`。
- FLEURS：`https://huggingface.co/datasets/google/fleurs`。
- 建议规模：冒烟 1-10 条；性能 LibriSpeech test-clean；精度 MLS/FLEURS 按语种子集或全量。

---
## 6. FireRedASR-AED

### 6.1 仓库观察与判断依据

- 随仓有 `infer.py`，README 指向上游 `https://github.com/FireRedTeam/FireRedASR.git` 和 ModelScope 权重 `FireRedASR-AED-L`。
- README 要把 `infer.py` 移动到官方仓，修改 `batch_uttid` 和 `batch_wav_path`，当前脚本仍偏示例。
- `infer.py` 使用 `transfer_to_npu`，调用 `FireRedAsr.from_pretrained('aed', ...)` 后 `model.transcribe`。
- 任务为中文 ASR，输出文本明确，CER/WER 指标成熟。
- README 已吸收 Canary-1B 的数据准备要求，强调 AISHELL/LibriSpeech 等正式评测需准备数据和评测分离。

### 6.2 后续适配

- 复杂度：中。
- CLI 化：`--model_dir --input_wav --manifest --output_csv --batch_size --device --dtype`。
- 增加批量目录递归、音频采样率检查、输出 JSON/CSV。
- 新增 `prepare_eval_data.py` 或脚本化复用现有 `scripts/prepare_librispeech_test_clean.sh`，支持 AISHELL-1 test。
- 新增 `eval_asr.py`，支持 CER/WER 和文本规范化配置。
- 明确 NPU device 设置，减少对 `transfer_to_npu` 的隐式依赖。

### 6.3 功能验证

- 已有验证脚本：有 `infer.py`，但需要去硬编码。
- 验证数据来源：官方 example `BAC009S0764W0121.wav`、自备中文 WAV、AISHELL-1 小样本。
- 权重获取：按 README 从 ModelScope 下载 `FireRedASR-AED-L` 到本地 `pretrained_models/` 或 `weights/`。
- 验证内容：单文件、目录、manifest、多采样率/长短音频。
- 验收：转写文本非空，批量成功率 100%，输出 CSV/JSON 包含 uttid、wav、text、耗时，异常音频报错清晰。

### 6.4 性能验证

- 已有性能脚本：无，需要新增 `benchmark_asr.py` 或给 `eval_asr.py` 增加 `--performance_mode`。
- 对比对象：FireRedASR 官方 CPU/CUDA 路径或论文/官方公开指标；最低要求同 checkpoint 本地 CPU/CUDA 与 NPU 对比。
- 对比数据集：至少使用 AISHELL-1 test；也可补 LibriSpeech test-clean 做英文兼容但中文模型主线应以 AISHELL 为准。
- 数据生成：从 AISHELL wav.scp/text 生成 manifest，固定 batch size 1/4/8。
- 指标：RTF、音频秒/秒、平均/P95 延迟、显存、特征提取/模型/解码分段耗时。

### 6.5 精度验证

- 已有精度脚本：无，需要新增 CER/WER 评测。
- 对比对象：FireRedASR 官方 AED-L 公开结果或同 checkpoint CPU/CUDA 输出。
- 对比数据集：AISHELL-1 test 是最低要求；可扩展 WenetSpeech/MagicData 子集。
- 指标：CER、WER、空转写率、NPU/CPU 文本一致率；中文评测需固定文本规范化规则。
- 验收：NPU CER 相比源仓 CPU/CUDA 无显著退化；解码差异需逐条输出 diff。

### 6.6 数据集获取

- 上游代码：`https://github.com/FireRedTeam/FireRedASR.git`。
- 权重：ModelScope `FireRedASR-AED-L`，按 README 下载到本地模型目录。
- 功能数据：官方 example 或任意中文 WAV。
- 精度/性能数据：AISHELL-1 test；可扩展 WenetSpeech、MagicData。
- 建议规模：冒烟 10 条；性能 1h/10h 两档；精度 AISHELL-1 test 全量。

---
## 7. MossFormer2_SE_48K

### 7.1 仓库观察与判断依据

- 仓库主要为 README，没有随仓推理脚本文件。
- README 指向上游 `https://github.com/modelscope/ClearerVoice-Studio.git`，要求下载 ModelScope `iic/ClearerVoice-Studio` checkpoints。
- 示例只展示 `torch.npu.is_available()` 和 `device = torch.device('npu:0')`，未完整展示 ClearVoice 内部模块 NPU 迁移 patch。
- 任务是 48 kHz 语音增强，输入输出 WAV，功能验证相对简单，但精度需要成对干净/带噪参考。
- README 说明 ModelScope 下载会包含 SE_48K、SS_16K、SR_48K 等全部模型。

### 7.2 后续适配

- 复杂度：中。
- 将 README 示例落为 `infer_npu.py`，支持 `--input --output --model_dir --device --sample_rate --batch_dir`。
- 检查 ClearerVoice 内部是否全部 `.to(device)`，必要时提供 patch。
- 新增 48 kHz 输入检查、重采样策略或快速失败策略，并记录选择原因。
- 新增 `eval_se.py`，支持 PESQ/STOI/SI-SDR/DNSMOS；新增 `benchmark_se.py`。
- 形成推理指导文档，说明 checkpoints 目录结构。

### 7.3 功能验证

- 已有验证脚本：无随仓脚本，仅 README 示例，需要补 `infer_npu.py`。
- 验证数据来源：ClearerVoice samples、自备带噪 WAV、合成噪声 WAV。
- 权重获取：`modelscope download --model iic/ClearerVoice-Studio --local_dir ./checkpoints`。
- 验证内容：单条 48 kHz 输入、16 kHz/44.1 kHz/48 kHz 输入、目录批处理。
- 验收：输出 WAV 存在，采样率 48 kHz 或符合明确策略，时长差 < 10 ms，无全零/NaN，听感噪声下降。

### 7.4 性能验证

- 已有性能脚本：无，需要新增 `benchmark_se.py`。
- 对比对象：ClearerVoice-Studio 源仓 CPU/CUDA 路径或官方报告；最低要求同 checkpoint 同音频 CPU/CUDA 与 NPU 对比。
- 对比数据集：至少用 VoiceBank+DEMAND 或 DNS Challenge 子集，固定 5s/30s/60s 三档。
- 数据生成：如原始数据非 48 kHz，需固定重采样脚本并记录；性能 manifest 不应随机变化。
- 指标：RTF、音频秒/秒、P50/P95、显存、CPU 占用、分块 overlap-add 耗时。

### 7.5 精度验证

- 已有精度脚本：无，需要新增 `eval_se.py`。
- 对比对象：ClearerVoice-Studio 官方/源仓输出，同 checkpoint CPU/CUDA 输出。
- 对比数据集：VoiceBank+DEMAND、DNS Challenge，或干净语音+噪声合成的成对数据。
- 指标：PESQ、STOI、SI-SDR、DNSMOS；NPU vs CPU/GPU 的波形/频谱差异。
- 注意：部分指标对采样率有要求，48 kHz 数据可能需按官方口径重采样后计算，不能静默改变指标定义。

### 7.6 数据集获取

- 上游代码：`https://github.com/modelscope/ClearerVoice-Studio.git`。
- 权重：ModelScope `iic/ClearerVoice-Studio`，目标 `./checkpoints`。
- 功能数据：ClearerVoice samples、自备 WAV。
- 精度/性能数据：VoiceBank+DEMAND、DNS Challenge；也可用干净语音与噪声按固定 SNR 合成并重采样到 48 kHz。
- 建议规模：冒烟 5 条；性能 100 条；精度 500-1000 对。

---
## 8. pyannote-speaker-diarization-3.1

### 8.1 仓库观察与判断依据

- 仓库主要为 README，无随仓 `infer.py`，但 README 给出完整示例脚本。
- 需要下载 `speaker-diarization-3.1`、`segmentation-3.0`、`wespeaker-voxceleb-resnet34-LM` 三个模型。
- README 要修改 `config.yaml` 中 embedding/segmentation 路径。
- 已知问题包括 `torchaudio.compliance.kaldi.py` complex abs 不支持和 numpy `np.NaN` 兼容。
- 输出为说话人分段，标准精度指标 DER/JER，需要 RTTM 标注和评测协议。

### 8.2 后续适配

- 复杂度：中。
- 将 README 示例落为 `infer_npu.py`，支持 `--wav --config --output_rttm --device`。
- 提供 `rewrite_config.py`，自动填本地 segmentation/embedding 路径。
- 将 torchaudio/numpy 修改整理为 patch 或严格版本约束。
- 新增 `eval_der.py`，支持参考 RTTM、collar、overlap 参数。
- 新增长音频性能 benchmark，分段统计 segmentation、embedding、clustering。

### 8.3 功能验证

- 已有验证脚本：README 示例有，但仓库无脚本文件，需要补齐。
- 验证数据来源：自制单说话人/双说话人 WAV、公开会议 sample。
- 权重获取：按 README 用 ModelScope 下载 `pyannote/speaker-diarization-3.1`、`pyannote/segmentation-3.0`、`pyannote/wespeaker-voxceleb-resnet34-LM`。
- 验证内容：单说话人、双说话人、三说话人、重叠语音、多声道输入。
- 验收：输出 RTTM 非空，格式可被 `dscore` 或 `pyannote.metrics` 读取，说话人数和分段大致合理。

### 8.4 性能验证

- 已有性能脚本：无，需要新增 `benchmark_diarization.py`。
- 对比对象：pyannote 官方 CPU/CUDA pipeline 或同 checkpoint 本地 CPU/CUDA 结果。
- 对比数据集：至少使用 AMI 会议子集；性能可选 5min/30min/1h 三档会议音频。
- 数据生成：统一音频采样率、单声道处理、manifest 和 RTTM 输出目录。
- 指标：RTF、每小时音频处理时间、segmentation/embedding/clustering 分段耗时、显存和 CPU 后处理占比。

### 8.5 精度验证

- 已有精度脚本：无，需要新增 DER/JER 评测。
- 对比对象：pyannote 官方 pipeline 在同数据上的 CPU/CUDA DER，或论文/model card 公开结果。
- 对比数据集：AMI、AISHELL-4、AliMeeting 或 VoxConverse，至少选一个带 RTTM 的公开数据集。
- 指标：DER、JER、miss/false alarm/confusion，明确 collar 0.25s/是否忽略 overlap。
- 验收：NPU DER/JER 与源仓 CPU/CUDA 对齐；聚类随机性需固定 seed 或多次运行统计。

### 8.6 数据集获取

- 权重：ModelScope `pyannote/speaker-diarization-3.1`、`pyannote/segmentation-3.0`、`pyannote/wespeaker-voxceleb-resnet34-LM`。
- 功能数据：自制两人对话或公开 sample。
- 精度数据：AMI、AISHELL-4、AliMeeting、VoxConverse，需 RTTM 标注。
- 性能数据：长会议音频，可从上述数据抽取 5min/30min/1h。
- 建议规模：冒烟 5 条；精度 10-100 场会议；性能 10 小时以上。

---
## 9. BUTSpeechFIT-DiariZen

### 9.1 仓库观察与判断依据

- 仓库主要为 README，没有随仓 `infer.py`。
- 依赖上游 `https://github.com/BUTSpeechFIT/DiariZen`、Hugging Face `BUT-FIT/diarizen-wavlm-large-s80-md`、submodule/工具 `dscore`。
- README 要安装/编译多套 pyannote 相关包，并修改 torchaudio kaldi complex abs。
- 还涉及 numpy `np.NaN` 兼容和本地模型软链接路径问题。
- 与 pyannote-speaker-diarization 相比，依赖更重、下载来源更多、工程化程度更低。

### 9.2 后续适配

- 复杂度：高。
- 补 `infer_npu.py` 和 `eval_der.py`，支持 wav 输入、RTTM 输出、参考 RTTM 评估。
- 固定 DiariZen、pyannote、dscore 版本，形成可复现 `requirements` 和 patch。
- 将 torchaudio/numpy 修改变成 patch 或版本约束。
- 去除软链接中个人路径假设，全部改为 `--model_dir` 和配置文件。
- 如果与 pyannote 仓库二选一推进，建议优先 pyannote-speaker-diarization-3.1；DiariZen 作为专项。

### 9.3 功能验证

- 已有验证脚本：无随仓脚本；README 参考 Hugging Face 使用说明创建 `infer.py`。
- 验证数据来源：DiariZen example，例如 `example/EN2002a_30s.wav`，以及自制 2/3/4 说话人样本。
- 权重获取：`git clone https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md.git`，必要时设置 `HF_ENDPOINT=https://hf-mirror.com`。
- 验证内容：冒烟输出 speaker turns、RTTM 格式、重叠语音和多说话人。
- 验收：RTTM 可被 dscore 读取，结果非空，说话人数合理，缺模型或路径错误快速失败。

### 9.4 性能验证

- 已有性能脚本：无。
- 对比对象：DiariZen 官方/源仓 CUDA 路径，或同 checkpoint 本地 CPU/CUDA 结果。
- 对比数据集：至少使用 AMI 或 VoxConverse 子集；性能用 30s/5min/30min 会议音频。
- 数据生成：准备 wav manifest 和输出 RTTM 目录，固定聚类/阈值参数。
- 指标：RTF、segmentation/embedding/clustering 耗时、显存、CPU 后处理占比。

### 9.5 精度验证

- 已有精度脚本：无，需要 `eval_der.py` 调 dscore 或 pyannote.metrics。
- 对比对象：官方 DiariZen 报告或源仓 CPU/CUDA DER。
- 对比数据集：AMI、DIHARD、VoxConverse、AliMeeting 至少选一个。
- 指标：DER、JER、miss/FA/confusion，明确 collar/overlap 策略。
- 验收：NPU 与源仓同 checkpoint 输出差异可解释，DER 不显著退化。

### 9.6 数据集获取

- 上游代码：`https://github.com/BUTSpeechFIT/DiariZen.git`。
- 权重：Hugging Face `BUT-FIT/diarizen-wavlm-large-s80-md`。
- dscore：README 使用 `https://githubfast.com/nryant/dscore.git`，正式文档应尽量给官方 GitHub 地址和镜像备选。
- 数据：AMI、DIHARD、VoxConverse、AliMeeting；功能可用 DiariZen example。
- 建议规模：冒烟 1-5 条；精度 10-100 场会议；性能 10-50 小时。

---
## 10. whisper-large-v3

### 10.1 仓库观察与判断依据

- README 提供较完整的批量 `infer.py`，支持单文件、多文件、目录输入和 CSV 输出。
- 仓库中 `whisper_v3.tar.gz` 是 Git LFS 指针，真实大小约 5.6GB，当前目录未展开出实际 `infer.py`。
- README 指定 torch/torch_npu 2.5.1、transformers 4.57.3、datasets 4.4.1、accelerate 1.12.0、soundfile、librosa、ModelScope 等依赖。
- 权重通过 ModelScope `AI-ModelScope/whisper-large-v3` 下载。
- ASR 任务指标成熟，中文 CER、英文 WER、多语言评估数据都较容易获取。

### 10.2 后续适配

- 复杂度：中。
- 拉取/解包 `whisper_v3.tar.gz`，把实际 `infer.py` 显式纳入仓库或改为 patch 交付。
- CLI 增加 `--language --task --chunk_length_s --return_timestamps --max_new_tokens --device`。
- 新增 `prepare_eval_data.py`，准备 AISHELL-1、LibriSpeech、FLEURS/CommonVoice 子集。
- 新增 `eval_asr.py`，支持 CER/WER 和文本规范化。
- 精简依赖，避免安装无关包。

### 10.3 功能验证

- 已有验证脚本：README 有脚本内容，但当前仓库实际文件需从 LFS 大包确认。
- 验证数据来源：自备中文/英文 WAV，README 示例音频，AISHELL/LibriSpeech 小样本。
- 权重获取：`modelscope download --model AI-ModelScope/whisper-large-v3 --local_dir ./whisper-large-v3`。
- 验证内容：单文件、多文件列表、目录递归、mp3/m4a/flac、长音频。
- 验收：CSV 输出完整，包含文件名、文本、耗时/状态；语言正确，空结果数为 0 或可解释。

### 10.4 性能验证

- 已有性能脚本：README 脚本有并发/批量基础，但需独立 benchmark 和分段计时。
- 对比对象：Hugging Face transformers Whisper large-v3 CPU/CUDA 路径或 OpenAI/Transformers 官方实现。
- 对比数据集：至少 LibriSpeech test-clean；中文可补 AISHELL-1 test。
- 数据生成：固定 manifest，测试 batch size 1/2/4、线程 1/2/4，10s/30s/5min 分桶。
- 指标：RTF、音频秒/秒、P95、显存峰值、processor/generate/I/O 分段耗时。

### 10.5 精度验证

- 已有精度脚本：无，需要新增 WER/CER 评测。
- 对比对象：transformers CPU/CUDA 同 checkpoint 输出或官方 Whisper large-v3 公共指标。
- 对比数据集：中文 AISHELL-1 test 计算 CER；英文 LibriSpeech test-clean/test-other 计算 WER；多语言用 FLEURS/CommonVoice。
- 指标：CER/WER、空识别率、NPU/CPU 文本一致率；规范化需固定，不可临时修改。
- 验收：NPU 相比 CPU/CUDA 无显著退化，长音频切块不漏段。

### 10.6 数据集获取

- 权重：ModelScope `AI-ModelScope/whisper-large-v3`。
- 中文数据：AISHELL-1、WenetSpeech 子集。
- 英文数据：LibriSpeech `https://www.openslr.org/12`。
- 多语言：CommonVoice、FLEURS。
- 建议规模：冒烟 10 条；性能 1-10 小时；精度 1000 条左右。

---
## 11. BEATs

### 11.1 仓库观察与判断依据

- 随仓有 `BEATs.py`、`infer_npu.py`、`infer_cpu.py`、`requirements.txt`。
- README 指示 clone `https://github.com/microsoft/unilm.git` 到 `BEATs/upstream`，替换官方 UniLM `beats/BEATs.py`，并把 `infer_npu.py` 拷到 UniLM 目录。
- `infer_npu.py` 显式使用 `torch.device('npu:0' if torch.npu.is_available() else 'cpu')`，并将模型和输入 `.to(device)`。
- 当前示例音频路径硬编码，分类标签依赖 checkpoint。
- README 已提出 ESC-50/AudioSet 数据准备不要触发无关 split 下载，AudioSet shard 下载大小要记录。

### 11.2 后续适配

- 复杂度：低-中。
- CLI 化：`--checkpoint --audio/--manifest --label_map --device --output_csv --top_k`。
- 增加目录批量推理和 batch 输入，当前单样本脚本无法体现 NPU 性能。
- 明确不同 BEATs checkpoint 对应 label set 和任务类型。
- 新增 `eval_beats.py`，支持 accuracy/mAP 和 NPU/CPU logits 对齐。
- 精简 `requirements.txt`，拆分推理最小依赖。

### 11.3 功能验证

- 已有验证脚本：有 `infer_npu.py` 和 `infer_cpu.py`。
- 验证数据来源：任意 16 kHz WAV、ESC-50 小样本、AudioSet 子集。
- 权重获取：按 BEATs/UniLM 官方 checkpoint 下载；文档需补具体 checkpoint URL、目标目录和 label map。
- 验证内容：CPU/NPU 单条 top5，静音音频，不同时长音频。
- 验收：概率维度正确，top-k 非空，无 NaN；CPU/NPU 都能加载同 checkpoint；输出 CSV 含 top-k label/prob。

### 11.4 性能验证

- 已有性能脚本：无，需要新增 batch benchmark。
- 对比对象：UniLM/BEATs 官方 PyTorch CPU/CUDA 路径或同 checkpoint CPU baseline。
- 对比数据集：至少 ESC-50 全量或 AudioSet eval 子集；性能可从公开 WAV 切片/复制构造 1s/10s/30s 三档。
- 数据生成：生成 manifest，固定采样率和 batch size 1/8/16/32。
- 指标：样本/s、音频秒/s、平均/P95 延迟、显存；同时记录特征加载和模型前向耗时。

### 11.5 精度验证

- 已有精度脚本：无，需要新增分类评估。
- 对比对象：官方 BEATs checkpoint 的源仓 CPU/CUDA 输出和公开任务结果。
- 对比数据集：ESC-50 较小易用；AudioSet 标准但下载复杂，需确保 label map 与 checkpoint 对齐。
- 指标：top1/top5 accuracy、mAP、logits MAE、top-k 一致率。
- 验收：NPU 与 CPU logits 差异在浮点容差内，top-k 一致率高；分类精度与源仓同口径一致。

### 11.6 数据集获取

- 上游代码：`https://github.com/microsoft/unilm.git`。
- 权重：BEATs 官方 checkpoint；需在 README 中补完整 URL 和 label map 获取方式。
- 数据：ESC-50、AudioSet eval 子集；AudioSet 需记录 shard 下载大小和 YouTube 失效风险。
- 功能数据：任意 16 kHz WAV。
- 建议规模：冒烟 5 条；性能 1000 条；精度 ESC-50 全量或 AudioSet eval 子集。

---
## 12. MOSS-TTSD-v0.5

### 12.1 仓库观察与判断依据

- 仓库主要为 README 和截图，没有随仓实际 patch 文件。
- README 要使用 `quay.io/ascend/vllm-ascend:v0.10.0rc1` 容器，硬件 2 卡/910B。
- 上游项目为 `https://github.com/OpenMOSS/MOSS-TTSD`；权重建议从 ModelScope 一键整合包 `xueshanlinghu/MOSS-TTSD-zhenghebao` 下载，包含代码和 MOSS-TTSD-v0.5 权重，需 7z 多卷解压。
- README 要修改 `generation_utils.py`、`gradio_demo.py`、`inference.py`、`podcast_generate.py`、`streamer.py`、`XY_Tokenizer/inference.py`、`xy_tokenizer/nn/quantizer.py`，以及一键包下 `xy_tokenizer/model.py`。
- 默认 bf16、flash_attention_2、vLLM-Ascend 等组合与 NPU 兼容性强相关。

### 12.2 后续适配

- 复杂度：高。
- 将所有 README 截图/描述的修改整理为 patch 文件，不再依赖人工照图改代码。
- 明确 vLLM-Ascend、torch_npu、CANN、模型 dtype、attention backend 的兼容矩阵。
- CLI 化 `inference.py`，加入 `--device --num_devices --model_dir --tokenizer_dir --jsonl --output_dir --seed`。
- 增加 tokenizer encode/decode device 行为单元测试。
- 增加一键环境检查脚本，验证容器、NPU 卡数、权重目录、JSONL、参考音频。

### 12.3 功能验证

- 已有验证脚本：上游/整合包有脚本，但本仓没有可应用 patch 后的完整脚本。
- 验证数据来源：整合包 `examples/examples.jsonl`、自建 JSONL 和参考音频。
- 权重获取：`modelscope download --model xueshanlinghu/MOSS-TTSD-zhenghebao`，默认下载到 `~/.cache/modelscope/hub/`，再按 README 解压多卷包。
- 验证内容：单条 JSONL、多条批量、不同 `silence_duration`、seed、normalize 开关、Gradio demo 如需保留。
- 验收：输出 WAV 数量正确，可播放，采样率/时长合理，无开头杂音、截断、全零/NaN。

### 12.4 性能验证

- 已有性能脚本：无，需要新增 `benchmark_ttsd.py`。
- 对比对象：OpenMOSS/MOSS-TTSD 源仓或整合包 CUDA/vLLM 路径；若无公开硬件表，则同 checkpoint 同 JSONL 的本地源仓结果。
- 对比数据集：至少使用整合包 examples 和自建 50 条 JSONL；正式可由 CSMSC/AISHELL-3 转成 prompt+text。
- 数据生成：固定 JSONL、参考音频、seed、dtype、attention backend 和卡数。
- 指标：RTF、首音频延迟、总延迟、tokenizer/LLM/vocoder 分段耗时、2 卡利用率、显存峰值。

### 12.5 精度验证

- 已有精度脚本：无，需要新增 TTS/语音生成评测。
- 对比对象：源仓 CUDA/vLLM 输出和人工听测基线。
- 对比数据集：CSMSC/AISHELL-3 子集或整合包 examples 扩展集。
- 指标：ASR 回识别 CER、speaker embedding cosine、DNSMOS/UTMOS、人工 MOS/CMOS、A/B 偏好率。
- 验收：NPU 与源仓输出在可懂度、音色、自然度上无明显退化；不以波形逐点一致作为唯一标准。

### 12.6 数据集获取

- 上游代码：`https://github.com/OpenMOSS/MOSS-TTSD`。
- 整合包：`https://www.modelscope.cn/models/xueshanlinghu/MOSS-TTSD-zhenghebao/summary`，命令 `modelscope download --model xueshanlinghu/MOSS-TTSD-zhenghebao`。
- 功能数据：整合包 examples、自建 JSONL 和参考音频。
- 精度/性能数据：CSMSC、AISHELL-3 改造成 prompt+text；人工听测抽 50-100 条。
- 建议规模：冒烟 5 条；性能 50 条；精度 100-500 条。

---
## 13. 综合优先级与落地建议

### 13.1 优先级排序

| 优先级 | 仓库 | 原因 |
|---|---|---|
| P0 | DNSMOS | 已有 CANNExecutionProvider 推理脚本，权重/数据/CSV 输出链路短，最快形成 Canary-1B 风格交付。 |
| P0 | BEATs | 已有 NPU/CPU 双脚本，适合快速建立音频分类的功能、性能、精度模板。 |
| P1 | FireRedASR-AED | 中文 ASR 指标成熟，工程化成本中等，适合作为 ASR 模板。 |
| P1 | whisper-large-v3 | README 脚本较完整，ASR 数据易得；需先解决 LFS 大包/实际脚本落库。 |
| P1 | Canary-1B | 已完成较完整输出件，可作为其他仓库 README/评测脚本标准模板。 |
| P2 | MossFormer2_SE_48K | 语音增强功能简单，但需补随仓脚本和成对数据评估。 |
| P2 | pyannote-speaker-diarization-3.1 | 可落地，但 DER 数据、RTTM 和依赖版本需投入。 |
| P3 | Index-TTS-2 | 能力强但环境/编译/多权重复杂，建议专项。 |
| P3 | BUTSpeechFIT-DiariZen | 依赖链重且当前偏文档，建议与 pyannote 二选一时优先 pyannote。 |
| P3 | MMAudio | 大包、2 卡、多模态生成、质量评价均复杂。 |
| P3 | MOSS-Speech | 多仓库和第三方源码补丁，适配风险高。 |
| P3 | MOSS-TTSD-v0.5 | 多文件 patch、大模型、2 卡、vLLM-Ascend 和主观验证，需专项资源。 |

### 13.2 建议执行节奏

1. 第一批：DNSMOS、BEATs、FireRedASR-AED。目标是统一产出 `infer_npu.py`、`prepare_eval_data.py`、`eval.py`、`benchmark.py`、`README_INFERENCE.md`。
2. 第二批：whisper-large-v3、MossFormer2_SE_48K、pyannote-speaker-diarization-3.1。目标是补数据准备、标准指标和源仓 CPU/CUDA 对比。
3. Canary-1B 作为模板维护：后续只补真实 NPU 运行结果、环境版本和常见问题。
4. 专项批：Index-TTS-2、MOSS-TTSD-v0.5、MOSS-Speech、MMAudio。先做 patch 化和权重/数据检查，再做主观/客观质量评估。

### 13.3 对所有仓库的共同要求

- 每个仓库的功能验证必须写明：已有脚本、验证数据 URL/生成方式、权重 URL/下载命令、运行命令、验收输出。
- 每个仓库的性能验证必须至少选一个数据集，与源仓 CPU/CUDA/官方报告做同 checkpoint、同数据、同参数对比。
- 每个仓库的精度验证必须至少选一个公开数据集或明确人工评测协议，与源仓 CPU/CUDA/官方指标做对比。
- 数据准备脚本必须支持离线复用，禁止在评测阶段隐式联网下载。
- 缺少官方指标或官方组件时应在文档中明确“缺失/待补”，不能用简化指标或第三方非官方实现冒充官方评测。
