# Ascend-SACT 语音/音频模型 NPU 适配仓库详细评估报告

分析日期：2026-05-22  
分析范围：`/home/pei/ModelZoo` 下已克隆的 12 个 GitCode 仓库。  
分析方式：静态代码/文档分析，未在 Ascend NPU 上实际执行。判断依据来自各仓库 `README.md`、随仓 `infer.py`/`infer_npu.py`/`run_web.sh`、`requirements.txt`、仓库文件规模、是否包含可直接运行脚本、是否依赖上游大仓/多权重/系统补丁、是否已有结果或性能描述。

## 0. 评价口径

### 0.1 标签含义

| 标签 | 含义 |
|---|---|
| 低 | 有可直接执行脚本或清晰命令；主要工作是下载权重、替换路径、小样本冒烟测试。一般 0.5-2 人日可完成单卡功能闭环。 |
| 中 | 需要补 CLI、批处理、指标脚本、上游仓库集成或依赖版本修正；一般 2-5 人日。 |
| 高 | 需要多仓库联调、多处源码补丁、复杂依赖/编译、大模型或主观评价方案；一般 5-10 人日。 |
| 很高 | 存在重大阻塞，如大文件未拉取、多卡/专用容器、上游代码大量 cuda 假设、评价体系主观且需人工听测；通常超过 10 人日或需专项资源。 |

### 0.2 必看判断依据

- 仓库完整度：是否只有 README，还是包含实际 NPU 脚本/patch 文件。
- NPU 适配形态：`torch_npu`、`transfer_to_npu`、`onnxruntime-cann`、`cuda -> npu` 替换、CPU fallback、dtype 补丁等。
- 外部依赖：是否必须 clone 官方上游仓库、HuggingFace/ModelScope 大权重、多模型组合、Git LFS 大包。
- 验证闭环：是否给了运行命令、示例输出、性能数字、CSV/RTTM/音频等结果形式。
- 数据集可获得性：是否可用公开小样本冒烟；精度验证是否需要标准标注集、人工 MOS、成对干净/带噪数据、说话人 RTTM 标注等。

---

## 1. DNSMOS

### 1.1 仓库观察与判断依据

- 随仓包含 `DNSMOS/infer.py`，README 中也完整粘贴了推理脚本。
- 推理后端明确为 `onnxruntime` 的 `CANNExecutionProvider`，不是简单的 `cuda` 字符串替换。
- 脚本包含 NPU 环境检查、`npu-smi info`、性能模式切换、批量处理、CSV 输出。
- README 指定测试集可用 VCC2018，并给出下载命令。
- 该任务本质是非侵入式语音质量评分，输入 WAV，输出 `MOS_SIG/MOS_BAK/MOS_OVRL/P808_MOS`，功能链路较短。

### 1.2 后续适配

- 评级：低。
- 建议任务：
  1. 固化 `onnxruntime-cann`、CANN、Python、numpy/librosa/soundfile/pandas 版本到最小 `requirements.txt`，避免当前文档依赖隐含。
  2. 增加 `--device_id` 参数，当前脚本默认操作 0 卡性能模式。
  3. 增加模型文件自动检查说明，要求 `DNSMOS/model_v8.onnx`、`DNSMOS/sig_bak_ovr.onnx`、可选 `pDNSMOS/sig_bak_ovr.onnx`。
  4. 增加 CPU/ONNXRuntime baseline 脚本用于精度比对。
- 规模/范围：单文件脚本增强，约 100-200 行内修改。
- 工作量：0.5-1.5 人日。
- 阻塞点：主要是 ONNX 权重是否能稳定下载；NPU 环境需包含 `CANNExecutionProvider`。

### 1.3 功能验证

- 评级：低。
- 可执行方案：
  1. 准备 5-10 条 WAV，覆盖短音频、小于 9 秒、长音频、不同采样率、单声道/双声道。
  2. 执行：`python infer.py -t ./dataset/vcc2018 -o ./csv/vcc2018.csv --model_root . --batch_size 4`。
  3. 检查输出 CSV 是否包含 `filename,len_in_sec,MOS_SIG,MOS_BAK,MOS_OVRL,P808_MOS`。
- 指标：成功率 100%；无空 CSV；MOS 数值在 0-5 合理范围；短音频 padding 不报错。
- 工作量：0.5 人日。

### 1.4 性能验证

- 评级：低-中。
- 可执行方案：
  1. 用 100/1000 条 9-30 秒 WAV 做批处理。
  2. batch size 取 1/2/4/8，记录端到端耗时、纯推理耗时、吞吐音频时长/s、NPU 利用率。
  3. 对比 CPU onnxruntime 与 CANNExecutionProvider。
- 指标：吞吐提升倍数、平均单文件耗时、P95 文件耗时、NPU 显存占用。
- 工作量：1 人日。
- 中等原因：预处理 librosa 在 CPU，端到端性能受 I/O 与 mel 计算影响，需拆分统计。

### 1.5 精度验证

- 评级：中。
- 可执行方案：
  1. CPU ONNXRuntime 与 NPU CANN 对同一批音频输出逐项对齐。
  2. 计算四个分数的 MAE、最大绝对误差、Pearson/Spearman 相关。
  3. 如有主观 MOS 标注，可计算与人工 MOS 的相关性；否则至少验证 NPU/CPU 数值一致性。
- 指标：NPU vs CPU MAE 建议 < 1e-3 或根据 ORT/CANN 浮点差异放宽到 < 1e-2；排序相关 > 0.99。
- 工作量：1-2 人日。
- 中等原因：DNSMOS 自身是代理指标，若要验证真实 MOS 相关性，需要带主观评分的数据，不能只看脚本运行成功。

### 1.6 数据集获取

- 评级：低-中。
- 可执行方案：
  1. 冒烟：任意公开 WAV 或自录音频。
  2. 批量：README 建议 VCC2018 converted speech。
  3. 精度：如要 MOS 相关性，需要找到带主观质量评分的数据或使用原 DNSMOS 官方测试集。
- 数据规模建议：冒烟 10 条；性能 1000 条；精度一致性 500-1000 条。
- 阻塞点：VCC2018 可下载但体积较大；人工 MOS 标签获取难度高于纯音频获取。

---

## 2. Index-TTS-2

### 2.1 仓库观察与判断依据

- 随仓有 `infer_v2.py` 和 `run_web.sh`，README 指向上游 `triomino/index-tts`。
- 环境要求 CANN 8.3.RC1、Python 3.11.13、torch 2.8.0、torch_npu v2.8.0-7.2.0，并要求编译安装 GCC 13.3.0、CMake 3.31.0、torch_npu。
- README 要下载多个模型：IndexTTS-2、w2v-bert-2.0、MaskGCT、campplus、bigvgan。
- README 说明需修改 `infer_v2.py`、新增/修改 `infer_v2_npu.py`、修改 `webui.py`。
- README 给出性能结论：warmup 后 RTF 约 0.6，但没有精度/音质指标。

### 2.2 后续适配

- 评级：高。
- 可追溯原因：多权重、多组件、要求编译 torch_npu，且 README 中还有“非本仓库文件”的修改项，说明适配不是随仓即可直接运行。
- 建议任务：
  1. 将 README 中手工修改固化为 patch 或脚本，避免用户手工编辑上游仓。
  2. 明确 `infer_v2.py` 与 `infer_v2_npu.py` 的差异，避免仅文档描述。
  3. 增加非 WebUI 的 CLI 推理入口，支持文本、说话人提示音频、情感提示音频、输出路径参数化。
  4. 增加模型路径配置文件，统一检查 5 组权重是否存在。
  5. 梳理 `use_cuda_kernel` 命名，避免 NPU 模式仍出现 CUDA kernel 语义。
- 规模/范围：涉及上游 index-tts、WebUI、声学模型、vocoder、speaker/emotion encoder，预计 5-10 个文件。
- 工作量：5-8 人日。
- 阻塞点：torch_npu v2.8 编译、GCC/CMake 编译、多个模型下载、NPU 上动态 shape/自回归稳定性。

### 2.3 功能验证

- 评级：中-高。
- 可执行方案：
  1. 冒烟：使用官方 example，测试自由生成和精确时长控制两种模式。
  2. 零样本音色：准备 3 个说话人参考音频，每人 3 条文本。
  3. 情感解耦：准备 3 种情感提示或文本软指令，各生成 3 条。
  4. WebUI：访问 7860，验证 example、上传 prompt、输出下载。
- 指标：生成成功率、无 NaN/空音频、输出采样率/时长正确、精确时长模式误差、异常恢复。
- 工作量：2-3 人日。
- 中高原因：TTS 不是单一输入输出；音色、情感、时长控制均需覆盖，且 WebUI/CLI 两条链路都要验证。

### 2.4 性能验证

- 评级：中。
- 可执行方案：
  1. 固定 20 条短中长文本，分别测试 cold start、warmup 后 RTF。
  2. 记录首 token/首音频延迟、总生成耗时、RTF、NPU 显存。
  3. 比较 fp16/static 与非 static 设置。
- 指标：RTF、P50/P95 延迟、显存峰值、失败率。README 已提 RTF 约 0.6，可作为复现目标。
- 工作量：1-2 人日。
- 中等原因：自回归生成性能受文本长度、采样策略、prompt 长度影响，需标准化测试集。

### 2.5 精度验证

- 评级：高。
- 可执行方案：
  1. CPU/GPU 参考实现与 NPU 输出做固定 seed 下的声学特征相似度比对。
  2. 用 ASR 回识别生成音频，计算 CER/WER，验证可懂度。
  3. 用 speaker embedding 计算参考音色相似度。
  4. 用情感分类器或人工 A/B 测试评估情感一致性。
  5. 时长控制计算目标时长误差均值/P95。
- 指标：CER/WER、speaker cosine similarity、duration error、MOS/CMOS 或人工偏好率。
- 工作量：5-8 人日。
- 高原因：TTS 精度不仅是数值一致，还涉及可懂度、自然度、音色、情感和时长，客观指标不充分。

### 2.6 数据集获取

- 评级：中-高。
- 可执行方案：
  1. 冒烟：官方 examples。
  2. 中文 TTS：AISHELL-3、CSMSC 或自建 20-50 条提示音频。
  3. 情感：公开情感语音或自选 prompt；若做严肃评估需要带情感标签数据。
- 数据规模建议：冒烟 10 条；性能 50-100 条；精度 100-500 条加人工抽检。
- 阻塞点：高质量说话人/情感提示音频与人工 MOS 评价成本较高。

---

## 3. MMAudio

### 3.1 仓库观察与判断依据

- 仓库主要是 README 和截图，实际大包 `mmaudio.tar.gz` 是 Git LFS 指针，显示真实大小约 10GB，未拉取。
- README 指向上游 `hkchengrex/MMAudio`，并要求下载 apple CLIP、nvidia bigvgan 等额外模型。
- 适配方式包含多处手工改 `cuda` 为 `npu`、替换 CLIP 本地模型、VAE、BigVGAN filter/resample/bigvgan dtype 改动。
- 硬件要求 G8600/910B2C 2 卡，比多数单卡仓库要求更高。
- 任务是视频/文本到音频生成，验证涉及生成质量。

### 3.2 后续适配

- 评级：高。
- 可追溯原因：多处源码手工补丁、外部大包未拉取、依赖多个大模型、2 卡硬件要求。
- 建议任务：
  1. 拉取 Git LFS 大包或改为明确 patch 文件，不依赖截图说明。
  2. 将 README 中所有代码修改整理成统一 diff，覆盖 `demo.py`、`features_utils.py`、`vae.py`、BigVGAN alias-free activation 等。
  3. 增加 `--device npu:0`、`--dtype float32/bf16` 参数，避免硬编码。
  4. 增加单卡降级路径或明确必须 2 卡的原因。
  5. 对 BF16 不支持点建立 op 清单，确认是否还需 CPU fallback。
- 规模/范围：预计 6-10 个上游文件，加模型路径和 demo 配置。
- 工作量：6-10 人日。
- 阻塞点：10GB LFS 包、多个权重下载、2 卡资源、生成式模型 op 支持。

### 3.3 功能验证

- 评级：中-高。
- 可执行方案：
  1. 文生音频：5 条短 prompt，验证生成 wav。
  2. 视频生音频：5 个短视频，验证视频特征、CLIP、本地 BigVGAN 链路。
  3. 异常输入：空 prompt、长 prompt、短/长视频。
- 指标：生成成功率、输出采样率/时长、无爆音/全零、日志无 NPU op fallback 严重错误。
- 工作量：2-3 人日。
- 中高原因：输入模态和模型组件多，任一权重路径或 dtype 不兼容都会失败。

### 3.4 性能验证

- 评级：中-高。
- 可执行方案：
  1. 固定 10 个 prompt/视频，统计生成 5s/10s/30s 音频耗时。
  2. 记录 RTF、NPU 利用率、显存、CPU 占用、数据拷贝时间。
  3. 比较 float32 转换前后性能损失。
- 指标：RTF、端到端延迟、显存峰值、2 卡利用率均衡程度。
- 工作量：2-4 人日。
- 中高原因：README 指出 BF16 支持不足需转 float32，这会显著影响性能，需要分模块定位瓶颈。

### 3.5 精度验证

- 评级：高。
- 可执行方案：
  1. 与 GPU/CPU 官方输出做相同 seed 下音频特征比对，如 mel L1、CLAPScore、FAD。
  2. 人工听测评估音频与文本/视频一致性、自然度、噪声。
  3. 检查 dtype 转换后是否引入音质退化。
- 指标：CLAPScore/FAD、人工 MOS、A/B 偏好率、mel 差异。
- 工作量：5-8 人日。
- 高原因：生成音频没有单一确定答案，且多模态一致性需要主客观结合。

### 3.6 数据集获取

- 评级：高。
- 可执行方案：
  1. 冒烟：手写 prompt 和少量公开视频。
  2. 性能：构造固定长度 prompt/视频集。
  3. 精度：AudioCaps、Clotho、VGGSound 等，但下载、授权、处理成本较高。
- 数据规模建议：冒烟 10 条；性能 50 条；精度 200-1000 条或人工抽检 50 条。
- 阻塞点：视频/音频版权、数据体积、评价标注、CLAP/FAD 评估工具链。

---

## 4. MOSS-Speech

### 4.1 仓库观察与判断依据

- 随仓有 `infer.py` 和超长 `requirements.txt`。
- README 指出依赖 MOSS-Speech、MOSS-Speech-Codec、HuggingFace Space 三套代码/权重。
- README 需要修改安装环境中的 `diffusers`、`transformers` 源码，以及 MOSS-Speech Space 内 Matcha-TTS、CosyVoice HiFiGAN 文件。
- `infer.py` 中仍可见 `device="cuda"`、`device_map="cuda"`、`.to("cuda")`，依赖 `transfer_to_npu` 自动迁移，说明适配不够显式。
- 任务是语音对话大模型，输出可能包含音频和文本，链路复杂。

### 4.2 后续适配

- 评级：高。
- 可追溯原因：三个外部项目、环境包源码补丁、脚本仍保留 cuda 设备字符串。
- 建议任务：
  1. 将 `infer.py` 中 `cuda` 设备显式改为 `npu` 或统一 device 参数，验证 `transfer_to_npu` 是否足够。
  2. 把 diffusers/transformers/Matcha-TTS/HiFiGAN 修改做成 patch，并记录兼容版本。
  3. 增加模型路径、codec 路径、prompt 音频路径 CLI 参数。
  4. 梳理 CPU fallback：Whisper 特征提取和 `istft` 被文档要求放 CPU，需要明确性能影响和数据搬运。
  5. 精简 requirements，当前文件像完整环境冻结，包含大量无关包和 CUDA/NVIDIA 包。
- 规模/范围：至少 5 个第三方文件补丁 + 推理主脚本 + 环境重构。
- 工作量：6-10 人日。
- 阻塞点：第三方库版本漂移、远程代码 trust_remote_code、CPU/NPU 混合执行。

### 4.3 功能验证

- 评级：高。
- 可执行方案：
  1. 文本输入生成文本输出。
  2. 文本输入生成音频输出。
  3. 带 prompt audio 的音色延续。
  4. 长对话/多轮上下文。
  5. 异常路径：权重缺失、codec 缺失、prompt 音频格式错误。
- 指标：生成成功率、音频可播放、采样率正确、文本非空、无明显截断。
- 工作量：3-5 人日。
- 高原因：链路跨 LLM、codec、TTS、Whisper 特征和 HiFiGAN，功能点多且环境补丁多。

### 4.4 性能验证

- 评级：中-高。
- 可执行方案：
  1. 固定 10 条输入，测试文本输出和音频输出两种模式。
  2. 分段统计 LLM generate、codec、vocoder、CPU 特征/istft 时间。
  3. 记录首 token、首音频、总延迟、RTF、显存。
- 指标：端到端延迟、RTF、NPU/CPU 时间占比、显存峰值。
- 工作量：2-4 人日。
- 中高原因：文档明确存在 CPU fallback，端到端性能可能被 CPU 和数据搬运限制。

### 4.5 精度验证

- 评级：高。
- 可执行方案：
  1. 文本输出用人工或任务集评价相关性。
  2. 音频输出用 ASR 回识别 CER/WER 评估可懂度。
  3. prompt 音色用 speaker embedding 相似度。
  4. 主观 MOS/CMOS 听测自然度。
- 指标：CER/WER、speaker similarity、人工 MOS、任务成功率。
- 工作量：5-8 人日。
- 高原因：语音对话模型没有单一标准答案，且音频自然度/音色需要人工或外部模型评价。

### 4.6 数据集获取

- 评级：高。
- 可执行方案：
  1. 冒烟：官方 prompt 音频与少量中文对话。
  2. 功能：自建 20-50 条对话脚本。
  3. 精度：需要带参考回复/音频或人工评价协议，公开集不能直接覆盖全部能力。
- 数据规模建议：冒烟 5 条；功能 50 条；精度人工 50-100 条。
- 阻塞点：MOSS-Speech/Codec/Space 权重多源下载，数据评价主观，人工听测组织成本高。

---

## 5. Canary-1B

### 5.1 仓库观察与判断依据

- 随仓有 `infer.py`，README 指向上游 NVIDIA NeMo 和 `nvidia/canary-1b` 权重。
- README 说明 `infer.py` 需移动到 NeMo 官方仓，并修改模型路径软链接。
- `infer.py` 通过 `transfer_to_npu`，调用 `EncDecMultiTaskModel.from_pretrained` 后执行 ASR 和翻译任务。
- 功能覆盖英语 ASR、指定语言 ASR、语音翻译。
- NeMo 依赖通常较重，且多语言/翻译任务验证需要不同数据。

### 5.2 后续适配

- 评级：中。
- 可追溯原因：脚本短但依赖 NeMo 大仓；路径硬编码和任务硬编码需要工程化。
- 建议任务：
  1. 将 `infer.py` 改成 CLI：`--model_dir --audio --source_lang --target_lang --task`。
  2. 增加 batch manifest 输入，支持 NeMo 常用 JSON manifest。
  3. 明确 NPU device 设置，减少对 `transfer_to_npu` 魔法迁移的依赖。
  4. 增加输出 JSON/CSV，包含文本、耗时、任务类型。
- 规模/范围：单脚本增强 + NeMo 环境文档。
- 工作量：2-4 人日。
- 阻塞点：NeMo 版本与 torch_npu 兼容；权重缓存路径软链接易出错。

### 5.3 功能验证

- 评级：中。
- 可执行方案：
  1. 英语 ASR：5 条 LibriSpeech 小样本。
  2. 西语/德语/法语等 ASR：每种 3-5 条。
  3. 语音翻译：英语到法语/德语各 5 条。
  4. 长音频切分或超过模型推荐长度时的行为。
- 指标：生成文本非空、语言正确、任务参数生效、批量成功率。
- 工作量：1-2 人日。
- 中等原因：不仅是 ASR，还包含多语言和翻译参数组合。

### 5.4 性能验证

- 评级：中。
- 可执行方案：
  1. 按 10s/30s/60s 音频测试 RTF。
  2. batch size 1/2/4，记录吞吐、显存、延迟。
  3. 对比 CPU 或 GPU 官方参考如可用。
- 指标：RTF、P95 延迟、音频秒/秒、显存峰值。
- 工作量：1-2 人日。
- 中等原因：NeMo 数据加载和预处理较重，ASR/翻译解码长度也影响性能。

### 5.5 精度验证

- 评级：中。
- 可执行方案：
  1. ASR：LibriSpeech test-clean/test-other 或 CommonVoice 子集，计算 WER/CER。
  2. 翻译：CoVoST 或 FLEURS 子集，计算 BLEU/chrF，人工抽检。
  3. NPU 与 CPU/GPU 参考输出对比，检查解码差异。
- 指标：WER/CER、BLEU/chrF、NPU vs reference 文本一致率。
- 工作量：2-4 人日。
- 中等原因：公开数据可得，但多语言和翻译任务使评估维度增多。

### 5.6 数据集获取

- 评级：中-高。
- 可执行方案：
  1. 冒烟：README 示例音频或 NeMo sample。
  2. 英语 ASR：LibriSpeech。
  3. 多语言/翻译：FLEURS、CoVoST、CommonVoice。
- 数据规模建议：冒烟 10 条；性能 100 条；精度每语种 100-1000 条。
- 阻塞点：多语言数据下载、许可证、转写/翻译参考文本格式统一。

---

## 6. FireRedASR-AED

### 6.1 仓库观察与判断依据

- 随仓有 `infer.py`，README 指向官方 `FireRedTeam/FireRedASR` 和 ModelScope 权重 `FireRedASR-AED-L`。
- README 要把 `infer.py` 移动到官方仓，修改 `batch_uttid` 和 `batch_wav_path`。
- `infer.py` 使用 `transfer_to_npu`，调用 `FireRedAsr.from_pretrained('aed', ...)` 后 `model.transcribe`。
- 任务为中文 ASR，输出文本明确，评价指标成熟。

### 6.2 后续适配

- 评级：中。
- 可追溯原因：核心脚本简单，但路径、音频列表、输出和评测均硬编码/缺失。
- 建议任务：
  1. CLI 化：`--model_dir --input_wav/--manifest --output_csv --batch_size`。
  2. 增加多文件目录递归、音频重采样检查。
  3. 增加 CER/WER 评估脚本，支持参考文本 TSV/JSON。
  4. 明确 NPU device 与 dtype。
- 规模/范围：1-2 个脚本。
- 工作量：2-3 人日。
- 阻塞点：FireRedASR 上游安装依赖与 torch_npu 兼容。

### 6.3 功能验证

- 评级：低-中。
- 可执行方案：
  1. 官方 example `BAC009S0764W0121.wav` 冒烟。
  2. 目录输入 10 条中文 WAV。
  3. 不同采样率/时长输入，检查是否自动处理或报出清晰错误。
- 指标：转写文本非空、批量成功率、输出 CSV 完整。
- 工作量：0.5-1 人日。
- 低中原因：单任务 ASR，链路短；但当前脚本硬编码，需要先改参数化。

### 6.4 性能验证

- 评级：中。
- 可执行方案：
  1. AISHELL-1 test 或 100 条中文音频，按 batch size 1/4/8 测试。
  2. 记录 RTF、平均/ P95 延迟、显存、吞吐。
  3. 分离音频读取、特征提取、模型推理、解码耗时。
- 指标：RTF、音频秒/秒、显存峰值、失败率。
- 工作量：1-2 人日。
- 中等原因：AED 解码自回归，性能受解码长度和 batch 策略影响。

### 6.5 精度验证

- 评级：中。
- 可执行方案：
  1. 用 AISHELL-1 test 计算 CER。
  2. 对比 CPU/GPU 官方结果或官方公开指标。
  3. NPU 与 CPU 同模型输出差异分析。
- 指标：CER、NPU/CPU 文本一致率、空转写率。
- 工作量：1-3 人日。
- 中等原因：中文 ASR 数据容易获取，但完整指标需清洗文本规范化。

### 6.6 数据集获取

- 评级：中。
- 可执行方案：
  1. 冒烟：官方 examples。
  2. 精度：AISHELL-1、WenetSpeech 子集、MagicData 子集。
  3. 性能：从上述数据抽 1-10 小时音频。
- 数据规模建议：冒烟 10 条；精度 AISHELL-1 test 全量；性能 1h/10h 两档。
- 阻塞点：部分中文数据集需注册或协议确认；文本规范化需统一。

---

## 7. MossFormer2_SE_48K

### 7.1 仓库观察与判断依据

- 仓库主要为 README，没有随仓推理脚本文件。
- README 指向 `modelscope/ClearerVoice-Studio` 上游，要求下载 `iic/ClearerVoice-Studio` checkpoints。
- 示例脚本中仅设置 `torch.npu.is_available()` 和 `device = torch.device('npu:0')`，但未展示对 ClearVoice 内部模型的完整 NPU patch。
- 任务是 48kHz 语音增强，输入输出 WAV，功能验证相对简单；精度验证需要干净参考音。

### 7.2 后续适配

- 评级：中。
- 可追溯原因：当前是部署指导型仓库，缺少随仓可执行 infer.py 和明确内部模块 NPU 迁移 patch。
- 建议任务：
  1. 将 README 示例落为 `infer_npu.py`，支持 `--input --output --model_dir --device`。
  2. 检查 ClearerVoice 模型内部是否全部 `.to(device)`，必要时添加 patch。
  3. 增加 48kHz 输入检查、重采样策略、批量目录处理。
  4. 增加增强前后音量、时长、采样率一致性检查。
- 规模/范围：1 个推理脚本 + 可能 1-3 个上游 patch。
- 工作量：2-4 人日。
- 阻塞点：ClearerVoice 依赖和 checkpoint 体积；内部是否有 CUDA 特定逻辑需实测确认。

### 7.3 功能验证

- 评级：低-中。
- 可执行方案：
  1. 单条 48kHz 带噪语音输入，检查增强 wav 输出。
  2. 16k/44.1k/48k 输入，验证是否重采样或报错。
  3. 目录批处理 20 条。
- 指标：输出文件存在、采样率 48k、时长差 < 10ms、无全零/NaN、听感噪声下降。
- 工作量：0.5-1 人日。
- 低中原因：音频到音频链路短，但当前需先补 infer 脚本。

### 7.4 性能验证

- 评级：中。
- 可执行方案：
  1. 按 5s/30s/60s 音频测试处理耗时。
  2. 记录 RTF、显存、CPU 占用。
  3. 目录批处理 100 条，统计 P50/P95。
- 指标：RTF、P95 延迟、显存峰值、音频秒/秒。
- 工作量：1-2 人日。
- 中等原因：48kHz 音频点数多，模型可能分块处理，性能受 I/O 和 overlap-add 影响。

### 7.5 精度验证

- 评级：中。
- 可执行方案：
  1. 使用带干净参考的噪声增强数据，计算 PESQ、STOI、SI-SDR、DNSMOS。
  2. 对比 CPU/GPU 上游输出，计算波形/频谱差异。
  3. 人工抽听增强是否过度降噪或语音失真。
- 指标：PESQ/STOI/SI-SDR 提升、DNSMOS 提升、NPU vs reference 差异。
- 工作量：2-4 人日。
- 中等原因：需要成对干净/带噪数据，且 48kHz 下部分常用指标对采样率有要求。

### 7.6 数据集获取

- 评级：中。
- 可执行方案：
  1. 冒烟：ClearerVoice samples。
  2. 精度：VoiceBank+DEMAND、DNS Challenge、自己混合干净语音与噪声并重采样到 48k。
  3. 性能：构造不同长度 wav。
- 数据规模建议：冒烟 5 条；性能 100 条；精度 500-1000 对。
- 阻塞点：48kHz 成对数据不如 16k 常见；需注意重采样是否影响指标。

---

## 8. pyannote-speaker-diarization-3.1

### 8.1 仓库观察与判断依据

- 仓库主要为 README，无随仓 infer.py 文件，但 README 给出完整示例脚本。
- 需要下载 `speaker-diarization-3.1`、`segmentation-3.0`、`wespeaker-voxceleb-resnet34-LM` 三个模型。
- README 要修改 `config.yaml` 中 embedding/segmentation 路径。
- 已知问题包括 `torchaudio.compliance.kaldi.py` 的 complex abs 不支持和 numpy `np.NaN` 兼容。
- 输出是说话人分段，标准精度指标是 DER，需要 RTTM 标注。

### 8.2 后续适配

- 评级：中。
- 可追溯原因：依赖多模型配置和第三方库补丁，虽脚本简单但环境容易踩坑。
- 建议任务：
  1. 将 README infer 示例落为 `infer_npu.py`，支持输入 wav、config、输出 RTTM。
  2. 提供 `config.yaml` 自动重写工具，填入本地模型路径。
  3. 将 torchaudio/numpy 补丁整理为版本约束或 patch。
  4. 增加多音频批处理和 RTTM 输出。
- 规模/范围：1-2 个脚本 + config 工具 + 依赖版本锁定。
- 工作量：2-5 人日。
- 阻塞点：pyannote 依赖版本组合、ModelScope 权重结构、本地路径配置。

### 8.3 功能验证

- 评级：中。
- 可执行方案：
  1. 单说话人、双说话人、三说话人混合各 3 条。
  2. 16k 单声道和多声道输入，验证自动混音/重采样。
  3. 输出 RTTM 和控制台分段。
- 指标：分段非空、说话人数合理、RTTM 格式可被 dscore/pyannote.metrics 读取。
- 工作量：1-2 人日。
- 中等原因：分段任务输出不是单文本，需人工查看边界和 speaker label 合理性。

### 8.4 性能验证

- 评级：中。
- 可执行方案：
  1. 5min/30min/1h 音频测试端到端耗时。
  2. 分别统计 segmentation、embedding、clustering 时间。
  3. 记录 RTF、显存、CPU 占用。
- 指标：RTF、每小时音频处理时间、显存峰值。
- 工作量：1-2 人日。
- 中等原因：pipeline 包含神经网络和聚类后处理，CPU 后处理可能成为瓶颈。

### 8.5 精度验证

- 评级：中-高。
- 可执行方案：
  1. 用 AMI、AISHELL-4、AliMeeting 或 VoxConverse 计算 DER。
  2. 设定 collar 0.25s、是否忽略 overlap，与公开指标保持一致。
  3. 对比 CPU/GPU pyannote 输出 DER 与分段差异。
- 指标：DER、JER、miss/false alarm/confusion 分项。
- 工作量：3-5 人日。
- 中高原因：需要 RTTM 标注和严格评测协议，且聚类随机性/阈值会影响结果。

### 8.6 数据集获取

- 评级：中-高。
- 可执行方案：
  1. 冒烟：自制两人对话或公开 sample。
  2. 精度：AMI、AISHELL-4、AliMeeting、VoxConverse。
  3. 性能：会议长音频。
- 数据规模建议：冒烟 5 条；精度 10-100 场会议；性能 10 小时以上。
- 阻塞点：会议数据体积大，RTTM 标注格式整理费时，部分数据需申请。

---

## 9. BUTSpeechFIT-DiariZen

### 9.1 仓库观察与判断依据

- 仓库主要为 README，没有随仓 infer.py。
- 依赖上游 `BUTSpeechFIT/DiariZen`、HuggingFace `diarizen-wavlm-large-s80-md`、submodule `dscore`。
- README 要安装/编译多套 pyannote 相关包，并修改 torchaudio kaldi complex abs。
- 还涉及 numpy `np.NaN` 兼容、本地模型软链接路径问题。
- 与 pyannote 相比，依赖更重，且下载来源更多。

### 9.2 后续适配

- 评级：高。
- 可追溯原因：不是完整代码适配仓，主要是安装说明；依赖 DiariZen + pyannote + dscore + HuggingFace 权重，多处手工操作。
- 建议任务：
  1. 补充 `infer_npu.py` 和 `eval_der.py`，支持 wav 输入、RTTM 输出、参考 RTTM 评估。
  2. 将 DiariZen 上游和 pyannote 依赖版本固定到可复现环境文件。
  3. 将 torchaudio/numpy 修改变成 patch 或版本约束。
  4. 去除软链接中手工路径假设，改为 `--model_dir`。
- 规模/范围：环境重构 + 2 个脚本 + 若干依赖 patch。
- 工作量：5-8 人日。
- 阻塞点：上游 submodule 下载、pyannote 版本冲突、HuggingFace 模型目录结构。

### 9.3 功能验证

- 评级：中-高。
- 可执行方案：
  1. 使用 `example/EN2002a_30s.wav` 冒烟输出 speaker turns。
  2. 输出 RTTM 并用 dscore 读取。
  3. 2/3/4 说话人样本覆盖，包含重叠语音。
- 指标：成功率、RTTM 格式合法、说话人数合理、无空结果。
- 工作量：2-3 人日。
- 中高原因：当前没有随仓脚本，且要先完成复杂环境和模型路径问题。

### 9.4 性能验证

- 评级：中。
- 可执行方案：
  1. 30s/5min/30min 会议音频测试。
  2. 记录 RTF、segmentation/embedding/clustering 分段耗时。
  3. 比较 NPU 与 CPU。
- 指标：RTF、每小时处理时间、显存、CPU 后处理占比。
- 工作量：1-2 人日。
- 中等原因：pipeline 类似 diarization，后处理复杂但测试指标明确。

### 9.5 精度验证

- 评级：中-高。
- 可执行方案：
  1. 使用 AMI/DIHARD/VoxConverse 等带 RTTM 数据。
  2. 用 dscore 计算 DER/JER，并明确 collar/overlap 策略。
  3. 对比官方 DiariZen 结果或 CPU 结果。
- 指标：DER、JER、miss/FA/confusion。
- 工作量：3-5 人日。
- 中高原因：DER 标准评估依赖 RTTM 和协议，且 DiariZen 模型可能对数据域敏感。

### 9.6 数据集获取

- 评级：高。
- 可执行方案：
  1. 冒烟：DiariZen example。
  2. 精度：AMI、DIHARD、VoxConverse、AliMeeting。
  3. 性能：长会议音频。
- 数据规模建议：冒烟 1-5 条；精度 10-100 场会议；性能 10-50 小时。
- 阻塞点：部分标准 diarization 数据需注册/申请；RTTM 协议整理耗时；HuggingFace 下载可能不稳定。

---

## 10. whisper-large-v3

### 10.1 仓库观察与判断依据

- README 提供了较完整的批量 `infer.py`，支持单文件、多文件、目录输入和 CSV 输出。
- 仓库文件列表中实际 `whisper_v3.tar.gz` 是 Git LFS 指针，真实大小约 5.6GB，未拉取；当前目录未展开出 `infer.py`。
- README 指定 torch/torch_npu 2.5.1、transformers/datasets/accelerate/soundfile/librosa/modelscope 等依赖。
- README 有推理结果示例：NPU 1 卡、并发线程 4、batch size 2、成功 7 个。
- 任务是 ASR，指标 WER/CER 成熟，数据可得。

### 10.2 后续适配

- 评级：中。
- 可追溯原因：README 脚本完整，但实际代码在未拉取的大包中；需确认 LFS 内容和脚本可运行。
- 建议任务：
  1. 拉取/解包 `whisper_v3.tar.gz`，把 `infer.py` 显式放入仓库。
  2. 固化 requirements 最小集，避免过度安装。
  3. 增加语言、任务、chunk/长音频参数。
  4. 增加 CER/WER 评测脚本。
- 规模/范围：1-2 个脚本，确认大包内容。
- 工作量：2-4 人日。
- 阻塞点：5.6GB LFS 文件下载；transformers 版本与 Whisper large-v3 权重格式。

### 10.3 功能验证

- 评级：低-中。
- 可执行方案：
  1. 单个中文/英文 WAV。
  2. 多文件列表。
  3. 目录递归。
  4. mp3/m4a/flac 等格式。
- 指标：CSV 输出完整、成功率、空结果数、语言正确、长音频不崩溃。
- 工作量：0.5-1 人日。
- 低中原因：README 脚本已覆盖多输入，但需要先获得实际脚本文件。

### 10.4 性能验证

- 评级：中。
- 可执行方案：
  1. batch size 1/2/4、线程 1/2/4 对比。
  2. 10s/30s/5min 音频测试 RTF。
  3. 记录 processor、generate、I/O 分段耗时。
- 指标：RTF、音频秒/秒、P95 延迟、显存峰值、并发失败率。
- 工作量：1-2 人日。
- 中等原因：Whisper 解码自回归，性能受 `max_new_tokens`、语言和音频长度影响。

### 10.5 精度验证

- 评级：中。
- 可执行方案：
  1. 中文 AISHELL-1 或 WenetSpeech 子集计算 CER。
  2. 英文 LibriSpeech test-clean/test-other 计算 WER。
  3. 对比 transformers CPU/GPU 输出。
- 指标：CER/WER、NPU/CPU 文本一致率、空识别率。
- 工作量：2-3 人日。
- 中等原因：ASR 数据和指标成熟，但多语言/文本规范化会影响可比性。

### 10.6 数据集获取

- 评级：低-中。
- 可执行方案：
  1. 冒烟：README 示例音频或自备 WAV。
  2. 中文：AISHELL-1、WenetSpeech 子集。
  3. 英文：LibriSpeech。
  4. 多语言：CommonVoice/FLEURS。
- 数据规模建议：冒烟 10 条；性能 1-10 小时；精度 1000 条左右。
- 阻塞点：大权重下载是首要阻塞，数据本身较容易。

---

## 11. BEATs

### 11.1 仓库观察与判断依据

- 随仓有 `BEATs.py`、`infer_npu.py`、`infer_cpu.py`、`requirements.txt`。
- README 指示替换官方 UniLM `beats/BEATs.py`，并把 `infer_npu.py` 拷到 UniLM 目录。
- `infer_npu.py` 显式使用 `torch.device('npu:0' if torch.npu.is_available() else 'cpu')`，并将模型和输入 `.to(device)`。
- 当前示例音频路径硬编码为 `tianjingcheng/NeMo/nvidia/2902-9008-0000_01.wav`，分类标签依赖 checkpoint。
- 任务是音频分类/特征提取，功能链路较短，但精度验证需标准分类数据。

### 11.2 后续适配

- 评级：低-中。
- 可追溯原因：已有 NPU/CPU 双脚本和替换文件，但仍需移动到上游 UniLM，路径硬编码。
- 建议任务：
  1. CLI 化：`--checkpoint --audio --label_map --device`。
  2. 增加目录批量推理和 top-k 输出 CSV。
  3. 明确不同 BEATs checkpoint 对应 label set。
  4. 精简 requirements，当前冻结环境非常庞大且含大量无关包。
- 规模/范围：1 个脚本改造 + 文档补充。
- 工作量：1-3 人日。
- 阻塞点：官方 checkpoint 和 label map 的匹配。

### 11.3 功能验证

- 评级：低。
- 可执行方案：
  1. 用 3-5 条 16k WAV 冒烟。
  2. 验证 CPU 与 NPU 都可输出 top5 label/prob。
  3. 测试不同音频长度和静音音频。
- 指标：输出概率维度正确、top5 非空、概率无 NaN、CPU/NPU 均可运行。
- 工作量：0.5-1 人日。

### 11.4 性能验证

- 评级：中。
- 可执行方案：
  1. batch size 1/8/16/32 测试。
  2. 1s/10s/30s 音频测试吞吐。
  3. CPU vs NPU 对比。
- 指标：样本/s、音频秒/s、平均延迟、显存。
- 工作量：1-2 人日。
- 中等原因：当前脚本是单样本，需要补 batch 才能体现 NPU 性能。

### 11.5 精度验证

- 评级：中-高。
- 可追溯原因：demo top5 只能证明运行，不能证明分类准确；需要与 checkpoint 训练任务匹配的数据和 label map。
- 可执行方案：
  1. ESC-50/AudioSet 子集/官方 BEATs eval 数据，按 checkpoint 对应任务评估。
  2. NPU vs CPU logits 差异，top1/top5 一致率。
  3. 计算 accuracy/mAP。
- 指标：top1/top5 accuracy、mAP、logits MAE、top-k 一致率。
- 工作量：3-5 人日。
- 中高原因：AudioSet 获取和标签映射复杂；不同 checkpoint 任务不同。

### 11.6 数据集获取

- 评级：中。
- 可执行方案：
  1. 冒烟：任意 WAV。
  2. 精度：ESC-50 较小易用；AudioSet 标准但下载复杂。
  3. 性能：可从公开 WAV 复制/切片构造。
- 数据规模建议：冒烟 5 条；性能 1000 条；精度 ESC-50 全量或 AudioSet eval 子集。
- 阻塞点：AudioSet YouTube 链接失效和标签映射。

---

## 12. MOSS-TTSD-v0.5

### 12.1 仓库观察与判断依据

- 仓库主要为 README 和截图，没有随仓实际补丁文件。
- README 要使用 `quay.io/ascend/vllm-ascend:v0.10.0rc1` 容器，硬件 2 卡/910B。
- 权重建议下载 ModelScope 一键整合包，含 GitHub 代码和 MOSS-TTSD-v0.5 权重，需 7z 多卷解压。
- README 要修改 `generation_utils.py`、`gradio_demo.py`、`inference.py`、`podcast_generate.py`、`streamer.py`、`XY_Tokenizer/inference.py`、`xy_tokenizer/nn/quantizer.py`，以及一键包下 `xy_tokenizer/model.py`。
- 任务是语音生成，默认 bf16、flash_attention_2 等实现可能与 NPU 兼容相关。

### 12.2 后续适配

- 评级：高。
- 可追溯原因：多文件手工 `cuda -> npu`，专用 vLLM-Ascend 容器，2 卡要求，大整合包下载解压，未随仓提供 patch。
- 建议任务：
  1. 将所有 README 截图/描述的修改整理成 patch 文件。
  2. 明确 vLLM-Ascend、torch_npu、CANN、模型 dtype/attention 后端兼容矩阵。
  3. CLI 化 `inference.py` 的 NPU 参数，避免硬编码卡号。
  4. 增加 tokenizer encode/decode device 行为单元测试。
  5. 增加一键环境检查脚本，验证权重目录、JSONL、参考音频。
- 规模/范围：8+ 文件补丁，容器和模型路径整理。
- 工作量：6-10 人日。
- 阻塞点：大整合包、多卷解压、2 卡资源、bf16/attention op 兼容。

### 12.3 功能验证

- 评级：中-高。
- 可执行方案：
  1. 使用 `examples/examples.jsonl` 冒烟生成 1 条。
  2. 测试多条 JSONL 批量生成。
  3. 测试不同 `silence_duration`、seed、normalize 开关。
  4. 校验输出目录、wav 可播放、无开头杂音/截断。
- 指标：生成成功率、输出数量、采样率/时长、无 NaN/全零、日志无严重 NPU 错误。
- 工作量：2-3 人日。
- 中高原因：输入 JSONL + 参考音频 + tokenizer + 主模型 + vocoder 链路较长。

### 12.4 性能验证

- 评级：中-高。
- 可执行方案：
  1. 固定 10 条 JSONL，测 cold/warm 性能。
  2. 统计预处理、tokenizer、LLM/generation、vocoder 各阶段耗时。
  3. 比较 dtype bf16/eager/sdpa/flash_attention_2 可用组合。
- 指标：RTF、首音频延迟、总延迟、显存峰值、2 卡利用率。
- 工作量：2-4 人日。
- 中高原因：生成链路长且有 attention 后端选择，性能瓶颈需分段定位。

### 12.5 精度验证

- 评级：高。
- 可执行方案：
  1. ASR 回识别生成音频计算 CER，验证文本可懂度。
  2. speaker embedding 与参考音频计算相似度。
  3. 人工 MOS/CMOS 听测自然度、音色、韵律、杂音。
  4. NPU 与官方 CPU/GPU 输出做同 seed A/B。
- 指标：CER、speaker cosine similarity、MOS/CMOS、A/B 偏好率。
- 工作量：5-8 人日。
- 高原因：TTS/语音生成没有逐点确定答案，主观音质和音色一致性必须参与。

### 12.6 数据集获取

- 评级：高。
- 可执行方案：
  1. 冒烟：整合包 examples。
  2. 功能：自建 20-50 条 JSONL 和参考音频。
  3. 精度：使用 CSMSC/AISHELL-3 等 TTS 数据改造成 prompt + text。
  4. 人工听测：抽 50-100 条。
- 数据规模建议：冒烟 5 条；性能 50 条；精度 100-500 条。
- 阻塞点：大模型权重获取和参考音频质量；人工听测成本。

---

## 13. 综合优先级与落地建议

### 13.1 优先级排序

| 优先级 | 仓库 | 原因 |
|---|---|---|
| P0 | DNSMOS | 完整 CANN 推理脚本，功能/性能闭环最快。 |
| P0 | BEATs | 有 NPU/CPU 脚本，适合快速建立音频分类验证模板。 |
| P1 | FireRedASR-AED | 中文 ASR，指标成熟，工程化成本中等。 |
| P1 | whisper-large-v3 | README 脚本完整，ASR 数据易得；需先解决 LFS 大包。 |
| P1 | Canary-1B | 多语言 ASR/翻译能力强，但 NeMo 依赖较重。 |
| P2 | MossFormer2_SE_48K | 语音增强功能简单，但需补随仓脚本和成对数据评估。 |
| P2 | pyannote-speaker-diarization-3.1 | 可落地，但 DER 数据和依赖版本需投入。 |
| P3 | Index-TTS-2 | 能力强但环境/编译/多权重复杂，建议专项做。 |
| P3 | BUTSpeechFIT-DiariZen | 依赖链重且当前偏文档，建议与 pyannote diarization 二选一优先。 |
| P3 | MMAudio | 大包、2 卡、多模态生成、质量评价均复杂。 |
| P3 | MOSS-Speech | 多仓库和第三方源码补丁，适配风险高。 |
| P3 | MOSS-TTSD-v0.5 | 多文件补丁、大模型、2 卡和主观验证，需专项资源。 |

### 13.2 建议执行节奏

1. 第一周：DNSMOS、BEATs、FireRedASR-AED 建立统一模板：`infer_npu.py`、`eval.py`、`benchmark.py`、`README_NPU.md`。
2. 第二周：whisper-large-v3、Canary-1B、MossFormer2_SE_48K 完成数据集和指标脚本。
3. 第三周：pyannote/DiariZen diarization 只选一个主线，优先 pyannote-speaker-diarization-3.1，因为路径更直接。
4. 后续专项：TTS/生成类按 Index-TTS-2 -> MOSS-TTSD -> MOSS-Speech/MMAudio 顺序推进。
