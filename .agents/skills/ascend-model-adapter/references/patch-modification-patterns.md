# 参考 patch 修改模式

来源：扩大到当前 checkout 的 `ACL_PyTorch/built-in` patch/diff 文件（本轮挖掘 115 个 patch/diff、80 个模型目录，覆盖 audio、cv、ocr、foundation_models、embodied_ai、nlp）。此文件只作为运行时写 patch 的内化参考；不要把“修改模式清单”作为用户交付物，除非用户明确要求说明依据。

## 目录

1. 读 patch 的顺序
2. 统计信号与任务模式
3. 设备/后端/checkpoint
4. 算子、shape、服务和依赖
5. 任务专用模式
6. 写 patch 前自检

## 读 patch 的默认顺序

1. 先读同任务/同路线样本 patch：`find <sample_dir> -maxdepth 3 -type f \( -name '*.patch' -o -name '*.diff' \)`。如需要扩大范围，先在完整或扩大 sparse checkout 上运行 `scripts/patch_pattern_miner.py <ACL_PyTorch/built-in> --out /tmp/modelzoo_patch_patterns.md`，再按当前任务筛选相近 patch。
2. 同时看 README 中 patch 应用位置、固定 commit、命令和 FAQ，避免只照搬代码差异。
3. 对每个 patch 归类为：设备迁移、推理后端替换、算子替换、shape/导出修正、依赖修正、服务化/缓存、评测/性能脚本修正。
4. 给新模型写 patch 时只保留必要改动；避免把调试日志、大段无关重构、一次性本地路径写进 patch。


## 扩大挖掘后的统计信号

在 115 个 patch/diff 中，模式出现频率大致为：不支持/低效算子替换 71、dtype/精度调整 50、CUDA/NPU 设备迁移 42、评测/性能脚本 34、服务化/cache/warmup 31、依赖与环境补丁 27、OM/InferSession 替换 23、ONNX 导出与图修正 17、vLLM/TorchAir 12。运行时优先把这些当作检查点，而不是事后补文档。

- audio patch 多集中在 ASR/TTS 的多子模型解码、beam search、kaldi/FFT、vLLM/TorchAir、cache 和 OM session。
- cv patch 多集中在 ONNX 导出、OM session、einsum/grid/sample/custom op、dtype 和评测脚本。
- ocr/document patch 多集中在 pipeline 中只替换 detector/recognizer/VLM 的核心推理、dynamic shape、timeout、NPU 预处理和 TorchAir 编译。
- nlp/foundation patch 多集中在 KV cache 结构、RoPE/cache dtype、固定 batch、ONNX dummy input、外部大模型 ONNX 数据和生成式解码接口。
- embodied/3D patch 多集中在 custom op/SDK、静态视觉缓存、真实 NPU tensor、动作模型服务化和矩阵重排。

## 样本 patch 对应的高频修改点

| 类型 | 典型样本 patch | 抽取出的修改模式 | 运行时默认动作 |
|---|---|---|---|
| 音频/TTS/vLLM | `Index-TTS-vLLM-v2/*.patch`、`MMAudio/diff_torchaudio_kaldi.patch`、`whisper/whisper_torchair/patches/kaldi.patch`、`YingMusic-SVC_for_Pytorch/diff.patch` | CUDA 设备选择改 NPU；speaker/cache 预热；kaldi/fbank/FFT 的复数取模拆解或局部 CPU fallback；vLLM-Ascend 环境变量 | 先查 `torch.cuda`、`torch.fft`、`torchaudio.compliance.kaldi`、`fft_device` 等信号；参考样本决定保留 NPU、拆 `rfft().abs()`，或显式 CPU fallback；把 CPU fallback 和端到端耗时分开写 |
| CV/生成/检测 | `InstantID/*.patch`、`D-FINE_NPU.patch`、`F3Net.patch`、`PromptIR.patch` | `torch_npu`/`transfer_to_npu` 注入；`cuda()`/`.to('cuda')` 替换；ONNXRuntime session 替换为 `ais_bench.InferSession`；dtype 从 fp64/bf16 调整到 fp32/fp16；grid/attention 相关算子转 fp16 | 先全局搜索 CUDA 假设和 onnxruntime session；统一设备参数；必要时用 `InferSession.infer` 替换 ORT `session.run`；确保输入 dtype/shape 与 OM 一致 |
| 3D/机器人/embodied | `FocalFormer3D_for_Pytorch/diff.patch`、`IsaacGR00T/diff.patch`、`GraspNet/*.patch` | custom CUDA op 条件编译；DrivingSDK/NPU op 需要真 NPU tensor；动态 split/attention/视觉静态量缓存；bf16 改 fp16；FFN split/矩阵重排提升 cube 利用率 | 遇 custom op 先判断能否跳过 CUDA build 或绑定 Ascend SDK；不要只依赖 CUDA alias；对视觉/动作模型优先缓存静态 shape、pos-embedding、cu_seqlens |
| OCR/Paddle/文档 | `PP-DocLayoutV2/3/paddlex.patch`、`UVDoc/*.patch` | PaddleX predictor 注入 OM session；动态 shape `dymshape` 和 `custom_sizes`；长 timeout；MagicONNX API 兼容；后处理/纹理变换张量搬迁 | 对 pipeline 只替换核心 detector/recognizer/VLM 子模型；保留原预后处理；动态 shape 写清 `custom_sizes` 和输入范围；第三方工具 API 变更用最小兼容 patch |
| foundation/CLIP | `Chinese_CLIP/cn_clip.patch`、`SigLIP2/TorchAir/*.patch`、`blip/*.patch` | ONNX dummy input 改真实 batch/shape；固定 batch 评测，尾 batch padding；fp16 转换工具兼容；评测 batch 调小避免显存/内存压力；生成式模型把 HuggingFace cache/inputs 转成 OM 可接受数组 | 导出脚本必须参数化 batch/context/resolution；评测脚本处理尾 batch；若 OM 固定 batch，padding 后再裁掉输出；生成式模型保留原 tokenizer/postprocess，只替换 decoder 核心调用 |
| ASR/流式解码 | `Conformer_for_Pytorch/export_acc.patch`、`EspNet_for_Pytoch/*.diff`、`Zipformer_streaming/om_infer.diff`、`whisper/*/kaldi.patch` | 为 encoder/decoder/joiner/LM/CTC scorer 增加 NPU adapter；beam search 支持 batch；ONNX 导出动态轴；流式状态拆成多个 OM；禁用 ORT int8 quantize 或不适配量化；kaldi/FFT 兼容 | 先识别 ASR 子模块边界，逐个导出/转换/替换；保持 beam search 和 tokenizer 逻辑；状态/cache shape、dtype、batch 要显式记录 |
| LLM/NLP | `Qwen2_for_Pytorch/*.patch`、`MiniCPM_for_Pytorch/*.patch`、`Pet_for_Pytorch/*.patch` | KV cache 长度/结构调整；RoPE sin/cos dtype 对齐；`use_cache` 输出从对象/tuple 改为列表或张量；导出脚本注入 NPU/FP16 | 先固定生成参数和 cache 结构；避免破坏 HuggingFace 输出语义；如果改 cache 布局，补 CPU vs NPU logits/token 对齐 |

## 可复用修改模式

### 1. 设备选择从 CUDA 单分支改为 NPU 优先但保留 fallback

- 搜索：`cuda`、`.cuda()`、`torch.cuda`、`device='cuda'`、`accelerator='gpu'`、`map_location`、`set_device`。
- 默认模式：推理/评测入口显式接受 `--device npu/cpu` 且默认 `npu`；只有 CPU fallback 入口需要延迟导入 torch_npu。不要用静默 CUDA/CPU fallback 掩盖 NPU 不可用；选择物理卡通过 `ASCEND_RT_VISIBLE_DEVICES=<id>` 控制。
- 对 Lightning/第三方框架保守处理：如果框架仍使用 `accelerator='gpu'` 才能触发 transfer shim，README 写清原因，不要盲目改成不存在的 `accelerator='npu'`。
- 设备选择优先参数化：通过 `--device` 暴露 CPU/NPU 切换，使 CPU baseline 和 NPU 推理尽量共用同一入口；默认值必须是 NPU（如 `--device npu`）。物理卡选择统一使用 `export ASCEND_RT_VISIBLE_DEVICES=<id>`，不要用 `npu:0`/`npu:<id>` 或脚本默认 `--device-id`。CPU 只能显式选择。

### 1b. 推理后端路由 patch

- 问题模式：上游代码按关键字或配置硬编码选择 ONNX Runtime、TF SavedModel、Paddle inference、OpenVINO 等后端，使组件无法走 torch_npu/TorchAir。
- 修改模式：优先查找同一项目内 PyTorch 等价类，patch 路由分支或 import，使该组件接受 `device` 并可 `.to(npu)`；源码 patch 通常比运行时 monkey-patch 更小、更可维护。
- 注意：在 ModelZoo Ascend 默认 PyTorch/torch_npu 适配场景中，ONNX Runtime 通常不能直接驱动 Ascend NPU，容易退化为 CPU 路径；若确需 CPU fallback，README 必须写具体阻塞原因。
- 若使用者提供 checkpoint，默认围绕该 checkpoint 做路由切换和加载适配，不擅自更换权重；checkpoint 加载方式可能随 PyTorch/ONNX 后端不同而变化。

### 1c. checkpoint 加载 patch

- 使用者提供 checkpoint 时优先使用该 checkpoint，并确认配置、类别数、tokenizer、speaker/label 映射等成套文件。
- PyTorch 2.6+ `torch.load` 默认安全策略使用 `weights_only=True`；可信旧 checkpoint 包含自定义类并报 `UnpicklingError` 时，可在对应加载点 patch `weights_only=False`。
- 只对可信来源 checkpoint 使用 `weights_only=False`；不可信模型优先转换为 state_dict 或使用安全 allowlist。

### 2. 推理后端从 ONNXRuntime/Paddle 原生 infer 替换为 OM InferSession

- 常见改法：导入 `from ais_bench.infer.interface import InferSession`；在当前可见卡下初始化 `InferSession(0, om_path)`；用 `session.infer(feeds=[...])` 或 `session.infer(batch_inputs, mode='dymshape', custom_sizes=...)` 替换 `session.run(...)` / `self.infer(...)`。
- 保留原预处理和后处理；只替换核心模型调用，降低精度漂移。
- OM 路径、batch、dynamic shape 不硬编码；物理卡选择使用 `ASCEND_RT_VISIBLE_DEVICES`。若 ais_bench API 需要 device id，在当前可见卡内传 `0`，不要用 `npu:<id>` 表示物理卡。

### 3. NPU 不支持或低效算子用等价表达/CPU fallback/拆图

- 复数/FFT：遇到 `torch.fft.rfft(...).abs()` 报错或数值异常时，可先 `r = torch.fft.rfft(x)`，再用 `sqrt(r.real**2 + r.imag**2)`；也可 `view_as_real(r)` 后对实部/虚部平方和开方。若 STFT/ISTFT 在 NPU 不可用，可明确把该前后处理放 CPU，再把结果送回 NPU。
- dtype：fp64 常改 fp32；bf16 不支持时改 fp16；grid/attention/sampling 等算子常需 fp16 输入。
- attention：可用 `torch_npu.npu_fusion_attention`、TorchAir/vLLM-Ascend 编译配置，或把动态 split 改 reshape + 静态缓存。
- custom op：先判断是否能条件跳过 CUDA extension；必要时绑定 Ascend SDK 或拆子图。不能改的 custom op 要标 CPU fallback，并把性能口径拆开。

### 3b. 音频 fbank/FFT patch 模式

- 搜索信号：`torchaudio.compliance.kaldi.fbank`、`torch.fft.rfft`、`.abs()`、`view_as_real`、`fft_device`、`device.type == "mps"`、`kaldi.py`。
- 已有样本有三类做法：保留 NPU 并拆 `rfft().abs()`；按 SOC/设备能力局部 CPU fallback；直接把 fbank 放 CPU 后将结果送回 NPU。
- 写 patch 前先确认实际设备路由，不要仅凭 warning 判断是否 fallback。若需改 `torchaudio/compliance/kaldi.py`，单独成 patch 并写明目标版本、应用路径和恢复方式；优先参考 MMAudio/Index-TTS/Whisper/YingMusic 的最小改法，并在 README 性能口径里标明 CPU fallback。

### 4. 动态 shape 和固定 batch 的处理

- 导出 ONNX 时尽量把 batch、resolution、context_length、input_shape 参数化；若 ATC/OM 固定 batch，则评测脚本 padding 尾 batch，输出后裁掉。
- 对 Paddle/OCR/doc layout 这类动态输入，使用 `dymshape` 时必须写 `custom_sizes`、输入范围和失败时调参方法。
- 对多图/多模态/机器人视觉，预计算并缓存静态量（pos embedding、rotary cos/sin、cu_seqlens、split sizes）以降低编译和运行风险。

### 5. vLLM/TorchAir 服务化补丁

- 补丁通常不仅改设备，还改启动参数、编译配置、cache、并发和预热。
- 默认检查并设置必要环境变量，例如 NPU 内存分配、vLLM-Ascend 调度、编译缓存目录；但把值写为脚本参数或 README 可覆盖项。
- 性能脚本区分：首次编译、warmup、speaker/style cache 命中、服务端生成耗时、客户端端到端耗时。

### 6. 依赖与第三方库兼容补丁

- 避免 requirements 覆盖镜像内 torch/torch_npu；业务依赖和评测依赖分开。
- requirements/setup/pyproject patch 必须最小化：只删除或替换会导致安装失败、拉取 CUDA/GPU runtime、或覆盖 Ascend 镜像核心栈的条目（`torch*`、不适配的 `onnxruntime-gpu` 等），只新增经 import/`--help`/单样例验证确实缺失的业务依赖。不要把上游依赖列表大幅改写成短白名单，除非已证明原依赖会整体阻塞且 README 解释原因。
- 有 editable 子包时，优先通过 README 安装顺序和 `--no-deps` 控制依赖解析；只有 metadata 本身会误导用户或 CI 时才 patch `setup.py/pyproject.toml`。典型顺序：业务 requirements → 子包 editable → 顶层 `pip install --no-deps -e .`。
- 第三方库 API 变更只做最小兼容，例如 `onnx.mapping` 改 `onnx.helper`、`np.int` 改 `int`。
- 如果必须 patch site-packages（如 torchaudio kaldi），要单独成 patch；用 `importlib.util.find_spec()` 定位包目录，diff 路径从 `<pkg>/` 开始，从 site-packages 根目录 `patch -p1`，README 写明目标版本、dry run、恢复方式。不要 import 包只为读取 `__file__`。
- torchaudio 2.9+ `torchaudio.load` 会使用 TorchCodec，`backend` 参数会被忽略；遇到 Ascend 镜像中的 torchcodec/FFmpeg/ABI 问题时，直接改用 `soundfile`/`librosa` 读写，不要只依赖 `backend='soundfile'`。

### 7. 评测/性能脚本补丁

- 评测脚本常见修改：batch 调小、尾 batch padding、输出保存格式修正、设备名从 GPU 改 NPU、删除参数统计中依赖 CUDA 的部分。
- 不把“脚本能运行”当 accuracy；仍要回到上游原始指标，或说明 CPU/upstream vs NPU 对齐替代口径。
- 性能中若包含 CPU 前后处理、服务端排队、cache miss，要在 README 表格拆列。
- 正式 NPU 性能至少独立执行 3 次并报告中位数；脚本保存每次结果，纯模型计时前后 synchronize。

### 8. ASR/多子模型 adapter 模式

- Conformer/EspNet/Zipformer 类 patch 常不是简单换 device，而是在 encoder、decoder、joiner、LM、CTC scorer、beam search 周围加 adapter/decorator。
- 默认先画出子模型和状态流：音频特征 → encoder → decoder/joiner/LM/CTC → beam search → text；只把可编译子模型换成 OM/NPU，保留搜索逻辑。
- 流式模型的 state/cache 输入输出要逐项命名和保存样例 shape；dynamic shape 必须给 `dymshape/custom_sizes` 或 ATC 动态档位。

### 9. Transformer/生成式 cache 模式

- Qwen/BLIP/LLM 类 patch 常改 `past_key_values`、RoPE sin/cos dtype、`use_cache` 输出结构和 max cache size。
- 默认用小 prompt 做逐 token 对齐：比较前几步 logits/top-k/token，而不是只看最终文本。
- cache 改成 list/flat tensor/OM inputs 时，README 写清 layout、层数、head、seq_len、dtype，并让脚本参数化 max length。

### 10. NPU 预处理加速模式

- MinerU/OCR 类 patch 会把 resize/normalize/grayscale/mask 等预处理搬到 NPU，并预分配 buffer。
- 默认只有在预处理成为瓶颈或 TorchAir 需要图内固定 shape 时才这样做；否则优先保留 CPU 预处理以降低精度风险。
- 若搬迁预处理，必须做 CPU 预处理 vs NPU 预处理的中间张量对齐。

### 11. 大 ONNX 与外部数据模式

- SAM 等大模型 patch 会拆 encoder/decoder，并把大 ONNX 转 external data。
- 默认检测 ONNX 文件大小和 initializer；超过单文件限制或 ATC 读入失败时，启用 external data，并在 README 写清文件清单不可拆散。

### 12. 量化禁用/回退模式

- 部分流式 ASR patch 会注释 ORT dynamic quantize/int8 导出，因为 ATC/OM 不一定支持该量化图。
- 默认不要沿用上游 ORT quantize 作为 Ascend 路线；先导出 FP32/FP16 可用 OM，再评估 Ascend 支持的量化方案。

## 写新 patch 前的自检

- `git apply --check <patch>` 在固定上游 commit 通过。
- patch 不包含本地绝对路径、临时打印、无关格式化、大段重复 diff、下载到个人目录的逻辑。
- 每个新增 import 都被使用；每个硬编码 device/batch/shape/soc_version 都有参数或 README 解释。
- 如果替换为 OM/TorchAir/vLLM 后端，CPU/upstream baseline 仍可单独运行，便于精度对齐。
- 如果有 CPU fallback，README 和 benchmark 明确标出，不能把端到端结果说成纯 NPU 性能。
