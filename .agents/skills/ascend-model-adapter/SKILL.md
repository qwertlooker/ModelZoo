---
name: ascend-model-adapter
description: 将 Hugging Face、GitHub、PyTorch、ONNX、Paddle、vLLM、TorchAir 或研究模型适配到华为昇腾 Ascend NPU，并产出或检视可提交 Ascend ModelZoo-PyTorch ACL_PyTorch/built-in 的材料。适用于给定模型链接或 PR 后需要完成/审查从 GPU/上游实现迁移到 NPU 的镜像环境、源码补丁、必要精度基线、ONNX/OM/ATC 转换、torch_npu/TorchAir/vLLM-Ascend 推理、精度性能验证、README、上库脚本和 PR dry run 的任务。
---

# Ascend Model Adapter

用这个 skill 把一个上游/GPU 模型链接适配成 `ModelZoo-PyTorch/ACL_PyTorch/built-in` 风格的昇腾 NPU 推理项目。核心目标是 **GPU/官方口径 → NPU** 迁移：有对比时默认与官方/GPU 精度和性能比较；当前通常没有本地 GPU 环境，因此使用源仓/论文/官方发布的精度与性能作为参考。默认在 **NPU + Ascend 镜像/容器** 中执行；推理、评测和性能入口默认走 NPU，CPU 只作为代码分析、无官方精度时的 CPU/upstream 精度 baseline、ONNX 导出和生成待验证材料的 fallback，且必须显式指定。CPU-only 运行不得宣称 NPU 验证通过；CPU 性能对比没有迁移意义，默认不在 README/PR 中体现。

## 工作方式

1. **确定模型与目标目录**：确认模型 URL、模型名、任务类型、目标目录、目标对外硬件型号（如 Atlas 800I A2）、数据集/指标、是否允许下载权重和数据；同时按 `references/adaptation-process-log.md` 在工作区创建不上库的适配过程记录。
2. **参考最新 ModelZoo 样本**：运行 `scripts/modelzoo_sampler.py --count 20 --clone --out /tmp/modelzoo_samples.md`；优先读取与当前任务同类型、且最近合入的 README、patch、requirements、导出/推理脚本；需要写源码补丁时同时读取 `references/patch-modification-patterns.md`；如果本地已有完整/扩大 checkout，可运行 `scripts/patch_pattern_miner.py <ACL_PyTorch/built-in> --out /tmp/modelzoo_patch_patterns.md` 扩大 patch 参考范围。PR 检视意见只从这些已采样目录对应的上库 PR 中抽取：优先使用 sampler 表格里的 PR 号，并用 PR diff/变更文件确认触达该 `ACL_PyTorch/built-in/<category>/<model>` 路径；无法确认路径对应关系的 PR 只能作为“补充风格参考”，不得当作该模型样本的审查依据。可用 `scripts/gitcode_pr_review_sampler.py --prs <PR号...> --paths <采样目录...>` 抽取并校验；网络失败时读取 `references/modelzoo-sampling.md`。
3. **确定适配路线并生成工程**：按模型结构直接选择 `onnx-om`、`torch-npu`、`torchair`、`vllm-ascend` 或拆图混合路线。新项目可用 `python scripts/scaffold_adapter.py <model_url> ACL_PyTorch/built-in/<category>/<model> --category <category> --route <route>` 直接在最终上库目录生成扁平脚手架，不要先放到 `ascend_adapter/` 等临时子目录再搬迁。
4. **实现 baseline、导出、转换和推理**：按下面“适配流程”补齐源码 patch、必要的精度 baseline、Ascend 推理脚本、精度脚本和性能脚本；官方已有可复现精度指标时，不额外要求 CPU 精度对比；性能不做 CPU 对比。
5. **整理上库材料**：读取 `references/output-contract.md`，按 ModelZoo 风格完成 README、requirements、patch、脚本、日志、结果表和 checklist；把过程记录中的可复用问题沉淀到 README FAQ/已知问题，不把过程记录本身上库。
6. **PR 检视与 dry run**：用户给出上库 PR 时，读取 `references/pr-review-heuristics.md`，优先拉取 PR 当前 head/merge ref，运行 `scripts/modelzoo_pr_quickcheck.py <repo> --target <模型目录>` 做冲突标记、ModeList 统计、Python 编译、ruff、残留命令和性能口径快检；再对 patch、site-packages patch、数据准备脚本、benchmark/help 做可复现 dry run。CPU-only dry run 不替代真实 NPU 精度/性能验证。
7. **反思改进 skill**：如果某问题耗时很久、反复出现、依赖外部提示才解决，或导致路线/patch 重大调整，回看过程记录，判断是否应更新本 skill 的启发式、patch 模式、交付契约或脚本。

## 从样本内化出的默认判断

这些规则不要作为用户交付物单独输出，而要在写脚本、README 和 patch 时自然体现：

- 看到上游或子模块的 `requirements.txt`、`setup.py`、`pyproject.toml` 可能安装 `torch/torch_npu/torchvision/torchaudio` 时，先做 `python3 -c "import ..."` 依赖预检，再按 `pip install -r requirements.txt` → 必要的 `pip install -e ./子包` → 顶层 `pip install --no-deps -e .` 顺序安装；默认生成最小过滤版依赖、patch 依赖约束或写明安装顺序，避免破坏 Ascend 镜像内置版本。过滤只删除会阻塞或覆盖镜像栈的包，补齐实际缺失业务依赖；过滤后要用推理入口做 import/`--help`/单样例 smoke test。
- 看到需要升级核心框架版本时，默认先找成套 Ascend 镜像/CANN/torch/torch_npu；不要把低版本 CANN 镜像里 pip 升级另一套 torch/torch_npu 写成推荐可复现环境，除非已实测并解释配套来源。
- 看到源仓 README、论文、release、脚本或日志中已有 accuracy、benchmark 或 performance 数据时，默认分开处理：精度直接参考并对齐源仓/官方（通常是 GPU）accuracy 口径，不再额外要求 CPU 精度对比；与官方精度对比时，所选数据集必须使用官方相同的完整数据集/split、同一 checkpoint 和评测脚本，不能用小样本结果冒充官方对比；若官方列出多个数据集，可只评测其中一部分并写明选择和未评测项。性能直接参考并对齐源仓/官方（通常是 GPU）benchmark/performance 口径或说明差异；没有官方性能时只报告 NPU 可复现性能。CPU 性能对比没有意义，不在 README/PR 中体现。
- 看到模型有多个变体、配置文件或论文实验设置时，默认确认当前 checkpoint 对应的评测配置、聚类/阈值/beam/search 参数和数据预处理；不要只按默认 config 猜测。
- 如果使用者提供了 checkpoint、权重目录或模型包，默认以使用者提供的 artifact 为准；只在它缺失、不成套或无法复现时，才建议替代下载源或官方默认权重，并在 README 写清差异。
- 看到 `cuda`、`torch.cuda`、CUDA extension、`setup.py` 编译扩展、custom op 时，默认检查是否需要 NPU 等价实现、第三方 Ascend SDK、拆图或明确 CPU fallback。
- 看到上游硬编码推理后端选择（`onnxruntime`、`tensorrt`、`tf.saved_model`、`paddle.inference`、`OpenVINO` 等）时，默认评估 PyTorch/torch_npu/TorchAir 等价路径是否能让组件部署到 NPU；源码路由 patch 通常比运行时 monkey-patch 更清洁。
- 写推理后端替换时，区分“同架构同权重的后端替换”和“换模型/换权重”：前者默认做数值等效性验证，后者必须重新按任务指标评测并声明不能直接比较源仓结果。
- 看到 `torch.load` 加载可信旧 checkpoint 报 `UnpicklingError` 时，默认检查是否因 PyTorch 2.6+ `weights_only=True` 默认安全策略导致；只有在 checkpoint 来源可信时才准备 `weights_only=False` patch，框架自带 load 包装也要同步检查。
- 看到音频模型使用 `torchaudio.load` 时，默认检查 torchaudio 2.9+ 改用 TorchCodec 且 `backend` 参数被忽略带来的兼容问题；必要时用 `soundfile`/`librosa` 直接替代音频 I/O。
- 看到动态输入、符号 shape、多输入多输出、控制流、attention、RoPE、后处理入图时，默认准备 ONNX fix/shape 固化/onnxsim/onnxslim/MagicONNX/msit surgeon 或拆子图。
- 看到 pipeline 模型（OCR/VLM、CLIP、VLA、TTS、检测+识别）时，默认拆组件分别评估 NPU 可行性，优先将可 NPU 化组件部署至 NPU；只有存在具体技术阻塞时才接受 CPU fallback，并在 README FAQ 记录原因。
- 看到 vLLM/TorchAir/服务化模型时，默认提供服务启动命令、客户端命令、预热/编译缓存说明、并发配置和端到端性能口径。
- 写推理、评测、benchmark 或数据集批处理入口时，默认设备必须是 NPU（设备字符串只写 `npu`）；需要选择物理卡时，在命令前使用 `export ASCEND_RT_VISIBLE_DEVICES=<id>` 控制可见卡，不写 `npu:0`/`npu:<id>`。CPU 模式只能作为显式 `--device cpu`/`--backend cpu` fallback 或无官方指标时的 baseline，不能成为 README 主命令或脚本默认值。
- 写 README、性能表、PR 描述和对外交付材料时，硬件字段使用对外产品/整机型号，例如 `Atlas 800I A2`、`Atlas 800I A3`、`Atlas 300I DUO/Pro`；不要给出详细芯片型号、芯片步进或内部代号。`SOC_VERSION` 只作为 ATC/脚本参数保留，不作为公开性能表的硬件型号。
- 看到离线部署、多权重、标准数据集或测试样例时，默认写清权重/数据/评测工具/protocol/reference label 的来源、目录树、生成命令、缓存方式和最小验证数据；音频/图像等预处理必须按上游 README/论文/评测脚本精确复现，不能只写“用户自行准备”。
- 数据准备优先做成单一 `prepare_data.py`，让 Python 直接处理 tar/zip、目录展开、manifest/RTTM/scp 生成和必要的音频/图像转换；只有依赖系统级工具编排时才加 shell 包装，避免 README 同时引用重复脚本。
- 看到首次编译、CPU 回退、长时间 ATC、环境隔离、patch 只能执行一次等情况时，默认先写入不上库的适配过程记录，再把用户需要复现/避坑的结论整理进 README FAQ/注意事项。
- 写源码 patch 时默认采用参考 patch 的最小化模式：设备选择参数化、推理后端只替换核心调用、保留原预后处理、对不支持算子用等价表达/拆图/明确 CPU fallback，并保证 `git apply --check` 可复现。优先 patch 上游推理/评测入口支持 NPU；只有上游没有统一入口时才新增脚本。
- 上库前默认按“已采样目录对应上库 PR”的检视口径自查：冲突标记是否清理、ModeList 统计是否和表格行数一致、精度数据是否可复现、性能单位与脚本默认参数是否匹配任务、对外硬件型号是否准确、上游 commit 是否固定、权重与配置是否成套、外部数据文件/子模块是否说明来源、debug code/重复文档/残留 import 是否清理。若 PR 与采样目录没有可验证路径对应关系，只吸收通用 CI/文档风格，不作为模型特定证据。

详细启发式见 `references/adaptation-heuristics.md`；patch 修改模式见 `references/patch-modification-patterns.md`；适配过程记录模板见 `references/adaptation-process-log.md`；上库前审查口径见 `references/pr-review-heuristics.md`。这些参考只用于影响实现和文档写法，不作为额外交付物；其中过程记录明确不上库。

## 环境原则

- 默认使用 Ascend 镜像；裸机只作为补充说明。
- 必须记录：`npu-smi info`、`source /usr/local/Ascend/ascend-toolkit/set_env.sh`、`atc --version`、Python、CANN、torch、torch_npu、torchvision/torchaudio、ais_bench、msit、TorchAir/vLLM-Ascend 版本。
- ONNX/OM PyTorch 任务优先参考近期样本中的 `swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-onnx-inference:*` 镜像；vLLM 任务优先参考 `quay.io/ascend/vllm-ascend:*`。无法确定 tag 时，把镜像 tag 参数化，并写明所需 CANN/PyTorch/torch_npu 版本。
- 不要在已配套的 Ascend 镜像里随意重装 `torch`/`torch_npu`。业务依赖单独 pin；如必须修复版本冲突，先解释原因并记录恢复命令。
- CPU fallback 可以 clone、分析源码、安装纯 Python 依赖、在无官方精度数据时跑必要的 CPU/upstream 精度 baseline、导出 ONNX；推理/评测/benchmark 的默认运行模式仍必须是 NPU，CPU fallback 需要显式参数和具体原因。不能伪造 ATC、OM、NPU 精度或 NPU 性能结果。CPU 性能数据不作为交付或对比项。

## 路线选择

- **ONNX → ATC → OM**：默认离线推理路线。适合输入 shape 可控、ONNX 可导出、需要 ais_bench/ACL 性能数据的模型。
- **TorchAir / torch_npu 图模式**：适合 ONNX 导出脆弱、输入动态、上游强依赖 PyTorch control flow，或参考项目已有 TorchAir 路线的模型。
- **vLLM-Ascend**：适合 LLM/VLM/TTS 服务化模型，特别是依赖 vLLM 调度、paged attention、OpenAI API 或 FastAPI server 的项目。
- **直接 torch_npu eager**：只作为初步正确性验证或上游已接受路线；仍需给出性能测试方式。
- **拆图/多子模型**：遇到多模态、CLIP、VLA、OCR pipeline、检测+识别、VLM+action expert 时，优先把文本、视觉、检测、专家网络等子图分开导出/转换/验证。

## 适配流程

1. **上游审计**
   - 固定源码 commit/revision、权重版本、license、模型任务、输入输出、预处理/后处理、评测数据和官方指标；如使用者提供 checkpoint，优先围绕该 checkpoint 确认配置、版本和评测口径。主动查找源仓 README/论文/release/benchmark 脚本/评测日志中的官方/GPU accuracy 与 performance 数据；分别记录精度命令/数据集/metric，以及性能命令/batch/并发/输入规格/warmup/loop/统计口径。
   - 同时检查依赖是否覆盖镜像栈、是否硬编码 CUDA、是否有 custom op、是否有动态 shape、是否在线下载权重/数据、是否存在多组件流水线；依赖检查要覆盖子模块和本地 vendor 包的 requirements/setup/pyproject。
   - 固定上游 commit，确认模型版本、权重、配置文件成套；如 README 示例使用变体版本，必须解释对应关系，并确认该变体对应的评测参数、数据预处理和后处理配置。
   - 审计推理后端路由：若上游硬编码 ONNX Runtime、TensorFlow、Paddle inference 等 CPU-only/非 Ascend 后端，先评估是否有 PyTorch/torch_npu/TorchAir 等价路径。
2. **Pipeline 组件部署分析**（多组件流水线默认执行）
   - 列出每个可独立推理的子模型/组件，评估是否有 PyTorch 实现、能否迁移 NPU、是否存在 NPU 不支持算子或框架限制。
   - 目标是最大化 NPU 利用率；CPU fallback 必须有具体阻塞原因，不能只写“上游默认用 CPU/ONNX Runtime”。
3. **建立精度 baseline**
   - 先判断源仓/论文/官方 README 是否已有可复现 accuracy 指标。若已有，默认不做 CPU 精度对比，只保留必要的 import/单样例/中间张量 smoke test 作为排障证据；NPU 精度直接对齐官方/GPU 指标。
   - 与官方精度对比时，必须使用官方相同的完整数据集或官方指定 split、同一 checkpoint/配置/阈值/随机种子/评测脚本。若官方列出多个数据集，可只选择其中一部分评测，但被选择的数据集必须完整评测；小样本/子集只能写成 smoke test 或输出对齐，不能写成官方指标对比。
   - 只有源仓没有可复现 accuracy/官方指标或原始指标无法复现时，才实现 CPU/upstream baseline，并使用同一输入集与 NPU 输出对齐；数值模型给 atol/rtol/cosine，任务模型给替代指标和原因。
   - 对 ASR/OCR/检测/分割/检索/生成等任务，按原始任务指标计算，不要用“能输出文件/截图”替代精度。
4. **实现 Ascend 路径**
   - 源码修改统一做成 `diff.patch` 或 `<model>_NPU.patch`，README 写明应用路径、固定 commit、`git apply --check` 和是否可重复执行。
   - 优先 patch 上游已有 `infer.py`、`inference.py`、`test.py`、`demo.py`、评测脚本或命令入口，增加 `--device`/环境变量/配置项支持 NPU，且默认值设为 NPU；CPU 只能显式选择。不要为了模板完整性另写重复脚本。
   - ONNX/OM 路线：需要离线推理时实现 `export_onnx.py`/`pth2onnx.py`、必要的 ONNX fix/optimize、`convert_om.sh`；推理/评测优先复用或 patch 上游入口，缺口才新增脚本。
   - TorchAir/vLLM 路线：实现镜像启动、环境变量、图编译/缓存、服务启动、客户端推理和性能统计。
   - 对 unsupported op 使用等价替换、拆图、MagicONNX/msit surgeon、custom op 或 CPU fallback，并在性能说明中区分纯 NPU 与端到端。
   - 暴露关键参数：权重路径、ONNX 输出、OM 输出、batch、soc_version、ASCEND_RT_VISIBLE_DEVICES、数据路径；避免硬编码本地路径。
5. **NPU 验证**
   - 默认在容器中跑：环境检查 → 导出 → 转换/编译 → 单样例 NPU 推理 → NPU 精度验证（优先官方完整数据集/split accuracy；无官方指标时才 CPU/NPU 对齐）→ NPU 性能；README 和脚本中的主命令默认不需要用户额外传 CPU/NPU 开关也应走 NPU。
   - 多数据集或多配置评测时，如有多张空闲 NPU，可把互不依赖的评测任务用 `export ASCEND_RT_VISIBLE_DEVICES=<id>` 分发并行，脚本内 `--device` 仍传 `npu`；先用 `npu-smi info -t usages` 或进程检查确认 HBM/AI Core 空闲，日志按数据集和可见卡 ID 分开保存。
   - 记录对外硬件型号（如 Atlas 800I A2）、batch/并发、输入 shape、精度模式、warmup、loop、latency/FPS/QPS/RTF、日志路径。
6. **文档与上库**
   - 按 `references/output-contract.md` 完成目录和 README。
   - 从适配过程记录中提取用户复现需要知道的 FAQ/已知问题，例如依赖冲突、cache 清理、patch 只能执行一次、环境隔离、动态库路径、下载替代源；不要把完整过程记录上库。
   - 清理 debug code、无用注释、重复段落、残留 import 和不清晰变量名；PR 描述按 Motivation/Modification/Self-test/BC-breaking/Checklist 填写，不能保留模板占位。
   - 结果没有实测时写 `待 NPU 验证`；不要用“理论支持”代替验证结果。
7. **反思与回写**
   - 如果某个问题排查超过约 30 分钟、需要外部提示才解决、反复出现、导致路线变更，或暴露 skill 没有覆盖的检查点，必须在过程记录中写明症状、无效尝试、根因、最终修复和可复用性。
   - 只有当问题具备通用性时，才反向更新 skill 的参考文件或脚本；模型私有路径、账号、临时日志和内部数据不得写入 skill。

## 精度与性能指标选择

- **Accuracy 默认使用上游原始指标**：优先复用原始项目或论文/官方 README 的评测数据、预处理后处理、metric 和阈值；如果源仓已经给出 accuracy 数字或评测表，默认把它作为官方/GPU 参考，NPU 结果应对齐同一 checkpoint、官方相同的完整数据集/split、随机种子、阈值和脚本口径，不需要再提供 CPU 精度对比。同类 ModelZoo 样本已有固定口径时，优先保持 ModelZoo 口径一致。
- 官方给出多个数据集/benchmark 表时，可选择其中一部分数据集评测，但必须明确列出已评测与未评测数据集；对已评测数据集必须使用完整官方数据集/split。仅用小样本或子集时，只能作为 smoke test/输出对齐，不能与官方完整集指标直接比较。
- 只有在源仓没有可复现 accuracy/官方指标或原始指标无法复现时，才使用同一输入集的 CPU/upstream baseline 与 NPU 输出对齐，并说明替代原因；数值模型给出 atol/rtol 或 cosine similarity，任务模型给出任务指标差异。
- 不要随意创造更好看的指标。指标口径不同（例如 UVDoc 类样本）时，明确写“不能与官方直接比较”。
- **Benchmark 默认只报告 NPU 性能，并与官方/GPU 口径对齐**；如果源仓已有官方/GPU 性能数据或 benchmark 脚本，NPU 结果应尽量复用同一输入规格、batch/并发、warmup/loop、统计区间、端到端/纯模型定义和单位，并说明硬件差异是否可直接比较。当前默认没有本地 GPU 环境，不要求重新跑 GPU；没有官方/GPU 性能口径时，只给 NPU 可复现性能，不做 CPU 性能对比，也不在 README/PR 中体现 CPU 性能。没有原始性能口径时，才按路线选择 ModelZoo 常用口径：OM 用 `ais_bench` latency/FPS，服务模型用 QPS/tokens/s/端到端 latency，音频/TTS/ASR 默认用 RTF（耗时/音频时长，越低越好）作为主指标，可补充 RTFx=1/RTF（实时倍速，越高越好）或任务吞吐，pipeline 同时给纯模型和端到端。
- 首次图编译/首次 warmup、数据加载、CPU fallback、后处理耗时要单独说明，不混入稳定纯推理性能。
- 性能表、脚本输出和 README/PR 描述中的数字与单位必须一致。

## 产物要求

至少交付：

- `README.md`：中文 ModelZoo 风格说明，包含镜像、环境、源码、权重、数据、转换/服务启动、推理、精度、性能和 FAQ。
- `requirements.txt`：只放业务依赖；镜像内置的 torch/torch_npu/torchvision/torchaudio 默认不要写入。
- 所有上游修改的 patch（`diff.patch` 或 `<model>_NPU.patch`）。
- ONNX/OM 路线额外提供导出/转换脚本；上游缺少推理、评测或性能入口时，才新增对应脚本。
- 数据准备脚本如需提供，默认单一 `prepare_data.py`，不要同时提交功能重复的 `.sh` 包装脚本。
- 环境与验证日志，或 CPU-only 的 `待 NPU 验证` 报告。

## 完成标准

只有满足以下条件，才能称为“可上库”：

- 目录符合 `ACL_PyTorch/built-in/<category>/<model>` 风格。
- 干净 clone + 指定镜像可复现主要命令。
- 已在目标 Ascend 硬件执行 NPU 推理；CPU-only 只算材料准备完成。
- 精度与源仓/官方指标有明确容差或任务指标对比；源仓已有 accuracy 数据时优先对齐官方完整数据集/split 口径，不额外要求 CPU 精度对比。无官方可复现指标时，才使用 CPU/upstream baseline 与 NPU 对齐。
- NPU 性能测试可复现；源仓已有官方/GPU benchmark/performance 数据时优先对齐源仓口径，并说明 warmup/loop/batch/并发/输入规格/对外硬件型号；不要求本地 GPU 复测，也不提供 CPU 性能对比。
- 已记录已知问题、支持/限制硬件型号、长时间编译、custom op、离线下载和依赖冲突；复杂排障和关键决策已写入不上库的适配过程记录，并提炼必要 FAQ。
- 已通过本地自检避免常见 PR 检视问题：CodeCheck/SCA/Antipoison、模板占位、缺精度数据、缺对外硬件型号说明、未固定 commit、性能口径不匹配、外部文件来源不明。
