# 适配启发式

这些启发式来自当前 ModelZoo `ACL_PyTorch/built-in` 中已适配项目的 README、patch、requirements 和脚本。它们不是额外交付步骤，而是写适配代码和文档时的默认判断。

## 环境与依赖

- 近期样本多次强调不要在 Ascend 镜像中重装 `torch`/`torch_npu`。处理上游 requirements 时，默认过滤或固定会覆盖镜像栈的包。
- README 中的安装命令必须明确当前工作目录和 requirements 来源；在上游仓、子模块或 vendor 包内执行 `pip install -r requirements.txt`、`pip install -e .`、`pip install .` 时，先审计 `requirements.txt`、`setup.py`、`pyproject.toml` 是否会安装/升级 `torch`、`torch_npu`、`torchvision`、`torchaudio`。否则提供过滤版 requirements、patch 依赖约束，或显式 `grep -vE`/`--no-deps` 命令。
- 过滤或 patch 依赖后，默认用上游推理/评测入口做 import smoke test（例如 `python -c` 导入入口模块或跑 `--help`/单样例），补齐被过滤掉但实际需要的业务依赖；不要只凭 requirements 文件静态推断。
- `torch_npu` 导入的 `undefined symbol`、`aclruntime` wheel ABI、Python 版本不匹配通常是版本栈问题，先查环境再改模型。
- 默认给出已验证的配套环境：镜像/CANN/torch/torch_npu/torchvision/torchaudio/Python 必须成套。不要在低版本 CANN 镜像中直接 pip 升级到另一套 torch/torch_npu 作为推荐环境；若确需自建环境，写成 Dockerfile 或明确“待验证”，并给出配套来源。
- 基于已有适配材料重新验证时，先列环境差异表（芯片、CANN、Python、torch、torch_npu、torchvision/torchaudio、关键第三方库），逐条判断旧 patch 是否仍必要或变成冗余；版本相关 FAQ 只能按当前验证环境和兼容环境分条件描述。
- README 的镜像、Python、CANN、torch/torch_npu/torchaudio 版本、FAQ 和 patch 说明必须相互一致；只在对应版本会触发的问题前标注版本条件，避免用 torch 2.1 镜像却无条件写 PyTorch 2.6+/torchaudio 2.9+ 问题。
- Paddle、vLLM、TorchAir、DrivingSDK、OpenCV、tesseract 等依赖容易互相冲突；多组件流水线默认允许拆环境。
- 使用者提供 checkpoint、权重目录或模型包时，默认以该 artifact 为准；只在文件缺失、配置不成套或无法复现时建议替代权重，并记录差异。
- PyTorch 2.6+ 的 `torch.load` 默认安全策略会使用 `weights_only=True`；可信旧 checkpoint 若包含自定义类并触发 `UnpicklingError`，可 patch 为 `weights_only=False`，但必须确认来源可信。lightning/pyannote 等框架自带 load 包装也要同步检查。
- torchaudio 2.9+ 的 `torchaudio.load` 会改用 TorchCodec，`backend` 参数会被忽略；Ascend 镜像中若因 torchcodec/FFmpeg/ABI 失败，直接改用 `soundfile`/`librosa` 读写音频。

## 源码 patch

- 大多数适配项目都有 patch。常见修改点：设备选择、CUDA 假设、custom op、导出脚本、attention/位置编码、数据加载、后处理、评测脚本。详细修改模式见 `patch-modification-patterns.md`。
- patch 要基于固定 commit，README 写明应用位置和 `git apply --check`。如果 patch 只能执行一次，写入 FAQ。默认保持最小补丁：只改适配必需路径，不把调试输出、本地路径、无关重构带入。
- 遇到 `cuda`、`torch.cuda`、`USE_CUDA`、CUDA extension、`setup.py` 编译扩展时，优先改为 NPU 等价路径；不能改的部分标明 CPU fallback。
- 优先 patch 上游推理/评测脚本：如果上游已有入口，通过 patch 增加 NPU 设备参数；只有上游没有统一入口或需要组合多个子模块时才新增脚本。
- 推理后端路由 patch：当上游根据配置硬编码 ONNX Runtime、TF SavedModel、Paddle inference、OpenVINO 等后端时，先评估同一项目内是否有 PyTorch 等价类可迁移到 NPU。在 ModelZoo Ascend 默认 PyTorch/torch_npu 适配场景中，ONNX Runtime 通常不能直接驱动 Ascend NPU，容易退化为 CPU 路径。
- 后端替换与模型替换要分清：同一架构同一权重从 ORT/TF/Paddle 切到 PyTorch/torch_npu 属于后端替换，默认做 logits/embedding/输出张量数值等效性验证；若更换 checkpoint、模型类、tokenizer、聚类策略、阈值或预后处理，则属于口径变化，必须重新跑任务指标并声明不能直接比较源仓结果。

## ONNX 与 OM

- 动态 shape、符号维、多输入多输出、control flow、attention、RoPE、后处理入图是导出高风险点。默认准备 ONNX checker、shape 固化、onnxsim/onnxslim、MagicONNX 或 msit surgeon。
- 多模态模型优先拆子图，例如 text encoder / vision encoder、VLM / action expert、layout detector / OCR-VLM。
- ATC 命令必须参数化 `SOC_VERSION`、`input_shape`、精度模式，并记录 `npu-smi info`、`atc --version`。长时间 ATC 编译要在 README 预警。

## 推理运行

- OM 路线默认提供导出/转换与可用 `ais_bench` 的 benchmark 说明；推理入口优先 patch 上游脚本，确需新增时使用单一脚本和 `--device npu/cpu` 参数，不默认拆成 `infer_cpu.py`/`infer_npu.py`。TorchAir/vLLM 路线默认提供服务启动、客户端脚本、编译缓存和并发说明。
- vLLM/TorchAir 首次图编译耗时不能直接算入稳定性能，除非表格单独标明。
- 若 pipeline 中存在 CPU 回退，性能结果要拆分纯 NPU 子模型与端到端耗时。

## 权重、数据和离线部署

- 多权重模型默认写权重清单、来源、目录树、离线缓存方式。
- 大数据集默认写容量、分包、官方来源、申请入口或生成脚本，以及最小验证子集；测试数据、评测工具、protocol、reference label/RTTM 都必须可追溯，不能只写“用户自行准备”。
- README 中提到的每个外部资源都要进入公网地址说明，包括 issue/release note、论文、数据集、评测工具、protocol、样例数据来源和关键预处理工具；不要依赖未列入交付件清单的相对文档链接。
- 音频/图像/视频预处理必须精确复现上游 README、论文实验设置或评测脚本，包括多声道到单声道、采样率、裁剪/resize/crop、归一化、padding、重采样工具和命令；不能用“等价直觉”替代。若改动预处理，必须做中间结果或任务指标对齐。
- 评测依赖和推理依赖可分开，例如 `requirements_eval.txt`。

## Pipeline 组件部署分析

- 多组件流水线（diarization、OCR、VLM、TTS、检测+识别等）默认逐组件评估 NPU 可行性。
- 对每个组件检查：是否有 PyTorch 实现、能否 `.to(device)` 迁移到 NPU、是否有不支持算子、是否被硬编码到 ONNX/TF/Paddle/OpenVINO 等后端、是否有等价 PyTorch 路径。
- CPU fallback 必须有具体技术阻塞，不能只因为上游默认 CPU-only 后端就照搬。
- 概述、组件表、性能表必须一致：如果聚类/后处理在 CPU，就不要写“全部组件均在 NPU”；可写“核心模型在 NPU，聚类/后处理 CPU fallback”。

## 源仓指标对齐

- 上游 README、论文、release notes、benchmark 脚本、评测日志中已有 accuracy/performance 数据时，默认分开处理：精度对齐源仓 accuracy 口径；性能对齐源仓 benchmark 口径并给出 NPU 结果。
- 精度对齐前记录：checkpoint/权重版本、模型变体、数据集或子集、随机种子、预处理/后处理、metric、阈值、聚类/beam/search 参数、评测工具版本和关键选项（如 collar、overlap、ignore 区域）。源仓没有可复现 accuracy 时，才使用 CPU/upstream baseline 与 NPU 输出对齐。
- 多模型变体或多套配置时，优先查 README benchmark 表、论文实验章节和 release 说明，确认当前 checkpoint 对应哪套评测配置；不要只看默认 config 或示例脚本。若默认 config 与 benchmark 口径不同，README 必须显式说明。
- 只有模型组件、checkpoint、预处理/后处理和评测脚本口径与源仓一致时，才能把 NPU 结果和源仓 accuracy 表直接对齐；若替换了嵌入模型、tokenizer、label map、聚类策略或阈值，只能写“参考源仓指标，当前口径不同”，并给出当前口径验证。
- 性能对齐前记录：输入规格、batch/并发、warmup/loop、统计区间、端到端/纯模型定义、是否包含数据加载/后处理/首次编译。性能只要求 NPU 可复现，不默认采集或比较本地 CPU 性能。
- NPU 结果优先复用上游评测/benchmark 脚本或 patch 后的同一入口；不能复用时，README 必须解释差异，不能直接换成更容易或更好看的口径。
- 若硬件不同导致性能不可直接比较，仍保留源仓数据作为参考基线，并明确写“硬件/口径不同，不直接比较”；同时给出 NPU 可复现命令和结果。

## 精度指标

- 默认复用原始项目的官方 metric、数据集和预处理后处理；源仓已有 accuracy 表或评测命令时，优先对齐该表/命令。
- 只有源仓没有可复现 accuracy/官方指标或官方评测不可复现时，才用 CPU/upstream baseline 与 NPU 对齐，并说明替代原因。
- CPU/NPU 输出对齐不能冒充任务指标：没有 reference label/RTTM/GT 时，不要把边界差、输出一致率写成 DER、WER、mAP 等官方 metric；应命名为“输出对齐/RTTM 边界差/cosine diff”等替代指标。
- 常见任务口径：分类 top-1/top-5，检测 mAP，分割 IoU/mIoU，OCR/VLM 用官方 end-to-end 指标，ASR 用 WER/CER/BLEU，TTS/音频生成同时给样例和 RTF/RTFx，embedding/reranker 用检索或排序指标，LLM/VLM 服务用任务评测或语义一致性样例。
- 指标计算方式不同则不与官方直接比较。

## 性能指标

- 优先复用原始项目已有 benchmark/performance 口径或同类 ModelZoo 样本口径；源仓有 benchmark 脚本时优先 patch 后在 NPU 运行。性能数据不要求与本地 CPU 对比。
- OM 纯模型默认 `ais_bench` latency/FPS；服务模型默认 QPS、tokens/s、端到端 latency；音频默认 RTF/RTFx；pipeline 默认同时给纯模型和端到端。
- 每个性能表都写芯片、batch/并发、输入规格、精度模式、warmup、loop、工具、是否包含数据加载/前处理/后处理/聚类/模型加载/首次编译/cache miss/CPU fallback；纯模型耗时和 pipeline 端到端耗时不能混写，必要时分两张表。
- 性能表默认只放 NPU；如为了排障临时测了 CPU，放入过程记录或附注，不作为主表对比项，除非用户或上库规范明确要求。

## 过程记录与反思改进

- ModelZoo README 中常见的 FAQ/注意事项（如 msit 安装失败、TorchAir cache 需删除、patch 只能执行一次、环境隔离、动态库路径、下载失败、依赖冲突）通常来自实际踩坑。适配时先把完整排障过程写入不上库的过程记录，再把用户需要复现的结论提炼到 README。
- 遇到耗时较久、反复出现、依赖外部提示、导致路线变更或 CPU fallback 的问题，必须记录：症状、阶段、失败尝试、根因、最终修复、耗时、是否外部提示、是否应反向改进 skill。
- 每个关键决策都要能回溯依据：为什么选 ONNX-OM/TorchAir/vLLM/拆图，为什么 patch 上游入口，为什么接受 CPU fallback，为什么替换依赖或权重。
- 适配完成后做一次反思：哪些问题本应由 skill 提前提示？如果是通用规律，更新 `adaptation-heuristics.md`、`patch-modification-patterns.md`、`output-contract.md` 或脚本；如果只是本模型私有问题，只保留在过程记录和 README FAQ。

## PR 检视默认自检

- 提交前按 `pr-review-heuristics.md` 的审查口径自检，但不要把它变成用户交付物。
- 默认补齐精度验证数据、性能单位、芯片/机器型号、上游 commit、权重/配置成套关系、外部数据文件来源。
- README 中的命令必须至少做静态自检，避免明显语法错误、错误环境变量名、错误相对路径、未定义脚本或未随仓交付的辅助文件。
- 默认清理 debug code、重复 README 段落、无用注释、残留 import、硬编码路径和不清晰变量名。
- 默认确保 PR 描述不保留模板占位，Self-test 写出环境、转换、推理、精度和性能证据。
