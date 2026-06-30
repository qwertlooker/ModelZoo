---
name: ascend-model-adapter
description: 将 Hugging Face、GitHub、PyTorch、ONNX、Paddle、vLLM、TorchAir 或研究模型适配到华为昇腾 Ascend NPU，并产出可提交 Ascend ModelZoo-PyTorch ACL_PyTorch/built-in 的材料。适用于给定模型链接后需要完成镜像环境、源码补丁、CPU/NPU 基线、ONNX/OM/ATC 转换、torch_npu/TorchAir/vLLM-Ascend 推理、精度性能验证、README 和上库脚本的任务。
---

# Ascend Model Adapter

用这个 skill 把一个上游模型链接适配成 `ModelZoo-PyTorch/ACL_PyTorch/built-in` 风格的昇腾 NPU 推理项目。默认在 **NPU + Ascend 镜像/容器** 中执行；CPU 只作为代码分析、CPU baseline、ONNX 导出和生成待验证材料的 fallback。CPU-only 运行不得宣称 NPU 验证通过。

## 工作方式

1. **确定模型与目标目录**：确认模型 URL、模型名、任务类型、目标目录、目标芯片、数据集/指标、是否允许下载权重和数据。
2. **参考最新 ModelZoo 样本**：运行 `scripts/modelzoo_sampler.py --count 20 --clone --out /tmp/modelzoo_samples.md`；优先读取与当前任务同类型、且最近合入的 README、patch、requirements、导出/推理脚本。若 PR discussion 可访问，可用 `scripts/gitcode_pr_review_sampler.py --prs <PR号...>` 抽取检视意见；网络失败时读取 `references/modelzoo-sampling.md`。
3. **确定适配路线并生成工程**：按模型结构直接选择 `onnx-om`、`torch-npu`、`torchair`、`vllm-ascend` 或拆图混合路线。新项目可用 `python scripts/scaffold_adapter.py <model_url> <project_dir> --category <category> --route <route>` 生成脚手架。
4. **实现 baseline、导出、转换和推理**：按下面“适配流程”补齐源码 patch、CPU/upstream baseline、Ascend 推理脚本、精度脚本和性能脚本。
5. **整理上库材料**：读取 `references/output-contract.md`，按 ModelZoo 风格完成 README、requirements、patch、脚本、日志、结果表和 checklist。

## 从样本内化出的默认判断

这些规则不要作为用户交付物单独输出，而要在写脚本、README 和 patch 时自然体现：

- 看到上游 `requirements.txt` 可能安装 `torch/torch_npu/torchvision/torchaudio` 时，默认生成过滤版依赖或写明安装顺序，避免破坏 Ascend 镜像内置版本。
- 看到 `cuda`、`torch.cuda`、CUDA extension、`setup.py` 编译扩展、custom op 时，默认检查是否需要 NPU 等价实现、第三方 Ascend SDK、拆图或明确 CPU fallback。
- 看到动态输入、符号 shape、多输入多输出、控制流、attention、RoPE、后处理入图时，默认准备 ONNX fix/shape 固化/onnxsim/onnxslim/MagicONNX/msit surgeon 或拆子图。
- 看到 pipeline 模型（OCR/VLM、CLIP、VLA、TTS、检测+识别）时，默认拆组件分别导出、转换、验证，并把环境冲突的组件隔离。
- 看到 vLLM/TorchAir/服务化模型时，默认提供服务启动命令、客户端命令、预热/编译缓存说明、并发配置和端到端性能口径。
- 看到离线部署或多权重依赖时，默认写清权重清单、目录树、下载源、缓存方式和最小验证数据。
- 看到首次编译、CPU 回退、长时间 ATC、环境隔离、patch 只能执行一次等情况时，默认写入 README 的 FAQ/注意事项。
- 上库前默认按近期 PR 检视口径自查：精度数据是否可复现、性能单位是否匹配任务、芯片/机器型号是否准确、上游 commit 是否固定、权重与配置是否成套、外部数据文件是否说明来源、debug code/重复文档/残留 import 是否清理。

详细启发式见 `references/adaptation-heuristics.md`；上库前审查口径见 `references/pr-review-heuristics.md`。这些参考只用于影响实现和文档写法，不作为额外交付物。

## 环境原则

- 默认使用 Ascend 镜像；裸机只作为补充说明。
- 必须记录：`npu-smi info`、`source /usr/local/Ascend/ascend-toolkit/set_env.sh`、`atc --version`、Python、CANN、torch、torch_npu、torchvision/torchaudio、ais_bench、msit、TorchAir/vLLM-Ascend 版本。
- ONNX/OM PyTorch 任务优先参考近期样本中的 `swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-onnx-inference:*` 镜像；vLLM 任务优先参考 `quay.io/ascend/vllm-ascend:*`。无法确定 tag 时，把镜像 tag 参数化，并写明所需 CANN/PyTorch/torch_npu 版本。
- 不要在已配套的 Ascend 镜像里随意重装 `torch`/`torch_npu`。业务依赖单独 pin；如必须修复版本冲突，先解释原因并记录恢复命令。
- CPU fallback 可以 clone、分析源码、安装纯 Python 依赖、跑 CPU baseline、导出 ONNX；不能伪造 ATC、OM、NPU 精度或 NPU 性能结果。

## 路线选择

- **ONNX → ATC → OM**：默认离线推理路线。适合输入 shape 可控、ONNX 可导出、需要 ais_bench/ACL 性能数据的模型。
- **TorchAir / torch_npu 图模式**：适合 ONNX 导出脆弱、输入动态、上游强依赖 PyTorch control flow，或参考项目已有 TorchAir 路线的模型。
- **vLLM-Ascend**：适合 LLM/VLM/TTS 服务化模型，特别是依赖 vLLM 调度、paged attention、OpenAI API 或 FastAPI server 的项目。
- **直接 torch_npu eager**：只作为初步正确性验证或上游已接受路线；仍需给出性能测试方式。
- **拆图/多子模型**：遇到多模态、CLIP、VLA、OCR pipeline、检测+识别、VLM+action expert 时，优先把文本、视觉、检测、专家网络等子图分开导出/转换/验证。

## 适配流程

1. **上游审计**
   - 固定源码 commit/revision、权重版本、license、模型任务、输入输出、预处理/后处理、评测数据和官方指标。
   - 同时检查依赖是否覆盖镜像栈、是否硬编码 CUDA、是否有 custom op、是否有动态 shape、是否在线下载权重/数据、是否存在多组件流水线。
   - 固定上游 commit，确认模型版本、权重、配置文件成套；如 README 示例使用变体版本，必须解释对应关系。
2. **建立 baseline**
   - 实现最小 CPU/upstream 推理，保存样例输入输出、shape、dtype、任务指标或语义输出。
   - baseline 默认使用上游原始预处理、后处理、数据集和指标；无法跑完整数据集时，至少提供单样例和小集合。
   - 对 ASR/OCR/检测/分割/检索/生成等任务，按原始任务指标计算，不要用“能输出文件/截图”替代精度。
3. **实现 Ascend 路径**
   - 源码修改统一做成 `diff.patch` 或 `<model>_NPU.patch`，README 写明应用路径、固定 commit、`git apply --check` 和是否可重复执行。
   - ONNX/OM 路线：实现 `export_onnx.py`/`pth2onnx.py`、必要的 ONNX fix/optimize、`convert_om.sh`、`infer_npu.py`、`eval_accuracy.py`、`benchmark.sh`。
   - TorchAir/vLLM 路线：实现镜像启动、环境变量、图编译/缓存、服务启动、客户端推理和性能统计。
   - 对 unsupported op 使用等价替换、拆图、MagicONNX/msit surgeon、custom op 或 CPU fallback，并在性能说明中区分纯 NPU 与端到端。
   - 暴露关键参数：权重路径、ONNX 输出、OM 输出、batch、soc_version、device_id、数据路径；避免硬编码本地路径。
4. **NPU 验证**
   - 默认在容器中跑：环境检查 → 导出 → 转换/编译 → 单样例推理 → CPU/NPU 对齐 → 数据集精度 → 性能。
   - 记录芯片、batch/并发、输入 shape、精度模式、warmup、loop、latency/FPS/QPS/RTF、日志路径。
5. **文档与上库**
   - 按 `references/output-contract.md` 完成目录和 README。
   - 清理 debug code、无用注释、重复段落、残留 import 和不清晰变量名；PR 描述不能保留模板占位。
   - 结果没有实测时写 `待 NPU 验证`；不要用“理论支持”代替验证结果。

## 精度与性能指标选择

- **Accuracy 默认使用上游原始指标**：优先复用原始项目或论文/官方 README 的评测数据、预处理后处理、metric 和阈值；同类 ModelZoo 样本已有固定口径时，优先保持 ModelZoo 口径一致。
- 如果原始指标无法复现，使用同一输入集的 CPU/upstream baseline 与 NPU 输出对齐，并说明替代原因；数值模型给出 atol/rtol 或 cosine similarity，任务模型给出任务指标差异。
- 不要随意创造更好看的指标。指标口径不同（例如 UVDoc 类样本）时，明确写“不能与官方直接比较”。
- **Benchmark 默认使用原始项目可比口径**；没有原始性能口径时，按路线选择 ModelZoo 常用口径：OM 用 `ais_bench` latency/FPS，服务模型用 QPS/tokens/s/端到端 latency，音频/TTS/ASR 用 RTF/RTFx 或任务吞吐，pipeline 同时给纯模型和端到端。
- 首次图编译/首次 warmup、数据加载、CPU fallback、后处理耗时要单独说明，不混入稳定纯推理性能。
- 性能表、脚本输出和 README/PR 描述中的数字与单位必须一致。

## 产物要求

至少交付：

- `README.md`：中文 ModelZoo 风格说明，包含镜像、环境、源码、权重、数据、转换、推理、精度、性能、FAQ 和上库 checklist。
- `requirements.txt`：只放业务依赖；镜像内置的 torch/torch_npu 默认不要写入。
- 导出/转换/推理/评测/性能脚本。
- 所有上游修改的 patch。
- 环境与验证日志，或 CPU-only 的 `待 NPU 验证` 报告。

## 完成标准

只有满足以下条件，才能称为“可上库”：

- 目录符合 `ACL_PyTorch/built-in/<category>/<model>` 风格。
- 干净 clone + 指定镜像可复现主要命令。
- 已在目标 Ascend 芯片执行 NPU 推理；CPU-only 只算材料准备完成。
- 精度与 CPU/官方指标有明确容差或任务指标对比。
- 性能测试可复现，说明 warmup/loop/batch/并发/输入规格/芯片。
- 已记录已知问题、限制芯片、长时间编译、custom op、离线下载和依赖冲突。
- 已通过本地自检避免常见 PR 检视问题：CodeCheck/SCA/Antipoison、模板占位、缺精度数据、缺芯片获取步骤、未固定 commit、性能口径不匹配、外部文件来源不明。
