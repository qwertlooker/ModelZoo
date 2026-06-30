# 适配启发式

这些启发式来自当前 ModelZoo `ACL_PyTorch/built-in` 中已适配项目的 README、patch、requirements 和脚本。它们不是额外交付步骤，而是写适配代码和文档时的默认判断。

## 环境与依赖

- 近期样本多次强调不要在 Ascend 镜像中重装 `torch`/`torch_npu`。处理上游 requirements 时，默认过滤或固定会覆盖镜像栈的包。
- `torch_npu` 导入的 `undefined symbol`、`aclruntime` wheel ABI、Python 版本不匹配通常是版本栈问题，先查环境再改模型。
- Paddle、vLLM、TorchAir、DrivingSDK、OpenCV、tesseract 等依赖容易互相冲突；多组件流水线默认允许拆环境。

## 源码 patch

- 大多数适配项目都有 patch。常见修改点：设备选择、CUDA 假设、custom op、导出脚本、attention/位置编码、数据加载、后处理、评测脚本。详细修改模式见 `patch-modification-patterns.md`。
- patch 要基于固定 commit，README 写明应用位置和 `git apply --check`。如果 patch 只能执行一次，写入 FAQ。默认保持最小补丁：只改适配必需路径，不把调试输出、本地路径、无关重构带入。
- 遇到 `cuda`、`torch.cuda`、`USE_CUDA`、CUDA extension、`setup.py` 编译扩展时，优先改为 NPU 等价路径；不能改的部分标明 CPU fallback。

## ONNX 与 OM

- 动态 shape、符号维、多输入多输出、control flow、attention、RoPE、后处理入图是导出高风险点。默认准备 ONNX checker、shape 固化、onnxsim/onnxslim、MagicONNX 或 msit surgeon。
- 多模态模型优先拆子图，例如 text encoder / vision encoder、VLM / action expert、layout detector / OCR-VLM。
- ATC 命令必须参数化 `SOC_VERSION`、`input_shape`、精度模式，并记录 `npu-smi info`、`atc --version`。长时间 ATC 编译要在 README 预警。

## 推理运行

- OM 路线默认提供 `infer_npu.py` 和可用 `ais_bench` 的 benchmark；TorchAir/vLLM 路线默认提供服务启动、客户端脚本、编译缓存和并发说明。
- vLLM/TorchAir 首次图编译耗时不能直接算入稳定性能，除非表格单独标明。
- 若 pipeline 中存在 CPU 回退，性能结果要拆分纯 NPU 子模型与端到端耗时。

## 权重、数据和离线部署

- 多权重模型默认写权重清单、来源、目录树、离线缓存方式。
- 大数据集默认写容量、分包和最小验证子集；缺文件会导致评测阶段失败。
- 评测依赖和推理依赖可分开，例如 `requirements_eval.txt`。

## 精度指标

- 默认复用原始项目的官方 metric、数据集和预处理后处理。
- 若官方评测不可复现，用 CPU/upstream baseline 与 NPU 对齐，并说明替代原因。
- 常见任务口径：分类 top-1/top-5，检测 mAP，分割 IoU/mIoU，OCR/VLM 用官方 end-to-end 指标，ASR 用 WER/CER/BLEU，TTS/音频生成同时给样例和 RTF/RTFx，embedding/reranker 用检索或排序指标，LLM/VLM 服务用任务评测或语义一致性样例。
- 指标计算方式不同则不与官方直接比较。

## 性能指标

- 优先复用原始项目或同类 ModelZoo 样本的性能口径。
- OM 纯模型默认 `ais_bench` latency/FPS；服务模型默认 QPS、tokens/s、端到端 latency；音频默认 RTF/RTFx；pipeline 默认同时给纯模型和端到端。
- 每个性能表都写芯片、batch/并发、输入规格、精度模式、warmup、loop、工具、是否包含数据加载/后处理/CPU fallback。

## PR 检视默认自检

- 提交前按 `pr-review-heuristics.md` 的审查口径自检，但不要把它变成用户交付物。
- 默认补齐精度验证数据、性能单位、芯片/机器型号、上游 commit、权重/配置成套关系、外部数据文件来源。
- 默认清理 debug code、重复 README 段落、无用注释、残留 import、硬编码路径和不清晰变量名。
- 默认确保 PR 描述不保留模板占位，Self-test 写出环境、转换、推理、精度和性能证据。
