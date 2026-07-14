# Ascend 适配领域启发式

## 目录

1. 环境与依赖
2. 权重和兼容性
3. Pipeline 与后端
4. ONNX/OM
5. 数据与预处理
6. 编码规范与代码质量
7. 服务和性能

这些是排查信号，不替代实测、官方兼容矩阵或项目日志证据。

## 1. 环境与依赖

- 镜像中的 CANN、Python、torch、torch_npu、torchvision/torchaudio 应成套；遇到 undefined symbol/ABI 错误先查版本栈。
- 非必要不创建 venv；只有同机需要多套不兼容 Python/PyTorch 时才隔离并记录原因。CPU 精度对照优先复用 NPU 环境，只切 `--device cpu`。
- 不先安装 CPU-only PyTorch 再追加 torch_npu；镜像预装时记录镜像 digest 和导入/NPU tensor 证据，不在快速上手中写泛化的 `pip install torch torch-npu`。
- 审计顶层、子模块和 vendor 的 requirements/setup/pyproject。优先顺序：业务 requirements → 安全的 editable 子包 → 顶层 `pip install --no-deps -e .`。
- 只过滤会覆盖核心栈、拉取 CUDA runtime 或阻塞安装的依赖；过滤后用真实入口 import、`--help` 和单样例补齐业务依赖。
- Paddle、vLLM、TorchAir、DrivingSDK、OpenCV、tesseract 等容易冲突；复杂 pipeline 可拆容器，但要记录组件边界和数据交换。
- 版本相关 FAQ 只在实际触发版本下出现，不把历史 workaround 无条件复制到新环境。
- 修改 site-packages 时用 `importlib.util.find_spec('<pkg>').submodule_search_locations[0]` 定位，patch 路径以 `<pkg>/` 开头并从 site-packages 根目录 `patch -p1`；不要为定位而 import 包。
- ONNX Runtime CANN EP 必须固定 CANN、onnxruntime-cann、Python/架构和安装命令或内部 wheel SHA，并验证 available/session provider；不能只写“安装配套 CANN EP”。

## 2. 权重和兼容性

- 用户提供的 checkpoint 优先；确认类别数、tokenizer、speaker/label map、配置和模型类成套。
- 可信旧 checkpoint 在 PyTorch 2.6+ 可能因 `weights_only=True` 默认策略失败；优先转换 state_dict/safetensors。只有来源和哈希可信且在隔离容器中，才考虑 `weights_only=False`。
- torchaudio 2.9+ 音频加载可能引入 TorchCodec/FFmpeg/ABI 变化；出现问题时检查实际版本，可用 soundfile/librosa 替代 I/O，但要对齐波形和任务指标。
- 模型变体、聚类/阈值/beam/search 和预后处理必须与 checkpoint 对应，不能只用默认 config 推断官方口径。

## 3. Pipeline 与后端

- 搜索硬编码 `cuda`、`.cuda()`、`torch.cuda`、CUDAExtension、custom op 和 map_location；按组件决定 NPU 等价实现、拆图或 CPU fallback。
- 上游硬编码 ONNX Runtime、TensorRT、TF、Paddle inference、OpenVINO 时，检查同项目的 PyTorch 等价路径。后端替换保持同架构同权重并做数值对齐。
- OCR/VLM/VLA/TTS/diarization/检测+识别先列组件表：实现、权重、输入输出、路线、目标设备、阻塞和耗时。
- CPU fallback 必须是评估后的技术选择；概述、组件表、FAQ 和性能表保持一致。
- 同架构后端替换比较 logits/embedding/中间张量；换模型、权重、tokenizer 或策略后重新跑任务指标。
- 官方 evaluator、normalizer、tokenizer 和预期字段严格失败；不要用简化 metric、宽泛 getattr/try-except 或远端自动 fallback 生成貌似成功但不可比的结果。

## 4. ONNX/OM

- 动态 shape、符号维、多输入输出、control flow、attention、RoPE、cache 和后处理入图是高风险点。
- 导出先跑 checker；需要时采用 shape 固化、onnxsim/onnxslim、MagicONNX/msit surgeon 或拆子图，并记录每一步输入输出 artifact hash。
- 固定 batch 评测处理尾 batch padding，并在输出后裁掉；dynamic shape 明确范围、档位和 custom_sizes。
- 大 ONNX 检查 external data，README 列出不可拆散的文件。
- ATC 命令参数化 SOC_VERSION、input shape 和精度模式；长编译与 cache 行为写入 FAQ。

## 5. 数据与预处理

- 权重、数据、评测工具、protocol、reference label/RTTM 都写来源、版本、目录树和生成命令。
- `prepare_data.py` 区分下载输入、生成中间文件和最终 manifest；每个 dataset/language 分支用最小 fixture dry run。
- 检查 GitHub zip 顶层目录、Hugging Face 原始文件 URL、压缩包类型、文件大小和可选 SHA256。
- 环境数据下载优先在独立脚本中使用 wget/curl 并即时 file/size/SHA 校验，不把 nltk.download、datasets.load_dataset 或 Hub 自动下载嵌入推理/评测。确需 Python 下载器时同时提供固定 revision 的 wget/curl 离线替代。
- 提供严格 `--offline` 或等价模式；缺文件时报告具体路径并失败，不联网 fallback。manifest/meta 记录远端 URL、config/split、本地文件、复用/离线状态和样本规模。
- 音频/图像/视频预处理精确复现采样率、声道、resize/crop、归一化、padding 和重采样工具；改动后对齐中间结果或任务指标。
- 原仓没有的 norm stats、label map 或配置必须提供生成依据和命令。

## 6. 编码规范与代码质量

所有交付脚本（推理、评测、导出、benchmark、数据准备）必须通过基础静态检查（pylint / ruff / flake8 等），不得在 S1 前遗留以下典型问题：

### 错误处理
- **禁止 `sys.exit()` 和 `raise SystemExit()`**：函数体内不得调用 `sys.exit()` 或 `raise SystemExit()`，主进程入口 `if __name__ == "__main__"` 除外。依赖缺失应 `raise RuntimeError(...)`，参数校验失败应 `raise argparse.ArgumentTypeError(...)` 或 `raise ValueError(...)`，验证不通过应 `raise SystemExit(...)` 仅限 `main()` 或其直接委托的终态判定。
- **异常链保留**：`except ... as exc` 后用 `raise ... from exc` 保留原始堆栈，不要吞掉上下文。

### 格式
- **算术操作符两侧必须有空格**：`a + b`，`a - b`，`a * b`，`a / b`，`a ** b`，`a // b`，`a % b`。`a+b`、`a-b` 均不合规范。
- **一行只写一条语句**：禁止用分号 `;` 将多条语句写在同一行。例如 `ps.infer([pi]); ms.infer([mi])` 应拆为两行。
- **紧凑计时模式**：性能测试中避免 `t = time.perf_counter(); func(); lat.append(time.perf_counter() - t)` 这类一行多语句写法，应拆为独立赋值、调用、记录三行。

### 导入
- 移除不再使用的 `import sys` 等废弃导入；`sys.exit` 全部替换后同步删除 `import sys`。

### 自检
- S1 门禁前对全部交付脚本执行 `ruff check` 或等价工具，零错误才可通过；发现的规则违反必须修复，不得用 `noqa` 压制后进入候选目录。

## 7. 服务和性能

- vLLM/TorchAir 提供 server、client、health check、预热/编译缓存、并发、显存和恢复说明。
- vLLM 嵌套配置按固定版本实际 `--help` 的 JSON/CLI 语法书写；server 的 served model、context、client 和 benchmark 名称保持一致。外部 evaluator/agent/tool 仓固定 commit。
- 纯模型 NPU 计时前后同步；端到端测量明确包含数据、网络、排队、后处理和 CPU 组件。
- 首次编译、cache miss、warmup 与稳定性能分开；服务模型同时考虑并发、tokens/s、latency 和错误请求。
- 正式性能至少独立运行 3 次并报告中位数，保存每次结果；CPU 性能不用于推断加速比。
- 音频默认 RTF，OM 常用 latency/FPS，服务常用 QPS/tokens/s；最终口径仍优先与官方 benchmark 对齐。
- 多卡任务用 `ASCEND_RT_VISIBLE_DEVICES` 分配，并给每个任务独立日志/cache/输出；检测不到空闲卡时要求显式指定，不静默占用 0 卡。
