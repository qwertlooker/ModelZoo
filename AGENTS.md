# 项目硬约束速查卡

- 本仓用于将推理模型规范迁移并适配至昇腾 NPU，正式目标路径为 `ACL_PyTorch/built-in/<领域>/<模型目录>`。
- 模型适配、验收和上库复核的完整权威要求是 `模型NPU 适配标准流程.md`；开始相关工作前必须重新读取并执行。
- 完成请求的修改后默认提交当前分支并推送已配置远程；不得强制推送或重写已发布历史。推送失败时报告确切错误并保留本地提交。
- 每次模型适配或上库复核都必须重新查询目标仓 `master` HEAD，按最新实质变更选择参考目录，不复用历史快照。
- `NPU_ADAPTATION.md` 必须记录检查日期、目标仓 commit、拟合入路径、参考目录及其最后实质变更 commit/date、选择原因、上库文件清单和排除项。
- 目标路径已存在时，必须先比较现有文件和 README，明确本次新增、替换或增量更新范围。
- 迁移工作区不等同于正式候选目录；`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、`README_old.md`、日志、权重、数据、虚拟环境、缓存、`upstream/` 和 `.codex-reference/` 默认不进入候选目录。
- 正式 `README.md` 必须自包含，能完成安装、推理、评测和结果理解，不得依赖默认排除的内部证据。
- 每个模型必须先盘点原始测试集和原始指标；不得编造官方结果、用自定义指标冒充或把 dummy/smoke 样例当作精度、质量、性能验收。
- 状态统一使用 S0-S4；S3 仅表示 NPU L2 精度/质量与性能验收完成，上库候选就绪还需目标路径、README、许可证、贡献门禁和 clean-room 重放通过。
- 用户要求“完成适配和验证”时默认目标至少是 S3；因硬件、权重、授权或外部服务阻塞时必须保留未完成/待验收并说明原因。
- 每个模型必须提供或复用可执行的数据准备、NPU 运行、结果比较和报告入口；只写 `ACCEPTANCE_PLAN.md` 不等于已交付工具。
- 提交前必须按文档声明目录和干净环境执行最低正式路径的 clean-room 重放；正式命令至少实际执行 `--help`、dry-run 或轻量 fixture。
- 使用 `python3 tools/audit_model_delivery.py <model_dir>` 执行结构门禁；准备上库时增加 `--target-readiness --target-path ACL_PyTorch/built-in/<领域>/<模型目录>`。
- **精度/性能验收原则**：NPU 适配项目一般不具备 GPU 环境。精度对比优先使用原始模型的公开/官方指标（论文、GitHub/HuggingFace 官方数据或业界公开数据），不要求同环境重跑 CUDA 原始路径；当无公开指标可引用时，使用 CPU 推理结果（`--device cpu`，复用同一 Python 环境）作为精度基线确认 NPU 适配正确性；性能数据按模型所属领域一般指标（如 TTS 的 RTF/RTFx、ASR 的 RTF/RTFx、CV 的 FPS/throughput 等）在 NPU 上实际测试，至少 3 次取中位数，不使用 CPU 性能数据推断加速比。验收主线为 NPU 必跑 + 公开指标对照；CUDA 对照组为可选增强，非强制要求。
- NPU 侧固定 CANN 配套 wheel 或基础镜像，不得先装 CPU-only PyTorch 再追加 `torch-npu`；CPU 精度对照可复用 NPU 环境（`--device cpu`），不强制独立 venv。
- **环境原则**：非必要不使用 venv 虚拟环境。当 CANN 基础镜像或系统 Python 已满足版本和依赖要求时直接使用，不额外创建 venv。仅在同一机器需要多套不兼容的 Python/PyTorch 版本时才使用 venv 隔离，并在文档中写明隔离原因。
- 使用 ONNX Runtime CANN EP 必须给出可执行 `onnxruntime-cann` 安装版本/命令或内部 wheel 文件名与 SHA。
- vLLM 嵌套配置必须按固定版本 `--help` 的真实 JSON/CLI 语法书写；外部数据集、评测器和 agent 仓库必须 checkout 固定 commit。
- 默认维护 `README.md`、`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md` 三类主文档；除非明确要求，不修改原始 `README_old.md`。
环境数据（权重、评测数据集、NLTK/spacy 语料、辅助模型 checkpoint 等）下载优先使用 `wget`/`curl`，不依赖 Python 库内置下载器（如 `nltk.download()`、`huggingface-cli download`、`datasets.load_dataset()` 等自动下载路径）。`pip install` 不在此限制范围——NPU 服务器有内部 PyPI 镜像可用，pip 包安装正常；问题在于 Python 库的专用下载通道（NLTK data server、Hugging Face hub API、HF datasets parquet 等）绕过内部镜像，走公网被代理/防火墙拦截后静默返回损坏文件（如 HTML 错误页被保存为 zip），且缺少校验环节，难以定位根因。`wget`/`curl` 下载的文件可用 `file`、`sha256sum`、`head` 等标准工具即时验证；下载命令可直接写入离线准备脚本，在可联网机器执行后通过 scp/USB/共享目录传输到目标服务器。若因上游项目结构原因必须使用 Python 下载器，必须在 README 和 NPU_ADAPTATION 中同时给出等价的 `wget`/`curl` 离线替代命令，并记录 Python 下载器在校验方面的已知局限。
- 修改 pip 安装的 site-packages 内第三方库时，定位包路径必须用 `importlib.util.find_spec('<pkg>').submodule_search_locations[0]`，禁止用 `import <pkg>` + `__file__`（import 会触发 `__init__.py`，若依赖链有问题则定位本身就崩溃）；补丁从 site-packages 根目录 `patch -p1` 应用，diff 路径以 `<pkg>/` 开头；详见标准流程 9.1。
- 只做 NPU 适配和验收必需的最小修改，保持 CPU/CUDA 行为不变；第三方源码变更用固定版本可重复应用的 patch/diff 或目标仓认可源码交付。
- 未经明确允许，不得在代码或 patch 中加入静态检查屏蔽注释；缺依赖、字段或官方评估组件时应暴露原始错误，不得静默回退或 monkey patch。
