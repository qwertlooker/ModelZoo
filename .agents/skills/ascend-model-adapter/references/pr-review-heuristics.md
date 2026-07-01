# PR 检视启发式

来源规则：只把“已采样 ModelZoo 目录对应的上库 PR”作为主要检视样本。对应关系必须来自目录页 PR 信号，或 PR diff/变更文件中能看到相同 `ACL_PyTorch/built-in/<category>/<model>` 路径。未能验证路径对应关系的近期 PR，不写成该样本来源；最多作为通用 CI/README 风格的补充参考。

这些不是额外交付步骤，而是生成适配工程时要默认规避的 review 问题。

## CI 与代码规范

- CodeCheck 是最常见失败项；提交前默认运行格式化、lint、基础 import 检查和脚本 help 检查。
- 检测到 `# noqa`、`pylint disable`、`flake8` 抑制注释时，默认删除；确实需要时在代码旁写清原因，因为 CI 会提示“请 Committer 检视其合理性”。
- 删除无用注释、debug code、临时打印、无意义变量名；变量名不要用 `m` 这类不清晰缩写。
- 删除或替换已移除模块的残留 import；删除文件后全仓 grep 一遍旧模块名。
- PR 必须让 Antipoison、CodeCheck、SCA、流水线全部通过。SCA/开源片段失败时，优先检查第三方代码片段、license、复制的大段源码和下载脚本。

## README 与文档结构

- 不重复写同一段获取源码/安装步骤；冗余段落会被要求删除。
- README 要写清“配套信息”：上游 commit、权重版本、配置文件版本、数据集版本、芯片/机器型号、CANN/torch_npu/镜像版本。
- README 要补充获取芯片型号的步骤，例如 `npu-smi info` 与 `SOC_VERSION`/`chip_name` 如何设置。
- 硬件字段要准确：区分芯片型号、机器型号、Atlas 300I DUO/Pro、Atlas 800I A2/A3、单芯/整卡等表述。
- 如果模型名/标题与实际示例版本不同（例如 SAM2 vs SAM2.1），必须说明模型、配置、权重成套使用，避免 review 质疑。
- 如果依赖外部小文件或清单（例如 `val_wav.scp`），README 必须说明来源、生成方式或上游自带路径。

## 可复现性

- 上游源码必须固定 commit/revision；否则 reviewer 会担心原仓更新后 patch 无法应用。
- 导出/转换/推理脚本不要只写死本地路径或输出名；把 onnx output、权重路径、batch、soc_version、device_id 等暴露为参数，并提供默认值。
- 如果把 shell 脚本改成 Python 脚本更易提供默认参数和跨环境复现，优先 Python。
- 下载脚本只有在原生 HF/ModelScope 命令在常见网络下不可用或需要特殊目录结构时才保留；README 解释必要性。

## 精度检视

- 默认要求有精度验证数据；不能只写“推理正常”。
- 精度对比优先基于论文、官方公开数据集或 ModelZoo 同类可复现数据集；如果源仓 README/论文/release 已给出 accuracy 表，默认要求说明 NPU 结果是否按同一 checkpoint、数据集和 metric 对齐。
- 如果无法使用官方指标或源仓没有可复现 accuracy，必须说明原因；此时才使用 CPU/upstream 与 NPU 对齐，并保证同一评测脚本、同一数据划分、同一随机种子/阈值/top-k/IoU 策略。
- 对 ASR 等任务，推理结果写文件不等于精度；必须单独计算 WER/CER/BLEU 等任务指标，并给出计算命令。
- 生成/多模态/机器人模型至少提供可复现的小集合评测或语义/数值对齐说明；不要只贴截图。
- Pipeline 模型若有 CPU fallback，必须说明具体技术阻塞（纯 NumPy/SciPy 算法、NPU 不支持算子、框架限制等）；不能只写“上游默认用 ONNX Runtime/CPU”。若有 PyTorch 等价路径，应优先评估 NPU 化。

## 性能检视

- 性能指标必须符合任务：ASR/TTS/音频默认 RTF/RTFx 或音频时长归一化指标；检测/分类/OM 默认 latency/FPS；服务模型默认 QPS/tokens/s/latency。源仓已有 benchmark/performance 数据时，reviewer 会关注是否复用了同一 benchmark 口径，或是否解释了硬件/输入/batch/统计差异；不会默认要求本地 CPU 性能对比。
- 指标单位要明确，性能表不要混淆 ms、s、FPS、RTF、QPS。
- 端到端耗时和纯模型耗时分开；包含数据加载、后处理、CPU fallback、首次编译时必须单列说明。
- 若更新性能结果，README 表格、脚本输出、PR 描述中的数字要一致。
- Pipeline 中包含 CPU 组件时，性能表要拆分纯 NPU 子模型与端到端；说明 CPU 组件是否已评估过迁移 NPU，CPU fallback 应是评估后的选择。

## Patch 与算子支持

- patch 必须覆盖实际不支持的算子或代码路径；例如原始代码中有 `split` 等 ATC 不支持算子时，patch 需要真正替换，而不是只在文档说明。
- 删除 custom op 或替换实现后，全链路 import、setup、requirements、README 都要同步。
- 对 ONNX/OM 导出脚本暴露关键参数；不要把内部调试脚本原样上库。

## PR 描述与自测

- PR 描述不要保留模板占位文字。Motivation、Modification、Self-test、BC-breaking 要写模型适配事实。
- Self-test 默认包含：环境、转换/编译、单样例推理、精度、性能；截图只能作为补充，不能替代命令和结果表。
- 如果有兼容性或依赖变化，必须在 BC-breaking/FAQ 中说明。
