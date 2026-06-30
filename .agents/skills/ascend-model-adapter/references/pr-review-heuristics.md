# PR 检视启发式

来源：抽样读取 GitCode `Ascend/ModelZoo-PyTorch` 近期合入 PR 的 discussions、CI 机器人评论、AI summary/AI review 和人工检视意见。样本包括 GraspNet #7630、InstantID #7624、PromptIR #7623、YingMusic-SVC #7609、F3Net #7610、IsaacGR00T #7616、SAM2 #7613、FocalFormer3D #7595、chronos-2 #7581、SAM3 #7604、Canary-1B #7592、PaddleOCR-VL/PP-DocLayout #7594、vla/pi0 #7590、Chinese_CLIP #7537、Index-TTS-vLLM-v2 #7579、D-FINE #7573、Fun-ASR-Nano #7556、Buffalo_l #7538、RT-DETRv2 #7553 等。

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
- 精度对比优先基于论文、官方公开数据集或 ModelZoo 同类可复现数据集。
- 如果无法使用官方指标，必须说明原因，并保证 CPU/upstream 与 NPU 使用同一评测脚本、同一数据划分、同一随机种子/阈值/top-k/IoU 策略。
- 对 ASR 等任务，推理结果写文件不等于精度；必须单独计算 WER/CER/BLEU 等任务指标，并给出计算命令。
- 生成/多模态/机器人模型至少提供可复现的小集合评测或语义/数值对齐说明；不要只贴截图。

## 性能检视

- 性能指标必须符合任务：ASR/TTS/音频默认 RTF/RTFx 或音频时长归一化指标；检测/分类/OM 默认 latency/FPS；服务模型默认 QPS/tokens/s/latency。
- 指标单位要明确，性能表不要混淆 ms、s、FPS、RTF、QPS。
- 端到端耗时和纯模型耗时分开；包含数据加载、后处理、CPU fallback、首次编译时必须单列说明。
- 若更新性能结果，README 表格、脚本输出、PR 描述中的数字要一致。

## Patch 与算子支持

- patch 必须覆盖实际不支持的算子或代码路径；例如原始代码中有 `split` 等 ATC 不支持算子时，patch 需要真正替换，而不是只在文档说明。
- 删除 custom op 或替换实现后，全链路 import、setup、requirements、README 都要同步。
- 对 ONNX/OM 导出脚本暴露关键参数；不要把内部调试脚本原样上库。

## PR 描述与自测

- PR 描述不要保留模板占位文字。Motivation、Modification、Self-test、BC-breaking 要写模型适配事实。
- Self-test 默认包含：环境、转换/编译、单样例推理、精度、性能；截图只能作为补充，不能替代命令和结果表。
- 如果有兼容性或依赖变化，必须在 BC-breaking/FAQ 中说明。
