# ModelZoo `ACL_PyTorch/built-in` 采样指南

快照来源：`https://gitcode.com/Ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in` 及各分类页面；检查日期为 2026-06-30。当前仓库暴露的 built-in 分类包括：`audio`、`cv`、`nlp`、`ocr`、`embedding`、`foundation_models`、`embodied_ai`。排除辅助/隐藏文件后，页面可见模型数量为：audio 27、cv 90、nlp 17、ocr 9、embedding 4、foundation_models 6、embodied_ai 4。

适配新模型时，先用 `scripts/modelzoo_sampler.py` 刷新清单，并优先参考最近合入的项目；README 风格、镜像、CANN 版本和验收预期变化很快，不要长期复用旧快照。

## 近期代表性样本集

该样本集有意略多于“约 20 个”项目，以覆盖全部类型并让近期合入项目占主导。

| 类型 | 项目 | 近期页面信号 | 参考价值 |
|---|---|---:|---|
| embodied_ai | GraspNet | 22 小时前，PR 7630 | 最新 embodied OM 路线；包含自定义 pointnet/点云工具、安装脚本、评测与推理分离。 |
| cv | InstantID | 3 天前，PR 7624 | 近期复杂 CV/VLM 类 pipeline；包含多个 patch、ONNX shape 修复、MagicONNX、ais_bench 组件。 |
| cv | PromptIR | 6 天前，PR 7623 | 近期图像恢复项目；图像优先 README、patch 流程、依赖 pinning。 |
| audio | YingMusic-SVC_for_Pytorch | 7 天前，PR 7609 | 近期音频/SVC；图像优先 README、离线权重说明、torch_npu 不匹配 FAQ。 |
| cv | F3Net | 13 天前，PR 7610 | 近期 CV 显著性检测；镜像契约、精度/性能脚本、CPU 与 NPU 指标对齐。 |
| cv | SAM2 | 19 天前，PR 7613 | 近期分割优化模式；刷新样本时若可访问可重点参考。 |
| embodied_ai | IsaacGR00T | 19 天前，PR 7616 | Torch/机器人模型；包含 patch、NPU 工具、HF 下载、环境较重的安装流程。 |
| cv | SAM3 | 22 天前，PR 7604 | 现代基础视觉模型；ONNX 导出/优化、FlashAttentionTik patch、转换脚本、COCO IoU 评测。 |
| cv | FocalFormer3D_for_Pytorch | 21 天前，PR 7595 | 3D 检测；包含大数据集说明、DrivingSDK/自定义依赖、图像优先风格。 |
| nlp | chronos-2 | 21 天前，PR 7581 | 近期 NLP/时间序列；直接 Ascend 推理并提供性能/精度脚本。 |
| audio | Canary-1B | 28 天前，PR 7592 | 近期 ASR/AST；CANN 8.5.1 + torch_npu 2.9.0 风格、数据准备、RTFx/WER/BLEU。 |
| embodied_ai | vla/pi0 | 30 天前，PR 7590 | 拆分式 VLA 模型；分别为 VLM 与 action expert 提供 ONNX/OM 验证脚本。 |
| audio | Index-TTS-vLLM-v2 | 1 个月前，PR 7579 | vLLM-Ascend TTS 服务路线、FastAPI 服务、环境变量、RTF。 |
| cv | D-FINE | 1 个月前，PR 7573 | 检测模型；包含 patch、ONNX/OM、`om_inf.py`、ais_bench 性能。 |
| audio | CosyVoice3 | 1 个月前，PR 7565 | TorchAir/vLLM 镜像路线；docker 启动和服务式推理。 |
| nlp | ProtBert_for_Pytorch | 1 个月前，PR 7569 | 经典 Hugging Face encoder；`TestProtbert_2onnx.py`、静态 shape ATC、`ais_bench`。 |
| ocr | PP-DocLayoutV2 | 1 个月前，PR 7594 | Paddle/PaddleX + ONNX/OM；msit surgeon、动态 batch、下游 OCR/VLM 精度。 |
| ocr | PP-DocLayoutV3 | 1 个月前，PR 7594 | Paddle 版版面检测器；动态输入 shape ATC 和 demo 推理。 |
| ocr | PaddleOCR-VL-1.5 | 1 个月前，PR 7594 | vLLM-Ascend VLM 服务和 OmniDocBench 端到端评测。 |
| ocr | UVDoc | 1 个月前，PR 7533 | OM 路线，包含 MagicONNX patch、tesseract 精度评测、自定义 benchmark。 |
| embedding | bge-m3 | 1 个月前，PR 7587 | TorchAir embedding 路线；HF 模型 clone、简洁 `infer.py`、NPU ID 参数。 |
| embedding | bge-reranker-v2-m3 | 1 个月前，PR 7587 | Reranker 变体；路线相似，但需要关注排序分数语义。 |
| foundation_models | Chinese_CLIP | 1 个月前，PR 7537 | 双 encoder 导出/OM/评测 shell 模式；包含 patch 和 CLIP 检索指标。 |
| foundation_models | SigLIP2 | 2 个月前，PR 7517 | 文本/视觉双 ONNX→OM 模型；预处理/后处理、ImageNet 精度、ais_bench。 |

## 可复用模式

- 新项目通常把 ModelZoo 目录作为权威交付包，要求用户在固定 commit 克隆上游源码，然后应用 `diff.patch` 或模型专用 patch。
- README 标题通常包括：概述、输入输出数据、推理环境、快速上手、获取源码、安装依赖、准备权重/数据、导出模型、转换 OM、运行推理、精度、性能、FAQ/已知问题。
- 近期图像优先 README 往往包含“版本配套表”，并提醒不要重装镜像已提供的 `torch`/`torch_npu`。
- OM 项目通常包含 `export_onnx.py`/`pth2onnx.py`、可选 ONNX 修复/优化脚本、`convert_om.sh` 或内嵌 `atc` 命令、`infer.py`、`eval_accuracy.py`、`eval_performance.py`/`benchmark.sh`。
- vLLM/TorchAir 项目重点记录容器启动、服务命令、NPU 显存/环境变量和任务专用客户端脚本，而不是 ATC 转换。
- 精度证据可以是数值张量 diff、任务指标（WER/BLEU/mAP/IoU/overall）或端到端服务评测；性能证据必须说明芯片、batch/并发、输入 shape、精度模式、warmup/loop 次数和测试工具。
