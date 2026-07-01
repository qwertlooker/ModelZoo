# ModelZoo 交付契约

完成 Ascend ModelZoo-PyTorch 适配并整理交付目录时，使用本契约作为收尾检查依据。

## 目录结构

优先在 `ACL_PyTorch/built-in/<category>/<model>` 下使用扁平的 ModelZoo 风格目录。只提交必要文件；能通过 patch 修改上游代码时，优先 patch 上游代码，避免复制重复脚本。部分 ModelZoo 项目接近 README + patch + requirements 的极简形态；另一些只额外保留少量辅助或修复脚本。

```text
<ModelName>/
├── README.md                         # 必需
├── requirements.txt                  # 必需；只放业务依赖
├── diff.patch 或 <model>_NPU.patch   # 修改上游代码时必需
└── 可选，仅在需要时保留
    ├── export_onnx.py / pth2onnx.py  # ONNX/OM 路线
    ├── convert_om.sh / atc.sh        # ONNX/OM 路线
    ├── infer.py / ascend_infer.py    # 仅当上游缺少推理入口时新增
    ├── validate_acc.py / eval_accuracy.py
    ├── validate_perf.py / benchmark.sh
    └── 上述脚本需要的 helper/fix 文件
```

关键原则：

- 如果上游已有推理/评测入口（`inference.py`、`infer.py`、`test.py`、`demo.py`、shell 命令等），优先 patch 这些入口以支持 NPU，不要新增重复脚本。
- 可行时使用单一 `--device npu/cpu` 参数或环境变量；不要默认拆成 `infer_cpu.py` 与 `infer_npu.py`。
- 不要把 agent 内部文件作为上库交付物，例如 `env_check.py`、`docker_run.sh`、`collect_report.py`、`adaptation_config.yaml`。环境检查、Docker 命令和证据收集命令应写入 README。
- 对用户提供的 checkpoint/weights，必须记录实际产物路径、期望目录树，以及 config/tokenizer/label-map 等配套关系；不得静默替换成其他 checkpoint。

## README 章节

包含以下章节或等价命名：

1. 标题：`<ModelName>-推理指导` 或 `<ModelName>(路线)-推理指导`。
2. 概述：任务、上游链接、固定 commit/revision、用户提供或官方 checkpoint 信息、许可证、适配范围、支持芯片。
3. 输入输出数据：tensor 名称、shape、dtype、layout；OM 路线必须提供。
4. 推理环境准备：固件/驱动、CANN、Python、PyTorch、torch_npu、torchvision/torchaudio、额外 SDK、vLLM/TorchAir/ais_bench/msit 版本。除非有明确理由，否则说明不要重装镜像已提供的 `torch/torch_npu`。
5. 镜像启动：docker pull/run、NPU 设备挂载、环境变量，以及 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
6. 快速上手：克隆 ModelZoo、按固定 commit 克隆上游、应用 patch、安装业务依赖、准备权重/数据。
7. 准备权重和数据：权重清单、来源或用户提供路径、目录树、离线缓存配置。
8. 模型导出/转换或服务启动：
   - ONNX/OM：导出 ONNX、校验 ONNX、携带 `--soc_version` 执行 ATC、提供 OM 样例推理。
   - TorchAir：图编译设置、缓存位置、首次运行编译说明、NPU ID。
   - vLLM-Ascend：镜像 tag、server 启动、显存/环境变量、client 命令。
9. 推理：NPU 上的精确命令；可行时 patch 上游命令。
10. 精度验证：数据集、原始/上游指标、源仓已有 accuracy 数据或表格、命令、CPU/官方结果、NPU 结果、容差或 delta。
11. 性能验证：源仓已有 benchmark/performance 数据、命令/工具、warmup、loop、batch/并发、精度模式、latency/FPS/QPS/RTF、芯片。
12. FAQ/已知问题：unsupported ops、ATC 长时间编译、依赖冲突、离线下载、CPU fallback 原因、patch 排障。
13. 公网地址说明：引用外部 URL 时提供。

复杂 pipeline 可选补充：

- Pipeline 组件部署：列出每个组件、上游 backend、选定 backend、NPU 可行性和 CPU fallback 原因。diarization/OCR/VLM/TTS pipeline 推荐提供；简单模型不要泛化成额外交付物。
- 交付件清单：在有帮助时列出提交文件及说明。

## 指标选择

- 精度：尽量使用上游原始指标、数据集 split、预处理、后处理和阈值。源仓已有 accuracy 数据时，NPU 结果优先对齐同一 checkpoint、数据集/子集、随机种子和评测脚本口径。若原始指标无法复现，则在相同输入上对比 NPU 与 CPU/upstream baseline，并说明替代原因。
- 性能：优先使用原始项目已有 benchmark/performance 脚本和可比性能指标，尽量对齐输入规格、batch/并发、warmup/loop、统计区间、端到端/纯模型定义和单位。否则按路线采用常用口径：OM 用 `ais_bench` latency/FPS，vLLM 服务用 QPS/tokens/s/latency，音频用 RTF/RTFx，pipeline 同时报告纯模型和端到端 latency。
- 始终说明 warmup、loop 次数、batch/并发、输入 shape、精度模式、芯片，以及是否包含首次编译或 CPU fallback。

## 容器命令模板

```bash
export IMAGE=<ascend-image-tag>
docker pull ${IMAGE}
docker run -it --rm --net=host --privileged=true --shm-size=256g \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v $PWD:/workspace -w /workspace \
  ${IMAGE} bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

根据 Atlas A2/A3 服务器和选定 NPU ID 调整设备数量。若宿主机使用 `/usr/local/bin/npu-smi`，需增加该路径挂载。

## PR 就绪自检

默认在认为目录可提交 PR 前，确保以下条件成立：

- PR/README 文本没有模板占位符、重复安装段落、debug code、过期注释、不清晰变量名或残留 import。
- 上游源码固定到 commit/revision；README 包含 checkout 和 patch 应用说明。
- 尊重用户提供的 checkpoint/weights，并与正确的 config/tokenizer/label map 配套；任何替换都必须显式说明。
- README 包含芯片/主机信息，以及在设置 `SOC_VERSION` 或 `chip_name` 前获取芯片名称的命令。
- 精度不能只用截图或输出文件表示；必须包含任务指标命令和结果。
- 性能指标和单位匹配任务，并在 README、脚本、PR 描述中保持一致。
- Pipeline CPU fallback 有具体技术原因，不能只写“上游默认 backend”。
- 本地 lint/import/help 检查通过；预期 Antipoison、CodeCheck、SCA 和 PR 流水线可通过。

## 验证证据清单

- [ ] 上游 URL 与 commit/revision 已固定。
- [ ] 许可证与再分发限制已检查。
- [ ] 容器镜像和宿主机 driver/CANN 兼容性已说明。
- [ ] CPU/upstream baseline 输出或指标已记录；源仓已有 accuracy/performance 数据时，已记录原始口径和 NPU 对齐结果或差异说明。
- [ ] 如适用，ONNX 导出成功且记录 ONNX checker/simplifier 结果。
- [ ] 如适用，ATC 成功且记录 `.om` 产物路径。
- [ ] NPU 单样例推理成功。
- [ ] 精度指标在容差内，或 delta 已合理解释。
- [ ] 性能命令和结果表已记录。
- [ ] CPU-only 限制标记为 `待 NPU 验证`，不得标记为通过。
