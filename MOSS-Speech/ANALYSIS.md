# MOSS-Speech NPU 适配分析

## 1. 参考原始仓库与适配版本边界

检查日期：2026-06-01。

| 资产 | 来源 | 分支/HEAD | 当前边界 |
|---|---|---|---|
| 主权重 | <https://modelscope.cn/models/openmoss/MOSS-Speech> / Git `https://www.modelscope.cn/openmoss/MOSS-Speech.git` | `master` / `270d64296cafb94ca1f35b14b8d7918a1c4a2dc0` | 当前适配仅针对 `openmoss/MOSS-Speech` / `fnlp/MOSS-Speech` 同一 MOSS-Speech 主模型，不覆盖 MOSS-TTSD、MOSS-TTSD-v0.5 或其他同名派生。 |
| Codec | <https://modelscope.cn/models/AI-ModelScope/MOSS-Speech-Codec> / Git `https://www.modelscope.cn/AI-ModelScope/MOSS-Speech-Codec.git` | `master` / `a5423645a66476da761bbbdbc2003ae34e3c31c4` | 当前适配绑定 `MOSS-Speech-Codec`，不替换为其他 codec/vocoder。 |
| Space 代码 | <https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech> | `main` / `92a89018a8aa6b36f08c366c2659c76ffdc3f980` | 当前适配使用该 Space 中 `cosyvoice/`、`utils/`、`assets/prompt_*.wav` 和可选 `Matcha-TTS/` 路径。 |

本地上游副本：`MOSS-Speech/upstream/`，通过 `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech MOSS-Speech/upstream` 获取。

权重 SHA256：本次未下载大权重，尚未记录权重文件级 SHA256。后续在正式 NPU 验收前必须对本地主模型和 codec 文件执行 `sha256sum` 并补回报告。

## 2. 现有文件审视

| 文件 | 问题 | 本次处理 |
|---|---|---|
| `README.md` | 原说明要求修改安装环境中的 `diffusers`、`transformers` 源码，并依赖 `transfer_to_npu` 自动迁移 CUDA 写法。 | 重写为可复现的版本边界、环境、下载、运行和验收说明。 |
| `infer.py` | 硬编码 `device="cuda"`、`device_map="cuda"`、`.to("cuda")`，路径和 prompt 写死。 | 改为参数化入口，默认 `--device npu`，CPU 验证显式 `--device cpu`，仅 NPU 后端条件导入 `torch_npu`。 |
| `requirements.txt` | 为环境冻结文件，包含训练、服务、CUDA/NVIDIA 包等大量非最小依赖。 | 保留作历史环境参考；部署以 README/NPU_ADAPTATION 中最小推理依赖为准。 |
| `patches/README.md` | 原先缺少 patch 策略。 | 新增说明：当前未修改上游；后续必须 patch 化，不允许手工改 site-packages。 |

## 3. 上游设备相关扫描

扫描命令：

```bash
grep -RIn "cuda\|device_map\|torch_npu\|istft\|bfloat16\|cached_download" MOSS-Speech/upstream --exclude-dir=.git
```

主要结论：

- Space `utils/interface.py` 默认 `device='cuda'`，但通过构造参数可传入设备，模型加载后 `.to(self.device)`；当前适配不直接使用该 Gradio 入口。
- `cosyvoice/hifigan/generator.py` 使用 `torch.istft`。旧 README 建议将 ISTFT 转 CPU；该做法属于 CPU fallback，不能作为默认官方 NPU 路径静默启用。若 NPU 算子不支持，应在 NPU 验证中记录原始错误，并用独立 patch/独立非官方模式处理。
- `cosyvoice` 训练、TensorRT、ONNX 导出路径存在多处 CUDA 逻辑；当前单请求推理入口不覆盖训练/TRT/ONNX 路径。
- `diffusers` / `transformers` site-packages 修改没有绑定精确版本，不符合当前项目标准；这不等于确认不需要，而是需要先复现具体版本错误，再固定版本并生成可检查 patch。

## 4. 本次适配策略

1. 以适配目录新增 `infer.py` 作为主入口，不改上游已有文件。
2. 默认 `--device npu`，实际卡号通过 `ASCEND_RT_VISIBLE_DEVICES` 控制；CPU 验证必须显式 `--device cpu`。
3. 必需依赖在文件顶层导入；只有 `torch_npu` 根据 `--device npu` 条件导入。
4. 不提供 CPU/远端自动 fallback；缺权重、缺官方 remote code、缺 codec、NPU 算子不支持时直接失败并暴露原始错误。
5. 输出音频和文本均落盘，便于功能、主观质量和后续客观指标评估。

## 5. 风险与限制

- 当前环境未下载主权重/codec，未完成端到端 CPU 或 NPU 生成验证。
- MOSS-Speech 是多仓库、多 remote-code、多组件链路，主观音频质量不能只通过 dummy smoke test 判定。
- 公开资料未给出统一的 NPU 性能或精度基线；验收应以同 checkpoint、同输入、同生成参数下 CPU/CUDA 与 NPU 的功能和质量对齐为主。
- 若 `processor.decode` 内部必须依赖 CPU ISTFT 或 Whisper CPU 特征，应先记录官方实现和错误边界，再作为明确的非官方兼容模式评估，不得混入默认路径。原 README 中的 Matcha-TTS bf16 修改也应按同样方式复现、patch 化和验收。
