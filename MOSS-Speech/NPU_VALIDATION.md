# MOSS-Speech NPU 验证记录

检查日期：2026-06-01。

## 1. 已完成验证

| 项目 | 命令/结果 |
|---|---|
| Git 工作区初始状态 | `git status --short --branch`：`main...origin/main`，开始时无未提交变更。 |
| 主模型远端 | `git ls-remote --symref https://www.modelscope.cn/openmoss/MOSS-Speech.git HEAD`：`master` / `270d64296cafb94ca1f35b14b8d7918a1c4a2dc0`。 |
| Codec 远端 | `git ls-remote --symref https://www.modelscope.cn/AI-ModelScope/MOSS-Speech-Codec.git HEAD`：`master` / `a5423645a66476da761bbbdbc2003ae34e3c31c4`。 |
| Space 远端 | `git ls-remote --symref https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech HEAD`：`main` / `92a89018a8aa6b36f08c366c2659c76ffdc3f980`。 |
| Space clone | `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 ... MOSS-Speech/upstream` 成功；`git -C MOSS-Speech/upstream rev-parse HEAD` 输出 `92a89018a8aa6b36f08c366c2659c76ffdc3f980`。 |
| 设备相关扫描 | 已扫描 `cuda/device_map/torch_npu/istft/bfloat16/cached_download`；风险点见 `ANALYSIS.md`。 |
| 上游 patch | 当前无 `.patch`；`patches/README.md` 已说明生成和检查方式。 |
| 语法检查 | `python3 -m py_compile MOSS-Speech/infer.py` 通过。 |
| README 回退 | 已将 `MOSS-Speech/README.md` 恢复为原始既有实现参考；新增推理说明改放 `README_INFERENCE.md`。 |
| README_INFERENCE | 已按 Canary-1B 文档结构生成 `MOSS-Speech/README_INFERENCE.md`，覆盖概述、输入输出、环境、目录、下载、推理、流程复查、验收和公网地址。 |
| 当前宿主 help 检查 | `python3 MOSS-Speech/infer.py --help` 在当前宿主因缺少顶层必需依赖 `torch` 直接报 `ModuleNotFoundError: No module named 'torch'`；这符合项目级严格失败原则。安装 README_INFERENCE 中依赖后需重新执行 help 和端到端命令。 |

## 2. 待完成验证

当前环境尚未下载 MOSS-Speech 主权重和 Codec，未执行端到端 CPU/NPU 生成。正式验收需补充：

```bash
python MOSS-Speech/infer.py --device cpu --output_modality text --max_new_tokens 64 \
  --model MOSS-Speech/weights/MOSS-Speech \
  --codec MOSS-Speech/weights/MOSS-Speech-Codec \
  --space_dir MOSS-Speech/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python MOSS-Speech/infer.py --device npu --output_modality audio \
  --model MOSS-Speech/weights/MOSS-Speech \
  --codec MOSS-Speech/weights/MOSS-Speech-Codec \
  --space_dir MOSS-Speech/upstream \
  --prompt_audio MOSS-Speech/upstream/assets/prompt_cn.wav
```

并记录：

- Python / torch / torch-npu / torchaudio / transformers / modelscope 版本；
- CANN / 驱动 / 固件版本；
- 主模型与 codec 文件 SHA256；
- 生成文本、输出 wav 路径、采样率、时长；
- 端到端耗时、首 token/首音频耗时、峰值 HBM/RSS；
- 如失败，保留原始 traceback，不切换到 CPU fallback。

## 3. 验收报告模板

```text
日期：
机器/卡型：
CANN/驱动/固件：
Python/torch/torch-npu/transformers/modelscope：
主模型路径与 SHA256：
Codec 路径与 SHA256：
Space commit：92a89018a8aa6b36f08c366c2659c76ffdc3f980
命令：
输入 prompt：
输出 modality：
输出文件/文本：
采样率/时长：
耗时/RTF/峰值内存：
结论：通过 / 失败
失败 traceback（如有）：
```
