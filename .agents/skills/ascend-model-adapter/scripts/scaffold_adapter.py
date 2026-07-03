#!/usr/bin/env python3
"""Create a minimal ModelZoo-PyTorch Ascend adaptation scaffold.

Generated files follow actual ACL_PyTorch/built-in conventions:
- Generate directly into the final flat project root, not an adapter subdir.
- Prefer upstream code + patch over duplicate scripts.
- Do not default-generate env_check.py, docker_run.sh, collect_report.py, or adaptation_config.yaml.
- Do not default-generate infer.py; use --with-infer only when upstream lacks an inference entry.
- ONNX/OM route gets export + convert scripts by default.
"""
from __future__ import annotations

import argparse
import re
import stat
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_IMAGE = "swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-onnx-inference:cann8.3.rc1_torch2.1.0-800I-A2-openeuler24.03-py3.11-aarch64"
ROUTES = ["auto", "onnx-om", "torch-npu", "torchair", "vllm-ascend"]


def infer_name(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    name = parts[-1] if parts else "model"
    name = re.sub(r"\.git$", "", name)
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", name).strip("-_") or "model"


def write(path: Path, content: str, executable: bool = False, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.lstrip(), encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def readme(model: str, url: str, category: str, route: str, image: str, hardware_model: str) -> str:
    if route == "onnx-om":
        route_note = f"""
## 模型导出/转换

```bash
python3 export_onnx.py --model-path <checkpoint_or_repo> --output model.onnx
export SOC_VERSION=<ATC转换所需soc_version>
bash convert_om.sh model.onnx model ${{SOC_VERSION}}
```

TODO：补充 ONNX checker/简化结果、ATC 日志、OM 路径和样例推理命令。
"""
    elif route == "torchair":
        route_note = """
## TorchAir 图编译/推理

TODO：补充上游推理入口 patch 后的 NPU 命令、编译缓存目录、首次编译说明和性能统计方式。
"""
    elif route == "vllm-ascend":
        route_note = """
## vLLM-Ascend 服务启动

TODO：补充服务端启动命令、客户端请求命令、并发/内存/缓存配置和性能统计方式。
"""
    else:
        route_note = """
## 模型导出/转换或服务启动

TODO：根据实际路线补充。ONNX/OM 需要导出和 ATC；torch_npu/TorchAir/vLLM 优先 patch 上游入口或服务脚本。
"""

    return f"""
# {model}-推理指导

> 状态：脚手架已生成。请删除 TODO 或明确标记 `待 NPU 验证`，并在 NPU 环境完成验证后再上库。

## 概述

- 上游模型：{url}
- ModelZoo 类别：{category or 'TODO'}
- 推荐适配路线：{route}
- 目标硬件型号：{hardware_model}
- 交付范围：Ascend NPU 镜像环境、源码 patch、推理、精度与性能验证。

TODO：补充任务简介、论文/项目链接、license、固定 commit/revision、checkpoint/权重版本、支持输入输出和限制。公开文档硬件字段使用 `Atlas 800I A2` 这类对外型号，不写详细芯片型号、芯片步进或内部代号；`SOC_VERSION` 仅作为 ATC 转换参数。复杂 pipeline 才补组件表；简单模型不要为了模板完整性新增冗余章节。

## 推理环境准备

| 组件 | 推荐/实测版本 | 说明 |
|---|---|---|
| 固件与驱动 | TODO | 与 CANN 匹配 |
| CANN | TODO | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Python | TODO | 镜像内置或业务环境 |
| PyTorch / torch_npu | TODO | 镜像内置，通常不要重装 |
| torchvision / torchaudio | TODO | 与镜像内 torch ABI 配套，通常不要重装 |
| 推理工具 | TODO | ATC / ais_bench / msit / TorchAir / vLLM-Ascend |

请勿在容器内用 pip 重装镜像内置的 `torch/torch_npu/torchvision/torchaudio`，除非 README 明确解释成套版本来源、恢复命令和验证范围。

### 创建并进入容器

将 `<container-name>` 替换为容器名，将 `<宿主机工程目录>` 替换为本目录在宿主机上的绝对路径。

```bash
export IMAGE={image}
docker pull ${{IMAGE}}

docker run -itd -u root --net=host --privileged=true \\
  --name <container-name> \\
  --shm-size=256g \\
  --ipc=host \\
  --device=/dev/davinci0 \\
  --device=/dev/davinci_manager \\
  --device=/dev/devmm_svm \\
  --device=/dev/hisi_hdc \\
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \\
  -v /usr/local/dcmi:/usr/local/dcmi \\
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \\
  -v /etc/ascend_install.info:/etc/ascend_install.info \\
  -v <宿主机工程目录>:<宿主机工程目录> \\
  -v /root/.cache:/root/.cache \\
  ${{IMAGE}} bash -i

docker exec -it <container-name> bash
cd <宿主机工程目录>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

## 快速上手

### 获取源码并应用 patch

```bash
git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/{category}/{model}
# TODO：clone 上游源码并固定 commit/revision
# git clone {url} <upstream_dir>
# cd <upstream_dir>
# git reset --hard <commit>
# git apply --check ../diff.patch
# git apply ../diff.patch
```

### 安装依赖

安装前先预检核心栈和已知业务依赖；`requirements.txt` 只做最小化修改，不要无理由删除不冲突依赖。

```bash
python3 - <<'PY'
mods = ["torch", "torch_npu"]  # 按任务追加 transformers/torchaudio/pyannote 等
missing = []
for name in mods:
    try:
        mod = __import__(name)
        print(name + ": " + str(getattr(mod, "__version__", "ok")))
    except Exception as exc:
        missing.append((name, repr(exc)))
if missing:
    raise SystemExit("missing/failed imports: " + str(missing))
PY

pip install -r requirements.txt
# 如有 editable 子包：
# pip install -e ./<subpkg>
# 顶层包默认禁止解析依赖，避免覆盖镜像 torch 栈：
# pip install --no-deps -e .

# smoke test
# python3 <infer_or_eval_entry>.py --help
```

## 准备权重和数据

TODO：优先记录使用者提供的 checkpoint/权重目录；列出配置、tokenizer、label-map、speaker-map 等成套文件和目录树。若替换为其他权重，必须说明原因和差异。

TODO：如需数据准备脚本，优先提供单一 `prepare_data.py`，直接处理 tar/zip、manifest/scp/reference 生成和必要的音频/图像转换；不要再加功能重复的 shell 包装。
{route_note}
## 模型推理

优先运行 patch 后的上游推理入口。推理、评测和 benchmark 脚本默认设备必须是 NPU；如果提供 `--device`，默认值设为 `npu`，CPU 仅用于显式 `--device cpu` baseline/fallback；切换物理卡时在命令前 `export ASCEND_RT_VISIBLE_DEVICES=<id>`，不要写 `npu:0`。输入输出 tensor 名称、shape、dtype、layout 可放在本节；OM 固定输入输出时必须列清。

```bash
# python3 <upstream_infer.py> --model <checkpoint_or_repo> --device npu --input <sample_input> --output outputs
```

仅当上游没有统一入口时，使用本目录通过 `--with-infer` 生成的 `infer.py`。

## 精度与性能验证

默认复用上游原始指标、官方完整数据集/split、预处理和后处理。本适配是 GPU/上游实现到 NPU 的迁移：源仓已有官方/GPU 精度指标时，不要求 CPU 精度对比，直接用 NPU 结果对齐官方/GPU 口径。若官方有多个数据集，可只评测其中一部分并列明未评测项；对已选择的数据集必须完整评测。无法复现原始指标或无官方指标时，才使用同一输入集的 CPU/upstream baseline 与 NPU 输出对齐，并说明替代原因。

| 数据集/split | 指标 | 官方/源仓/GPU 精度 | NPU 结果 | 差异 | 结论 |
|---|---|---:|---:|---:|---|
| TODO（完整官方 split） | 原始指标/TODO | TODO | TODO | TODO | 待 NPU 验证 |

性能默认优先复用上游官方/GPU 或同类 ModelZoo benchmark 口径；默认没有本地 GPU 环境时使用官方发布性能作为参考，没有官方/GPU 性能时只报告 NPU 性能。CPU 性能对比对 GPU→NPU 迁移没有意义，默认不在 README/PR 中体现。没有可比口径时，OM 用 `ais_bench` latency/FPS，服务模型用 QPS/tokens/s/latency，音频默认用 RTF（耗时/音频时长，越低越好）为主，可补充 RTFx=1/RTF（实时倍速，越高越好），pipeline 同时给纯模型和端到端。首次编译、CPU fallback、数据加载/后处理耗时需单列。

| 硬件型号 | Batch/并发 | 输入规格 | 精度模式 | 工具/loop | 性能口径 | NPU 性能 |
|---|---:|---|---|---|---|---:|
| {hardware_model} | TODO | TODO | TODO | TODO | 原始口径/ais_bench/E2E | 待 NPU 验证 |

多数据集/多 split 可在多张空闲 NPU 上并行运行，例如为不同任务分别执行 `export ASCEND_RT_VISIBLE_DEVICES=0`、`export ASCEND_RT_VISIBLE_DEVICES=1`，脚本参数仍传 `--device npu`，日志和输出目录按数据集/可见卡 ID 区分。

## FAQ 与已知问题

- TODO：列出不支持算子、动态 shape、长时间 ATC 编译、依赖冲突、离线权重下载问题、CPU fallback 原因。
- CPU-only 运行只能生成材料，不能声明 NPU 验证通过。

## 公网地址说明

只列本 README 实际使用或实测相关的源码、权重、数据集、评测工具、protocol、测试样例、论文、issue/release note、关键预处理工具 URL；不要堆砌未验证地址。

| 名称 | 地址 | 说明 |
|---|---|---|
| 上游源码 | {url} | 固定到 TODO commit |

## 上库自检

- [ ] 脚手架直接位于 `ACL_PyTorch/built-in/<category>/<model>` 根目录，README 中无 `ascend_adapter/` 等开发期路径。
- [ ] 固定上游 URL 和 commit/revision，并提供 checkout 命令。
- [ ] 使用者提供 checkpoint 时已记录目录树和配套配置；没有静默替换权重。
- [ ] patch 可通过 `git apply --check`。
- [ ] 依赖安装不会覆盖镜像内 `torch/torch_npu/torchvision/torchaudio`，并已做 import/`--help`/单样例 smoke test。
- [ ] 推理、评测和 benchmark 入口默认设备为 NPU，CPU 只作为显式 fallback/baseline。
- [ ] README/PR/性能表使用对外硬件型号（如 Atlas 800I A2），未暴露详细芯片型号或内部代号。
- [ ] 源仓有官方精度时，NPU 精度使用相同完整数据集/split 对齐官方指标，无需 CPU 对比；无官方指标时才做 CPU/upstream baseline 对齐。
- [ ] NPU 性能结果可复现，指标和单位适合该任务；如有官方/GPU 性能，已记录参考口径；未展示 CPU 性能对比。
- [ ] CPU fallback 有具体技术阻塞说明。
- [ ] README 无未处理 TODO 或已明确标记 `待 NPU 验证`。
"""


def export_onnx(model: str) -> str:
    return f'''
#!/usr/bin/env python3
"""Export {model} to ONNX. Replace TODOs with model-specific loader and inputs."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="User-provided checkpoint/repository path")
    parser.add_argument("--output", default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("TODO: load model, build sample inputs, torch.onnx.export, run onnx.checker.")


if __name__ == "__main__":
    main()
'''


def convert_om() -> str:
    return r'''
#!/usr/bin/env bash
set -euo pipefail
ONNX_PATH=${1:-model.onnx}
OUTPUT_PREFIX=${2:-model}
SOC_VERSION=${3:-${SOC_VERSION:-}}
if [ -z "${SOC_VERSION}" ]; then
  echo "ERROR: set SOC_VERSION for ATC conversion." >&2
  exit 2
fi
INPUT_SHAPE=${INPUT_SHAPE:-"TODO_input:1,3,224,224"}
INPUT_FORMAT=${INPUT_FORMAT:-NCHW}
PRECISION_MODE=${PRECISION_MODE:-mixed_float16}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
npu-smi info || true
atc --framework=5 \
  --model="${ONNX_PATH}" \
  --output="${OUTPUT_PREFIX}" \
  --soc_version="${SOC_VERSION}" \
  --input_shape="${INPUT_SHAPE}" \
  --input_format="${INPUT_FORMAT}" \
  --precision_mode_v2="${PRECISION_MODE}"
'''


def infer(model: str, route: str) -> str:
    return f'''
#!/usr/bin/env python3
"""Optional inference entry for {model}.

Generate this file only when upstream lacks a usable inference entry. Prefer patching
upstream inference/evaluation scripts instead.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="{model} inference")
    parser.add_argument("--model-path", required=True, help="checkpoint/repository/OM path depending on route")
    parser.add_argument("--input", required=True, help="input data path")
    parser.add_argument("--output", default="outputs", help="output directory")
    parser.add_argument("--device", choices=["npu", "cpu"], default="npu")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.device == "npu":
        import torch_npu  # noqa: F401
        raise NotImplementedError("TODO: implement NPU inference for route={route}.")
    raise NotImplementedError("TODO: implement CPU/upstream baseline inference.")


if __name__ == "__main__":
    main()
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_url")
    ap.add_argument("output_dir", type=Path, help="Final ModelZoo project directory to create")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--category", default="TODO")
    ap.add_argument("--route", choices=ROUTES, default="auto")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--hardware-model", default="Atlas 800I A2", help="Public-facing hardware model for README/performance tables")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--with-infer", action="store_true", help="Generate optional infer.py only when upstream lacks an inference entry")
    args = ap.parse_args()

    model = args.model_name or infer_name(args.model_url)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "README.md": readme(model, args.model_url, args.category, args.route, args.image, args.hardware_model),
        "requirements.txt": "# Add business dependencies here with minimal changes from upstream.\n# Do not list image-provided torch/torch_npu/torchvision/torchaudio unless intentionally changing the whole verified stack.\n# Remove only blocking/conflicting GPU/CUDA-only packages; keep harmless upstream dependencies to reduce review risk.\n",
    }
    executable: set[str] = set()
    if args.route == "onnx-om":
        files["export_onnx.py"] = export_onnx(model)
        files["convert_om.sh"] = convert_om()
        executable.update({"export_onnx.py", "convert_om.sh"})
    if args.with_infer:
        files["infer.py"] = infer(model, args.route)
        executable.add("infer.py")

    for rel, content in files.items():
        write(out / rel, content, executable=rel in executable, force=args.force)
    print(f"Created Ascend ModelZoo adaptation scaffold: {out}")
    print(f"  Route: {args.route}")
    print(f"  Files: {', '.join(sorted(files.keys()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
