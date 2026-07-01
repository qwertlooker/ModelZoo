#!/usr/bin/env python3
"""Create a minimal ModelZoo-PyTorch Ascend adaptation scaffold.

Generated files follow actual ACL_PyTorch/built-in conventions:
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

DEFAULT_IMAGE = "swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-onnx-inference:cann8.3.rc1_torch2.1.0-800I-A2-ubuntu22.04-py3.11-aarch64"
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


def readme(model: str, url: str, category: str, route: str, image: str, soc: str) -> str:
    if route == "onnx-om":
        route_note = f"""
## 模型导出/转换

```bash
python3 export_onnx.py --model-path <checkpoint_or_repo> --output model.onnx
bash convert_om.sh model.onnx model {soc}
```
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

> 状态：脚手架已生成。请补齐 TODO，并在 NPU 环境完成验证后再上库。

## 概述

- 上游模型：{url}
- ModelZoo 类别：{category or 'TODO'}
- 推荐适配路线：{route}
- 目标芯片：{soc}
- 交付范围：Ascend NPU 镜像环境、源码 patch、推理、精度与性能验证。

TODO：补充任务简介、论文/项目链接、license、固定 commit/revision、支持输入输出和限制。

## Pipeline 组件部署（多组件模型填写）

| 组件 | 上游默认后端 | 选定后端 | NPU 可行性 | CPU fallback 原因 |
|---|---|---|---|---|
| TODO | TODO | torch_npu/OM/TorchAir/CPU | TODO | TODO |

CPU fallback 必须有具体技术阻塞，不能只写“上游默认 CPU”。

## 输入输出数据

| 名称 | dtype | shape | layout | 说明 |
|---|---|---|---|---|
| TODO_input | TODO | TODO | TODO | TODO |
| TODO_output | TODO | TODO | TODO | TODO |

## 推理环境准备

| 组件 | 推荐/实测版本 | 说明 |
|---|---|---|
| 固件与驱动 | TODO | 与 CANN 匹配 |
| CANN | TODO | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Python | TODO | 镜像内置或业务环境 |
| PyTorch / torch_npu | TODO | 镜像内置，通常不要重装 |
| 推理工具 | TODO | ATC / ais_bench / msit / TorchAir / vLLM-Ascend |

请勿在容器内用 pip 重装镜像内置的 `torch/torch_npu`，除非 README 明确解释原因。

### 默认镜像

```bash
export IMAGE={image}
docker pull ${{IMAGE}}
docker run -it --rm --net=host --privileged=true --shm-size=256g \\
  --device=/dev/davinci0 \\
  --device=/dev/davinci_manager \\
  --device=/dev/devmm_svm \\
  --device=/dev/hisi_hdc \\
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \\
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \\
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \\
  -v $PWD:/workspace -w /workspace \\
  ${{IMAGE}} bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

## 快速上手

```bash
git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git
# TODO：clone 上游源码并固定 commit/revision
# git clone {url}
# cd <upstream_dir>
# git reset --hard <commit>
# git apply ../diff.patch
pip install -r requirements.txt
```

## 准备权重和数据

TODO：优先记录使用者提供的 checkpoint/权重目录；列出配置、tokenizer、label-map、speaker-map 等成套文件和目录树。若替换为其他权重，必须说明原因和差异。
{route_note}
## 模型推理

优先运行 patch 后的上游推理入口，例如：

```bash
# python3 <upstream_infer.py> --model <checkpoint_or_repo> --device npu --input <sample_input>
```

仅当上游没有统一入口时，使用本目录通过 `--with-infer` 生成的 `infer.py`。

## 精度验证

默认复用上游原始指标、数据集切分、预处理和后处理；无法复现原始指标时，使用同一输入集的 CPU/upstream baseline 与 NPU 输出对齐，并说明替代原因。

| 数据集 | 指标 | CPU/官方精度 | NPU 精度 | 差异 | 结论 |
|---|---|---:|---:|---:|---|
| TODO | 原始指标/TODO | TODO | TODO | TODO | 待 NPU 验证 |

## 性能验证

默认优先复用上游或同类 ModelZoo benchmark 口径；没有可比口径时，OM 用 `ais_bench` latency/FPS，服务模型用 QPS/tokens/s/latency，音频用 RTF/RTFx，pipeline 同时给纯模型和端到端。首次编译、CPU fallback、数据加载/后处理耗时需单列。

| 芯片 | Batch/并发 | 输入规格 | 精度模式 | 工具/loop | 性能口径 | 性能 |
|---|---:|---|---|---|---|---:|
| {soc} | TODO | TODO | TODO | TODO | 原始口径/ais_bench/E2E | 待 NPU 验证 |

## FAQ 与已知问题

- TODO：列出不支持算子、动态 shape、长时间 ATC 编译、依赖冲突、离线权重下载问题、CPU fallback 原因。
- CPU-only 运行只能生成材料，不能声明 NPU 验证通过。

## 上库自检

- [ ] 固定上游 URL 和 commit/revision，并提供 checkout 命令。
- [ ] 使用者提供 checkpoint 时已记录目录树和配套配置；没有静默替换权重。
- [ ] patch 可通过 `git apply --check`。
- [ ] NPU 精度与 CPU/官方指标对齐；精度不是只靠截图或输出文件证明。
- [ ] NPU 性能结果可复现，指标和单位适合该任务。
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
SOC_VERSION=${3:-${SOC_VERSION:-Ascend910B}}
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
    parser.add_argument("--device-id", type=int, default=0)
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
    ap.add_argument("--soc-version", default="Ascend910B")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--with-infer", action="store_true", help="Generate optional infer.py only when upstream lacks an inference entry")
    args = ap.parse_args()

    model = args.model_name or infer_name(args.model_url)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "README.md": readme(model, args.model_url, args.category, args.route, args.image, args.soc_version),
        "requirements.txt": "# Add business dependencies here. Do not list image-provided torch/torch_npu unless explicitly required.\n# If torchvision/torchaudio is needed, install with --no-deps unless intentionally changing torch stack.\n",
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
