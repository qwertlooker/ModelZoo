#!/usr/bin/env python3
"""Create a ModelZoo-PyTorch Ascend adaptation scaffold for one upstream model."""
from __future__ import annotations

import argparse
import os
import re
import stat
import sys
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


def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def readme(model: str, url: str, category: str, route: str, image: str, soc: str) -> str:
    return f"""
# {model}-推理指导

> 状态：脚手架已生成，需补齐模型专属 TODO 并在 NPU 环境完成验证后再上库。

## 概述

- 上游模型：{url}
- ModelZoo 类别：{category or 'TODO'}
- 推荐适配路线：{route}
- 目标芯片：{soc}
- 交付范围：Ascend NPU 镜像环境、模型导出/转换/推理、精度与性能验证、CPU fallback。

TODO：补充模型任务、论文/项目简介、license、固定 commit/revision、支持输入输出和限制。

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

### 默认镜像

```bash
export IMAGE={image}
docker pull ${{IMAGE}}
docker run -it --rm --net=host --privileged=true --shm-size=256g \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v $PWD:/workspace -w /workspace \
  ${{IMAGE}} bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 env_check.py
```

## 快速上手

```bash
git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git
# TODO：clone 上游源码并固定 commit/revision
# git clone {url}
# git reset --hard <commit>
# git apply ../diff.patch
pip install -r requirements.txt
```

## 模型导出/转换

### ONNX/OM 路线

```bash
python3 export_onnx.py --model-path <weights_or_repo> --output model.onnx
bash convert_om.sh model.onnx model {soc}
```

### TorchAir/vLLM 路线

TODO：如选择 TorchAir 或 vLLM-Ascend，替换为服务启动、graph compile、NPU ID、内存配置和客户端命令。

## 模型推理

```bash
python3 infer_npu.py --model model.om --input <sample_input> --output outputs/npu
```

CPU fallback：

```bash
python3 infer_cpu.py --model-path <weights_or_repo> --input <sample_input> --output outputs/cpu
```

## 精度验证

默认复用上游原始指标、数据集切分、预处理和后处理；无法复现原始指标时，使用同一输入集的 CPU/upstream baseline 与 NPU 输出对齐，并说明替代原因。

```bash
python3 eval_accuracy.py --cpu-output outputs/cpu --npu-output outputs/npu --metric <upstream_metric>
```

| 数据集 | 指标 | CPU/官方精度 | NPU 精度 | 差异 | 结论 |
|---|---|---:|---:|---:|---|
| TODO | 原始指标/TODO | TODO | TODO | TODO | 待 NPU 验证 |

## 性能验证

默认优先复用上游或同类 ModelZoo 的 benchmark 口径；没有可比口径时，OM 用 `ais_bench` latency/FPS，服务模型用 QPS/tokens/s/latency，音频用 RTF/RTFx，pipeline 同时给纯模型和端到端。首次编译、CPU fallback、数据加载/后处理耗时需单列。

```bash
bash benchmark.sh model.om
```

| 芯片 | Batch/并发 | 输入规格 | 精度模式 | 工具/loop | 性能口径 | 性能 |
|---|---:|---|---|---|---|---:|
| {soc} | TODO | TODO | TODO | TODO | 原始口径/ais_bench/E2E | 待 NPU 验证 |

## FAQ 与已知问题

- TODO：列出不支持算子、动态 shape、长时间 ATC 编译、依赖冲突、离线权重下载问题。
- CPU-only 运行只能生成材料，不能声明 NPU 验证通过。

## 上库自检

- [ ] 固定上游 URL 和 commit/revision，并提供 `git checkout`/`git reset --hard` 命令。
- [ ] 完成镜像环境验证并记录 `env_check.py` 输出。
- [ ] 导出/转换/推理脚本可从干净环境执行，关键路径、batch、soc_version、device_id 可配置。
- [ ] NPU 精度与 CPU/官方指标对齐；精度不是只靠截图或输出文件证明。
- [ ] NPU 性能结果可复现，指标和单位适合该任务。
- [ ] README 写清芯片/机器型号、权重与配置配套关系、外部数据文件来源。
- [ ] 已清理 debug code、重复段落、无用注释、残留 import 和不清晰变量名。
- [ ] README 无未处理 TODO 或已明确标记 `待 NPU 验证`。
"""


def env_check() -> str:
    return r'''
#!/usr/bin/env python3
import json
import os
import platform
import shutil
import subprocess
import sys


def run(cmd):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
        return {"cmd": cmd, "returncode": p.returncode, "output": p.stdout.strip()}
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}

info = {
    "python": sys.version,
    "platform": platform.platform(),
    "env": {k: os.environ.get(k) for k in [
        "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "LD_LIBRARY_PATH", "PATH",
        "ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES", "PYTORCH_NPU_ALLOC_CONF",
    ]},
    "commands": {},
    "python_packages": {},
}
for cmd in (["npu-smi", "info"], ["atc", "--version"], [sys.executable, "-m", "ais_bench", "--help"], [sys.executable, "-m", "msit", "--help"]):
    if shutil.which(cmd[0]) or cmd[0] == sys.executable:
        info["commands"][" ".join(cmd)] = run(cmd)

for pkg in ["torch", "torch_npu", "torchvision", "torchaudio", "onnx", "onnxruntime", "aclruntime", "ais_bench", "msit", "torchair", "vllm", "vllm_ascend"]:
    try:
        mod = __import__(pkg)
        info["python_packages"][pkg] = getattr(mod, "__version__", "imported")
    except Exception as exc:
        info["python_packages"][pkg] = "unavailable: " + str(exc)

try:
    import torch
    info["torch_npu_available"] = bool(hasattr(torch, "npu") and torch.npu.is_available())
    if hasattr(torch, "npu"):
        info["torch_npu_device_count"] = torch.npu.device_count()
except Exception as exc:
    info["torch_npu_available"] = "error: " + repr(exc)

print(json.dumps(info, ensure_ascii=False, indent=2))
'''


def export_onnx(model: str) -> str:
    return f'''
#!/usr/bin/env python3
"""Export {model} to ONNX.

Replace TODO sections with the model-specific loader and dummy/sample inputs.
Run `python3 export_onnx.py --help` for arguments.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Upstream checkpoint/repository path")
    parser.add_argument("--output", default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic axes if supported")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("TODO: load upstream model, build sample inputs, call torch.onnx.export, then run onnx.checker.")


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


def infer_cpu(model: str) -> str:
    return f'''
#!/usr/bin/env python3
"""CPU/upstream baseline inference for {model}."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/cpu")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("TODO: run upstream CPU/PyTorch inference and save baseline outputs.")


if __name__ == "__main__":
    main()
'''


def infer_npu(model: str, route: str) -> str:
    return f'''
#!/usr/bin/env python3
"""Ascend NPU inference for {model}; route={route}."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help=".om path, model dir, or service endpoint depending on route")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/npu")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("TODO: implement OM/aclruntime, TorchAir, torch_npu, or vLLM-Ascend inference path.")


if __name__ == "__main__":
    main()
'''


def eval_accuracy() -> str:
    return r'''
#!/usr/bin/env python3
"""Compare CPU/upstream and NPU outputs or compute a task metric."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-output", required=False)
    parser.add_argument("--npu-output", required=True)
    parser.add_argument("--dataset", required=False)
    parser.add_argument("--metric", default="upstream_original", help="Default to the upstream/original task metric; use cpu_npu_diff only when the original metric is unavailable")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    raise NotImplementedError("TODO: implement the upstream/original task metric first; if unavailable, implement CPU-vs-NPU diff with documented tolerance.")


if __name__ == "__main__":
    main()
'''


def benchmark() -> str:
    return r'''
#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-model.om}
LOOP=${LOOP:-100}
WARMUP=${WARMUP:-5}
BATCHSIZE=${BATCHSIZE:-1}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
npu-smi info || true
if python3 -c 'import ais_bench' >/dev/null 2>&1; then
  python3 -m ais_bench --model "${MODEL}" --loop "${LOOP}" --batchsize "${BATCHSIZE}"
else
  echo "TODO: ais_bench unavailable; implement route-specific benchmark with warmup=${WARMUP}, loop=${LOOP}." >&2
  exit 2
fi
'''


def collect_report() -> str:
    return r'''
#!/usr/bin/env python3
"""Collect adaptation evidence into a Markdown report."""
from pathlib import Path
import datetime as dt

report = Path("adaptation_report.md")
report.write_text(f"""# Adaptation report

Generated: {dt.datetime.now().isoformat(timespec='seconds')}

## Evidence

- [ ] env_check.py output attached
- [ ] CPU/upstream baseline attached
- [ ] ONNX checker / ATC log attached
- [ ] NPU inference output attached
- [ ] accuracy result attached
- [ ] performance result attached
""", encoding="utf-8")
print(report)
'''


def docker_run(image: str) -> str:
    return f'''
#!/usr/bin/env bash
set -euo pipefail
IMAGE=${{IMAGE:-{shell_quote(image)}}}
WORKDIR=${{WORKDIR:-$PWD}}
NPU_ID=${{NPU_ID:-0}}

docker pull "${{IMAGE}}"
docker run -it --rm --net=host --privileged=true --shm-size=256g \
  --device=/dev/davinci${{NPU_ID}} \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "${{WORKDIR}}":/workspace -w /workspace \
  "${{IMAGE}}" bash
'''


def config(model: str, url: str, category: str, route: str, image: str, soc: str) -> str:
    return f'''
model_name: "{model}"
upstream_url: "{url}"
upstream_commit: "TODO"
category: "{category}"
route: "{route}"
default_image: "{image}"
target_soc_version: "{soc}"
status:
  cpu_baseline: "TODO"
  npu_inference: "待 NPU 验证"
  accuracy: "待 NPU 验证"
  performance: "待 NPU 验证"
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
    args = ap.parse_args()

    model = args.model_name or infer_name(args.model_url)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "README.md": readme(model, args.model_url, args.category, args.route, args.image, args.soc_version),
        "requirements.txt": "# Add business dependencies here. Do not list image-provided torch/torch_npu unless explicitly required.\n",
        "adaptation_config.yaml": config(model, args.model_url, args.category, args.route, args.image, args.soc_version),
        "env_check.py": env_check(),
        "docker_run.sh": docker_run(args.image),
        "export_onnx.py": export_onnx(model),
        "convert_om.sh": convert_om(),
        "infer_cpu.py": infer_cpu(model),
        "infer_npu.py": infer_npu(model, args.route),
        "eval_accuracy.py": eval_accuracy(),
        "benchmark.sh": benchmark(),
        "collect_report.py": collect_report(),
    }
    executable = {"env_check.py", "docker_run.sh", "convert_om.sh", "infer_cpu.py", "infer_npu.py", "export_onnx.py", "eval_accuracy.py", "benchmark.sh", "collect_report.py"}
    for rel, content in files.items():
        write(out / rel, content, executable=rel in executable, force=args.force)
    print(f"Created Ascend ModelZoo adaptation scaffold: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
