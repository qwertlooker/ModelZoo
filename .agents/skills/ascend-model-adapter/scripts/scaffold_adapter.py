#!/usr/bin/env python3
"""Create an Ascend adaptation workspace or a flat ACL target scaffold."""

from __future__ import annotations

import argparse
import os
import re
import stat
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_IMAGE = "<ascend-image-tag>"
DEFAULT_HARDWARE = "<target-hardware-model>"
ROUTES = ["onnx-om", "torch-npu", "torchair", "vllm-ascend", "hybrid"]


def infer_name(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    name = re.sub(r"\.git$", "", parts[-1] if parts else "model")
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", name).strip("-_") or "model"


def atomic_write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content.lstrip(), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def route_section(route: str) -> str:
    sections = {
        "onnx-om": """## 模型导出与转换

```bash
python3 export_onnx.py --model-path <checkpoint-or-repo> --output model.onnx
export SOC_VERSION=<soc-version>
bash convert_om.sh model.onnx model "${SOC_VERSION}"
```

TODO：实现模型加载和样例输入；记录 ONNX checker、ATC 日志和 OM SHA256。
""",
        "torch-npu": """## torch_npu 推理

TODO：patch 上游入口支持默认 `--device npu`，补充单样例、精度和性能命令。
""",
        "torchair": """## TorchAir 图编译与推理

TODO：补充上游入口、编译配置、cache 冷热、首次编译、稳定推理和性能命令。
""",
        "vllm-ascend": """## vLLM-Ascend 服务

TODO：补充 server、health check、client、并发、显存、cache 和性能命令。
""",
        "hybrid": """## 混合/拆图路线

TODO：列出每个组件的实现、权重、输入输出、目标设备、CPU fallback 阻塞和端到端数据流。
""",
    }
    return sections[route]


def readme(
    model: str, url: str, category: str, route: str, image: str, hardware: str
) -> str:
    return f"""
# {model}-推理指导

> 当前为脚手架，状态不是 NPU 验证通过。删除所有 TODO，或将缺失验证明确标为 `待 NPU 验证`。

## 概述

- 上游：{url}
- 固定 commit/revision：TODO
- 类别/目录：`ACL_PyTorch/built-in/{category}/{model}`
- 路线：{route}
- 对外硬件型号：{hardware}
- checkpoint/config/tokenizer/label map：TODO
- license/再分发限制：TODO

## 环境

| 项目 | 实测版本 |
|---|---|
| 镜像 | `{image}` |
| 对外硬件型号 | {hardware} |
| CANN / Python | TODO |
| torch / torch_npu | TODO |
| 路线相关工具 | TODO |

不要用 pip 覆盖镜像配套的 torch/torch_npu/torchvision/torchaudio。

### 容器启动

按项目实际设备和权限调整模板；只挂载任务需要的缓存、凭据和宿主机目录。

```bash
export IMAGE={image}
docker pull ${{IMAGE}}
docker run -itd -u root --net=host --privileged=true \
  --name <container-name> --shm-size=256g --ipc=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v <宿主机工程目录>:<宿主机工程目录> \
  -v /root/.cache:/root/.cache \
  ${{IMAGE}} bash -i
docker exec -it <container-name> bash
cd <宿主机工程目录>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

## 源码、patch 与依赖

```bash
git clone {url} <upstream-dir>
cd <upstream-dir>
git checkout <fixed-commit>
git apply --check ../diff.patch
git apply ../diff.patch

# 回到本 ModelZoo 目录，先审计上游 requirements/setup/pyproject。
pip install -r requirements.txt
# 有顶层 editable 包时：pip install --no-deps -e <upstream-dir>
```

TODO：补充 import、`--help`、错误路径和单样例 smoke test。

## 权重与数据

TODO：列出来源、SHA256、目录树、生成者/消费者、完整数据集/split 和评测工具。生成的统计或 label 文件必须给生成命令。

{route_section(route)}
## NPU 推理

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
# python3 <patched-upstream-entry>.py --device npu ...
```

TODO：列出输入输出 name/shape/dtype/layout、命令、输出和日志。

## 精度与性能

| 数据集/split | Metric | 官方/GPU 参考与来源 | NPU 实测 | 结论 |
|---|---|---|---|---|
| TODO | TODO | TODO | 待 NPU 验证 | TODO |

| 对外硬件型号 | Batch/并发 | Warmup/loop | 计时边界 | NPU 性能 |
|---|---:|---:|---|---:|
| {hardware} | TODO | TODO | 纯模型/端到端 TODO | 待 NPU 验证 |

纯模型 NPU 计时前后 synchronize；端到端结果写明数据、网络、排队、后处理和 CPU fallback。正式性能至少独立执行 3 次，保存每次结果并报告中位数。

## FAQ

- CPU-only 只能准备材料，不能声明 NPU 通过。
- TODO：记录实际依赖、cache、长编译、unsupported op 和 fallback。

## 公网地址

| 名称 | URL | 用途 |
|---|---|---|
| 上游源码 | {url} | 固定到 TODO commit |
"""


def export_onnx(model: str) -> str:
    return f'''#!/usr/bin/env python3
"""Export {model} to ONNX after replacing the model-specific TODO."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("TODO: load model, export ONNX, run checker, and record artifact hash")


if __name__ == "__main__":
    main()
'''


def convert_om() -> str:
    return r"""#!/usr/bin/env bash
set -euo pipefail
ONNX_PATH=${1:-model.onnx}
OUTPUT_PREFIX=${2:-model}
SOC_VERSION=${3:-${SOC_VERSION:-}}
if [ -z "${SOC_VERSION}" ]; then
  echo "ERROR: set SOC_VERSION" >&2
  exit 2
fi
INPUT_SHAPE=${INPUT_SHAPE:-"TODO_input:1,3,224,224"}
INPUT_FORMAT=${INPUT_FORMAT:-NCHW}
PRECISION_MODE=${PRECISION_MODE:-mixed_float16}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model="${ONNX_PATH}" --output="${OUTPUT_PREFIX}" \
  --soc_version="${SOC_VERSION}" --input_shape="${INPUT_SHAPE}" \
  --input_format="${INPUT_FORMAT}" --precision_mode_v2="${PRECISION_MODE}"
"""


def infer(model: str, route: str) -> str:
    return f'''#!/usr/bin/env python3
"""Optional {model} inference entry; prefer patching an upstream entry."""

import argparse
import importlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="{model} inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--device", choices=["npu", "cpu"], default="npu")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.device == "npu":
        importlib.import_module("torch_npu")
        raise NotImplementedError("TODO: implement NPU inference for route={route}")
    raise NotImplementedError("TODO: implement explicit CPU/upstream fallback")


if __name__ == "__main__":
    main()
'''


def npu_adaptation(model: str, url: str, category: str) -> str:
    return f"""
# {model} NPU 适配事实

## 目标仓基线

- 检查日期：TODO
- ModelZoo-PyTorch master commit：TODO
- 拟合入路径：`ACL_PyTorch/built-in/{category}/{model}`
- 目标路径：TODO（不存在/新增，或存在/增量更新范围）
- 近期同领域同路线参考目录及最后实质 commit/date：TODO
- 当前贡献门禁、license、自测试、modelzoo_level 适用结论：TODO

## 版本边界

- Upstream：{url}
- Upstream commit：TODO
- 权重 revision/文件/SHA：TODO
- Config/tokenizer/辅助模型：TODO
- 非目标变体：TODO

## 适配与验证事实

- 设备节点、route 和 patch：TODO
- 依赖/import/py_compile/help：TODO
- 权重/数据路径与校验：TODO
- CPU/CUDA/NPU 已执行命令和结果：TODO
- 未执行项、原因和补验方式：TODO

## 正式候选清单与排除项

- 候选文件：TODO
- 默认排除：`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、`upstream/`、日志、权重、数据、cache、`.ascend-adaptation/`

## 当前状态

- S0–S4：未定级
- 下一等级缺失证据：TODO
- target_ready：false
"""


def acceptance_plan(model: str) -> str:
    return f"""
# {model} 验收方案与结果

## 验收目标与版本边界

TODO：当前模型/权重/变体、排除项、目标 S3 或用户要求的 S4。

## 原始公开/官方基线

TODO：dataset/split/样本规模、metric、normalizer/后处理、decode 参数、checkpoint 和来源。

## 数据准备与固定输入

TODO：独立准备命令、manifest/meta、离线模式、SHA、样本数和可读性检查。

## 功能验证（S2）

TODO：干净环境安装、真实权重/输入、CPU/CUDA/NPU 端到端命令、输出 sidecar。

## L2 精度/质量与性能（S3）

TODO：NPU evaluator、自动 compare、阈值来源；公开基线不可比或 patch 改语义时补 CPU 精度对照。

性能至少独立执行 3 次，保存每次结果并报告中位数；说明 pure model/E2E、同步、warmup、batch/并发和峰值 HBM。

## 扩展验收（S4，可选）

TODO：长稳、L3、业务集、并发或用户明确扩展项。

## 最低正式验收清单

- [ ] 数据准备和 metadata 可执行。
- [ ] 功能推理输出独立目录和 sidecar。
- [ ] NPU 精度/质量结果和 compare 非零失败门禁。
- [ ] NPU 性能 3 次和中位数。
- [ ] 报告记录 S0–S4、未完成项和结论。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_url")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--layout",
        choices=["workspace", "target"],
        required=True,
        help="workspace includes internal adaptation/acceptance docs; target is the flat ACL candidate",
    )
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--hardware-model", default=DEFAULT_HARDWARE)
    parser.add_argument("--with-infer", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model = args.model_name or infer_name(args.model_url)
    output = args.output_dir
    files = {
        "README.md": readme(
            model,
            args.model_url,
            args.category,
            args.route,
            args.image,
            args.hardware_model,
        ),
        "requirements.txt": (
            "# Business dependencies only. Keep the image-provided torch stack intact.\n"
        ),
    }
    if args.layout == "workspace":
        files["NPU_ADAPTATION.md"] = npu_adaptation(
            model, args.model_url, args.category
        )
        files["ACCEPTANCE_PLAN.md"] = acceptance_plan(model)
    executable: set[str] = set()
    if args.route == "onnx-om":
        files["export_onnx.py"] = export_onnx(model)
        files["convert_om.sh"] = convert_om()
        executable.update({"export_onnx.py", "convert_om.sh"})
    if args.with_infer:
        files["infer.py"] = infer(model, args.route)
        executable.add("infer.py")

    collisions = [
        output / relative for relative in files if (output / relative).exists()
    ]
    if collisions and not args.force:
        raise FileExistsError(
            "Refusing partial scaffold write; existing files: "
            + ", ".join(str(path) for path in collisions)
        )
    output.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        atomic_write(output / relative, content, relative in executable)

    print(f"Created scaffold: {output}")
    print(f"Route: {args.route}; layout: {args.layout}")
    print("Files: " + ", ".join(sorted(files)))
    print(
        "Initialize a separate project work log with scripts/project_log.py; do not put it here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
