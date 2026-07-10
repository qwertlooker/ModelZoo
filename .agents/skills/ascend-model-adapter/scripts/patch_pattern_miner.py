#!/usr/bin/env python3
"""Mine ModelZoo patch/diff files and summarize recurring Ascend adaptation patterns."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

PATTERN_RULES = {
    "device_cuda_npu": [
        "torch.cuda",
        ".cuda()",
        "cuda:",
        'cuda"',
        "cuda'",
        ".npu()",
        "torch.npu",
        "torch_npu",
        "transfer_to_npu",
    ],
    "ais_bench_om": ["InferSession", "ais_bench", ".om", "dymshape", "custom_sizes"],
    "onnx_export_fix": [
        "torch.onnx.export",
        "dynamic_axes",
        "onnxsim",
        "onnxslim",
        "keep_initializers",
        "opset",
        "onnx.mapping",
        "np_dtype_to_tensor_dtype",
    ],
    "dtype_precision": [
        "float16",
        "fp16",
        "bfloat16",
        "bf16",
        "float32",
        "float64",
        "half()",
        ".half",
    ],
    "unsupported_ops": [
        "grid_sample",
        "torch.fft",
        "rfft",
        "stft",
        "istft",
        "npu_fusion_attention",
        "attention",
        "einsum",
        "split",
        "reshape",
        "contiguous",
        "mmcv",
        "CUDAExtension",
        "load_inline",
    ],
    "vllm_torchair": [
        "vllm",
        "torchair",
        "torch.compile",
        "cudagraph",
        "npugraph",
        "compilation_config",
        "VLLM_ASCEND",
    ],
    "service_cache": [
        "server",
        "client",
        "grpc",
        "FastAPI",
        "cache",
        "warmup",
        "timeout",
    ],
    "requirements_env": [
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "torch==",
        "torchvision",
        "torchaudio",
    ],
    "eval_benchmark": [
        "accuracy",
        "eval",
        "metric",
        "benchmark",
        "latency",
        "FPS",
        "QPS",
        "RTF",
    ],
}

LABELS = {
    "device_cuda_npu": "CUDA/NPU 设备迁移",
    "ais_bench_om": "OM/InferSession 替换",
    "onnx_export_fix": "ONNX 导出与图修正",
    "dtype_precision": "dtype/精度调整",
    "unsupported_ops": "不支持/低效算子替换",
    "vllm_torchair": "vLLM/TorchAir 编译服务",
    "service_cache": "服务化/cache/warmup",
    "requirements_env": "依赖与环境补丁",
    "eval_benchmark": "评测与性能脚本",
}


@dataclass
class PatchInfo:
    path: Path
    category: str
    model: str
    files: list[str]
    adds: int
    dels: int
    patterns: list[str]


def patch_files(root: Path) -> list[Path]:
    return sorted([p for ext in ("*.patch", "*.diff") for p in root.rglob(ext)])


def touched_files(text: str) -> list[str]:
    files = []
    for m in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", text, re.M):
        files.append(m.group(2))
    if not files:
        for m in re.finditer(r"^\+\+\+\s+(?:b/)?([^\t\n]+)", text, re.M):
            f = m.group(1).strip()
            if f != "/dev/null":
                files.append(f)
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def classify(text: str) -> list[str]:
    low = text.lower()
    hits = []
    for key, needles in PATTERN_RULES.items():
        if any(n.lower() in low for n in needles):
            hits.append(key)
    return hits


def analyze(root: Path) -> list[PatchInfo]:
    infos = []
    for p in patch_files(root):
        text = p.read_text(errors="ignore")
        rel = p.relative_to(root)
        parts = rel.parts
        category = parts[0] if parts else "unknown"
        model = "/".join(parts[:2]) if len(parts) >= 2 else str(rel.parent)
        lines = text.splitlines()
        added_text = "\n".join(
            line[1:]
            for line in lines
            if line.startswith("+") and not line.startswith("+++")
        )
        classification_text = added_text + "\n" + "\n".join(touched_files(text))
        infos.append(
            PatchInfo(
                path=rel,
                category=category,
                model=model,
                files=touched_files(text),
                adds=sum(
                    1
                    for line in lines
                    if line.startswith("+") and not line.startswith("+++")
                ),
                dels=sum(
                    1
                    for line in lines
                    if line.startswith("-") and not line.startswith("---")
                ),
                patterns=classify(classification_text),
            )
        )
    return infos


def render(infos: list[PatchInfo], root: Path, max_examples: int) -> str:
    by_cat = Counter(i.category for i in infos)
    by_model = Counter(i.model for i in infos)
    by_pat = Counter(p for i in infos for p in i.patterns)
    by_cat_pat: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    for i in infos:
        for p in i.patterns:
            by_cat_pat[i.category][p] += 1
            if len(examples[p]) < max_examples:
                examples[p].append(str(i.path))

    out = []
    out.append("# ModelZoo patch 模式挖掘报告")
    out.append("")
    out.append(f"根目录：`{root}`")
    out.append(f"patch/diff 文件数：{len(infos)}；涉及模型目录：{len(by_model)}")
    out.append("")
    out.append("## 按类型统计")
    for cat, n in sorted(by_cat.items()):
        out.append(f"- {cat}: {n}")
    out.append("")
    out.append("## 高频模式")
    for key, n in by_pat.most_common():
        out.append(f"- {LABELS.get(key, key)} (`{key}`): {n}")
    out.append("")
    out.append("## 类型 × 模式")
    out.append("| 类型 | " + " | ".join(LABELS[k] for k in PATTERN_RULES) + " |")
    out.append("|---" + "|---" * len(PATTERN_RULES) + "|")
    for cat in sorted(by_cat):
        out.append(
            "| "
            + cat
            + " | "
            + " | ".join(str(by_cat_pat[cat].get(k, 0)) for k in PATTERN_RULES)
            + " |"
        )
    out.append("")
    out.append("## 每类模式示例")
    for key in PATTERN_RULES:
        out.append(f"### {LABELS[key]}")
        for ex in examples.get(key, []):
            out.append(f"- `{ex}`")
        out.append("")
    out.append("## patch 文件清单")
    for i in infos:
        pats = ", ".join(i.patterns) if i.patterns else "unclassified"
        touched = "; ".join(i.files[:5])
        if len(i.files) > 5:
            touched += f"; ...(+{len(i.files) - 5})"
        out.append(f"- `{i.path}`: +{i.adds}/-{i.dels}; {pats}; files: {touched}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root", type=Path, help="ACL_PyTorch/built-in root or a sampled subset root"
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=12)
    args = ap.parse_args()
    if not args.root.is_dir():
        ap.error(f"root is not a directory: {args.root}")
    infos = analyze(args.root)
    if not infos:
        ap.error(f"no patch/diff files found under: {args.root}")
    md = render(infos, args.root, args.max_examples)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
