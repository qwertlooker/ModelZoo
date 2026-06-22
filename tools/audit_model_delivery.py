#!/usr/bin/env python3
"""Audit the minimum evidence required for a ModelZoo NPU model delivery."""

import argparse
import re
import shlex
from pathlib import Path


README_HEADINGS = (
    "概述",
    "输入输出",
    "推理环境",
    "文件目录",
    "快速上手",
    "模型推理性能",
    "公网地址",
)
ACCEPTANCE_GROUPS = (
    ("原始测试集", "原始数据"),
    ("官方指标", "官方分数", "官方/公开"),
    ("CPU", "CUDA"),
    ("NPU",),
    ("功能验证", "功能矩阵"),
    ("L2",),
    ("精度", "质量"),
    ("性能",),
    ("最低正式验收",),
    ("报告模板",),
)
ADAPTATION_GROUPS = (
    ("版本边界", "版本与来源", "来源与边界"),
    ("验证",),
    ("未执行",),
    ("S0", "S1", "S2", "S3", "S4"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one model delivery directory.")
    parser.add_argument("model_dir")
    return parser.parse_args()


def missing_groups(
    text: str, groups: tuple[tuple[str, ...], ...]
) -> list[tuple[str, ...]]:
    return [group for group in groups if not any(term in text for term in group)]


def referenced_python_files(text: str) -> set[str]:
    pattern = re.compile(r"\bpython(?:3)?(?:\s+-\S+)*\s+([A-Za-z0-9_./-]+\.py)\b")
    return {match.group(1) for match in pattern.finditer(text)}


def external_clones_without_pin(text: str) -> list[str]:
    failures = []
    normalized = re.sub(r"\\\s*\n\s*", " ", text)
    lines = normalized.splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"\bgit clone\b(.*)", line)
        if not match:
            continue
        try:
            tokens = shlex.split(match.group(1))
        except ValueError:
            failures.append(f"cannot parse git clone command: {line.strip()}")
            continue

        positionals = []
        clone_ref = None
        option_takes_value = {
            "-b",
            "--branch",
            "--depth",
            "--filter",
            "--origin",
            "--reference",
            "--reference-if-able",
            "--separate-git-dir",
            "--shallow-since",
            "--shallow-exclude",
            "--template",
            "-j",
            "--jobs",
        }
        token_index = 0
        while token_index < len(tokens):
            token = tokens[token_index]
            if token in option_takes_value:
                if token_index + 1 >= len(tokens):
                    failures.append(
                        f"git clone option has no value: {line.strip()}"
                    )
                    positionals = []
                    break
                if token in {"-b", "--branch"}:
                    clone_ref = tokens[token_index + 1]
                token_index += 2
                continue
            if token.startswith("-"):
                token_index += 1
                continue
            positionals.append(token)
            token_index += 1

        if not positionals:
            continue
        url = positionals[0]
        if "ModelZoo-PyTorch" in url:
            continue
        target = positionals[1] if len(positionals) > 1 else None
        window = "\n".join(lines[index : index + 12])
        pinned_by_clone = bool(
            clone_ref
            and re.fullmatch(
                r"(?:[0-9a-f]{7,40}|v?\d[\w.-]*|[\w.-]+@[0-9a-f]{7,40})",
                clone_ref,
            )
        )
        pinned_after_clone = bool(
            re.search(
            r"\bgit (?:-C\s+\S+\s+)?(?:checkout|reset\s+--hard)\s+"
            r"(?:[0-9a-f]{7,40}|v?\d|[A-Za-z0-9_.-]+@[0-9a-f]{7,40})",
            window,
            )
        )
        if not pinned_by_clone and not pinned_after_clone:
            failures.append(f"unversioned external clone: {line.strip()}")
    return failures


def invalid_commands(text: str) -> list[str]:
    failures = []
    if "--speculative-config.method" in text:
        failures.append(
            "vLLM speculative config must be passed as one JSON "
            "--speculative-config value"
        )
    if re.search(
        r"pip install torch(?:==\S+)? torchaudio(?:==\S+)?\s+\\\n"
        r"\s+--index-url https://download\.pytorch\.org/whl/cpu.*?"
        r"pip install torch-npu",
        text,
        flags=re.DOTALL,
    ):
        failures.append(
            "NPU setup installs CPU-only PyTorch before torch-npu; split CPU/NPU "
            "environments and use a CANN-matched NPU wheel set"
        )
    return failures


def count_claim_failures(model_dir: Path, acceptance_text: str) -> list[str]:
    failures = []
    service_prompts = model_dir / "test_data" / "service_prompts.jsonl"
    if service_prompts.is_file():
        actual = sum(1 for line in service_prompts.read_text().splitlines() if line.strip())
        claimed = re.search(
            r"\|\s*功能验证\s*\|\s*(?:仓内\s*)?(\d+)\s*条", acceptance_text
        )
        if claimed and int(claimed.group(1)) != actual:
            failures.append(
                f"functional validation claims {claimed.group(1)} prompts but "
                f"committed input has {actual}"
            )
    smiles = model_dir / "test_data" / "smiles_functional.txt"
    if smiles.is_file():
        actual = sum(1 for line in smiles.read_text().splitlines() if line.strip())
        claimed = re.search(
            r"\|\s*功能验证\s*\|\s*(?:仓内\s*)?(\d+)\s*条", acceptance_text
        )
        if claimed and int(claimed.group(1)) != actual:
            failures.append(
                f"functional validation claims {claimed.group(1)} SMILES but "
                f"committed input has {actual}"
            )
    return failures


def resolve_reference(repo_root: Path, model_dir: Path, value: str) -> Path | None:
    if value.startswith("/workspace/ModelZoo/"):
        value = value.removeprefix("/workspace/ModelZoo/")
    candidates = (repo_root / value, model_dir / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if "/" not in value:
        for patch in (model_dir / "patches").glob("*.patch"):
            patch_text = patch.read_text(encoding="utf-8", errors="replace")
            if re.search(rf"^\+\+\+ b/{re.escape(value)}$", patch_text, re.MULTILINE):
                return patch
    return None


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    failures = []

    documents = {
        "README.md": tuple((term,) for term in README_HEADINGS),
        "NPU_ADAPTATION.md": ADAPTATION_GROUPS,
        "ACCEPTANCE_PLAN.md": ACCEPTANCE_GROUPS,
    }
    all_text = ""
    document_text = {}
    for filename, terms in documents.items():
        path = model_dir / filename
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        text = path.read_text(encoding="utf-8")
        document_text[filename] = text
        all_text += "\n" + text
        missing = missing_groups(text, terms)
        if missing:
            failures.append(f"{filename}: missing required sections/terms: {missing}")

    for reference in sorted(referenced_python_files(all_text)):
        if reference.startswith("/") and not reference.startswith("/workspace/ModelZoo/"):
            continue
        if resolve_reference(repo_root, model_dir, reference) is None:
            failures.append(f"documented Python entry does not exist: {reference}")

    patch_dir = model_dir / "patches"
    if patch_dir.is_dir() and not (patch_dir / "README.md").is_file():
        failures.append("patches/ exists without patches/README.md")

    readme_text = document_text.get("README.md", "")
    acceptance_text = document_text.get("ACCEPTANCE_PLAN.md", "")
    for failure in external_clones_without_pin(readme_text + "\n" + acceptance_text):
        failures.append(failure)
    for failure in invalid_commands(readme_text):
        failures.append(failure)
    failures.extend(count_claim_failures(model_dir, acceptance_text))

    patch_files = sorted(patch_dir.glob("*.patch")) if patch_dir.is_dir() else []
    if patch_files:
        baseline_terms = (
            ("原始 baseline", "未应用 patch", "原始 CPU", "原始 CUDA"),
            ("patch 后", "应用 patch 后", "回归 baseline"),
            ("NPU",),
        )
        missing = missing_groups(readme_text + "\n" + acceptance_text, baseline_terms)
        if missing:
            failures.append(
                "patch delivery does not document original, patched same-device, "
                f"and NPU result paths: {missing}"
            )

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"PASS: {model_dir}")


if __name__ == "__main__":
    main()
