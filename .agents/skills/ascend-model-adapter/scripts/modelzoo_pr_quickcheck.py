#!/usr/bin/env python3
"""Quick local checks for Ascend ModelZoo PR review.

This script intentionally runs only cheap, source-level checks. It does not
replace patch dry-run, data dry-run, or real NPU accuracy/performance tests.
"""

from __future__ import annotations

import argparse
import ast
import json
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".patch",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
}


@dataclass
class Finding:
    level: str
    message: str


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def check_conflict_markers(root: Path, paths: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for base in paths:
        if not base.exists():
            continue
        files = (
            [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        )
        for path in files:
            if "__pycache__" in path.parts or path.suffix not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            conflict = re.search(
                r"^<<<<<<< .*$.*?^=======$.*?^>>>>>>> .*$", text, flags=re.M | re.S
            )
            if conflict:
                line = text.count("\n", 0, conflict.start()) + 1
                findings.append(
                    Finding(
                        "ERROR", f"{rel(path, root)}:{line}: unresolved conflict block"
                    )
                )
    return findings


def check_modelist(root: Path) -> list[Finding]:
    path = root / "ACL_PyTorch" / "ModeList.md"
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = [
        line for line in text.splitlines() if line.startswith("| ") and "](" in line
    ]
    gpl_rows = [line for line in rows if "modelzoo-GPL" in line]
    builtin_contrib = len(rows) - len(gpl_rows)
    total = len(rows)

    findings: list[Finding] = []
    m1 = re.search(r"built-in.*?contrib.*?合计(\d+)个模型", text)
    if m1 and int(m1.group(1)) != builtin_contrib:
        findings.append(
            Finding(
                "ERROR",
                f"ACL_PyTorch/ModeList.md: built-in+contrib header={m1.group(1)} but table count={builtin_contrib}",
            )
        )
    m2 = re.search(r"项目中合计共(\d+)个模型", text)
    if m2 and int(m2.group(1)) != total:
        findings.append(
            Finding(
                "ERROR",
                f"ACL_PyTorch/ModeList.md: total header={m2.group(1)} but table count={total}",
            )
        )
    return findings


def check_py_compile(target: Path) -> list[Finding]:
    py_files = (
        sorted(target.rglob("*.py"))
        if target.is_dir()
        else ([target] if target.suffix == ".py" else [])
    )
    if not py_files:
        return []
    findings: list[Finding] = []
    with tempfile.TemporaryDirectory(prefix="modelzoo-pycompile-") as temp_dir:
        for index, path in enumerate(py_files):
            try:
                py_compile.compile(
                    str(path), cfile=str(Path(temp_dir) / f"{index}.pyc"), doraise=True
                )
            except py_compile.PyCompileError as exc:
                findings.append(
                    Finding(
                        "ERROR",
                        f"{rel(path, target)}: Python compilation failed: {exc.msg}",
                    )
                )
    return findings


def run_ruff(target: Path) -> list[Finding]:
    ruff = shutil.which("ruff")
    if ruff is None:
        return [Finding("WARN", "ruff not found; skip lint quickcheck")]
    proc = subprocess.run(
        [ruff, "check", str(target), "--output-format=concise"],
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return []
    detail = (proc.stdout + proc.stderr).strip().splitlines()
    summary = "; ".join(detail[:5])
    return [Finding("ERROR", f"ruff failed: {summary}")]


def check_stale_commands(root: Path, target: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in target.rglob("*"):
        if (
            "__pycache__" in path.parts
            or not path.is_file()
            or path.suffix.lower() not in {".md", ".py", ".sh"}
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "run_eval.sh" in text:
            findings.append(
                Finding(
                    "ERROR",
                    f"{rel(path, root)}: references run_eval.sh; ensure the script exists or remove it",
                )
            )
        if "dscore_tool/" in text:
            findings.append(
                Finding(
                    "ERROR",
                    f"{rel(path, root)}: references dscore_tool/; verify actual tool/submodule path",
                )
            )
    return findings


def _argparse_defaults(path: Path) -> dict[str, object]:
    defaults: dict[str, object] = {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        arg_name = None
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                arg_name = arg.value[2:].replace("-", "_")
                break
        if not arg_name:
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                defaults[arg_name] = kw.value.value
    return defaults


def check_batch_size_consistency(root: Path, target: Path) -> list[Finding]:
    readme = target / "README.md"
    if not readme.exists():
        return []
    text = readme.read_text(encoding="utf-8", errors="ignore")
    mentioned = {
        int(x)
        for x in re.findall(
            r"(?:默认|default)[^\n]{0,40}?batch[_-]?size\s*[=：]\s*(\d+)",
            text,
            flags=re.I,
        )
    }
    if not mentioned:
        return []
    findings: list[Finding] = []
    for script in target.rglob("*.py"):
        defaults = _argparse_defaults(script)
        if "batch_size" in defaults and isinstance(defaults["batch_size"], int):
            default = int(defaults["batch_size"])
            if default not in mentioned:
                findings.append(
                    Finding(
                        "WARN",
                        f"{rel(script, root)} default --batch_size={default}, README mentions {sorted(mentioned)}; verify commands/results use one口径",
                    )
                )
    return findings


def check_readme_references(root: Path, target: Path) -> list[Finding]:
    readme = target / "README.md"
    if not readme.exists():
        return [Finding("ERROR", f"{rel(target, root)}: README.md is missing")]
    text = readme.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"(?:python3?|bash)\s+([A-Za-z0-9_./-]+\.(?:py|sh))",
        r"git\s+apply(?:\s+--check)?\s+([A-Za-z0-9_./-]+\.(?:patch|diff))",
        r"pip\s+install\s+-r\s+([A-Za-z0-9_./-]+\.txt)",
    ]
    findings: list[Finding] = []
    seen: set[str] = set()
    for pattern in patterns:
        for raw in re.findall(pattern, text):
            if raw in seen:
                continue
            seen.add(raw)
            if "<" in raw or raw.startswith(("http://", "https://")):
                continue
            candidate = (target / raw).resolve()
            if candidate.exists():
                continue
            findings.append(
                Finding(
                    "INFO",
                    f"{rel(readme, root)}: referenced file is not in the PR directory; verify clone/cwd provides it: {raw}",
                )
            )
    return findings


def _markdown_heading_slugs(text: str) -> set[str]:
    slugs: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    fence_char = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        fence = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_char = marker
            elif marker == fence_char:
                in_fence = False
                fence_char = ""
            continue
        if in_fence:
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not heading:
            continue
        base = re.sub(r"[^\w\- ]", "", heading.group(1).lower())
        base = re.sub(r"\s+", "-", base.strip())
        if not base:
            continue
        duplicate = counts.get(base, 0)
        counts[base] = duplicate + 1
        slugs.add(base if duplicate == 0 else f"{base}-{duplicate}")
    return slugs


def check_readme_markdown(root: Path, target: Path) -> list[Finding]:
    readme = target / "README.md"
    if not readme.exists():
        return []
    raw = readme.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    findings: list[Finding] = []
    if raw and not raw.endswith(b"\n"):
        findings.append(
            Finding("ERROR", f"{rel(readme, root)}: Markdown file lacks final newline")
        )

    in_fence = False
    fence_char = ""
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        fence = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)
        if not fence:
            continue
        marker = fence.group(1)[0]
        if not in_fence:
            in_fence = True
            fence_char = marker
            if not fence.group(2).strip():
                findings.append(
                    Finding(
                        "ERROR",
                        f"{rel(readme, root)}:{line_number}: fenced code block lacks a language",
                    )
                )
        elif marker == fence_char:
            in_fence = False
            fence_char = ""

    slugs = _markdown_heading_slugs(text)
    for match in re.finditer(r"\]\(#([^)]+)\)", text):
        anchor = urllib.parse.unquote(match.group(1)).lower()
        if anchor not in slugs:
            line = text.count("\n", 0, match.start()) + 1
            findings.append(
                Finding(
                    "ERROR",
                    f"{rel(readme, root)}:{line}: internal Markdown anchor does "
                    f"not match a heading: #{anchor}",
                )
            )
    return findings


def check_review_patterns(root: Path, target: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in target.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "# noqa" in text or "pylint: disable" in text:
            findings.append(
                Finding(
                    "WARN",
                    f"{rel(path, root)}: lint suppression requires explicit review",
                )
            )
        if (
            re.search(r"(?:time\.time|perf_counter)\s*\(", text)
            and "npu" in text.lower()
            and "synchronize" not in text
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"{rel(path, root)}: NPU timing detected without synchronize; verify E2E vs device timing",
                )
            )
        if (
            "subprocess.run" in text
            and "check=True" not in text
            and ".returncode" not in text
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"{rel(path, root)}: subprocess.run may not propagate child failure",
                )
            )
        if re.search(r"\bInferSession\s*\(", text) and ".free_resource" not in text:
            findings.append(
                Finding(
                    "WARN",
                    f"{rel(path, root)}: ais_bench InferSession is created without explicit free_resource cleanup",
                )
            )
        if re.search(r"\bInferenceSession\s*\(", text) and not any(
            marker in text
            for marker in ("finally:", "ExitStack", "contextmanager", ".clear()")
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"{rel(path, root)}: ONNX Runtime InferenceSession ownership has no explicit scope-exit cleanup",
                )
            )
    for path in target.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".py", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "huggingface.co" in text and "/blob/" in text:
            findings.append(
                Finding(
                    "ERROR",
                    f"{rel(path, root)}: Hugging Face /blob/ URL may download HTML",
                )
            )
    return findings


def check_template_placeholders(root: Path, target: Path) -> list[Finding]:
    findings: list[Finding] = []
    placeholder = re.compile(
        r"<(?:container-name|宿主机工程目录|upstream[^>]*|fixed-commit|checkpoint[^>]*|soc-version|target[^>]*)>",
        flags=re.I,
    )
    for path in target.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".md",
            ".py",
            ".sh",
            ".txt",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "TODO" in text:
            findings.append(
                Finding("WARN", f"{rel(path, root)}: contains unresolved TODO")
            )
        if placeholder.search(text):
            findings.append(
                Finding(
                    "WARN",
                    f"{rel(path, root)}: contains unresolved template placeholder",
                )
            )
    return findings


def check_modelzoo_constraints(root: Path, target: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in target.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".py", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for device in ("npu:0", "cuda:0"):
            if device in text:
                findings.append(
                    Finding(
                        "ERROR",
                        f"{rel(path, root)}: hard-coded device {device}; use visibility env vars",
                    )
                )
        if path.suffix.lower() == ".py":
            lowered_name = path.name.lower()
            runtime_download = any(
                pattern in text
                for pattern in (
                    "nltk.download(",
                    "datasets.load_dataset(",
                    "hf_hub_download(",
                    "snapshot_download(",
                )
            )
            if runtime_download and not any(
                token in lowered_name for token in ("prepare", "download")
            ):
                findings.append(
                    Finding(
                        "ERROR",
                        f"{rel(path, root)}: runtime/assessment entry embeds an online data download",
                    )
                )
            if re.search(
                r"except\s+Exception(?:\s+as\s+\w+)?\s*:\s*(?:pass|continue|return\b)",
                text,
            ):
                findings.append(
                    Finding(
                        "WARN",
                        f"{rel(path, root)}: broad exception may silently hide a required failure",
                    )
                )
        if path.name == "README.md":
            if "download.pytorch.org/whl/cpu" in text and re.search(
                r"torch[-_]npu", text, flags=re.I
            ):
                findings.append(
                    Finding(
                        "ERROR",
                        f"{rel(path, root)}: CPU-only PyTorch install is mixed with torch_npu guidance",
                    )
                )
    return findings


def check_trailing_whitespace(root: Path, target: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in target.rglob("*"):
        if (
            "__pycache__" in path.parts
            or not path.is_file()
            or path.suffix.lower() not in {".md", ".py", ".sh", ".patch", ".txt"}
        ):
            continue
        for i, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            if line.endswith((" ", "\t")):
                findings.append(
                    Finding("WARN", f"{rel(path, root)}:{i}: trailing whitespace")
                )
                break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cheap Ascend ModelZoo PR quick checks"
    )
    parser.add_argument("repo", type=Path, help="ModelZoo repository root")
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        help="Changed model directory, relative to repo or absolute",
    )
    parser.add_argument(
        "--no-ruff", action="store_true", help="Skip ruff even if installed"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings, including unavailable ruff, as failures",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional structured findings output",
    )
    args = parser.parse_args()

    root = args.repo.resolve()
    target = args.target if args.target.is_absolute() else root / args.target
    target = target.resolve()
    if not root.exists() or not target.exists():
        print("ERROR: repo or target path does not exist", file=sys.stderr)
        return 2
    try:
        target.relative_to(root)
    except ValueError:
        print("ERROR: --target must be inside the repository root", file=sys.stderr)
        return 2

    findings: list[Finding] = []
    findings += check_conflict_markers(
        root, [root / "ACL_PyTorch" / "ModeList.md", target]
    )
    findings += check_modelist(root)
    findings += check_py_compile(target)
    if not args.no_ruff:
        findings += run_ruff(target)
    findings += check_stale_commands(root, target)
    findings += check_batch_size_consistency(root, target)
    findings += check_readme_references(root, target)
    findings += check_readme_markdown(root, target)
    findings += check_review_patterns(root, target)
    findings += check_template_placeholders(root, target)
    findings += check_modelzoo_constraints(root, target)
    findings += check_trailing_whitespace(root, target)

    errors = [f for f in findings if f.level == "ERROR"]
    warnings = [f for f in findings if f.level == "WARN"]
    infos = [f for f in findings if f.level == "INFO"]
    for finding in findings:
        print(f"{finding.level}: {finding.message}")
    print(
        f"SUMMARY: {len(errors)} error(s), {len(warnings)} warning(s), {len(infos)} info item(s)"
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                [finding.__dict__ for finding in findings], ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
