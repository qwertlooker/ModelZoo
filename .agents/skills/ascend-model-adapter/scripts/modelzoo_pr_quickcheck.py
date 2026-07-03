#!/usr/bin/env python3
"""Quick local checks for Ascend ModelZoo PR review.

This script intentionally runs only cheap, source-level checks. It does not
replace patch dry-run, data dry-run, or real NPU accuracy/performance tests.
"""

from __future__ import annotations

import argparse
import ast
import compileall
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".py", ".sh", ".patch", ".txt", ".yaml", ".yml", ".toml", ".json"}


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
    markers = ("<<<<<<<", "=======", ">>>>>>>")
    findings: list[Finding] = []
    for base in paths:
        if not base.exists():
            continue
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for path in files:
            if "__pycache__" in path.parts or path.suffix not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if any(m in line for m in markers):
                    findings.append(Finding("ERROR", f"{rel(path, root)}:{i}: unresolved conflict marker"))
    return findings


def check_modelist(root: Path) -> list[Finding]:
    path = root / "ACL_PyTorch" / "ModeList.md"
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = [line for line in text.splitlines() if line.startswith("| ") and "](" in line]
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
            Finding("ERROR", f"ACL_PyTorch/ModeList.md: total header={m2.group(1)} but table count={total}")
        )
    return findings


def check_py_compile(target: Path) -> list[Finding]:
    py_files = sorted(target.rglob("*.py")) if target.is_dir() else ([target] if target.suffix == ".py" else [])
    if not py_files:
        return []
    ok = compileall.compile_dir(str(target), quiet=1) if target.is_dir() else compileall.compile_file(str(target), quiet=1)
    return [] if ok else [Finding("ERROR", f"{target}: Python compilation failed")]


def run_ruff(target: Path) -> list[Finding]:
    ruff = shutil.which("ruff")
    if ruff is None:
        return [Finding("WARN", "ruff not found; skip lint quickcheck")]
    proc = subprocess.run([ruff, "check", str(target), "--output-format=concise"], text=True, capture_output=True)
    if proc.returncode == 0:
        return []
    detail = (proc.stdout + proc.stderr).strip().splitlines()
    summary = "; ".join(detail[:5])
    return [Finding("ERROR", f"ruff failed: {summary}")]


def check_stale_commands(root: Path, target: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in target.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file() or path.suffix.lower() not in {".md", ".py", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "run_eval.sh" in text:
            findings.append(Finding("ERROR", f"{rel(path, root)}: references run_eval.sh; ensure the script exists or remove it"))
        if "dscore_tool/" in text:
            findings.append(Finding("ERROR", f"{rel(path, root)}: references dscore_tool/; verify actual tool/submodule path"))
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
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
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
    mentioned = {int(x) for x in re.findall(r"batch_size\s*[=：]\s*(\d+)", text)}
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


def check_trailing_whitespace(root: Path, target: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in target.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file() or path.suffix.lower() not in {".md", ".py", ".sh", ".patch", ".txt"}:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if line.endswith((" ", "\t")):
                findings.append(Finding("WARN", f"{rel(path, root)}:{i}: trailing whitespace"))
                break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cheap Ascend ModelZoo PR quick checks")
    parser.add_argument("repo", type=Path, help="ModelZoo repository root")
    parser.add_argument("--target", required=True, type=Path, help="Changed model directory, relative to repo or absolute")
    parser.add_argument("--no-ruff", action="store_true", help="Skip ruff even if installed")
    args = parser.parse_args()

    root = args.repo.resolve()
    target = args.target if args.target.is_absolute() else root / args.target
    target = target.resolve()
    if not root.exists() or not target.exists():
        print("ERROR: repo or target path does not exist", file=sys.stderr)
        return 2

    findings: list[Finding] = []
    findings += check_conflict_markers(root, [root / "ACL_PyTorch" / "ModeList.md", target])
    findings += check_modelist(root)
    findings += check_py_compile(target)
    if not args.no_ruff:
        findings += run_ruff(target)
    findings += check_stale_commands(root, target)
    findings += check_batch_size_consistency(root, target)
    findings += check_trailing_whitespace(root, target)

    errors = [f for f in findings if f.level == "ERROR"]
    warnings = [f for f in findings if f.level != "ERROR"]
    for finding in findings:
        print(f"{finding.level}: {finding.message}")
    print(f"SUMMARY: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
