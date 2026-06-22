#!/usr/bin/env python3
"""Audit the minimum evidence required for a ModelZoo NPU model delivery."""

import argparse
import re
import shlex
import subprocess
from pathlib import Path


README_HEADING_GROUPS = (
    ("概述",),
    ("输入输出",),
    ("推理环境",),
    ("文件目录", "目录结构"),
    ("快速上手",),
    ("模型推理性能", "模型推理性能&精度", "性能与精度"),
    ("精度", "质量"),
    ("公网地址",),
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
TARGET_ADAPTATION_GROUPS = (
    ("目标仓快照", "目标仓 commit", "目标仓版本"),
    ("拟合入路径", "目标上库路径", "目标路径"),
    ("最新参考目录", "近期参考目录", "参考模型"),
    ("最后实质变更", "参考 commit", "参考版本"),
    ("上库文件清单",),
    ("许可证", "License"),
    ("PR 门禁", "贡献门禁"),
    ("modelzoo_level.txt",),
)
INTERNAL_ONLY_FILES = {
    "ACCEPTANCE_PLAN.md",
    "ANALYSIS.md",
    "NPU_ADAPTATION.md",
    "NPU_VALIDATION.md",
    "README_old.md",
    "readme_old.md",
}
INTERNAL_ONLY_PARTS = {
    ".codex-reference",
    ".venv",
    ".venv-cpu",
    ".venv-npu",
    "__pycache__",
    "eval_data",
    "eval_results",
    "outputs",
    "outputs_cpu",
    "results",
    "upstream",
    "weights",
}
DELIVERY_SUFFIXES = {
    ".cfg",
    ".csv",
    ".diff",
    ".flac",
    ".ini",
    ".jpg",
    ".jpeg",
    ".json",
    ".jsonl",
    ".md",
    ".mp3",
    ".patch",
    ".png",
    ".py",
    ".sh",
    ".tsv",
    ".txt",
    ".wav",
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one model delivery directory.")
    parser.add_argument("model_dir")
    parser.add_argument(
        "--target-readiness",
        action="store_true",
        help="Run additional checks for an ACL_PyTorch/built-in merge candidate.",
    )
    parser.add_argument(
        "--target-path",
        help="Expected target path, for example ACL_PyTorch/built-in/audio/Canary-1B.",
    )
    return parser.parse_args()


def missing_groups(
    text: str, groups: tuple[tuple[str, ...], ...]
) -> list[tuple[str, ...]]:
    return [group for group in groups if not any(term in text for term in group)]


def markdown_headings(text: str) -> list[str]:
    headings = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        heading = re.sub(r"<a\b[^>]*>.*?</a>", "", match.group(1))
        headings.append(heading.strip())
    return headings


def level_one_heading_count(text: str) -> int:
    count = 0
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and re.match(r"^#\s+\S", line):
            count += 1
    return count


def missing_heading_groups(
    text: str, groups: tuple[tuple[str, ...], ...]
) -> list[tuple[str, ...]]:
    headings = markdown_headings(text)
    return [
        group
        for group in groups
        if not any(term in heading for term in group for heading in headings)
    ]


def referenced_python_files(text: str) -> set[str]:
    pattern = re.compile(r"\bpython(?:3)?(?:\s+-\S+)*\s+([A-Za-z0-9_./-]+\.py)\b")
    return {match.group(1) for match in pattern.finditer(text)}


def tree_referenced_files(text: str) -> set[str]:
    references = set()
    parents: list[str] = []
    for line in text.splitlines():
        match = re.match(
            r"^((?:(?:│   |    ))*)(?:├──|└──)\s+`?([^`\s#]+)`?",
            line,
        )
        if not match:
            continue
        level = len(match.group(1)) // 4
        value = match.group(2).rstrip("/")
        comment = line[match.end() :]
        parents = parents[:level]
        if Path(value).suffix not in DELIVERY_SUFFIXES:
            parents.append(value)
            continue
        if any(term in comment for term in ("下载后", "运行后", "按需生成", "生成后")):
            continue
        if parents and parents[0] in INTERNAL_ONLY_PARTS | {
            "assets",
            "source",
            "third_party",
            "upstream-original",
            "upstream-npu",
            "validation_reports",
        }:
            continue
        references.add(Path(*parents, value).as_posix())
    return references


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


def unfinished_placeholders(text: str) -> list[str]:
    patterns = (
        r"<[A-Z0-9_ /.-]+>",
        r"待补充",
        r"待验收",
        r"\bTODO\b",
        r"\bTBD\b",
    )
    return [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]


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
    candidates = [repo_root / value, model_dir / value]
    if "/" not in value:
        candidates.append(repo_root / "tools" / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if "/" not in value:
        for patch in (model_dir / "patches").glob("*.patch"):
            patch_text = patch.read_text(encoding="utf-8", errors="replace")
            if re.search(rf"^\+\+\+ b/{re.escape(value)}$", patch_text, re.MULTILINE):
                return patch
    return None


def tracked_model_files(repo_root: Path, model_dir: Path) -> list[Path]:
    relative_model = model_dir.relative_to(repo_root)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            str(relative_model),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        Path(line).relative_to(relative_model)
        for line in result.stdout.splitlines()
        if line.strip()
        and (repo_root / line).is_file()
    ]


def is_internal_only(path: Path) -> bool:
    if path.name in INTERNAL_ONLY_FILES:
        return True
    if path.suffix in {".log", ".pyc"}:
        return True
    if any(part in INTERNAL_ONLY_PARTS or part.startswith(".venv") for part in path.parts):
        return True
    return path.parts[:2] == ("patches", "README.md")


def target_candidate_files(repo_root: Path, model_dir: Path) -> list[Path]:
    return [
        path
        for path in tracked_model_files(repo_root, model_dir)
        if not is_internal_only(path)
    ]


def target_readiness_failures(
    repo_root: Path,
    model_dir: Path,
    target_path: str | None,
    document_text: dict[str, str],
) -> tuple[list[str], list[Path]]:
    failures = []
    if not target_path:
        failures.append("--target-readiness requires --target-path")
    elif not re.fullmatch(
        r"ACL_PyTorch/built-in/[^/\s]+/[^/\s]+(?:/[^/\s]+)*", target_path
    ):
        failures.append(
            "target path must be under ACL_PyTorch/built-in/<domain>/<model>"
        )

    readme_text = document_text.get("README.md", "")
    adaptation_text = document_text.get("NPU_ADAPTATION.md", "")
    missing = missing_groups(adaptation_text, TARGET_ADAPTATION_GROUPS)
    if missing:
        failures.append(
            "NPU_ADAPTATION.md: missing target-delivery evidence: "
            f"{missing}"
        )

    if not re.search(
        r"当前(?:交付|验收)?状态\s*[:：|]\s*S[34]\b", adaptation_text
    ):
        failures.append(
            "target readiness requires a canonical '当前状态: S3' or S4 record"
        )
    if re.search(
        r"\b(?:NPU_ADAPTATION|ACCEPTANCE_PLAN|README_old|NPU_VALIDATION|ANALYSIS)"
        r"\.md\b",
        readme_text,
    ):
        failures.append(
            "README.md references internal evidence that is excluded from the "
            "default target candidate"
        )
    if "/workspace/ModelZoo/" in readme_text:
        failures.append(
            "README.md hard-codes the migration workspace path /workspace/ModelZoo"
        )

    h1_count = level_one_heading_count(readme_text)
    if h1_count != 1:
        failures.append(f"README.md must contain exactly one level-1 title, found {h1_count}")

    commit_ids = re.findall(
        r"\bcommit_id\s*=\s*([0-9a-fA-F]+)\b", readme_text
    )
    for commit_id in commit_ids:
        if len(commit_id) != 40:
            failures.append(
                f"README.md commit_id must be a full 40-character SHA: {commit_id}"
            )
    if not commit_ids:
        failures.append("README.md does not declare a commit_id")

    for filename, text in document_text.items():
        placeholders = unfinished_placeholders(text)
        if placeholders:
            failures.append(
                f"{filename} contains unfinished target-delivery placeholders: "
                f"{placeholders}"
            )

    candidate_files = target_candidate_files(repo_root, model_dir)
    candidate_set = {path.as_posix() for path in candidate_files}
    if "README.md" not in candidate_set:
        failures.append("target candidate does not contain README.md")
    if not any(
        path.suffix in {".py", ".sh", ".patch", ".diff"}
        for path in candidate_files
    ):
        failures.append(
            "target candidate has no runnable script or reproducible patch/diff"
        )
    for reference in sorted(tree_referenced_files(readme_text)):
        if reference not in candidate_set:
            failures.append(
                "README.md directory tree includes a file outside the target "
                f"candidate: {reference}"
            )

    tracked_files = tracked_model_files(repo_root, model_dir)
    forbidden_files = [
        path
        for path in tracked_files
        if path.suffix in {".log", ".pyc"}
        or any(
            part.startswith(".venv")
            or part in {"__pycache__", ".codex-reference", "upstream", "weights"}
            for part in path.parts
        )
    ]
    if forbidden_files:
        failures.append(
            "unignored local/development artifacts must not enter delivery: "
            + ", ".join(path.as_posix() for path in forbidden_files)
        )

    level_path = model_dir / "modelzoo_level.txt"
    if level_path.is_file():
        level_text = level_path.read_text(encoding="utf-8")
        for key in ("FuncStatus", "PerfStatus", "PrecisionStatus"):
            if not re.search(rf"^{key}:(?:PERFECT|OK|POK|NOK)$", level_text, re.MULTILINE):
                failures.append(f"modelzoo_level.txt has no valid {key} entry")

    return failures, candidate_files


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    failures = []

    documents = {
        "README.md": README_HEADING_GROUPS,
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
        missing = (
            missing_heading_groups(text, terms)
            if filename == "README.md"
            else missing_groups(text, terms)
        )
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
    for reference in sorted(tree_referenced_files(readme_text)):
        if resolve_reference(repo_root, model_dir, reference) is None:
            failures.append(
                f"README.md directory tree references a missing delivery file: {reference}"
            )
    for failure in external_clones_without_pin(readme_text + "\n" + acceptance_text):
        failures.append(failure)
    for failure in invalid_commands(readme_text):
        failures.append(failure)
    failures.extend(count_claim_failures(model_dir, acceptance_text))

    patch_files = sorted(patch_dir.glob("*.patch")) if patch_dir.is_dir() else []
    if patch_files:
        baseline_terms = (
            (
                "原始 baseline",
                "官方/公开基线",
                "官方基线",
                "公开基线",
                "原始测试集",
                "官方指标",
            ),
            ("NPU",),
        )
        missing = missing_groups(readme_text + "\n" + acceptance_text, baseline_terms)
        if missing:
            failures.append(
                "patch delivery does not document the original/public baseline and "
                f"NPU candidate: {missing}"
            )

    candidate_files = []
    if args.target_readiness:
        target_failures, candidate_files = target_readiness_failures(
            repo_root,
            model_dir,
            args.target_path,
            document_text,
        )
        failures.extend(target_failures)

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        if args.target_readiness:
            print("CANDIDATE FILES:")
            for path in candidate_files:
                print(f"- {path.as_posix()}")
        raise SystemExit(1)
    print(f"PASS: {model_dir}")
    if args.target_readiness:
        print(f"TARGET: {args.target_path}")
        print("CANDIDATE FILES:")
        for path in candidate_files:
            print(f"- {path.as_posix()}")


if __name__ == "__main__":
    main()
