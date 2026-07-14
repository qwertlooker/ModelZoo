#!/usr/bin/env python3
"""Audit README.md for executability, data prep completeness, and URL disclosure.

Usage:
    python3 audit_readme.py <README.md> [--json-out <path>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Finding:
    level: str
    category: str
    message: str
    line: int | None = None
    line_text: str | None = None


@dataclass
class AuditReport:
    blockers: list[Finding] = field(default_factory=list)
    suggestions: list[Finding] = field(default_factory=list)

    def add_blocker(self, category: str, message: str, line: int | None = None, line_text: str | None = None) -> None:
        self.blockers.append(Finding("BLOCKER", category, message, line, line_text))

    def add_suggestion(self, category: str, message: str, line: int | None = None, line_text: str | None = None) -> None:
        self.suggestions.append(Finding("SUGGESTION", category, message, line, line_text))


def parse_code_blocks(lines: list[str]) -> list[tuple[int, int, str, list[str]]]:
    """Extract fenced code blocks: (start_line, end_line, lang, code_lines).

    start_line/end_line are 1-indexed line numbers of the opening/closing fences.
    """
    blocks: list[tuple[int, int, str, list[str]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^```(\w*)\s*$", line)
        if m:
            lang = m.group(1)
            start = i + 1
            code_lines: list[str] = []
            i += 1
            while i < len(lines):
                if re.match(r"^```\s*$", lines[i]):
                    end = i + 1
                    blocks.append((start, end, lang, code_lines))
                    break
                code_lines.append(lines[i])
                i += 1
        i += 1
    return blocks


def collect_exports_before(lines: list[str], before_line: int) -> set[str]:
    """Collect variable names defined via export VAR= in lines before before_line (1-indexed)."""
    exports: set[str] = set()
    export_re = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=")
    for idx in range(before_line - 1):
        if idx >= len(lines):
            break
        m = export_re.search(lines[idx])
        if m:
            exports.add(m.group(1))
    return exports


def check_placeholder_executability(readme_lines: list[str], report: AuditReport) -> None:
    """Check all bash code blocks for undefined <...> placeholders."""
    blocks = parse_code_blocks(readme_lines)
    bash_langs = {"bash", "sh", "shell", ""}
    placeholder_re = re.compile(r"<([^<>]+)>")

    findings: list[tuple[int, str]] = []

    for start, end, lang, code in blocks:
        if lang.lower() not in bash_langs:
            continue
        for offset, cline in enumerate(code):
            line_no = start + offset
            exports = collect_exports_before(readme_lines, line_no)
            for pm in placeholder_re.finditer(cline):
                placeholder = pm.group(1).strip()
                if re.match(r"^https?://", placeholder):
                    continue
                if placeholder in {
                    "宿主机工程目录",
                    "container-name",
                    "ascend-image-tag",
                }:
                    continue
                var_name = placeholder
                if var_name not in exports:
                    is_env_ref = "${" + var_name + "}" in cline or "$" + var_name in cline
                    if not is_env_ref or var_name not in exports:
                        stripped = cline.rstrip("\n")
                        findings.append((line_no, stripped))

    if findings:
        for line_no, text in findings:
            report.add_blocker(
                "占位符可执行性",
                f"L{line_no}: {text.strip()} — 未定义占位符，前文无 export",
                line_no,
                text,
            )


def extract_all_urls(text: str) -> list[str]:
    """Extract all http(s) URLs and swr.* container image addresses from text."""
    urls: list[str] = []
    http_re = re.compile(r"https?://[^\s)\]>\"'`]+")
    for m in http_re.finditer(text):
        url = m.group(0).rstrip(".,;:'\")]")
        urls.append(url)
    swr_re = re.compile(r"swr\.[a-z0-9-]+\.myhuaweicloud\.com/[^\s)\]>\"'`]+")
    for m in swr_re.finditer(text):
        url = m.group(0).rstrip(".,;:'\")]")
        urls.append(url)
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def find_public_url_section(lines: list[str]) -> tuple[int | None, int | None, list[str]]:
    """Find the section (by heading) that declares public URLs.

    Returns (start_line, end_line, lines_in_section) or (None, None, []).
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = heading_re.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2)
            headings.append((i + 1, level, title))

    target_idx: int | None = None
    for idx, (ln, level, title) in enumerate(headings):
        if ("公网地址" in title or "公网链接" in title or "公开地址" in title or
            "URL" in title or "链接清单" in title):
            target_idx = idx
            break

    if target_idx is None:
        return None, None, []

    start_line, target_level, _ = headings[target_idx]
    if target_idx + 1 < len(headings):
        next_line = headings[target_idx + 1][0]
        end_line = next_line - 1
    else:
        end_line = len(lines)

    section_lines = lines[start_line:end_line]
    return start_line, end_line, section_lines


def check_public_urls(readme_text: str, readme_lines: list[str], report: AuditReport) -> None:
    """Check that the public URL section lists all URLs actually referenced."""
    all_urls = extract_all_urls(readme_text)
    start, end, section_lines = find_public_url_section(readme_lines)
    section_text = "\n".join(section_lines) if section_lines else ""

    generic_phrases = [
        "本 README 中的 GitHub",
        "本 README 中的 GitCode",
        "所有地址",
        "见上文",
        "详见正文",
    ]

    if start is None:
        report.add_blocker(
            "公网地址清单",
            "未找到公网地址/公网链接章节",
        )
        return

    has_generic_only = any(phrase in section_text for phrase in generic_phrases)
    section_urls = extract_all_urls(section_text)

    if has_generic_only and not section_urls:
        report.add_blocker(
            "公网地址清单",
            f'章节"公网地址声明"（L{start}）仅含泛化描述，未列出具体 URL',
            start,
        )

    listed_set = set(section_urls)
    missing = [u for u in all_urls if u not in listed_set]
    if missing:
        msg = f"{len(missing)} 个 URL 未在声明中列出:\n"
        for u in missing:
            msg += f"      - `{u}`\n"
        msg = msg.rstrip()
        report.add_blocker("公网地址清单", msg)


def find_dataset_section(lines: list[str]) -> tuple[int | None, int | None, list[str]]:
    """Find the data preparation section."""
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = heading_re.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2)
            headings.append((i + 1, level, title))

    target_idx: int | None = None
    for idx, (ln, level, title) in enumerate(headings):
        if ("数据" in title or "dataset" in title.lower() or "data" == title.lower().strip() or
            "数据准备" in title or "准备数据" in title):
            target_idx = idx
            break

    if target_idx is None:
        return None, None, []

    start_line, target_level, _ = headings[target_idx]
    if target_idx + 1 < len(headings):
        next_line = headings[target_idx + 1][0]
        end_line = next_line - 1
    else:
        end_line = len(lines)

    section_lines = lines[start_line:end_line]
    return start_line, end_line, section_lines


def extract_accuracy_table_categories(lines: list[str]) -> list[str]:
    """Extract category names from accuracy/performance tables."""
    categories: list[str] = []
    in_table = False
    header_seen = False
    cat_col_idx = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not in_table:
                in_table = True
                for i, c in enumerate(cells):
                    if c in {"类别", "class", "category", "类型", "数据集", "dataset", "场景"}:
                        cat_col_idx = i
                        break
                header_seen = True
                continue
            if header_seen and re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            if cat_col_idx is not None and len(cells) > cat_col_idx:
                cat = cells[cat_col_idx]
                if cat and cat != "---" and not cat.startswith("-") and not cat.isdigit():
                    if "合计" not in cat and "总计" not in cat and "平均" not in cat:
                        categories.append(cat)
        else:
            if in_table and not stripped.startswith("|"):
                in_table = False
                header_seen = False
                cat_col_idx = None
    return categories


def check_dataset_prep(readme_text: str, readme_lines: list[str], report: AuditReport) -> None:
    """Check that data preparation section exists and covers what tables reference."""
    start, end, section_lines = find_dataset_section(readme_lines)
    categories = extract_accuracy_table_categories(readme_lines)

    placeholder_paths = [
        "your_audio_folder",
        "your_data_folder",
        "your_dataset",
        "数据集路径",
        "数据路径",
    ]

    if start is None:
        if categories:
            cats_str = ", ".join(categories)
            report.add_blocker(
                "数据准备完整性",
                f"未找到数据准备章节\n      精度表引用类别: {cats_str}",
            )
        return

    section_text = "\n".join(section_lines)

    has_tree = ("├──" in section_text or "└──" in section_text or
                "目录结构" in section_text or "directory structure" in section_text.lower())
    has_source = ("http://" in section_text or "https://" in section_text or
                  "下载" in section_text or "download" in section_text.lower() or
                  "生成" in section_text or "合成" in section_text)
    has_command = ("```" in section_text or "bash" in section_text or
                   "python" in section_text or "wget" in section_text or
                   "curl" in section_text)

    is_placeholder_only = any(ph in section_text for ph in placeholder_paths) and not has_tree

    if not has_tree and categories:
        cats_str = ", ".join(categories)
        report.add_blocker(
            "数据准备完整性",
            f"数据准备章节（L{start}）缺少目录结构树\n      精度表引用类别: {cats_str}",
            start,
        )

    if is_placeholder_only:
        report.add_blocker(
            "数据准备完整性",
            f"数据准备章节（L{start}）仅使用示例占位路径，未给出真实获取方式",
            start,
        )


def check_code_doc_cross_reference(readme_text: str, readme_path: Path, report: AuditReport) -> None:
    """Check for parent.name/rglob/glob patterns in project scripts and warn if undocumented."""
    project_dir = readme_path.parent
    py_files = list(project_dir.rglob("*.py"))
    path_derivation_patterns: list[tuple[Path, int, str]] = []

    for py in py_files:
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if (".parent.name" in line or ".parent.stem" in line or
                "rglob(" in line or "glob(" in line):
                if "# audit-skip" not in line:
                    path_derivation_patterns.append((py, i, line.strip()))

    if path_derivation_patterns:
        start, _, _ = find_dataset_section(readme_text.splitlines())
        for py, ln, code in path_derivation_patterns[:3]:
            rel_py = str(py.relative_to(project_dir)) if project_dir in py.parents else str(py)
            if start is None:
                report.add_blocker(
                    "代码-文档交叉验证",
                    f"{rel_py} L{ln}: \"{code}\" — 目录结构必须文档化（无数据准备章节）",
                )


def check_code_block_language(readme_lines: list[str], report: AuditReport) -> None:
    """Count code blocks without language specification."""
    blocks = parse_code_blocks(readme_lines)
    no_lang = [(s, e) for s, e, lang, _ in blocks if not lang]
    if no_lang:
        report.add_suggestion(
            "代码块语言标注",
            f"{len(no_lang)} 个代码块缺少语言标识",
        )


def audit(readme_path: Path) -> AuditReport:
    report = AuditReport()
    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    check_placeholder_executability(lines, report)
    check_public_urls(text, lines, report)
    check_dataset_prep(text, lines, report)
    check_code_doc_cross_reference(text, readme_path, report)
    check_code_block_language(lines, report)

    return report


def print_report(report: AuditReport) -> int:
    print("=== README Audit Report ===")
    print()

    if not report.blockers and not report.suggestions:
        print("[PASS] No issues found.")
        return 0

    if report.blockers:
        by_cat: dict[str, list[Finding]] = {}
        for f in report.blockers:
            by_cat.setdefault(f.category, []).append(f)
        for cat, items in by_cat.items():
            print(f"[BLOCKER] {cat} ({len(items)}处):")
            for f in items:
                for line in f.message.split("\n"):
                    print(f"  {line}")
            print()

    if report.suggestions:
        by_cat: dict[str, list[Finding]] = {}
        for f in report.suggestions:
            by_cat.setdefault(f.category, []).append(f)
        for cat, items in by_cat.items():
            print(f"[SUGGESTION] {cat}:")
            for f in items:
                for line in f.message.split("\n"):
                    print(f"  {line}")
            print()

    return 1 if report.blockers else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit README.md for ModelZoo delivery quality")
    parser.add_argument("readme", type=Path, help="Path to README.md")
    parser.add_argument("--json-out", type=Path, default=None, help="Write JSON report to this path")
    args = parser.parse_args()

    if not args.readme.exists():
        print(f"ERROR: {args.readme} does not exist", file=sys.stderr)
        return 2

    report = audit(args.readme)

    if args.json_out:
        data: dict[str, Any] = {
            "readme": str(args.readme),
            "blockers": [
                {"category": f.category, "message": f.message, "line": f.line}
                for f in report.blockers
            ],
            "suggestions": [
                {"category": f.category, "message": f.message, "line": f.line}
                for f in report.suggestions
            ],
        }
        args.json_out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return print_report(report)


if __name__ == "__main__":
    sys.exit(main())
