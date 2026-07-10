#!/usr/bin/env python3
"""Sample recent Ascend ModelZoo-PyTorch ACL built-in projects.

The script intentionally has no third-party dependencies. It parses GitCode's
server-rendered category pages, ranks visible projects by recency/PR signal, and
can optionally sparse-clone selected project directories for local inspection.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
import tempfile
import urllib.request
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CATEGORIES = [
    "audio",
    "cv",
    "nlp",
    "ocr",
    "embedding",
    "foundation_models",
    "embodied_ai",
]
DEFAULT_REPO_PAGE = "https://gitcode.com/Ascend/ModelZoo-PyTorch"
DEFAULT_GIT_URL = "https://gitcode.com/Ascend/ModelZoo-PyTorch.git"
EXCLUDE_NAMES = {"master", "utils"}


@dataclass(frozen=True)
class Entry:
    category: str
    name: str
    href: str
    time_text: str
    days: float
    pr: int | None
    commit: str

    @property
    def sparse_path(self) -> str:
        return f"ACL_PyTorch/built-in/{self.category}/{self.name}"

    @property
    def url(self) -> str:
        return (
            "https://gitcode.com" + self.href
            if self.href.startswith("/")
            else self.href
        )


def fetch(url: str, timeout: int = 45, retries: int = 3) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "ascend-model-adapter/1.0"}
    )
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # network can be flaky on GitCode pages
            last_exc = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 5))
    assert last_exc is not None
    raise last_exc


def relative_days(text: str) -> float:
    text = text.strip()
    m = re.search(r"(\d+)", text)
    n = int(m.group(1)) if m else 0
    if "分钟" in text or "秒" in text or "小时" in text:
        return 0
    if "天" in text:
        return n
    if "个月" in text:
        return n * 30
    if "年" in text:
        return n * 365
    return 10_000


def parse_category(
    category: str,
    repo_page: str = DEFAULT_REPO_PAGE,
    timeout: int = 45,
    retries: int = 3,
) -> list[Entry]:
    url = f"{repo_page}/tree/master/ACL_PyTorch/built-in/{category}"
    text = fetch(url, timeout=timeout, retries=retries)
    entries: list[Entry] = []
    for tr in re.findall(r'<tr class="repo-code-table-row".*?</tr>', text, flags=re.S):
        m = re.search(
            r'<a href="([^"]+)" class="repo-code-table-name" title="([^"]*)"', tr
        )
        if not m:
            continue
        href, name = m.group(1), html.unescape(m.group(2))
        if name.startswith(".") or name in EXCLUDE_NAMES:
            continue
        time_m = re.search(
            r'class="text-\(--theme-aide-text\)"[^>]*>(.*?)</span>', tr, flags=re.S
        )
        time_text = (
            html.unescape(re.sub(r"<.*?>", "", time_m.group(1)).strip())
            if time_m
            else ""
        )
        commit_m = re.search(
            r'<span class="commit-message-text" title="([^"]*)"', tr, flags=re.S
        )
        commit = (
            html.unescape(commit_m.group(1)).replace("\n", " ").strip()
            if commit_m
            else ""
        )
        pr_m = re.search(r"/pull/(\d+)", tr)
        pr = int(pr_m.group(1)) if pr_m else None
        entries.append(
            Entry(category, name, href, time_text, relative_days(time_text), pr, commit)
        )
    return entries


def filter_entries(
    entries: Iterable[Entry],
    categories: set[str] | None = None,
    keywords: list[str] | None = None,
) -> list[Entry]:
    """Filter entries before recency ranking.

    Keywords are matched case-insensitively against project name and the visible
    commit-message signal.  Filtering before category coverage prevents a request
    for one task family from being diluted by unrelated recent projects.
    """
    selected = list(entries)
    if categories:
        selected = [entry for entry in selected if entry.category in categories]
    if keywords:
        lowered = [keyword.casefold() for keyword in keywords]
        selected = [
            entry
            for entry in selected
            if any(
                keyword in f"{entry.name} {entry.commit}".casefold()
                for keyword in lowered
            )
        ]
    return selected


def select_entries(
    entries: Iterable[Entry], count: int, per_category_min: int = 1
) -> list[Entry]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if per_category_min < 0:
        raise ValueError("per_category_min must be >= 0")
    all_entries = sorted(
        entries, key=lambda e: (e.days, -(e.pr or 0), e.category, e.name.lower())
    )
    selected: list[Entry] = []
    seen: set[tuple[str, str]] = set()
    for cat in CATEGORIES:
        cat_entries = [e for e in all_entries if e.category == cat]
        for e in cat_entries[:per_category_min]:
            if len(selected) >= count:
                break
            selected.append(e)
            seen.add((e.category, e.name))
    for e in all_entries:
        if len(selected) >= count:
            break
        key = (e.category, e.name)
        if key not in seen:
            selected.append(e)
            seen.add(key)
    return sorted(
        selected, key=lambda e: (e.days, -(e.pr or 0), e.category, e.name.lower())
    )


def markdown(entries: list[Entry], all_entries: list[Entry]) -> str:
    counts = {
        cat: sum(1 for e in all_entries if e.category == cat) for cat in CATEGORIES
    }
    lines = [
        "# Ascend ModelZoo built-in sample",
        "",
        f"Parsed {sum(counts.values())} visible model directories from `ACL_PyTorch/built-in`.",
        "",
        "## Category counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    lines += [f"| {cat} | {counts[cat]} |" for cat in CATEGORIES]
    lines += [
        "",
        "## Selected recent projects",
        "",
        "| Category | Project | Last update | PR | Link | Commit/message signal |",
        "|---|---|---:|---:|---|---|",
    ]
    for e in entries:
        commit = e.commit[:120].replace("|", "\\|")
        pr = e.pr if e.pr is not None else ""
        lines.append(
            f"| {e.category} | {e.name} | {e.time_text} | {pr} | {e.url} | {commit} |"
        )
    return "\n".join(lines) + "\n"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def clone_sparse(
    entries: list[Entry],
    dest: Path,
    git_url: str = DEFAULT_GIT_URL,
    history_depth: int = 200,
) -> None:
    # Never recursively delete a caller-provided directory.  An existing empty
    # directory is safe because `git clone <url> <dir>` accepts it; a non-empty
    # directory must be handled explicitly by the caller.
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Refusing to replace non-empty clone directory: {dest}")
    if history_depth < 1:
        raise ValueError("history_depth must be >= 1")
    run(
        [
            "git",
            "clone",
            f"--depth={history_depth}",
            "--filter=blob:none",
            "--sparse",
            git_url,
            str(dest),
        ]
    )
    paths = ["ACL_PyTorch/built-in/README.md", "ACL_PyTorch/built-in/README.en.md"] + [
        e.sparse_path for e in entries
    ]
    run(["git", "sparse-checkout", "set", "--skip-checks", *paths], cwd=dest)


def append_local_summary(md_path: Path, clone_dir: Path, entries: list[Entry]) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clone_dir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    with md_path.open("a", encoding="utf-8") as f:
        f.write("\n## Local file/headings summary\n\n")
        f.write(f"Sparse clone: `{clone_dir}`\n\n")
        f.write(f"Sampled repository HEAD: `{head}`\n\n")
        for e in entries:
            project = clone_dir / e.sparse_path
            readme = project / "README.md"
            last_change = subprocess.run(
                [
                    "git",
                    "log",
                    "-1",
                    "--date=iso-strict",
                    "--format=%H %ad %s",
                    "--",
                    e.sparse_path,
                ],
                cwd=clone_dir,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            if not last_change:
                last_change = (
                    "UNAVAILABLE IN SHALLOW HISTORY; DEEPEN BEFORE USING AS EVIDENCE"
                )
            files = (
                sorted(p.name for p in project.iterdir()) if project.exists() else []
            )
            headings: list[str] = []
            if readme.exists():
                for line in readme.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    if line.startswith("#"):
                        headings.append(re.sub(r"^#+\s*", "", line).strip())
                    if len(headings) >= 8:
                        break
            f.write(f"### {e.category}/{e.name}\n\n")
            f.write(f"Last path change: `{last_change}`\n\n")
            f.write("Files: " + ", ".join(files[:30]) + "\n\n")
            f.write("Headings: " + " | ".join(headings) + "\n\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of projects to select after category coverage",
    )
    ap.add_argument(
        "--per-category-min",
        type=int,
        default=1,
        help="Minimum selected projects per category when available",
    )
    ap.add_argument(
        "--category",
        action="append",
        choices=CATEGORIES,
        help="Restrict selection to one or more categories; may be repeated",
    )
    ap.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Restrict project/name commit signals by case-insensitive keyword; may be repeated",
    )
    ap.add_argument("--out", type=Path, default=None, help="Markdown output path")
    ap.add_argument(
        "--clone",
        action="store_true",
        help="Sparse-clone selected project directories for inspection",
    )
    ap.add_argument(
        "--clone-dir",
        type=Path,
        default=None,
        help="Clone destination; default is a temp directory",
    )
    ap.add_argument(
        "--repo-page", default=DEFAULT_REPO_PAGE, help="GitCode repository page root"
    )
    ap.add_argument("--git-url", default=DEFAULT_GIT_URL, help="Git clone URL")
    ap.add_argument(
        "--history-depth",
        type=int,
        default=200,
        help="Shallow history depth used to resolve per-directory last-change commit/date",
    )
    ap.add_argument(
        "--timeout", type=int, default=45, help="HTTP timeout seconds per request"
    )
    ap.add_argument(
        "--retries", type=int, default=3, help="HTTP retries per category page"
    )
    args = ap.parse_args()

    all_entries: list[Entry] = []
    for cat in CATEGORIES:
        try:
            all_entries.extend(
                parse_category(
                    cat, args.repo_page, timeout=args.timeout, retries=args.retries
                )
            )
        except Exception as exc:
            print(f"warning: failed to parse category {cat}: {exc}", file=sys.stderr)
    if not all_entries:
        print(
            "error: no ModelZoo entries parsed; use the bundled fallback reference",
            file=sys.stderr,
        )
        return 2
    candidates = filter_entries(all_entries, set(args.category or []), args.keyword)
    if not candidates:
        print("error: filters matched no ModelZoo entries", file=sys.stderr)
        return 2
    selected = select_entries(candidates, args.count, args.per_category_min)
    content = markdown(selected, all_entries)

    out = args.out
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
    else:
        print(content)

    if args.clone:
        clone_dir = args.clone_dir or Path(
            tempfile.mkdtemp(prefix="modelzoo-built-in-")
        )
        clone_sparse(selected, clone_dir, args.git_url, args.history_depth)
        print(f"Sparse clone: {clone_dir}", file=sys.stderr)
        if out:
            append_local_summary(out, clone_dir, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
