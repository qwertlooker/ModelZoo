#!/usr/bin/env python3
"""Fetch public GitCode PR discussions and summarize ModelZoo review signals.

This is a best-effort helper. GitCode APIs may require browser-like cookies; the
script first opens the PR page to initialize a session, then calls the public
issuepr discussion endpoints used by the web UI.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REPO = "Ascend/ModelZoo-PyTorch"
DEFAULT_BASE = "https://gitcode.com"
DISCUSSION_PAGE_SIZE = 100

KEYWORDS = [
    "CodeCheck",
    "运行失败",
    "代码风格",
    "开源片段",
    "SCA",
    "Antipoison",
    "flake8",
    "抑制注释",
    "精度",
    "性能",
    "README",
    "readme",
    "芯片",
    "机器型号",
    "commit_id",
    "commit",
    "patch",
    "链接",
    "失效",
    "下载",
    "数据",
    "WER",
    "RTF",
    "debug",
    "注释",
    "格式",
    "变量命名",
    "配套信息",
    "芯片型号",
    "退出码",
    "returncode",
    "synchronize",
    "拼写",
    "未定义",
    "语法",
    "blob",
    "resolve",
    "配置文件",
]

NOISE_COMMANDS = {
    "compile",
    "/compile",
    "lgtm",
    "/lgtm",
    "approve",
    "/approve",
    "/check-cla",
}


@dataclass
class Note:
    pr: int
    title: str
    author: str
    body: str
    created_at: str = ""


@dataclass
class PrRecord:
    detail: dict
    expected_paths: list[str] = field(default_factory=list)
    touched_paths: list[str] = field(default_factory=list)
    matched_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class HttpResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text[:200]}")


class HttpSession:
    def __init__(self):
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar())
        )

    def get(self, url: str, headers: dict, timeout: int) -> HttpResponse:
        request = urllib.request.Request(url, headers=headers)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return HttpResponse(
                    response.status, response.read().decode("utf-8", errors="replace")
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(exc.code, exc.read().decode("utf-8", errors="replace"))


def clean_text(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_noise(body: str) -> bool:
    b = clean_text(body).strip().lower()
    if not b or b in NOISE_COMMANDS:
        return True
    if (
        b.startswith("thanks for your pull-request")
        or "pr approval progress" in b
        or "cla signature pass" in b
    ):
        return True
    if (
        b.startswith("create merge request")
        or b.startswith("add label")
        or b.startswith("delete label")
    ):
        return True
    if b.startswith("merged from codehub"):
        return True
    if b.startswith("update merge request") or re.match(r"added \d+ commits?", b):
        return True
    if (
        b.startswith("changed the description")
        or b.startswith("changed this line")
        or b.startswith("changed title")
    ):
        return True
    if b.startswith("#### 变更摘要"):
        return True
    if "代码审查执行失败" in b or "审查未能完成" in b:
        return True
    return False


def note_iter(item: dict):
    for note in item.get("notes") or []:
        yield note
    if item.get("body"):
        yield item


def get_json(session: HttpSession, url: str, headers: dict) -> dict:
    try:
        resp = session.get(url, headers=headers, timeout=30)
    except Exception as exc:
        return {"_error": repr(exc)}
    if resp.status_code != 200:
        return {"_error": f"{resp.status_code}: {resp.text[:200]}"}
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - diagnostic helper
        return {"_error": repr(exc), "_text": resp.text[:500]}


def fetch_pr(
    session: HttpSession, base: str, repo: str, pr: int, headers: dict
) -> tuple[dict, list[Note]]:
    encoded_repo = urllib.parse.quote(repo, safe="")
    try:
        session.get(f"{base}/{repo}/pull/{pr}", headers=headers, timeout=30)
    except Exception:
        pass
    detail = get_json(
        session,
        f"{base}/issuepr/api/v1/projects/{encoded_repo}/isource/merge_requests/{pr}",
        headers,
    )
    discussion_query = urllib.parse.urlencode(
        {"page": 1, "per_page": DISCUSSION_PAGE_SIZE}
    )
    discussions = get_json(
        session,
        f"{base}/issuepr/api/v1/projects/{encoded_repo}/merge_requests/"
        f"{pr}/discussions?{discussion_query}",
        headers,
    )
    title = detail.get("title", "") if isinstance(detail, dict) else ""
    notes: list[Note] = []
    data = []
    if isinstance(discussions, dict) and isinstance(discussions.get("content"), dict):
        data = discussions["content"].get("data") or []
    for item in data:
        for note in note_iter(item):
            body = clean_text(note.get("body", ""))
            author = (note.get("author") or item.get("author") or {}).get(
                "username", ""
            )
            if not is_noise(body):
                notes.append(
                    Note(
                        pr=pr,
                        title=title,
                        author=author,
                        body=body,
                        created_at=str(
                            note.get("created_at") or item.get("created_at") or ""
                        ),
                    )
                )
    total = discussions.get("total") if isinstance(discussions, dict) else None
    detail["_discussion_total"] = total
    detail["_discussion_fetched"] = len(data)
    if isinstance(total, int) and total > len(data):
        discussions["_error"] = (
            f"discussion response truncated: fetched {len(data)} of {total}; "
            f"increase DISCUSSION_PAGE_SIZE or add cursor pagination"
        )
    if isinstance(discussions, dict) and discussions.get("_error"):
        detail["_discussion_error"] = discussions["_error"]
    return detail, notes


def fetch_diff_paths(
    session: HttpSession,
    base: str,
    repo: str,
    pr: int,
    headers: dict,
    timeout: int = 30,
) -> tuple[list[str], str | None]:
    """Return changed file paths from a public .diff URL, best effort."""
    url = f"{base}/{repo}/pull/{pr}.diff"
    try:
        response = session.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        text = response.text
    except Exception as exc:
        return [], f"failed to fetch PR diff: {exc}"
    paths: list[str] = []
    for m in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", text, flags=re.M):
        path = m.group(2)
        if path not in paths:
            paths.append(path)
    if not paths:
        return [], "PR diff contained no changed-file paths"
    return paths, None


def match_paths(touched: list[str], required: list[str]) -> list[str]:
    if not required:
        return []
    out: list[str] = []
    normalized = [p.rstrip("/") for p in required]
    for path in touched:
        for prefix in normalized:
            if path == prefix or path.startswith(prefix + "/"):
                if prefix not in out:
                    out.append(prefix)
    return out


def parse_pr_path(values: list[str]) -> dict[int, list[str]]:
    """Parse repeatable PR=PATH arguments into an exact evidence mapping."""
    mapping: dict[int, list[str]] = collections.defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --pr-path {value!r}; expected PR=PATH")
        pr_text, path = value.split("=", 1)
        try:
            pr = int(pr_text)
        except ValueError as exc:
            raise ValueError(f"invalid PR number in --pr-path {value!r}") from exc
        path = path.strip().strip("/")
        if not path:
            raise ValueError(f"empty path in --pr-path {value!r}")
        if path not in mapping[pr]:
            mapping[pr].append(path)
    return dict(mapping)


def newest_notes_first(notes: list[Note]) -> list[Note]:
    return sorted(notes, key=lambda note: note.created_at, reverse=True)


def dedupe_notes(notes: list[Note]) -> list[Note]:
    unique: list[Note] = []
    seen: set[tuple[int, str, str]] = set()
    for note in newest_notes_first(notes):
        key = (note.pr, note.author, note.body)
        if key not in seen:
            seen.add(key)
            unique.append(note)
    return unique


def remove_ci_history(notes: list[Note]) -> list[Note]:
    """Drop historical pipeline events; current head status comes from PR metadata."""
    return [
        note
        for note in notes
        if not (note.author == "ascend-robot" and "流水线" in note.body)
    ]


def classify(notes: list[Note]) -> dict[str, list[Note]]:
    groups: dict[str, list[Note]] = collections.defaultdict(list)
    for note in notes:
        b = note.body.lower()
        if any(
            k.lower() in b
            for k in [
                "codecheck",
                "flake8",
                "抑制注释",
                "代码风格",
                "格式",
                "变量命名",
                "debug",
                "注释",
            ]
        ):
            groups["代码规范/CI"].append(note)
        if any(
            k.lower() in b
            for k in [
                "退出码",
                "returncode",
                "check=true",
                "未定义",
                "语法",
                "拼写",
                "参数解析",
                "subprocess.run",
            ]
        ):
            groups["执行正确性/错误传播"].append(note)
        if any(
            k.lower() in b
            for k in ["精度", "wer", "cer", "公开数据集", "论文", "评测脚本"]
        ):
            groups["精度口径"].append(note)
        if any(k.lower() in b for k in ["性能", "rtf", "fps", "单位", "耗时"]):
            groups["性能口径"].append(note)
        if any(k.lower() in b for k in ["synchronize", "异步", "record=0", "除零"]):
            groups["NPU计时/边界"].append(note)
        if any(
            k.lower() in b
            for k in ["commit", "patch", "版本", "配套", "sam2.1", "权重", "配置"]
        ):
            groups["版本/patch/配套"].append(note)
        if any(
            k.lower() in b
            for k in [
                "readme",
                "文档",
                "芯片",
                "机器型号",
                "获取芯片",
                "源码",
                "下载",
                "数据文件",
            ]
        ):
            groups["README/可复现"].append(note)
        if any(
            k.lower() in b
            for k in ["/blob/", "/resolve/", "配置文件", "数据集下载", "自己准备"]
        ):
            groups["下载/数据/配置来源"].append(note)
        if any(k.lower() in b for k in ["sca", "开源片段", "antipoison", "license"]):
            groups["开源合规/安全"].append(note)
    return groups


def render_markdown(records: list[PrRecord], notes: list[Note]) -> str:
    lines = ["# GitCode PR review sample", ""]
    details = [r.detail for r in records]
    lines.append(
        f"Sampled PRs: {', '.join('#' + str(d.get('iid')) for d in details if d.get('iid'))}"
    )
    lines.append("")
    lines += [
        "## PR details",
        "",
        "| PR | Title | Head SHA | Current pipeline | Discussions | Expected path | Path match | Evidence errors |",
        "|---:|---|---|---|---:|---|---|---|",
    ]
    for r in records:
        d = r.detail
        expected = ", ".join(r.expected_paths) if r.expected_paths else "not supplied"
        match = ", ".join(r.matched_paths) if r.matched_paths else "UNVERIFIED"
        errors = "; ".join(r.errors).replace("|", "\\|")
        title = str(d.get("title", "")).replace("|", "\\|")
        head = str(d.get("sha", ""))[:12]
        pipeline_value = d.get("pipeline_status") or d.get(
            "pipeline_status_with_code_quality"
        )
        pipeline = str(pipeline_value or "unknown")
        if not pipeline_value and d.get("head_pipeline_id"):
            pipeline += f" (id={d['head_pipeline_id']})"
        discussion_total = d.get("_discussion_total", "")
        discussion_fetched = d.get("_discussion_fetched", "")
        discussion_count = f"{discussion_fetched}/{discussion_total}"
        lines.append(
            f"| {d.get('iid', '')} | {title} | {head} | {pipeline} | {discussion_count} | "
            f"{expected} | {match} | {errors} |"
        )
    lines.append("")
    counts = collections.Counter()
    for note in notes:
        for keyword in KEYWORDS:
            if keyword.lower() in note.body.lower():
                counts[keyword] += 1
    lines += ["## Keyword counts", "", "| Keyword | Count |", "|---|---:|"]
    for k, v in counts.most_common():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    summaries = [note for note in notes if note.body.startswith("#### 代码审查")]
    if summaries:
        lines += ["## Review summaries", ""]
        for note in summaries:
            lines.append(f"- PR{note.pr} `{note.author}`: {note.body[:500]}")
        lines.append("")
    lines += ["## Grouped review signals", ""]
    detail_notes = [note for note in notes if not note.body.startswith("#### 代码审查")]
    for group, items in classify(detail_notes).items():
        lines.append(f"### {group}")
        seen = set()
        for note in items[:20]:
            snippet = note.body[:220]
            key = (group, snippet)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- PR{note.pr} `{note.author}`: {snippet}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument(
        "--prs", nargs="*", type=int, default=[], help="PR numbers to sample"
    )
    ap.add_argument(
        "--paths",
        nargs="*",
        default=[],
        help="Sampled ModelZoo paths that each PR should touch, e.g. ACL_PyTorch/built-in/cv/F3Net",
    )
    ap.add_argument(
        "--pr-path",
        action="append",
        default=[],
        metavar="PR=PATH",
        help="Bind a PR to its exact sampled ModelZoo path; repeat for multiple PRs",
    )
    ap.add_argument(
        "--require-path-match",
        action="store_true",
        help="Legacy strict mode for --paths; --pr-path is strict by default",
    )
    ap.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Return success even if PR metadata, discussion, diff, or exact path evidence is unavailable",
    )
    ap.add_argument("--out", type=Path, default=None, help="Markdown output path")
    ap.add_argument(
        "--json-out", type=Path, default=None, help="Raw notes JSON output path"
    )
    args = ap.parse_args()

    try:
        exact_paths = parse_pr_path(args.pr_path)
    except ValueError as exc:
        ap.error(str(exc))
    prs = list(dict.fromkeys([*args.prs, *exact_paths]))
    if not prs:
        ap.error("provide --prs and/or at least one --pr-path PR=PATH")

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": f"{args.base}/{args.repo}/pull/{prs[0]}",
        "X-Requested-With": "XMLHttpRequest",
    }
    session = HttpSession()
    records: list[PrRecord] = []
    notes: list[Note] = []
    bad: list[int] = []
    for pr in prs:
        detail, pr_notes = fetch_pr(session, args.base, args.repo, pr, headers)
        expected = exact_paths.get(pr, args.paths)
        touched: list[str] = []
        errors: list[str] = []
        if detail.get("_error"):
            errors.append(f"PR metadata: {detail['_error']}")
        if detail.get("_discussion_error"):
            errors.append(f"PR discussions: {detail['_discussion_error']}")
        if expected:
            touched, diff_error = fetch_diff_paths(
                session, args.base, args.repo, pr, headers
            )
            if diff_error:
                errors.append(diff_error)
        matched = match_paths(touched, expected)
        if expected and not matched:
            errors.append("changed files do not prove the expected ModelZoo path")
        if errors:
            bad.append(pr)
        records.append(
            PrRecord(
                detail=detail,
                expected_paths=expected,
                touched_paths=touched,
                matched_paths=matched,
                errors=errors,
            )
        )
        notes.extend(pr_notes)
        time.sleep(0.1)
    notes = remove_ci_history(dedupe_notes(notes))
    md = render_markdown(records, notes)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
    else:
        print(md)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "records": [r.__dict__ for r in records],
                    "notes": [note.__dict__ for note in notes],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    strict = bool(exact_paths) or args.require_path_match
    if bad and strict and not args.allow_unverified:
        print(f"error: unverified PR evidence: {bad}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
