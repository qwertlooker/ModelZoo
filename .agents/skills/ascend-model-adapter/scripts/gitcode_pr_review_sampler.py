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
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_REPO = "Ascend/ModelZoo-PyTorch"
DEFAULT_BASE = "https://gitcode.com"

KEYWORDS = [
    "CodeCheck", "运行失败", "代码风格", "开源片段", "SCA", "Antipoison",
    "flake8", "抑制注释", "精度", "性能", "README", "readme", "芯片", "机器型号",
    "commit_id", "commit", "patch", "链接", "失效", "下载", "数据", "WER", "RTF",
    "debug", "注释", "格式", "变量命名", "配套信息", "芯片型号",
]

NOISE_COMMANDS = {"compile", "/compile", "lgtm", "/lgtm", "approve", "/approve", "/check-cla"}


@dataclass
class Note:
    pr: int
    title: str
    author: str
    body: str


def clean_text(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_noise(body: str) -> bool:
    b = clean_text(body).strip().lower()
    if not b or b in NOISE_COMMANDS:
        return True
    if b.startswith("thanks for your pull-request") or "pr approval progress" in b or "cla signature pass" in b:
        return True
    if b.startswith("create merge request") or b.startswith("add label") or b.startswith("delete label"):
        return True
    if b.startswith("merged from codehub"):
        return True
    return False


def note_iter(item: dict):
    for note in item.get("notes") or []:
        yield note
    if item.get("body"):
        yield item


def get_json(session: requests.Session, url: str, headers: dict) -> dict:
    resp = session.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        return {"_error": f"{resp.status_code}: {resp.text[:200]}"}
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - diagnostic helper
        return {"_error": repr(exc), "_text": resp.text[:500]}


def fetch_pr(session: requests.Session, base: str, repo: str, pr: int, headers: dict) -> tuple[dict, list[Note]]:
    encoded_repo = urllib.parse.quote(repo, safe="")
    session.get(f"{base}/{repo}/pull/{pr}", headers=headers, timeout=30)
    detail = get_json(session, f"{base}/issuepr/api/v1/projects/{encoded_repo}/isource/merge_requests/{pr}", headers)
    discussions = get_json(session, f"{base}/issuepr/api/v1/projects/{encoded_repo}/merge_requests/{pr}/discussions", headers)
    title = detail.get("title", "") if isinstance(detail, dict) else ""
    notes: list[Note] = []
    data = []
    if isinstance(discussions, dict) and isinstance(discussions.get("content"), dict):
        data = discussions["content"].get("data") or []
    for item in data:
        for note in note_iter(item):
            body = clean_text(note.get("body", ""))
            author = (note.get("author") or item.get("author") or {}).get("username", "")
            keep_ci = any(k in body for k in ["CodeCheck", "Antipoison", "SCA", "开源片段", "运行失败"])
            if keep_ci or not is_noise(body):
                notes.append(Note(pr=pr, title=title, author=author, body=body))
    detail["_discussion_total"] = discussions.get("total") if isinstance(discussions, dict) else None
    return detail, notes


def classify(notes: list[Note]) -> dict[str, list[Note]]:
    groups: dict[str, list[Note]] = collections.defaultdict(list)
    for note in notes:
        b = note.body.lower()
        if any(k.lower() in b for k in ["codecheck", "flake8", "抑制注释", "代码风格", "格式", "变量命名", "debug", "注释"]):
            groups["代码规范/CI"].append(note)
        if any(k.lower() in b for k in ["精度", "wer", "cer", "公开数据集", "论文", "评测脚本"]):
            groups["精度口径"].append(note)
        if any(k.lower() in b for k in ["性能", "rtf", "fps", "单位", "耗时"]):
            groups["性能口径"].append(note)
        if any(k.lower() in b for k in ["commit", "patch", "版本", "配套", "sam2.1", "权重", "配置"]):
            groups["版本/patch/配套"].append(note)
        if any(k.lower() in b for k in ["readme", "文档", "芯片", "机器型号", "获取芯片", "源码", "下载", "数据文件"]):
            groups["README/可复现"].append(note)
        if any(k.lower() in b for k in ["sca", "开源片段", "antipoison", "license"]):
            groups["开源合规/安全"].append(note)
    return groups


def render_markdown(details: list[dict], notes: list[Note]) -> str:
    lines = ["# GitCode PR review sample", ""]
    lines.append(f"Sampled PRs: {', '.join('#'+str(d.get('iid')) for d in details if d.get('iid'))}")
    lines.append("")
    lines += ["## PR details", "", "| PR | Title | Discussions |", "|---:|---|---:|"]
    for d in details:
        lines.append(f"| {d.get('iid','')} | {str(d.get('title','')).replace('|','\\|')} | {d.get('_discussion_total','')} |")
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
    lines += ["## Grouped review signals", ""]
    for group, items in classify(notes).items():
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
    ap.add_argument("--prs", nargs="+", type=int, required=True, help="PR numbers to sample")
    ap.add_argument("--out", type=Path, default=None, help="Markdown output path")
    ap.add_argument("--json-out", type=Path, default=None, help="Raw notes JSON output path")
    args = ap.parse_args()

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": f"{args.base}/{args.repo}/pull/{args.prs[0]}",
        "X-Requested-With": "XMLHttpRequest",
    }
    session = requests.Session()
    details: list[dict] = []
    notes: list[Note] = []
    for pr in args.prs:
        detail, pr_notes = fetch_pr(session, args.base, args.repo, pr, headers)
        details.append(detail)
        notes.extend(pr_notes)
        time.sleep(0.1)
    md = render_markdown(details, notes)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
    else:
        print(md)
    if args.json_out:
        args.json_out.write_text(json.dumps([note.__dict__ for note in notes], ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
