#!/usr/bin/env python3
"""Compare deterministic OpenAI-compatible service evaluation outputs."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two service evaluation JSONL files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_token_agreement", type=float, default=0.995)
    parser.add_argument(
        "--require_logprobs",
        action="store_true",
        help="Fail when no aligned token-logprob records are available.",
    )
    parser.add_argument(
        "--require_exact_tool_calls",
        action="store_true",
        help="Fail unless all tool-call payloads are byte-equivalent after JSON normalization.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate id {row_id}")
            rows[row_id] = row
    if not rows:
        raise ValueError(f"Empty result file: {path}")
    return rows


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = read_rows(baseline_path)
    candidate = read_rows(candidate_path)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"Result id mismatch: missing={missing}, extra={extra}")

    content_exact = 0
    finish_reason_exact = 0
    tool_calls_exact = 0
    token_equal = 0
    token_total = 0
    token_record_count = 0
    per_prompt = []

    for row_id in sorted(baseline):
        left = baseline[row_id]
        right = candidate[row_id]
        same_content = left.get("content") == right.get("content")
        same_finish = left.get("finish_reason") == right.get("finish_reason")
        same_tools = canonical(left.get("tool_calls") or []) == canonical(
            right.get("tool_calls") or []
        )
        content_exact += int(same_content)
        finish_reason_exact += int(same_finish)
        tool_calls_exact += int(same_tools)

        left_tokens = left.get("tokens") or []
        right_tokens = right.get("tokens") or []
        aligned = min(len(left_tokens), len(right_tokens))
        equal = sum(
            left_tokens[index] == right_tokens[index] for index in range(aligned)
        )
        if aligned:
            token_record_count += 1
            token_equal += equal
            token_total += max(len(left_tokens), len(right_tokens))
        per_prompt.append(
            {
                "id": row_id,
                "content_exact": same_content,
                "finish_reason_exact": same_finish,
                "tool_calls_exact": same_tools,
                "baseline_tokens": len(left_tokens),
                "candidate_tokens": len(right_tokens),
                "aligned_equal_tokens": equal,
            }
        )

    count = len(baseline)
    token_agreement = token_equal / token_total if token_total else None
    summary = {
        "baseline": str(baseline_path),
        "baseline_sha256": sha256(baseline_path),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "sample_count": count,
        "content_exact_rate": content_exact / count,
        "finish_reason_exact_rate": finish_reason_exact / count,
        "tool_calls_exact_rate": tool_calls_exact / count,
        "token_record_count": token_record_count,
        "token_agreement": token_agreement,
        "thresholds": {
            "min_token_agreement": args.min_token_agreement,
            "require_logprobs": args.require_logprobs,
            "require_exact_tool_calls": args.require_exact_tool_calls,
        },
        "per_prompt": per_prompt,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failures = []
    if args.require_logprobs and token_record_count != count:
        failures.append(
            f"token logprobs available for {token_record_count}/{count} records"
        )
    if token_agreement is not None and token_agreement < args.min_token_agreement:
        failures.append(
            f"token agreement {token_agreement:.6f} < {args.min_token_agreement:.6f}"
        )
    if args.require_exact_tool_calls and tool_calls_exact != count:
        failures.append(f"exact tool calls {tool_calls_exact}/{count}")
    if failures:
        raise SystemExit("Comparison failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
