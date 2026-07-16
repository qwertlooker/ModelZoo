#!/usr/bin/env python3
"""Fail closed when a MechVQA NPU result misses the declared threshold."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_JUDGES = {"GPT-OSS-120B", "DeepSeek-V3.2", "Kimi-k2"}


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MechVQA score with the public baseline."
    )
    parser.add_argument("--stats", required=True, help="Official evaluator stats JSON.")
    parser.add_argument(
        "--evaluated",
        required=True,
        help="Official evaluated JSONL; used to fail closed on judge errors.",
    )
    parser.add_argument("--output", required=True, help="Comparison report JSON.")
    parser.add_argument("--model", default="MechVL-4B-RL")
    parser.add_argument("--baseline", type=probability, default=0.8485)
    parser.add_argument(
        "--max-absolute-drop",
        type=probability,
        default=0.01,
        help="Provisional engineering tolerance; calibrate before formal acceptance.",
    )
    parser.add_argument("--expected-count", type=positive_int, default=1185)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_evaluated(
    path: Path, model: str, expected_count: int
) -> tuple[int, int, list[str], float]:
    record_indices: set[int] = set()
    record_count = 0
    protocol_error_count = 0
    examples: list[str] = []
    voted_score_sum = 0.0

    def reject(message: str) -> None:
        nonlocal protocol_error_count
        protocol_error_count += 1
        if len(examples) < 10:
            examples.append(message)

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                reject(f"line {line_number}: invalid JSON")
                continue
            if not isinstance(record, dict):
                reject(f"line {line_number}: record must be a JSON object")
                continue

            index = record.get("record_idx")
            if not isinstance(index, int) or index in record_indices:
                reject(f"line {line_number}: invalid or duplicate record_idx={index!r}")
            else:
                record_indices.add(index)

            result = record.get("evaluation_result", {}).get(model)
            if not isinstance(result, dict):
                reject(f"record {index}: missing evaluation_result for {model}")
                continue
            if result.get("error") or not str(result.get("model_answer", "")).strip():
                reject(f"record {index}: target response error or empty model_answer")

            voted_score = result.get("voted_score")
            if voted_score not in (0, 0.0, 1, 1.0):
                reject(f"record {index}: invalid voted_score")
            else:
                voted_score_sum += float(voted_score)

            judge_results = result.get("judge_results")
            if (
                not isinstance(judge_results, dict)
                or set(judge_results) != EXPECTED_JUDGES
            ):
                actual = (
                    sorted(judge_results) if isinstance(judge_results, dict) else []
                )
                reject(f"record {index}: judge set mismatch: {actual}")
                continue
            judge_scores = []
            for judge_name, judge_result in judge_results.items():
                if not isinstance(judge_result, dict) or judge_result.get("error"):
                    reject(f"record {index}: {judge_name} returned an error")
                    continue
                if judge_result.get("score") not in (0, 0.0, 1, 1.0):
                    reject(f"record {index}: {judge_name} returned an invalid score")
                    continue
                judge_scores.append(float(judge_result["score"]))
            if len(judge_scores) == len(EXPECTED_JUDGES):
                majority = Counter(judge_scores).most_common(1)[0][0]
                if (
                    voted_score not in (0, 0.0, 1, 1.0)
                    or float(voted_score) != majority
                ):
                    reject(f"record {index}: voted_score does not match judge majority")

    if record_count != expected_count:
        reject(f"evaluated record count is {record_count}, expected {expected_count}")
    if record_indices != set(range(expected_count)):
        reject("record_idx coverage is not exactly 0..expected_count-1")
    recomputed_score = voted_score_sum / record_count if record_count else 0.0
    return record_count, protocol_error_count, examples, recomputed_score


def main() -> None:
    args = parse_args()
    stats_path = Path(args.stats)
    evaluated_path = Path(args.evaluated)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    model_stats = stats.get("model_stats", {}).get(args.model)
    if not isinstance(model_stats, dict):
        raise KeyError(f"model '{args.model}' is missing from {stats_path}")

    score = float(model_stats["overall_voted_score"])
    total = int(model_stats["total"])
    errors = int(model_stats.get("error_count", 0))
    (
        evaluated_count,
        protocol_errors,
        protocol_error_examples,
        recomputed_score,
    ) = validate_evaluated(evaluated_path, args.model, args.expected_count)
    minimum = args.baseline - args.max_absolute_drop
    stats_consistent = abs(score - recomputed_score) <= 1e-12
    passed = (
        total == args.expected_count
        and evaluated_count == args.expected_count
        and errors == 0
        and protocol_errors == 0
        and stats_consistent
        and score >= minimum
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stats": str(stats_path.resolve()),
        "evaluated": str(evaluated_path.resolve()),
        "model": args.model,
        "metric": "MechVQA three-judge majority-voted accuracy",
        "public_baseline": args.baseline,
        "max_absolute_drop": args.max_absolute_drop,
        "threshold_source": "provisional engineering tolerance; maintainer calibration required",
        "minimum_score": minimum,
        "candidate_score": score,
        "recomputed_score": recomputed_score,
        "stats_consistent": stats_consistent,
        "difference": score - args.baseline,
        "expected_count": args.expected_count,
        "actual_count": total,
        "error_count": errors,
        "evaluated_count": evaluated_count,
        "required_judges": sorted(EXPECTED_JUDGES),
        "protocol_error_count": protocol_errors,
        "protocol_error_examples": protocol_error_examples,
        "passed": passed,
    }
    write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("MechVQA comparison failed")


if __name__ == "__main__":
    main()
