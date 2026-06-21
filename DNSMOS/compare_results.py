#!/usr/bin/env python3
"""Compare DNSMOS CPU/CUDA and NPU CSV outputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "SIG_raw",
    "BAK_raw",
    "OVRL_raw",
    "SIG",
    "BAK",
    "OVRL",
    "P808_MOS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two DNSMOS result CSV files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_abs_error", type=float, default=1e-4)
    parser.add_argument("--mean_abs_error", type=float, default=1e-5)
    parser.add_argument("--min_spearman", type=float, default=0.9999)
    return parser.parse_args()


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    result = {}
    for row in rows:
        key = row["filename"]
        if key in result:
            raise ValueError(f"Duplicate filename in {path}: {key}")
        result[key] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 1.0 if np.array_equal(left, right) else float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = read_csv(baseline_path)
    candidate = read_csv(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate filename sets differ.")

    report = {
        "baseline": str(baseline_path),
        "baseline_sha256": sha256(baseline_path),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "sample_count": len(baseline),
        "fields": {},
    }
    failures = []
    keys = sorted(baseline)
    for field in FIELDS:
        left = np.array([float(baseline[key][field]) for key in keys])
        right = np.array([float(candidate[key][field]) for key in keys])
        errors = np.abs(left - right)
        metrics = {
            "max_abs_error": float(errors.max()),
            "mean_abs_error": float(errors.mean()),
            "spearman": spearman(left, right),
        }
        report["fields"][field] = metrics
        if metrics["max_abs_error"] > args.max_abs_error:
            failures.append(f"{field} max_abs_error")
        if metrics["mean_abs_error"] > args.mean_abs_error:
            failures.append(f"{field} mean_abs_error")
        if not np.isnan(metrics["spearman"]) and metrics["spearman"] < args.min_spearman:
            failures.append(f"{field} spearman")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Comparison failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
