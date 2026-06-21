#!/usr/bin/env python3
"""Evaluate SpeechOcean762 correlations and optional backend alignment."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate speechscorer CSV output.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_abs_error", type=float, default=1e-4)
    parser.add_argument("--mean_abs_error", type=float, default=1e-5)
    parser.add_argument("--min_spearman", type=float, default=0.9999)
    return parser.parse_args()


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


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(rankdata(left), rankdata(right))


def read_results(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        utterance_id = row["utterance_id"]
        if utterance_id in result:
            raise ValueError(f"Duplicate utterance_id in {path}: {utterance_id}")
        result[utterance_id] = row
    if not result:
        raise ValueError(f"Empty results CSV: {path}")
    return result


def read_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["utterance_id"]] = row
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    manifest_path = Path(args.manifest)
    results = read_results(results_path)
    manifest = read_manifest(manifest_path)
    if set(results) != set(manifest):
        raise ValueError("Result and manifest utterance_id sets differ.")
    ids = sorted(results)
    total = np.array([float(manifest[key]["scores"]["total"]) for key in ids])
    report = {
        "results": str(results_path),
        "results_sha256": sha256(results_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "sample_count": len(ids),
        "correlation_with_human_total": {},
    }
    for field in ("entropy", "perplexity"):
        values = np.array([float(results[key][field]) for key in ids])
        report["correlation_with_human_total"][field] = {
            "pearson": pearson(values, total),
            "spearman": spearman(values, total),
        }
    age_groups = {}
    for utterance_id in ids:
        age = str(manifest[utterance_id]["scores"]["age"])
        age_groups.setdefault(age, []).append(utterance_id)
    report["notebook_grouped_by_age"] = [
        {
            "age": age,
            "sample_count": len(group_ids),
            "total_mean": float(
                np.mean(
                    [
                        float(manifest[key]["scores"]["total"])
                        for key in group_ids
                    ]
                )
            ),
            "entropy_mean": float(
                np.mean([float(results[key]["entropy"]) for key in group_ids])
            ),
            "perplexity_mean": float(
                np.mean([float(results[key]["perplexity"]) for key in group_ids])
            ),
        }
        for age, group_ids in sorted(age_groups.items())
    ]

    failures = []
    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline = read_results(baseline_path)
        if set(baseline) != set(results):
            raise ValueError("Baseline and candidate utterance_id sets differ.")
        alignment = {}
        for field in ("entropy", "perplexity"):
            left = np.array([float(baseline[key][field]) for key in ids])
            right = np.array([float(results[key][field]) for key in ids])
            errors = np.abs(left - right)
            metrics = {
                "max_abs_error": float(errors.max()),
                "mean_abs_error": float(errors.mean()),
                "spearman": spearman(left, right),
            }
            alignment[field] = metrics
            if metrics["max_abs_error"] > args.max_abs_error:
                failures.append(f"{field} max_abs_error")
            if metrics["mean_abs_error"] > args.mean_abs_error:
                failures.append(f"{field} mean_abs_error")
            if metrics["spearman"] < args.min_spearman:
                failures.append(f"{field} spearman")
        report["baseline"] = str(baseline_path)
        report["baseline_sha256"] = sha256(baseline_path)
        report["baseline_alignment"] = alignment

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Evaluation failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
