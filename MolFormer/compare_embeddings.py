#!/usr/bin/env python3
"""Compare MoLFormer embedding JSONL outputs by record id."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MoLFormer embeddings.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_cosine", type=float, default=0.99999)
    parser.add_argument("--max_abs_error", type=float, default=1e-4)
    parser.add_argument("--mean_abs_error", type=float, default=1e-5)
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[str, dict]:
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
        raise ValueError(f"Empty embedding file: {path}")
    return rows


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
    baseline = read_jsonl(baseline_path)
    candidate = read_jsonl(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate ids differ.")

    per_sample = []
    failures = []
    all_errors = []
    for row_id in sorted(baseline):
        left = np.asarray(baseline[row_id]["embedding"], dtype=np.float64)
        right = np.asarray(candidate[row_id]["embedding"], dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError(f"{row_id}: shape mismatch {left.shape} != {right.shape}")
        errors = np.abs(left - right)
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        cosine = float(np.dot(left, right) / denominator)
        metrics = {
            "id": row_id,
            "shape": list(left.shape),
            "cosine": cosine,
            "max_abs_error": float(errors.max()),
            "mean_abs_error": float(errors.mean()),
        }
        per_sample.append(metrics)
        all_errors.append(errors)
        if cosine < args.min_cosine:
            failures.append(f"{row_id} cosine")
        if metrics["max_abs_error"] > args.max_abs_error:
            failures.append(f"{row_id} max_abs_error")
        if metrics["mean_abs_error"] > args.mean_abs_error:
            failures.append(f"{row_id} mean_abs_error")

    concatenated = np.concatenate(all_errors)
    report = {
        "baseline": str(baseline_path),
        "baseline_sha256": sha256(baseline_path),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "sample_count": len(per_sample),
        "global_max_abs_error": float(concatenated.max()),
        "global_mean_abs_error": float(concatenated.mean()),
        "minimum_cosine": min(row["cosine"] for row in per_sample),
        "per_sample": per_sample,
    }
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
