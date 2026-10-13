# Licensed under the Apache License, Version 2.0.
"""Compare matching real CPU/NPU outputs; fixtures cannot produce an S3 report."""

import argparse
import json
import statistics
from pathlib import Path

from adapter_utils import (
    RESULT_COLUMNS,
    finite_float,
    read_csv_rows,
    sha256,
    validate_predictions,
    write_report,
)

PROVENANCE_KEYS = (
    "manifest_sha256",
    "weights_sha256",
    "source_revision",
    "mhcflurry",
    "torch",
    "task",
    "batch_size",
    "optimization_level",
    "compile",
)


def validate_placement(meta):
    device = meta["device"]
    if (
        device not in ("cpu", "npu", "cuda")
        or not meta["network_devices"]
        or any(devices != [device] for devices in meta["network_devices"])
    ):
        raise ValueError("Missing actual network placement for " + str(device))


def compare_provenance(baseline, candidate):
    for key in PROVENANCE_KEYS:
        if baseline[key] != candidate[key]:
            raise ValueError("Incomparable provenance: " + key)


def read_result(directory):
    directory = Path(directory)
    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if meta["simulation"]:
        raise ValueError("Simulation output cannot be accepted as model evidence")
    validate_placement(meta)
    if sha256(directory / "predictions.csv") != meta["predictions_sha256"]:
        raise ValueError("Prediction file hash mismatch")
    _, rows = read_csv_rows(directory / "predictions.csv")
    validate_predictions(rows, meta["task"])
    by_id = {row["id"]: row for row in rows}
    if not rows or len(by_id) != len(rows) or len(rows) != meta["rows"]:
        raise ValueError("Empty, duplicate or inconsistent result IDs")
    return meta, by_id


def compare(reference, candidate, atol, rtol):
    atol, rtol = finite_float(atol), finite_float(rtol)
    if atol < 0 or rtol < 0:
        raise ValueError("Negative tolerance")
    left_meta, left = read_result(reference)
    right_meta, right = read_result(candidate)
    if left_meta["device"] != "cpu" or right_meta["device"] != "npu":
        raise ValueError("Requires explicit CPU baseline and real NPU candidate")
    compare_provenance(left_meta, right_meta)
    if set(left) != set(right):
        raise ValueError("Result ID mismatch")
    times = [finite_float(value) for value in right_meta["run_seconds"]]
    if len(times) < 3 or any(value <= 0 for value in times):
        raise ValueError("Requires >=3 valid NPU timing runs")
    columns = RESULT_COLUMNS[right_meta["task"]]
    errors = {}
    passed = True
    allele_failures = 0
    if right_meta["task"] == "presentation":
        for identity in sorted(left):
            a, b = left[identity]["best_allele"], right[identity]["best_allele"]
            if not a.strip() or not b.strip():
                raise ValueError("Missing best_allele")
            allele_failures += int(a != b)
        passed = allele_failures == 0
    for column in columns:
        absolute = []
        failures = 0
        for identity in sorted(left):
            a, b = (
                finite_float(left[identity][column]),
                finite_float(right[identity][column]),
            )
            error = abs(a - b)
            absolute.append(error)
            failures += int(error > atol + rtol * abs(a))
        errors[column] = {
            "max_abs": max(absolute),
            "mean_abs": statistics.mean(absolute),
            "failed_rows": failures,
        }
        passed = passed and failures == 0
    return {
        "numeric_pass": passed,
        "status": "S3 pending L2 cohort/metric review",
        "rows": len(left),
        "atol": atol,
        "rtol": rtol,
        "threshold_source": "provisional engineering tolerance; baseline calibration required",
        "errors": errors,
        "best_allele_failed_rows": allele_failures,
        "npu_median_seconds": statistics.median(times),
        "npu_peptides_per_second": len(right) / statistics.median(times),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--atol", type=finite_float, default=0.001)
    parser.add_argument("--rtol", type=finite_float, default=0.001)
    args = parser.parse_args()
    if Path(args.report).exists():
        raise FileExistsError(args.report)
    report = compare(args.reference, args.candidate, args.atol, args.rtol)
    write_report(args.report, report)
    print(report)
    if not report["numeric_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
