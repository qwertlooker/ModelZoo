# Licensed under the Apache License, Version 2.0.
"""Equal-patient presentation AP/AUROC/PPV@N, including score-tie policy."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from adapter_utils import (
    finite_float,
    presentation_groups,
    read_inputs,
    sha256,
    write_json,
)
from compare_results import (
    PROVENANCE_KEYS,
    compare_provenance,
    read_result,
    validate_placement,
)


def ppv_at_n(labels, scores):
    positives = sum(labels)
    if positives == 0:
        raise ValueError("PPV@N requires positive examples")
    groups = defaultdict(list)
    for label, score in zip(labels, scores, strict=True):
        groups[score].append(label)
    remaining = positives
    expected_hits = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        take = min(remaining, len(group))
        expected_hits += take * sum(group) / len(group)
        remaining -= take
        if remaining == 0:
            break
    return expected_hits / positives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--reference-metrics")
    parser.add_argument("--max-metric-drop", type=finite_float, default=0.001)
    args = parser.parse_args()
    if args.max_metric_drop < 0:
        raise ValueError("Negative metric tolerance")
    # Required evaluation dependencies fail immediately on the execution path.
    from sklearn.metrics import average_precision_score, roc_auc_score

    meta, predictions = read_result(args.results)
    if (
        meta["manifest_sha256"] != sha256(args.manifest)
        or meta["task"] != "presentation"
    ):
        raise ValueError("Metric evaluation requires matching presentation results")
    rows = read_inputs(args.manifest)
    groups = presentation_groups(rows)
    if {row["id"] for row in rows} != set(predictions):
        raise ValueError("Metric cohort/result mismatch")
    per_patient = {}
    for patient, patient_rows in groups.items():
        labels = [int(row["label"]) for row in patient_rows]
        if set(labels) != {0, 1}:
            raise ValueError("Each patient requires both labels")
        scores = [
            finite_float(predictions[row["id"]]["presentation_score"])
            for row in patient_rows
        ]
        per_patient[patient] = {
            "AP": float(average_precision_score(labels, scores)),
            "AUROC": float(roc_auc_score(labels, scores)),
            "PPV@N": ppv_at_n(labels, scores),
        }
    metrics = {
        key: statistics.mean(value[key] for value in per_patient.values())
        for key in ("AP", "AUROC", "PPV@N")
    }
    report = {
        **{key: meta[key] for key in PROVENANCE_KEYS},
        "simulation": False,
        "network_devices": meta["network_devices"],
        "device": meta["device"],
        "rows": len(rows),
        "patients": len(groups),
        "policy": "equal-patient means; expected precision across ties at N positives",
        "metrics": metrics,
        "per_patient": per_patient,
        "metric_pass": None,
    }
    if args.reference_metrics:
        baseline = json.loads(Path(args.reference_metrics).read_text(encoding="utf-8"))
        if baseline["simulation"]:
            raise ValueError("Simulation metrics cannot be model evidence")
        validate_placement(baseline)
        if baseline["device"] != "cpu" or meta["device"] != "npu":
            raise ValueError("Metric comparison requires CPU -> NPU")
        compare_provenance(baseline, report)
        for key in ("rows", "patients", "policy"):
            if baseline[key] != report[key]:
                raise ValueError("Metric baseline mismatch: " + key)
        report["metric_pass"] = all(
            metrics[key]
            >= finite_float(baseline["metrics"][key]) - args.max_metric_drop
            for key in metrics
        )
        report["max_metric_drop"] = args.max_metric_drop
        report["threshold_source"] = "provisional; baseline calibration required"
    write_json(args.report, report)
    print(report)
    if report["metric_pass"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
