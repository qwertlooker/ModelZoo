"""Persist synchronized timings and summarize at least three independent NPU runs."""
import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import uuid
from importlib.metadata import version
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_report(output, metadata, durations_ms, samples):
    if not durations_ms or samples < 1 or any(not math.isfinite(t) or t <= 0 for t in durations_ms):
        raise ValueError("performance needs positive finite timings and samples")
    identity = dict(metadata)
    identity.update(python_version=platform.python_version(),
                    visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    if metadata["device"] == "npu":
        import torch
        identity.update(torch_npu_version=version("torch-npu"),
                        device_name=torch.npu.get_device_name(),
                        logical_device=torch.npu.current_device())
    elapsed = sum(durations_ms) / 1000
    report = {"schema": "synchronized-model-timing-v1", "run_id": str(uuid.uuid4()),
              "identity": identity, "samples": samples, "batch_durations_ms": durations_ms,
              "model_seconds": elapsed, "samples_per_second": samples / elapsed,
              "scope": "synchronized model calls; excludes loading, warmup, input/output and file IO",
              "result_sha256": file_sha256(output), "command": sys.argv}
    Path(str(output) + ".performance.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def summarize(paths):
    if len(paths) < 3:
        raise ValueError("at least three independent NPU runs required")
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    seen = set()
    for report in reports:
        if report["schema"] != "synchronized-model-timing-v1" or report["identity"]["device"] != "npu":
            raise ValueError("only NPU synchronized timing reports are accepted")
        if report["run_id"] in seen:
            raise ValueError("duplicate run ID; do not reuse one run")
        seen.add(report["run_id"])
        if report["identity"] != reports[0]["identity"] or report["samples"] != reports[0]["samples"]:
            raise ValueError("run model, data, device or benchmark settings differ")
        timings = report["batch_durations_ms"]
        if (not timings or report["samples"] < 1
                or any(not math.isfinite(t) or t <= 0 for t in timings)):
            raise ValueError("invalid timings or sample count")
        elapsed = sum(timings) / 1000
        if (not math.isclose(report["model_seconds"], elapsed)
                or not math.isclose(report["samples_per_second"], report["samples"] / elapsed)):
            raise ValueError("inconsistent performance statistics")
    return {"identity": reports[0]["identity"], "runs": len(reports),
            "scope": reports[0]["scope"], "samples_per_run": reports[0]["samples"],
            "median_model_seconds": statistics.median(r["model_seconds"] for r in reports),
            "median_samples_per_second": statistics.median(r["samples_per_second"] for r in reports),
            "reports": [{"path": str(p), "sha256": file_sha256(p)} for p in paths]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="infer output .performance.json sidecars")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(args.runs)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
