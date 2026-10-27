"""Prepare deterministic ETTh1 windows for TTM-R2."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

KIND = "TTM-R2"
SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def prepare(path, output, split, stride, task, horizon, expected_sha):
    import numpy as np
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError("dataset SHA256 mismatch")
    rows = []
    previous = None
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["date", *CHANNELS]:
            raise ValueError("invalid ETTh1 columns")
        for row in reader:
            timestamp = datetime.fromisoformat(row["date"])
            if previous is not None and timestamp - previous != timedelta(hours=1):
                raise ValueError("non-hourly timestamps")
            previous = timestamp
            values = [float(row[key]) for key in CHANNELS]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("non-finite ETTh1 value")
            rows.append(values)
    values = np.asarray(rows, dtype=np.float64)
    if len(values) < 14400 or stride <= 0:
        raise ValueError("ETTh1 needs 14400 rows and positive stride")
    if KIND == "TTM-R2" and horizon != 96:
        raise ValueError("this TTM checkpoint requires horizon 96")
    if KIND == "MOMENT" and task == "reconstruction" and horizon != 512:
        raise ValueError("reconstruction target is the 512 input points")
    if KIND != "MOMENT" and task != "forecast":
        raise ValueError("this model supports forecasting only")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    scaling = "none"
    center = np.zeros(7)
    scale = np.ones(7)
    if KIND in ("TTM-R2", "MOMENT"):
        center = values[:8640].mean(axis=0)
        scale = values[:8640].std(axis=0)
        scale[scale == 0] = 1
        values = (values - center) / scale
        scaling = "etth1_train_standard"
    start, end = (512, 8640) if split == "train" else (11520, 14400)
    end_horizon = 0 if task == "reconstruction" else horizon
    indexes = range(start, end - end_horizon + 1, stride)
    if not indexes:
        raise ValueError("split is too short")
    inputs = np.stack([values[i - 512:i] for i in indexes]).astype(np.float32)
    if task == "reconstruction":
        targets = inputs.copy()
    else:
        targets = np.stack([values[i:i + horizon] for i in indexes]).astype(np.float32)
    if KIND != "TTM-R2":
        inputs = inputs.transpose(0, 2, 1)
        targets = targets.transpose(0, 2, 1)
    info = {"schema": "etth1-windows-v1", "model": KIND, "dataset_sha256": actual,
            "split": split, "split_start": start, "split_end": end, "context": 512,
            "horizon": horizon, "stride": stride, "task": task, "scaling": scaling,
            "scaler_fit_rows": [0, 8640] if scaling != "none" else None,
            "scaler_mean": center.tolist(), "scaler_scale": scale.tolist(),
            "window_count": len(indexes), "channels": list(CHANNELS)}
    if Path(output).suffix != ".npz":
        raise ValueError("output must end in .npz")
    np.savez_compressed(output, inputs=inputs, targets=targets, preparation=json.dumps(info, sort_keys=True))
    print(json.dumps(info))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--task", choices=("forecast", "reconstruction"), default="forecast")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--expected-sha256", default=SHA256)
    args = p.parse_args()
    prepare(args.dataset, args.output, args.split, args.stride, args.task, args.horizon, args.expected_sha256)


if __name__ == "__main__":
    main()
