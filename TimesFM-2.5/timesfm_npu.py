#!/usr/bin/env python3
"""TimesFM 2.5 local inference and ETTh1 evaluation; NPU is never implicit fallback."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path

DATA_SHA = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
MODEL_SHA = "b53f6d52114e2ad786890f3c4637ce05f580b7800d6e24401f88b398b76035ef"
CONFIG_SHA = "452ecae918f67b2e7d0f2892ab424d1876939e70d077db3708a4fe8ca03a7de5"
CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_series(path):
    columns = {name: [] for name in CHANNELS}
    previous = None
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["date", *CHANNELS]:
            raise ValueError("ETTh1 CSV columns do not match the official dataset")
        for row in reader:
            timestamp = datetime.fromisoformat(row["date"])
            if previous is not None and timestamp - previous != timedelta(hours=1):
                raise ValueError(f"non-hourly timestamps at {row['date']}")
            previous = timestamp
            for name in CHANNELS:
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite value in {name} at {row['date']}")
                columns[name].append(value)
    return columns


def prepare(args):
    if args.context <= 0 or args.context > min(args.test_start, 16384):
        raise ValueError("context must fit before the test split")
    if not 1 <= args.horizon <= 128 or args.stride <= 0:
        raise ValueError("horizon must be 1..128 and stride must be positive")
    digest = sha256(args.dataset)
    if digest != args.expected_sha256:
        raise ValueError(f"dataset SHA mismatch: {digest}")
    columns = read_series(args.dataset)
    if args.test_end > len(columns[CHANNELS[0]]) or args.test_end - args.test_start < args.horizon:
        raise ValueError("test split does not fit the dataset or horizon")
    starts = range(args.test_start, args.test_end - args.horizon + 1, args.stride)
    manifest = {
        "schema": "timesfm-etth1-v1", "dataset_path": str(Path(args.dataset).resolve()),
        "dataset_sha256": digest, "dataset_rows": len(columns[CHANNELS[0]]),
        "test_start": args.test_start, "test_end": args.test_end,
        "context": args.context, "horizon": args.horizon, "stride": args.stride,
        "channels": list(CHANNELS), "window_count": len(starts),
        "series_count": len(starts) * len(CHANNELS),
        "preprocessing": "raw values; TimesFM model performs internal normalization",
    }
    Path(args.output).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {args.output}; {manifest['series_count']} series")


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "timesfm-etth1-v1":
        raise ValueError("unsupported manifest schema")
    if manifest["channels"] != list(CHANNELS):
        raise ValueError("manifest channel list differs from ETTh1")
    if sha256(manifest["dataset_path"]) != manifest["dataset_sha256"]:
        raise ValueError("dataset has changed since manifest creation")
    if not 1 <= manifest["horizon"] <= 128 or manifest["stride"] <= 0:
        raise ValueError("invalid manifest horizon or stride")
    if not 0 < manifest["context"] <= min(manifest["test_start"], 16384) or not manifest["test_start"] < manifest["test_end"] <= manifest["dataset_rows"]:
        raise ValueError("invalid manifest context or test split")
    starts = range(manifest["test_start"], manifest["test_end"] - manifest["horizon"] + 1, manifest["stride"])
    if not starts or manifest["window_count"] != len(starts) or manifest["series_count"] != len(starts) * len(CHANNELS):
        raise ValueError("manifest sample counts do not match split")
    return manifest


def batches(manifest, columns, batch_size):
    pending = []
    for start in range(manifest["test_start"], manifest["test_end"] - manifest["horizon"] + 1, manifest["stride"]):
        for name in manifest["channels"]:
            values = columns[name]
            pending.append((f"{start}:{name}", values[start - manifest["context"]:start],
                            values[start:start + manifest["horizon"]]))
            if len(pending) == batch_size:
                yield pending
                pending = []
    if pending:
        yield pending


def infer(args):
    if args.batch_size < 1 or args.warmup < 0:
        raise ValueError("batch-size must be positive and warmup cannot be negative")
    manifest = load_manifest(args.manifest)
    columns = read_series(manifest["dataset_path"])
    model_dir = Path(args.model_dir)
    for filename, expected in (("model.safetensors", MODEL_SHA), ("config.json", CONFIG_SHA)):
        digest = sha256(model_dir / filename)
        if digest != expected:
            raise ValueError(f"{filename} SHA mismatch: {digest}")
    import torch
    if args.device == "npu":
        import importlib
        importlib.import_module("torch_npu")
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but torch.npu.is_available() is false")
        sync = torch.npu.synchronize
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        sync = torch.cuda.synchronize
    else:
        sync = lambda: None
    from transformers import TimesFm2_5ModelForPrediction
    model, loading = TimesFm2_5ModelForPrediction.from_pretrained(
        str(model_dir), local_files_only=True, attn_implementation="eager", output_loading_info=True
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise ValueError(f"model loading {key}: {loading[key]}")
    model = model.to(args.device).eval()
    for parameter in model.parameters():
        if parameter.device.type != args.device or parameter.dtype != torch.float32:
            raise ValueError("model parameters are not on the requested device in float32")
    metadata = {"type": "metadata", "manifest_sha256": sha256(args.manifest), "dataset_sha256": manifest["dataset_sha256"],
                "model_sha256": MODEL_SHA, "config_sha256": CONFIG_SHA, "device": args.device,
                "dtype": "float32", "batch_size": args.batch_size, "warmup": args.warmup,
                "context": manifest["context"], "horizon": manifest["horizon"],
                "torch_version": torch.__version__}
    latencies = []
    written = 0
    with Path(args.output).open("w", encoding="utf-8", newline="\n") as out, torch.inference_mode():
        out.write(json.dumps(metadata) + "\n")
        for i, batch in enumerate(batches(manifest, columns, args.batch_size)):
            inputs = [torch.tensor(x[1], dtype=torch.float32, device=args.device) for x in batch]
            if i == 0:
                for _ in range(args.warmup):
                    model(past_values=inputs, forecast_context_len=manifest["context"])
                sync()
            sync()
            start = time.perf_counter()
            result = model(past_values=inputs, forecast_context_len=manifest["context"])
            sync()
            latencies.append((time.perf_counter() - start) * 1000)
            predictions = result.mean_predictions.detach().to("cpu").tolist()
            if len(predictions) != len(batch) or any(len(p) < manifest["horizon"] for p in predictions):
                raise ValueError("model output shape does not match requested horizon")
            for (sample_id, _, truth), pred in zip(batch, predictions):
                pred = pred[:manifest["horizon"]]
                if not all(math.isfinite(v) for v in pred):
                    raise ValueError(f"non-finite prediction: {sample_id}")
                out.write(json.dumps({"id": sample_id, "truth": truth, "prediction": pred}) + "\n")
                written += 1
    if written != manifest["series_count"]:
        raise ValueError(f"incomplete output: {written} != {manifest['series_count']}")
    print(json.dumps({"series": written, "batches": len(latencies), "batch_latency_median_ms": statistics.median(latencies),
                      "scope": "model call with device synchronization; host preparation and output transfer excluded"}))


def read_results(path):
    with Path(path).open(encoding="utf-8") as f:
        metadata = json.loads(next(f))
        if metadata.get("type") != "metadata":
            raise ValueError("missing result metadata")
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate predictions")
    for row in rows:
        if len(row["truth"]) != metadata["horizon"] or len(row["prediction"]) != metadata["horizon"]:
            raise ValueError("invalid prediction or truth length")
        if not all(math.isfinite(x) for x in row["truth"] + row["prediction"]):
            raise ValueError("non-finite output")
    return metadata, rows


def evaluate(args):
    manifest = load_manifest(args.manifest)
    meta, rows = read_results(args.results)
    if meta["manifest_sha256"] != sha256(args.manifest) or len(rows) != manifest["series_count"]:
        raise ValueError("result manifest or count mismatch")
    expected = (sample for batch in batches(manifest, read_series(manifest["dataset_path"]), 64) for sample in batch)
    square, absolute, count = 0.0, 0.0, 0
    for row, (sample_id, _, truth) in zip(rows, expected):
        if row["id"] != sample_id or row["truth"] != truth:
            raise ValueError("result sample identity/order or ground truth differs from dataset")
        for pred, truth in zip(row["prediction"], row["truth"]):
            delta = pred - truth
            square += delta * delta
            absolute += abs(delta)
            count += 1
    report = {"dataset": "ETTh1", "device": meta["device"], "series": len(rows), "values": count,
              "mse": square / count, "mae": absolute / count, "unit": "original input scale",
              "manifest_sha256": meta["manifest_sha256"]}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


def compare(args):
    if not math.isfinite(args.max_absolute) or args.max_absolute < 0:
        raise ValueError("max-absolute must be finite and nonnegative")
    left_meta, left = read_results(args.reference)
    right_meta, right = read_results(args.candidate)
    for key in ("manifest_sha256", "dataset_sha256", "model_sha256", "config_sha256", "dtype", "context", "horizon"):
        if left_meta[key] != right_meta[key]:
            raise ValueError(f"incompatible result metadata: {key}")
    if len(left) != len(right):
        raise ValueError("different sample counts")
    largest = 0.0
    for a, b in zip(left, right):
        if a["id"] != b["id"] or a["truth"] != b["truth"]:
            raise ValueError("sample identity/order or ground truth differs")
        largest = max(largest, *(abs(x - y) for x, y in zip(a["prediction"], b["prediction"])))
    report = {"max_absolute_error": largest, "threshold": args.max_absolute, "passed": largest <= args.max_absolute,
              "reference_device": left_meta["device"], "candidate_device": right_meta["device"]}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if not report["passed"]:
        raise ValueError("comparison threshold exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="create fixed ETTh1 test manifest")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-sha256", default=DATA_SHA)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--test-start", type=int, default=11520)
    p.add_argument("--test-end", type=int, default=14400)
    p.set_defaults(run=prepare)
    p = sub.add_parser("infer", help="run offline model inference")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("npu", "cpu", "cuda"), default="npu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.set_defaults(run=infer)
    p = sub.add_parser("evaluate", help="compute ETTh1 MSE and MAE")
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=evaluate)
    p = sub.add_parser("compare", help="compare identical model outputs across devices")
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--max-absolute", type=float, required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
