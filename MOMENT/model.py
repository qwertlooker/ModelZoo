"""MOMENT-1-large reconstruction or forecasting with an explicit trained head."""

import json
import sys
import argparse
import hashlib
import importlib
import math
import statistics
import time
from pathlib import Path
import os
import platform
import uuid
from importlib.metadata import version

MODEL = "AutonLab/MOMENT-1-large@ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"
ASSETS = {
    "model.safetensors": "a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572",
    "config.json": "b0111650e8718e31d7e89575666aa3e140747d35279ec1a7c9f6ed7616c29433",
}


def validate_input(args, x, targets):
    import numpy as np
    from pathlib import Path

    if args.task not in ("forecast", "reconstruction"):
        raise ValueError("MOMENT task must be forecast or reconstruction")
    if x.ndim != 3 or x.shape[-1] != 512:
        raise ValueError("MOMENT inputs require [batch,channels,512]")
    if targets is None or targets.ndim != 3 or targets.shape[:2] != x.shape[:2]:
        raise ValueError("targets must have shape [batch,channels,horizon]")
    if args.task == "forecast" and (not args.head or not args.head_sha256):
        raise ValueError("forecast requires a trained head and its SHA256")
    if args.task == "forecast":
        head_meta = json.loads(Path(args.head + ".json").read_text(encoding="utf-8"))
        if (
            head_meta.get("head_sha256") != args.head_sha256
            or head_meta.get("steps", 0) < 1
            or head_meta.get("train_split") != "train"
            or (head_meta.get("horizon") != targets.shape[-1])
            or (head_meta.get("upstream_model_sha256") != ASSETS["model.safetensors"])
        ):
            raise ValueError(
                "trained head metadata does not match model or forecast task"
            )
    if args.task == "reconstruction" and (args.head or targets.shape[-1] != 512):
        raise ValueError("reconstruction requires 512 targets and no forecast head")
    with np.load(args.input, allow_pickle=False) as data:
        info = json.loads(str(data["preparation"].item()))
    if (
        info.get("model") != "MOMENT"
        or info.get("task") != args.task
        or info.get("scaling") != "etth1_train_standard"
    ):
        raise ValueError(
            "MOMENT requires its standardized ETTh1 data and matching task"
        )


def load_model(args, model_dir, device):
    require_package("momentfm", "0.1.5")
    require_package("transformers", "4.57.6")
    require_torch_range("2.0.1")
    from momentfm import MOMENTPipeline

    require_source(
        MOMENTPipeline,
        "70510f443376c9e15bc41326cf7ed3374b1f9e43d9cda2ac448e6004c0943820",
    )
    from safetensors.torch import load_file, load_model as load_safetensors

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    options = {"task_name": "reconstruction"}
    if args.task == "forecast":
        import numpy as np

        with np.load(args.input, allow_pickle=False) as data:
            horizon = int(data["targets"].shape[-1])
        options = {"task_name": "forecasting", "forecast_horizon": horizon}
    model = MOMENTPipeline(config, model_kwargs=options)
    load_safetensors(model, str(model_dir / "model.safetensors"), strict=True)
    model.init()
    if args.task == "forecast":
        model.head.load_state_dict(load_file(args.head, device="cpu"), strict=True)
    return model.to(device).eval()


def predict(args, model, batch, target):
    import torch

    mask = torch.ones(
        (len(batch), batch.shape[-1]), dtype=torch.bool, device=batch.device
    )
    if args.task == "forecast":
        result = model.forecast(x_enc=batch, input_mask=mask).forecast
    else:
        result = model.reconstruct(x_enc=batch, input_mask=mask).reconstruction
    if tuple(result.shape) != tuple(target.shape):
        raise ValueError("unexpected MOMENT output shape")
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_file(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual}")


def require_package(name, version):
    from importlib.metadata import version as package_version

    actual = package_version(name)
    if actual != version:
        raise RuntimeError(f"{name} must be {version}, got {actual}")


def require_source(cls, expected_sha256):
    import inspect

    source = inspect.getsourcefile(cls)
    if source is None:
        raise RuntimeError("model source file unavailable")
    checked_file(source, expected_sha256)


def require_torch_range(minimum, maximum=None):
    import torch
    from packaging.version import Version

    actual = Version(torch.__version__)
    if actual < Version(minimum) or (
        maximum is not None and actual >= Version(maximum)
    ):
        raise RuntimeError(f"unsupported torch version: {actual}")


def input_arrays(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        x = data["inputs"]
        y = data["targets"] if "targets" in data else None
    if (
        x.ndim == 0
        or x.dtype != np.float32
        or (not np.isfinite(x).all())
        or (x.shape[0] == 0)
    ):
        raise ValueError("inputs must be nonempty, finite float32")
    if y is not None and (
        y.ndim == 0 or not np.isfinite(y).all() or y.shape[0] != x.shape[0]
    ):
        raise ValueError("invalid target values or sample count")
    return (x, y)


def device_and_sync(name):
    import torch

    if name == "npu":
        importlib.import_module("torch_npu")
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
        return (torch.device("npu"), torch.npu.synchronize)
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return (torch.device("cuda"), torch.cuda.synchronize)
    return (torch.device("cpu"), lambda: None)


def check_model_device(model, device):
    for parameter in model.parameters():
        if (
            parameter.device.type != device.type
            or parameter.dtype != __import__("torch").float32
        ):
            raise ValueError("model parameters must be on selected device and float32")


def parse_args():
    p = argparse.ArgumentParser(
        description="Strict offline inference, evaluation and comparison"
    )
    sub = p.add_subparsers(dest="command", required=True)
    inf = sub.add_parser("infer")
    inf.add_argument("--model-dir", required=True)
    inf.add_argument("--input", required=True)
    inf.add_argument("--output", required=True)
    inf.add_argument("--device", choices=("npu", "cpu", "cuda"), default="npu")
    inf.add_argument("--batch-size", type=int, default=8)
    inf.add_argument("--warmup", type=int, default=3)
    inf.add_argument("--head")
    inf.add_argument("--head-sha256")
    inf.add_argument(
        "--task", choices=("forecast", "reconstruction"), default="reconstruction"
    )
    score = sub.add_parser("evaluate")
    score.add_argument("--input", required=True)
    score.add_argument("--results", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--metric", choices=("regression",), required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--reference", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--max-absolute", type=float, required=True)
    comp.add_argument("--output", required=True)
    perf = sub.add_parser(
        "performance", help="summarize three independent NPU timing reports"
    )
    perf.add_argument("--runs", nargs="+", required=True)
    perf.add_argument("--output", required=True)
    return p.parse_args()


def read_prediction(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        prediction = data["prediction"]
        meta = json.loads(str(data["metadata"].item()))
    if prediction.size == 0 or not np.isfinite(prediction).all():
        raise ValueError("invalid predictions")
    return (prediction, meta)


def infer(args):
    import numpy as np
    import torch

    if args.batch_size <= 0 or args.warmup < 0:
        raise ValueError("invalid batch-size or warmup")
    if Path(args.output).suffix != ".npz":
        raise ValueError("output must end in .npz")
    if Path(args.output).resolve() == Path(args.input).resolve():
        raise ValueError("output must not overwrite input")
    x, targets = input_arrays(args.input)
    validate_input(args, x, targets)
    model_dir = Path(args.model_dir)
    for filename, expected in ASSETS.items():
        checked_file(model_dir / filename, expected)
    if args.head:
        if not args.head_sha256 or len(args.head_sha256) != 64:
            raise ValueError("a head requires its explicit SHA256")
        checked_file(args.head, args.head_sha256)
    elif args.head_sha256:
        raise ValueError("head SHA provided without head file")
    device, synchronize = device_and_sync(args.device)
    model = load_model(args, model_dir, device)
    check_model_device(model, device)
    output, durations = ([], [])
    with torch.inference_mode():
        for start in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(x[start : start + args.batch_size]).to(device)
            target_batch = (
                None if targets is None else targets[start : start + args.batch_size]
            )
            if start == 0:
                for _ in range(args.warmup):
                    predict(args, model, batch, target_batch)
                synchronize()
            synchronize()
            t0 = time.perf_counter()
            result = predict(args, model, batch, target_batch)
            synchronize()
            durations.append((time.perf_counter() - t0) * 1000)
            output.append(result.detach().to("cpu").float().numpy())
    prediction = np.concatenate(output, axis=0)
    if prediction.shape[0] != len(x) or not np.isfinite(prediction).all():
        raise ValueError("invalid model output")
    meta = {
        "model": MODEL,
        "model_sha256": ASSETS["model.safetensors"],
        "config_sha256": ASSETS["config.json"],
        "input_sha256": digest(args.input),
        "head_sha256": args.head_sha256,
        "device": args.device,
        "dtype": "float32",
        "task": args.task,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "torch_version": torch.__version__,
        "adapter_sha256": digest(__file__),
    }
    np.savez_compressed(
        args.output, prediction=prediction, metadata=json.dumps(meta, sort_keys=True)
    )
    write_report(args.output, meta, durations, len(x))
    print(
        json.dumps(
            {
                "samples": len(x),
                "batch_latency_median_ms": statistics.median(durations),
                "scope": "synchronized model calls only; single run, not independent-run median",
            }
        )
    )


def evaluate(args):
    import numpy as np

    _, targets = input_arrays(args.input)
    pred, meta = read_prediction(args.results)
    if meta["input_sha256"] != digest(args.input) or targets is None:
        raise ValueError("missing targets or input digest mismatch")
    if pred.ndim == 0 or len(pred) != len(targets):
        raise ValueError("prediction and target sample counts differ")
    if pred.shape != targets.shape:
        raise ValueError(f"regression shape mismatch: {pred.shape} != {targets.shape}")
    delta = pred.astype(np.float64) - targets
    report = {
        "mse": float(np.mean(delta * delta)),
        "mae": float(np.mean(np.abs(delta))),
        "samples": len(pred),
    }
    report.update(
        model=meta["model"],
        model_sha256=meta["model_sha256"],
        device=meta["device"],
        input_sha256=digest(args.input),
        results_sha256=digest(args.results),
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


def compare(args):
    import numpy as np

    if not math.isfinite(args.max_absolute) or args.max_absolute < 0:
        raise ValueError("max-absolute must be finite and nonnegative")
    ref, a = read_prediction(args.reference)
    out, b = read_prediction(args.candidate)
    for key in (
        "model",
        "model_sha256",
        "config_sha256",
        "head_sha256",
        "input_sha256",
        "dtype",
        "task",
    ):
        if a[key] != b[key]:
            raise ValueError(f"incompatible metadata: {key}")
    if ref.shape != out.shape:
        raise ValueError("prediction shapes differ")
    error = float(np.max(np.abs(ref.astype(np.float64) - out.astype(np.float64))))
    report = {
        "max_absolute": error,
        "threshold": args.max_absolute,
        "passed": error <= args.max_absolute,
        "reference_device": a["device"],
        "candidate_device": b["device"],
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if not report["passed"]:
        raise ValueError("comparison threshold exceeded")


def write_report(output, metadata, durations_ms, samples):
    if (
        not durations_ms
        or samples < 1
        or any((not math.isfinite(t) or t <= 0 for t in durations_ms))
    ):
        raise ValueError("performance needs positive finite timings and samples")
    identity = dict(metadata)
    identity.update(
        python_version=platform.python_version(),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    )
    if metadata["device"] == "npu":
        import torch

        identity.update(
            torch_npu_version=version("torch-npu"),
            device_name=torch.npu.get_device_name(),
            logical_device=torch.npu.current_device(),
        )
    elapsed = sum(durations_ms) / 1000
    report = {
        "schema": "synchronized-model-timing-v1",
        "run_id": str(uuid.uuid4()),
        "identity": identity,
        "samples": samples,
        "batch_durations_ms": durations_ms,
        "model_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "scope": "synchronized model calls; excludes loading, warmup, input/output and file IO",
        "result_sha256": digest(output),
        "command": sys.argv,
    }
    Path(str(output) + ".performance.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def summarize_performance(paths):
    if len(paths) < 3:
        raise ValueError("at least three independent NPU runs required")
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    seen = set()
    for report in reports:
        if (
            report["schema"] != "synchronized-model-timing-v1"
            or report["identity"]["device"] != "npu"
        ):
            raise ValueError("only NPU synchronized timing reports are accepted")
        if report["run_id"] in seen:
            raise ValueError("duplicate run ID; do not reuse one run")
        seen.add(report["run_id"])
        if (
            report["identity"] != reports[0]["identity"]
            or report["samples"] != reports[0]["samples"]
        ):
            raise ValueError("run model, data, device or benchmark settings differ")
        timings = report["batch_durations_ms"]
        if (
            not timings
            or report["samples"] < 1
            or any((not math.isfinite(t) or t <= 0 for t in timings))
        ):
            raise ValueError("invalid timings or sample count")
        elapsed = sum(timings) / 1000
        if not math.isclose(report["model_seconds"], elapsed) or not math.isclose(
            report["samples_per_second"], report["samples"] / elapsed
        ):
            raise ValueError("inconsistent performance statistics")
    return {
        "identity": reports[0]["identity"],
        "runs": len(reports),
        "scope": reports[0]["scope"],
        "samples_per_run": reports[0]["samples"],
        "median_model_seconds": statistics.median(
            (r["model_seconds"] for r in reports)
        ),
        "median_samples_per_second": statistics.median(
            (r["samples_per_second"] for r in reports)
        ),
        "reports": [{"path": str(p), "sha256": digest(p)} for p in paths],
    }


def main():
    args = parse_args()
    if args.command == "infer":
        infer(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "compare":
        compare(args)
    else:
        report = summarize_performance(args.runs)
        Path(args.output).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
