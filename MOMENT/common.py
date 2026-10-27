"""Offline inputs, strict device selection, comparison and scoring for one model."""
import argparse
import hashlib
import importlib
import json
import math
import statistics
import time
from pathlib import Path


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
    if actual < Version(minimum) or (maximum is not None and actual >= Version(maximum)):
        raise RuntimeError(f"unsupported torch version: {actual}")


def input_arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        x = data["inputs"]
        y = data["targets"] if "targets" in data else None
    if x.dtype != np.float32 or not np.isfinite(x).all() or x.shape[0] == 0:
        raise ValueError("inputs must be nonempty, finite float32")
    if y is not None and (not np.isfinite(y).all() or y.shape[0] != x.shape[0]):
        raise ValueError("invalid target values or sample count")
    return x, y


def device_and_sync(name):
    import torch
    if name == "npu":
        importlib.import_module("torch_npu")
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
        return torch.device("npu"), torch.npu.synchronize
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda"), torch.cuda.synchronize
    return torch.device("cpu"), lambda: None


def check_model_device(model, device):
    for parameter in model.parameters():
        if parameter.device.type != device.type or parameter.dtype != __import__("torch").float32:
            raise ValueError("model parameters must be on selected device and float32")


def parse_args():
    p = argparse.ArgumentParser(description="Strict offline inference, evaluation and comparison")
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
    inf.add_argument("--task", choices=("reconstruction", "forecast", "classification"), default="reconstruction")
    score = sub.add_parser("evaluate")
    score.add_argument("--input", required=True)
    score.add_argument("--results", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--metric", choices=("regression", "classification"), required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--reference", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--max-absolute", type=float, required=True)
    comp.add_argument("--output", required=True)
    return p.parse_args()


def read_prediction(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        prediction = data["prediction"]
        meta = json.loads(str(data["metadata"].item()))
    if prediction.size == 0 or not np.isfinite(prediction).all():
        raise ValueError("invalid predictions")
    return prediction, meta


def infer(args, implementation):
    import numpy as np
    import torch
    if args.batch_size <= 0 or args.warmup < 0:
        raise ValueError("invalid batch-size or warmup")
    if Path(args.output).suffix != ".npz":
        raise ValueError("output must end in .npz")
    x, targets = input_arrays(args.input)
    implementation.validate_input(args, x, targets)
    model_dir = Path(args.model_dir)
    for filename, expected in implementation.ASSETS.items():
        checked_file(model_dir / filename, expected)
    if args.head:
        if not args.head_sha256 or len(args.head_sha256) != 64:
            raise ValueError("a head requires its explicit SHA256")
        checked_file(args.head, args.head_sha256)
    elif args.head_sha256:
        raise ValueError("head SHA provided without head file")
    device, synchronize = device_and_sync(args.device)
    model = implementation.load_model(args, model_dir, device)
    check_model_device(model, device)
    output, durations = [], []
    with torch.inference_mode():
        for start in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(x[start:start + args.batch_size]).to(device)
            target_batch = None if targets is None else targets[start:start + args.batch_size]
            if start == 0:
                for _ in range(args.warmup):
                    implementation.predict(args, model, batch, target_batch)
                synchronize()
            synchronize()
            t0 = time.perf_counter()
            result = implementation.predict(args, model, batch, target_batch)
            synchronize()
            durations.append((time.perf_counter() - t0) * 1000)
            output.append(result.detach().to("cpu").float().numpy())
    prediction = np.concatenate(output, axis=0)
    if prediction.shape[0] != len(x) or not np.isfinite(prediction).all():
        raise ValueError("invalid model output")
    meta = {"model": implementation.MODEL, "model_sha256": implementation.ASSETS["model.safetensors"],
            "config_sha256": implementation.ASSETS["config.json"], "input_sha256": digest(args.input),
            "head_sha256": args.head_sha256, "device": args.device, "dtype": "float32",
            "task": args.task, "batch_size": args.batch_size, "warmup": args.warmup,
            "torch_version": torch.__version__}
    np.savez_compressed(args.output, prediction=prediction, metadata=json.dumps(meta, sort_keys=True))
    print(json.dumps({"samples": len(x), "batch_latency_median_ms": statistics.median(durations),
                      "scope": "synchronized model calls only; single run, not independent-run median"}))


def evaluate(args):
    import numpy as np
    _, targets = input_arrays(args.input)
    pred, meta = read_prediction(args.results)
    if meta["input_sha256"] != digest(args.input) or targets is None:
        raise ValueError("missing targets or input digest mismatch")
    if args.metric == "regression":
        if pred.shape != targets.shape:
            raise ValueError(f"regression shape mismatch: {pred.shape} != {targets.shape}")
        delta = pred.astype(np.float64) - targets
        report = {"mse": float(np.mean(delta * delta)), "mae": float(np.mean(np.abs(delta))), "samples": len(pred)}
    else:
        if pred.ndim != 2 or pred.shape[1] < 5 or targets.ndim != 1 or targets.dtype.kind not in "iu":
            raise ValueError("classification requires logits [N, classes] and integer target [N]")
        if np.any(targets < 0) or np.any(targets >= pred.shape[1]):
            raise ValueError("classification target outside output classes")
        top = np.argsort(pred, axis=1)[:, -5:]
        report = {"top1": float(np.mean(top[:, -1] == targets)),
                  "top5": float(np.mean(np.any(top == targets[:, None], axis=1))), "samples": len(pred)}
    report.update(model=meta["model"], model_sha256=meta["model_sha256"], device=meta["device"],
                  input_sha256=digest(args.input), results_sha256=digest(args.results))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


def compare(args):
    import numpy as np
    if not math.isfinite(args.max_absolute) or args.max_absolute < 0:
        raise ValueError("max-absolute must be finite and nonnegative")
    ref, a = read_prediction(args.reference)
    out, b = read_prediction(args.candidate)
    for key in ("model", "model_sha256", "config_sha256", "head_sha256", "input_sha256", "dtype", "task"):
        if a[key] != b[key]:
            raise ValueError(f"incompatible metadata: {key}")
    if ref.shape != out.shape:
        raise ValueError("prediction shapes differ")
    error = float(np.max(np.abs(ref.astype(np.float64) - out.astype(np.float64))))
    report = {"max_absolute": error, "threshold": args.max_absolute, "passed": error <= args.max_absolute,
              "reference_device": a["device"], "candidate_device": b["device"]}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if not report["passed"]:
        raise ValueError("comparison threshold exceeded")


def main(implementation):
    args = parse_args()
    if args.command == "infer":
        infer(args, implementation)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        compare(args)
