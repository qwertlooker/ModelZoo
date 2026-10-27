"""DINOv2 ViT-L/14 with a separately supplied official linear4 classifier."""

import sys
import argparse
import hashlib
import importlib
import json
import math
import statistics
import time
from pathlib import Path
import os
import platform
import uuid
from importlib.metadata import version

MODEL = "facebook/dinov2-large@47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
ASSETS = {
    "model.safetensors": "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
    "config.json": "12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
}


def validate_input(args, x, targets):
    import json
    import numpy as np

    if x.ndim != 4 or x.shape[1:] != (3, 224, 224):
        raise ValueError("DINO inputs require preprocessed float32 [batch,3,224,224]")
    if targets is None or targets.shape != (len(x),) or targets.dtype.kind not in "iu":
        raise ValueError("DINO requires one integer class label per input")
    if not args.head or not args.head_sha256:
        raise ValueError("DINO requires the official linear4 head and its SHA256")
    if args.task != "classification":
        raise ValueError("DINO requires the classification task")
    with np.load(args.input, allow_pickle=False) as data:
        info = json.loads(str(data["preparation"].item()))
    if info.get("preprocessor_sha256") != ASSETS[
        "preprocessor_config.json"
    ] or info.get("count") != len(x):
        raise ValueError("DINO input preparation metadata mismatch")


def load_model(args, model_dir, device):
    import torch

    require_package("transformers", "4.57.6")
    from transformers import Dinov2Model

    require_source(
        Dinov2Model, "e1376e0decc4cf3fd60042a17464a4118f1802a4e1ca2af4496c211fabb6d34b"
    )

    class Classifier(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            output = self.backbone(pixel_values=x, output_hidden_states=True)
            layers = [self.backbone.layernorm(h) for h in output.hidden_states[-4:]]
            features = torch.cat(
                [h[:, 0] for h in layers] + [layers[-1][:, 1:].mean(dim=1)], dim=1
            )
            return self.head(features)

    backbone, info = Dinov2Model.from_pretrained(
        str(model_dir), local_files_only=True, output_loading_info=True
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"checkpoint {key}: {info[key]}")
    head = torch.nn.Linear(5120, 1000)
    head.load_state_dict(
        torch.load(args.head, map_location="cpu", weights_only=True), strict=True
    )
    return Classifier(backbone, head).to(device).eval()


def predict(args, model, batch, target):
    output = model(batch)
    if tuple(output.shape) != (len(batch), 1000):
        raise ValueError("unexpected classifier output shape")
    return output


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
    inf.add_argument("--task", choices=("classification",), default="classification")
    score = sub.add_parser("evaluate")
    score.add_argument("--input", required=True)
    score.add_argument("--results", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--metric", choices=("classification",), required=True)
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
    if (
        pred.ndim != 2
        or pred.shape[1] < 5
        or targets.ndim != 1
        or (targets.dtype.kind not in "iu")
    ):
        raise ValueError(
            "classification requires logits [N, classes] and integer target [N]"
        )
    if np.any(targets < 0) or np.any(targets >= pred.shape[1]):
        raise ValueError("classification target outside output classes")
    top = np.argsort(pred, axis=1)[:, -5:]
    report = {
        "top1": float(np.mean(top[:, -1] == targets)),
        "top5": float(np.mean(np.any(top == targets[:, None], axis=1))),
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
