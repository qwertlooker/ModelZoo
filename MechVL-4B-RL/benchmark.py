#!/usr/bin/env python3
"""Measure repeatable end-to-end latency against a MechVL NPU service."""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import platform
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infer import RL_SUFFIX


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three independent end-to-end benchmark rounds on an NPU service."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="MechVL-4B-RL")
    parser.add_argument(
        "--api-key", default=os.environ.get("VQA_TARGET_API_KEY", "EMPTY")
    )
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--record", type=positive_int, default=10)
    parser.add_argument("--timeout", type=positive_int, default=600)
    parser.add_argument("--temperature", type=nonnegative_float, default=0.6)
    parser.add_argument("--top-p", type=probability, default=0.95)
    parser.add_argument("--top-k", type=positive_int, default=20)
    parser.add_argument("--max-tokens", type=positive_int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


def question(record: dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    raise ValueError("record has no user question")


def image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def payload_for(
    record: dict[str, Any], image_root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("record has no images")
    content = []
    for relative_name in images:
        relative = Path(str(relative_name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe image path: {relative}")
        image_path = (image_root / relative).resolve()
        if image_root.resolve() not in image_path.parents:
            raise ValueError(f"image escapes image root: {relative}")
        if not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        content.append(
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}}
        )
    content.append({"type": "text", "text": question(record).strip() + RL_SUFFIX})
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "stream": False,
    }


def get_json(url: str, api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def post_json(
    url: str, payload: dict[str, Any], api_key: str, timeout: int
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def request_once(
    payload: dict[str, Any], args: argparse.Namespace
) -> tuple[float, int]:
    started = time.perf_counter()
    response = post_json(
        f"{args.base_url.rstrip('/')}/chat/completions",
        payload,
        args.api_key,
        args.timeout,
    )
    elapsed = time.perf_counter() - started
    choices = response["choices"]
    if len(choices) != 1 or not (choices[0]["message"].get("content") or "").strip():
        raise ValueError("service returned an empty or ambiguous response")
    usage = response.get("usage")
    if not isinstance(usage, dict) or int(usage.get("completion_tokens", 0)) <= 0:
        raise ValueError("service did not return a positive completion token count")
    completion_tokens = int(usage["completion_tokens"])
    return elapsed, completion_tokens


def run_round(
    payloads: list[dict[str, Any]], args: argparse.Namespace, index: int
) -> dict[str, Any]:
    for warmup_index in range(args.warmup):
        request_once(payloads[warmup_index % len(payloads)], args)

    latencies = []
    completion_tokens = 0
    started = time.perf_counter()
    for payload in payloads:
        latency, tokens = request_once(payload, args)
        latencies.append(latency)
        completion_tokens += tokens
    wall_seconds = time.perf_counter() - started
    return {
        "run": index,
        "request_count": len(payloads),
        "warmup_count": args.warmup,
        "wall_seconds": wall_seconds,
        "mean_latency_seconds": statistics.fmean(latencies),
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "requests_per_second": len(payloads) / wall_seconds,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / wall_seconds,
        "latencies_seconds": latencies,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not args.dry_run and args.runs < 3:
        raise ValueError("formal performance requires at least three independent runs")

    manifest = Path(args.manifest).resolve()
    image_root = Path(args.image_root).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    records = read_jsonl(manifest)
    if args.record > len(records):
        raise ValueError(f"record={args.record} exceeds manifest size {len(records)}")
    selected = records[: args.record]
    payloads = [payload_for(record, image_root, args) for record in selected]

    base = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "npu",
        "provider": "vllm-ascend-openai-compatible",
        "measurement_scope": "end-to-end HTTP, queueing, preprocessing, model, generation, transfer",
        "base_url": args.base_url,
        "model": args.model,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "image_root": str(image_root),
        "record_count": len(payloads),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
        },
    }
    output_path = Path(args.output)
    if args.dry_run:
        base.update({"status": "dry_run", "performance_valid": False})
        write_json(output_path, base)
        print(json.dumps(base, ensure_ascii=False, indent=2))
        return

    models = get_json(f"{args.base_url.rstrip('/')}/models", args.api_key, args.timeout)
    served_models = {str(item["id"]) for item in models.get("data", [])}
    if args.model not in served_models:
        raise ValueError(
            f"served model mismatch: expected {args.model}, got {sorted(served_models)}"
        )

    runs = [run_round(payloads, args, index + 1) for index in range(args.runs)]
    summary = {
        "median_mean_latency_seconds": statistics.median(
            run["mean_latency_seconds"] for run in runs
        ),
        "median_p95_latency_seconds": statistics.median(
            run["p95_latency_seconds"] for run in runs
        ),
        "median_requests_per_second": statistics.median(
            run["requests_per_second"] for run in runs
        ),
        "median_completion_tokens_per_second": statistics.median(
            run["completion_tokens_per_second"] for run in runs
        ),
    }
    base.update(
        {
            "status": "completed",
            "performance_valid": True,
            "runs": runs,
            "summary": summary,
        }
    )
    write_json(output_path, base)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
