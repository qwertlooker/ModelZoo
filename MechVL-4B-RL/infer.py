#!/usr/bin/env python3
"""Query a MechVL-4B-RL vLLM-Ascend service with one drawing."""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RL_SUFFIX = (
    r" \n\nA conversation between User and Assistant. The user asks a question, and the "
    r"Assistant solves it. The assistant first thinks about the reasoning process in the "
    r"mind and then provides the user with the answer. The reasoning process and answer "
    r"are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    r"<think> reasoning process here </think><answer> answer here </answer>. \n\nNow let's "
    r"solve the question step by step.\n"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
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
        description="Run one MechVL-4B-RL image question through an NPU vLLM service."
    )
    parser.add_argument("--image", required=True, help="Local mechanical drawing path.")
    parser.add_argument("--question", required=True, help="Question about the drawing.")
    parser.add_argument("--output", required=True, help="Result JSON path.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="MechVL-4B-RL")
    parser.add_argument("--device", choices=("npu",), default="npu")
    parser.add_argument(
        "--api-key", default=os.environ.get("VQA_TARGET_API_KEY", "EMPTY")
    )
    parser.add_argument("--temperature", type=nonnegative_float, default=0.6)
    parser.add_argument("--top-p", type=probability, default=0.95)
    parser.add_argument("--top-k", type=positive_int, default=20)
    parser.add_argument("--max-tokens", type=positive_int, default=4096)
    parser.add_argument("--timeout", type=positive_int, default=600)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def extract_answer(text: str) -> str:
    lowered = text.lower()
    start = lowered.find("<answer>")
    if start >= 0:
        start += len("<answer>")
        end = lowered.find("</answer>", start)
        return text[start : end if end >= 0 else None].strip()
    think_end = lowered.find("</think>")
    if think_end >= 0:
        return text[think_end + len("</think>") :].strip()
    return text.strip()


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


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")

    models = get_json(f"{args.base_url.rstrip('/')}/models", args.api_key, args.timeout)
    served_models = {str(item["id"]) for item in models.get("data", [])}
    if args.model not in served_models:
        raise ValueError(
            f"served model mismatch: expected {args.model}, got {sorted(served_models)}"
        )

    prompt = args.question.strip() + RL_SUFFIX
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_path)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = post_json(
        f"{args.base_url.rstrip('/')}/chat/completions",
        payload,
        args.api_key,
        args.timeout,
    )
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    raw_text = choice["message"].get("content") or ""
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "device": args.device,
        "provider": "vllm-ascend-openai-compatible",
        "base_url": args.base_url,
        "model": args.model,
        "image": str(image_path),
        "image_sha256": sha256(image_path),
        "question": args.question,
        "prompt_format": "MechVL RL mech_r1",
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
        },
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage", {}),
        "raw_response": raw_text,
        "answer": extract_answer(raw_text),
    }
    if not result["answer"]:
        raise ValueError("service returned an empty answer")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {"answer": result["answer"], "output": str(output_path)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
