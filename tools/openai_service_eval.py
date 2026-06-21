#!/usr/bin/env python3
"""Run a deterministic JSONL prompt set against an OpenAI-compatible server."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REQUEST_FIELDS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI-compatible chat-completions endpoint."
    )
    parser.add_argument("--base_url", required=True, help="Server base URL ending in /v1.")
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument("--prompts", required=True, help="Input JSONL prompt manifest.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--api_key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key. Defaults to OPENAI_API_KEY or EMPTY.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--request_logprobs",
        action="store_true",
        help="Request token logprobs/top_logprobs=1 for backend token alignment.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "messages" not in row:
                raise ValueError(f"{path}:{line_number}: fields 'id' and 'messages' are required")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty prompt manifest: {path}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate prompt ids in {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_payload(row: dict[str, Any], model: str, request_logprobs: bool) -> dict[str, Any]:
    payload = {"model": model}
    for key in REQUEST_FIELDS:
        if key in row:
            payload[key] = row[key]
    if "extra_body" in row:
        extra_body = row["extra_body"]
        if not isinstance(extra_body, dict):
            raise TypeError(f"Prompt {row['id']}: extra_body must be an object")
        payload.update(extra_body)
    payload.setdefault("stream", False)
    if request_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1
    return payload


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
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
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def token_list(choice: dict[str, Any]) -> list[str]:
    logprobs = choice.get("logprobs")
    if not logprobs:
        return []
    content = logprobs.get("content") or []
    return [str(item["token"]) for item in content]


def normalize_result(
    prompt: dict[str, Any], response: dict[str, Any], elapsed: float
) -> dict[str, Any]:
    choices = response["choices"]
    if len(choices) != 1:
        raise ValueError(f"Prompt {prompt['id']}: expected one choice, got {len(choices)}")
    choice = choices[0]
    message = choice["message"]
    result = {
        "id": str(prompt["id"]),
        "checks": prompt.get("checks", {}),
        "elapsed_seconds": elapsed,
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "tokens": token_list(choice),
        "usage": response.get("usage", {}),
        "response_id": response.get("id"),
    }
    checks = prompt.get("checks", {})
    if checks.get("json"):
        parsed = json.loads(result["content"])
        if not isinstance(parsed, dict):
            raise ValueError(f"Prompt {prompt['id']}: response is not a JSON object")
    if checks.get("tool_call"):
        if not result["tool_calls"]:
            raise ValueError(f"Prompt {prompt['id']}: expected at least one tool call")
        for tool_call in result["tool_calls"]:
            json.loads(tool_call["function"]["arguments"])
    return result


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompts)
    prompts = read_jsonl(prompt_path)
    if args.limit:
        prompts = prompts[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    started = time.perf_counter()
    with temporary_path.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts, 1):
            payload = request_payload(prompt, args.model, args.request_logprobs)
            request_started = time.perf_counter()
            response = post_json(endpoint, payload, args.api_key, args.timeout)
            result = normalize_result(
                prompt, response, time.perf_counter() - request_started
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(prompts)}] id={prompt['id']} ok")
    temporary_path.replace(output_path)

    metadata = {
        "base_url": args.base_url,
        "model": args.model,
        "prompts": str(prompt_path),
        "prompts_sha256": sha256(prompt_path),
        "prompt_count": len(prompts),
        "request_logprobs": args.request_logprobs,
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output={output_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
