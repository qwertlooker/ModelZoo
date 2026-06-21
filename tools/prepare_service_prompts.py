#!/usr/bin/env python3
"""Expand a small service smoke manifest into a deterministic L1 prompt set."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


EN_WORDS = ("ready", "stable", "verified", "complete", "aligned")
ZH_WORDS = ("就绪", "稳定", "通过", "完成", "对齐")
CITIES = ("Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Hangzhou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic OpenAI-compatible L1 prompt manifest."
    )
    parser.add_argument("--base", required=True, help="Base JSONL prompt manifest.")
    parser.add_argument("--output", required=True, help="Generated JSONL path.")
    parser.add_argument("--count", type=int, default=100)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "messages" not in row:
                raise ValueError(
                    f"{path}:{line_number}: fields 'id' and 'messages' are required"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty base manifest: {path}")
    return rows


def replace_user_message(row: dict, content: str) -> None:
    for message in reversed(row["messages"]):
        if message["role"] == "user":
            message["content"] = content
            return
    raise ValueError(f"Prompt {row['id']}: no user message")


def specialize(row: dict, index: int) -> dict:
    result = copy.deepcopy(row)
    source_id = str(row["id"])
    result["id"] = f"{source_id}-{index:03d}"
    result["source_id"] = source_id
    variant = index // 4

    if source_id == "direct-en":
        word = EN_WORDS[variant % len(EN_WORDS)]
        replace_user_message(result, f"Return exactly the word: {word}")
    elif source_id == "direct-zh":
        word = ZH_WORDS[variant % len(ZH_WORDS)]
        replace_user_message(result, f"只回复：{word}")
    elif source_id == "json":
        answer = (variant % 97) + 1
        replace_user_message(
            result,
            f"Return a JSON object with integer field answer equal to {answer}.",
        )
    elif source_id == "tool":
        city = CITIES[variant % len(CITIES)]
        replace_user_message(
            result,
            f"What is the weather in {city}? Use the provided tool.",
        )
    elif source_id == "reasoning":
        left = (variant % 29) + 11
        right = (variant % 31) + 17
        replace_user_message(
            result,
            f"Compute {left} * {right} and provide the final number.",
        )
    return result


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    base_path = Path(args.base)
    base_rows = read_rows(base_path)
    rows = [
        specialize(base_rows[index % len(base_rows)], index)
        for index in range(args.count)
    ]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated duplicate prompt ids")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "base": str(base_path),
        "base_sha256": sha256(base_path),
        "count": len(rows),
        "output_sha256": sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
