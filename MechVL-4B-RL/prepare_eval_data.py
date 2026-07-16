#!/usr/bin/env python3
"""Validate the fixed public MechVQA test split and create a local manifest."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UPSTREAM_REVISION = "8841ee083c2704f2d8ccf426a8c0bb61ad911890"
SOURCE_MANIFEST_SHA256 = (
    "e9ff49a26742d24ac6c1cdff5279aa4eb75e3a17787da54efe75faff2adaeba2"
)
SOURCE_RECORD_COUNT = 1185
SOURCE_IMAGE_COUNT = 562


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MechVQA public_test data from the fixed upstream commit."
    )
    parser.add_argument(
        "--upstream-dir", required=True, help="Fixed MechVQA Git checkout."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Manifest and metadata directory."
    )
    parser.add_argument(
        "--limit",
        type=nonnegative_int,
        default=0,
        help="Write the first N records; 0 writes the full 1,185-record split.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            records.append(record)
    return records


def validate_record(record: dict[str, Any], image_root: Path, index: int) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"record {index}: messages must be a list")
    roles = {message.get("role") for message in messages if isinstance(message, dict)}
    if not {"user", "assistant"}.issubset(roles):
        raise ValueError(f"record {index}: user and assistant messages are required")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"record {index}: metadata must be an object")
    for field in ("capability", "subcategory", "difficulty", "language"):
        if not metadata.get(field):
            raise ValueError(f"record {index}: metadata.{field} is required")

    images = record.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"record {index}: at least one image is required")
    normalized = []
    root = image_root.resolve()
    for image in images:
        relative = Path(str(image))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"record {index}: unsafe image path {relative}")
        resolved = (root / relative).resolve()
        if root not in resolved.parents:
            raise ValueError(f"record {index}: image escapes image root: {relative}")
        if not resolved.is_file():
            raise FileNotFoundError(f"record {index}: image does not exist: {resolved}")
        normalized.append(relative.as_posix())
    return normalized


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    upstream_dir = Path(args.upstream_dir).resolve()
    revision = git_revision(upstream_dir)
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"expected upstream {UPSTREAM_REVISION}, got {revision}")

    image_root = upstream_dir / "benchmark_data"
    source_manifest = image_root / "vqa_benchmark" / "mechvqa_benchmark.jsonl"
    manifest_sha = sha256(source_manifest)
    if manifest_sha != SOURCE_MANIFEST_SHA256:
        raise ValueError(f"unexpected source manifest SHA256: {manifest_sha}")

    records = read_records(source_manifest)
    if len(records) != SOURCE_RECORD_COUNT:
        raise ValueError(f"expected {SOURCE_RECORD_COUNT} records, got {len(records)}")
    if args.limit > len(records):
        raise ValueError(f"limit={args.limit} exceeds source size {len(records)}")

    all_images = set()
    for index, record in enumerate(records):
        all_images.update(validate_record(record, image_root, index))
    if len(all_images) != SOURCE_IMAGE_COUNT:
        raise ValueError(
            f"expected {SOURCE_IMAGE_COUNT} unique images, got {len(all_images)}"
        )

    selected = records[: args.limit or None]
    selected_images = {str(image) for record in selected for image in record["images"]}
    output_dir = Path(args.output_dir)
    output_manifest = output_dir / "mechvqa_public_test.jsonl"
    output_metadata = output_dir / "mechvqa_public_test.meta.json"
    write_jsonl(output_manifest, selected)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MechVQA",
        "split": "public_test",
        "upstream_url": "https://github.com/xiaofengShi/MechVQA",
        "upstream_revision": revision,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": manifest_sha,
        "image_root": str(image_root),
        "source_record_count": len(records),
        "source_unique_image_count": len(all_images),
        "selected_record_count": len(selected),
        "selected_unique_image_count": len(selected_images),
        "selection": "full" if not args.limit else f"first-{args.limit}",
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": sha256(output_manifest),
        "offline": True,
    }
    write_json(output_metadata, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
