# Licensed under the Apache License, Version 2.0.
"""Validate a local frozen cohort without loading models or downloading data."""

import argparse
import csv
from pathlib import Path

from adapter_utils import presentation_groups, read_inputs, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task",
        choices=("affinity", "processing", "presentation"),
        default="presentation",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--purpose", choices=("smoke", "l2"), required=True)
    args = parser.parse_args()
    rows = read_inputs(args.input, args.task)
    if args.purpose == "l2" and len(rows) < 1000:
        raise ValueError(
            "L2 requires a frozen representative cohort of at least 1000 rows"
        )
    if args.purpose == "l2" and args.task == "presentation":
        presentation_groups(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "purpose": args.purpose,
        "task": args.task,
        "rows": len(rows),
        "source_sha256": sha256(args.input),
        "manifest_sha256": sha256(manifest),
        "offline": True,
        "limit": None,
        "flanks": "n_flank" in rows[0],
    }
    write_json(output / "manifest.meta.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
