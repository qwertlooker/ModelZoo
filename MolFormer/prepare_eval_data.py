#!/usr/bin/env python3
"""Create a deterministic SMILES manifest and inventory IBM split files."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a MoLFormer SMILES manifest.")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--smiles_file")
    sources.add_argument("--csv")
    sources.add_argument("--generate_l1_count", type=int)
    parser.add_argument("--smiles_column", default="smiles")
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--official_data_dir",
        help="Optional extracted IBM finetune_datasets directory to inventory.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_smiles(args: argparse.Namespace) -> list[str]:
    if args.generate_l1_count is not None:
        if args.generate_l1_count < 1 or args.generate_l1_count > 100:
            raise ValueError("--generate_l1_count must be between 1 and 100")
        candidates = (
            ["C" * length for length in range(1, 41)]
            + ["C" * length + "O" for length in range(1, 21)]
            + ["C" * length + "N" for length in range(1, 21)]
            + ["C" * length + "(=O)O" for length in range(1, 11)]
            + ["C" * length + "#N" for length in range(1, 11)]
        )
        if len(candidates) != len(set(candidates)):
            raise AssertionError("Generated duplicate SMILES")
        return candidates[: args.generate_l1_count]
    if args.smiles_file:
        path = Path(args.smiles_file)
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [value for value in values if value]
    path = Path(args.csv)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if args.smiles_column not in (reader.fieldnames or []):
            raise ValueError(
                f"{path}: missing SMILES column {args.smiles_column!r}; "
                f"columns={reader.fieldnames}"
            )
        return [
            row[args.smiles_column].strip()
            for row in reader
            if row[args.smiles_column].strip()
        ]


def inventory_csv_files(root: Path) -> list[dict]:
    inventory = []
    for path in sorted(root.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        inventory.append(
            {
                "path": str(path.relative_to(root)),
                "rows": rows,
                "columns": header,
                "sha256": sha256(path),
            }
        )
    if not inventory:
        raise ValueError(f"No CSV files found below {root}")
    return inventory


def main() -> None:
    args = parse_args()
    smiles = read_smiles(args)
    if args.limit:
        smiles = smiles[: args.limit]
    if not smiles:
        raise ValueError("No SMILES records found.")
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, value in enumerate(smiles):
            handle.write(
                json.dumps(
                    {"id": f"{index:06d}", "smiles": value},
                    ensure_ascii=False,
                )
                + "\n"
            )
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "sample_count": len(smiles),
        "limit": args.limit,
        "source": (
            args.smiles_file
            or args.csv
            or f"deterministic-linear-smiles:{args.generate_l1_count}"
        ),
        "manifest_sha256": sha256(output_path),
    }
    if args.official_data_dir:
        official_root = Path(args.official_data_dir)
        if not official_root.is_dir():
            raise NotADirectoryError(official_root)
        metadata["official_split_inventory"] = inventory_csv_files(official_root)
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
