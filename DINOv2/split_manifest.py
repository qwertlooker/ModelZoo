"""Split an image manifest into bounded shards for full validation."""
import argparse
import csv
import json
from pathlib import Path

import common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-size", type=int, default=256)
    args = p.parse_args()
    if args.shard_size < 1:
        raise ValueError("shard-size must be positive")
    original = Path(args.manifest).resolve(strict=True)
    with original.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["path", "label"]:
            raise ValueError("manifest must have path,label columns")
        rows = list(reader)
    for row in rows:
        path = Path(row["path"])
        row["path"] = str((path if path.is_absolute() else original.parent / path).resolve())
    if not rows or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("manifest is empty or has duplicate images")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    for offset in range(0, len(rows), args.shard_size):
        chunk = rows[offset:offset + args.shard_size]
        name = f"shard-{len(shards):05d}.csv"
        path = output / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(chunk)
        shards.append({"name": name, "count": len(chunk), "sha256": common.digest(path)})
    index = {"manifest_sha256": common.digest(args.manifest), "count": len(rows), "shards": shards}
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(rows), "shards": len(shards)}))


if __name__ == "__main__":
    main()
