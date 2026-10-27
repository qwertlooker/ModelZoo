"""Combine sharded top-k results after validating data and result provenance."""
import argparse
import csv
import json
import math
from pathlib import Path

import common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--expected-count", type=int, default=50000)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    import numpy as np
    shard_dir, work = Path(args.shard_dir), Path(args.work_dir)
    index = json.loads((shard_dir / "index.json").read_text(encoding="utf-8"))
    common.checked_file(args.manifest, index["manifest_sha256"])
    if index["count"] != args.expected_count or args.expected_count < 1:
        raise ValueError("manifest count does not match expected count")
    original = Path(args.manifest).resolve(strict=True)
    with original.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["path", "label"]:
            raise ValueError("invalid original manifest")
        original_rows = list(reader)
    for row in original_rows:
        path = Path(row["path"])
        row["path"] = str((path if path.is_absolute() else original.parent / path).resolve())
    total, top1, top5 = 0, 0.0, 0.0
    all_rows = []
    baseline = None
    seen = set()
    for shard in index["shards"]:
        name = shard["name"]
        if Path(name).name != name or not name.endswith(".csv") or name in seen:
            raise ValueError("invalid or duplicate shard name")
        seen.add(name)
        manifest = shard_dir / name
        common.checked_file(manifest, shard["sha256"])
        with manifest.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["path", "label"]:
                raise ValueError("invalid shard manifest")
            all_rows.extend(reader)
        stem = Path(name).stem
        inputs = work / "inputs" / (stem + ".npz")
        result = work / "results" / (stem + ".npz")
        score = json.loads((work / "scores" / (stem + ".json")).read_text(encoding="utf-8"))
        _, result_meta = common.read_prediction(result)
        identity = {k: result_meta[k] for k in ("model", "model_sha256", "config_sha256", "head_sha256", "dtype", "task", "device")}
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise ValueError("shards use different model, head, task or device")
        with np.load(inputs, allow_pickle=False) as data:
            prep = json.loads(str(data["preparation"].item()))
            count = len(data["targets"])
        if (prep["manifest_sha256"] != shard["sha256"] or count != shard["count"]
                or score["samples"] != count or score["input_sha256"] != common.digest(inputs)
                or score["results_sha256"] != common.digest(result)):
            raise ValueError(f"shard provenance mismatch: {name}")
        if any(not math.isfinite(score[k]) or not 0 <= score[k] <= 1 for k in ("top1", "top5")):
            raise ValueError("invalid classification score")
        total += count
        top1 += score["top1"] * count
        top5 += score["top5"] * count
    if total != args.expected_count or all_rows != original_rows:
        raise ValueError("incomplete shard set")
    report = {"samples": total, "top1": top1 / total, "top5": top5 / total,
              "manifest_sha256": index["manifest_sha256"], "shards": len(seen), "identity": baseline}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
