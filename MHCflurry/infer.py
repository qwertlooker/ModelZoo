# Licensed under the Apache License, Version 2.0.
"""Explicit CPU/CUDA/NPU inference; --dry-run never loads a model."""

import argparse
import os
from pathlib import Path

from adapter_utils import WEIGHT_NAME, positive_int, read_inputs, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--weight-archive", default=str(Path("weights") / WEIGHT_NAME))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("npu", "cpu", "cuda"), default="npu")
    parser.add_argument(
        "--task",
        choices=("affinity", "processing", "presentation"),
        default="presentation",
    )
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be >= 3")
    rows = read_inputs(args.manifest, args.task)
    if Path(args.output_dir).exists():
        raise FileExistsError(args.output_dir)
    if args.dry_run:
        print(
            {
                "simulation": True,
                "rows": len(rows),
                "device": args.device,
                "task": args.task,
                "manifest_sha256": sha256(args.manifest),
                "batch_size": args.batch_size,
                "runs": args.runs,
            }
        )
        return
    os.environ["MHCFLURRY_OPTIMIZATION_LEVEL"] = "0"
    os.environ["MHCFLURRY_TORCH_COMPILE"] = "0"
    from runtime import run

    run(args, rows)


if __name__ == "__main__":
    main()
