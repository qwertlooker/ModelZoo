import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="MoLFormer feature extraction")
    parser.add_argument(
        "--model",
        default="ibm-research/MoLFormer-XL-both-10pct",
        help="Local model directory or Hugging Face repository.",
    )
    parser.add_argument("--smiles", nargs="*", default=[])
    parser.add_argument(
        "--input",
        help="UTF-8 text file containing one SMILES string per line.",
    )
    parser.add_argument(
        "--manifest",
        help="JSONL manifest containing id and smiles fields.",
    )
    parser.add_argument(
        "--device", choices=["npu", "cpu", "cuda"], default="npu"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="embeddings.jsonl")
    return parser.parse_args()


def load_records(values, input_path, manifest_path):
    records = [
        {"id": f"arg-{index:06d}", "smiles": value}
        for index, value in enumerate(values)
    ]
    if input_path:
        with Path(input_path).open(encoding="utf-8") as handle:
            records.extend(
                {"id": f"line-{index:06d}", "smiles": line.strip()}
                for index, line in enumerate(handle)
                if line.strip()
            )
    if manifest_path:
        with Path(manifest_path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "id" not in row or "smiles" not in row:
                    raise ValueError(
                        f"{manifest_path}:{line_number}: id and smiles are required"
                    )
                records.append({"id": str(row["id"]), "smiles": row["smiles"]})
    if not records:
        raise ValueError("Provide at least one SMILES string.")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate input ids.")
    return records


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.device == "npu":
        import torch_npu

    device = torch.device(args.device)
    records = load_records(args.smiles, args.input, args.manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model,
        deterministic_eval=True,
        trust_remote_code=True,
    ).eval()
    model.to(device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for offset in range(0, len(records), args.batch_size):
            batch = records[offset : offset + args.batch_size]
            inputs = tokenizer(
                [record["smiles"] for record in batch],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                embeddings = model(**inputs).pooler_output.float().cpu()
            for record, embedding in zip(batch, embeddings):
                handle.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "smiles": record["smiles"],
                            "embedding": embedding.tolist(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    elapsed = time.perf_counter() - start
    print(f"device={device} samples={len(records)}")
    print(
        f"elapsed_seconds={elapsed:.3f} "
        f"samples_per_second={len(records) / elapsed:.3f}"
    )
    print(f"output={output_path}")
    metadata = {
        "command": " ".join(sys.argv),
        "device": args.device,
        "model": args.model,
        "manifest": args.manifest,
        "manifest_sha256": sha256(args.manifest) if args.manifest else None,
        "sample_count": len(records),
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(records) / elapsed,
        "output_sha256": sha256(output_path),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
