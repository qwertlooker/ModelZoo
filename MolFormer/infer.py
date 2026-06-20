import argparse
import json
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
        "--device", choices=["npu", "cpu", "cuda"], default="npu"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="embeddings.jsonl")
    return parser.parse_args()


def load_smiles(values, input_path):
    smiles = list(values)
    if input_path:
        with Path(input_path).open(encoding="utf-8") as handle:
            smiles.extend(line.strip() for line in handle if line.strip())
    if not smiles:
        raise ValueError("Provide at least one SMILES string.")
    return smiles


def main():
    args = parse_args()
    if args.device == "npu":
        import torch_npu

    device = torch.device(args.device)
    smiles = load_smiles(args.smiles, args.input)
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
        for offset in range(0, len(smiles), args.batch_size):
            batch = smiles[offset : offset + args.batch_size]
            inputs = tokenizer(batch, padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                embeddings = model(**inputs).pooler_output.float().cpu()
            for value, embedding in zip(batch, embeddings):
                handle.write(
                    json.dumps(
                        {"smiles": value, "embedding": embedding.tolist()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    elapsed = time.perf_counter() - start
    print(f"device={device} samples={len(smiles)}")
    print(
        f"elapsed_seconds={elapsed:.3f} "
        f"samples_per_second={len(smiles) / elapsed:.3f}"
    )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
