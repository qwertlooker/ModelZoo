import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torchaudio

from diarizen.pipelines.inference import DiariZenPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="DiariZen inference")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", nargs="+")
    inputs.add_argument(
        "--manifest",
        help="JSONL manifest containing id and audio_path fields.",
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument(
        "--device", choices=["npu", "cpu", "cuda"], default="npu"
    )
    return parser.parse_args()


def read_manifest(path):
    manifest_path = Path(path)
    records = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "audio_path" not in row:
                raise ValueError(
                    f"{manifest_path}:{line_number}: id and audio_path are required"
                )
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            records.append((str(row["id"]), audio_path.resolve()))
    if not records:
        raise ValueError(f"Empty manifest: {manifest_path}")
    ids = [record[0] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in {manifest_path}")
    return records


def main():
    args = parse_args()
    if args.device == "npu":
        import torch_npu

    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    embedding_model = Path(args.embedding_model)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not embedding_model.is_file():
        raise FileNotFoundError(embedding_model)

    records = (
        read_manifest(args.manifest)
        if args.manifest
        else [(Path(value).stem, Path(value)) for value in args.audio]
    )
    for _, audio_path in records:
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

    pipeline = DiariZenPipeline(
        diarizen_hub=model_dir,
        embedding_model=str(embedding_model),
        rttm_out_dir=args.output_dir,
        device=device,
    )
    embedding_providers = pipeline._embedding.session_.get_providers()
    if args.device == "npu" and embedding_providers[0] != "CANNExecutionProvider":
        raise RuntimeError(
            f"Expected CANNExecutionProvider, got {embedding_providers}"
        )
    print(f"embedding_providers={embedding_providers}")
    start = time.perf_counter()
    audio_seconds = 0.0
    output_rows = []
    for session_id, audio_path in records:
        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate
        audio_seconds += duration
        result = pipeline(str(audio_path), sess_name=session_id)
        output_rows.append(
            {
                "id": session_id,
                "audio_path": str(audio_path),
                "duration": duration,
                "tracks": len(result),
            }
        )
        print(f"{audio_path}: tracks={len(result)}")
    elapsed = time.perf_counter() - start
    print(f"device={device} files={len(records)}")
    print(
        f"audio_seconds={audio_seconds:.3f} elapsed_seconds={elapsed:.3f} "
        f"rtf={elapsed / audio_seconds:.6f}"
    )
    output_dir = Path(args.output_dir)
    metadata = {
        "command": " ".join(sys.argv),
        "device": args.device,
        "embedding_providers": embedding_providers,
        "model_dir": str(model_dir),
        "embedding_model": str(embedding_model),
        "manifest": args.manifest,
        "files": len(records),
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed,
        "rtf": elapsed / audio_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "records": output_rows,
    }
    (output_dir / "run.meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
