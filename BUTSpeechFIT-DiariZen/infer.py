import argparse
import time
from pathlib import Path

import torch
import torchaudio

from diarizen.pipelines.inference import DiariZenPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="DiariZen inference")
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument(
        "--device", choices=["npu", "cpu", "cuda"], default="npu"
    )
    return parser.parse_args()


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

    audio_paths = [Path(value) for value in args.audio]
    for audio_path in audio_paths:
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

    pipeline = DiariZenPipeline(
        diarizen_hub=model_dir,
        embedding_model=str(embedding_model),
        rttm_out_dir=args.output_dir,
        device=device,
    )
    start = time.perf_counter()
    audio_seconds = 0.0
    for audio_path in audio_paths:
        info = torchaudio.info(str(audio_path))
        audio_seconds += info.num_frames / info.sample_rate
        result = pipeline(str(audio_path), sess_name=audio_path.stem)
        print(f"{audio_path}: tracks={len(result)}")
    elapsed = time.perf_counter() - start
    print(f"device={device} files={len(audio_paths)}")
    print(
        f"audio_seconds={audio_seconds:.3f} elapsed_seconds={elapsed:.3f} "
        f"rtf={elapsed / audio_seconds:.6f}"
    )


if __name__ == "__main__":
    main()
