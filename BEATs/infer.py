#!/usr/bin/env python3
"""BEATs inference script for Ascend NPU or CPU.

This file is maintained in the adaptation repository and is not part of the
upstream patch. Copy it to the upstream `beats/` directory after applying the
NPU fbank patch.
"""

import argparse
import time

import torch
import torchaudio

from BEATs import BEATs, BEATsConfig


def resolve_device(device_name: str) -> torch.device:
    if device_name == "npu":
        import torch_npu  # noqa: F401 - registers the NPU backend
    if device_name not in {"cpu", "cuda", "npu"}:
        raise ValueError("--device must be one of: cpu, cuda, npu")
    return torch.device(device_name)


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="BEATs inference on NPU/CPU/CUDA")
    parser.add_argument("--checkpoint", required=True, help="Path to a fine-tuned BEATs checkpoint")
    parser.add_argument("--wav", required=True, help="Path to a wav file; resampled to 16 kHz if needed")
    parser.add_argument("--device", default="npu", help="Explicit torch device. Default: npu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)

    audio_input_16khz, sample_rate = torchaudio.load(args.wav)
    if sample_rate != 16000:
        audio_input_16khz = torchaudio.functional.resample(audio_input_16khz, sample_rate, 16000)
    padding_mask = torch.zeros_like(audio_input_16khz).bool()
    audio_input_16khz = audio_input_16khz.to(device)
    padding_mask = padding_mask.to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
        synchronize(device)

        start = time.time()
        probs = None
        for _ in range(args.repeat):
            probs = model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
        synchronize(device)
        elapsed = time.time() - start

    print(f"Elapsed: {elapsed:.4f}s, repeat={args.repeat}, avg={elapsed / max(args.repeat, 1):.4f}s")
    if probs is None:
        return
    label_dict = checkpoint.get("label_dict")
    for i, (topk_prob, topk_idx) in enumerate(zip(*probs.topk(k=args.topk))):
        labels = [label_dict[idx.item()] if label_dict else str(idx.item()) for idx in topk_idx]
        print(f"Top {args.topk} labels of audio {i}: {labels}, probs={topk_prob.cpu().tolist()}")


if __name__ == "__main__":
    main()
