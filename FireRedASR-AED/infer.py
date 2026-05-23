#!/usr/bin/env python3
"""FireRedASR-AED NPU inference script kept outside upstream patch.

Copy this file to the root of an applied FireRedASR upstream checkout, or run it
from this directory after setting PYTHONPATH to the upstream checkout.
"""

import argparse

import torch_npu

from fireredasr.models.fireredasr import FireRedAsr


def main() -> None:
    parser = argparse.ArgumentParser(description="FireRedASR-AED inference on CPU/CUDA/NPU")
    parser.add_argument("--model_dir", default="pretrained_models/FireRedASR-AED-L")
    parser.add_argument("--wav_path", default="examples/wav/BAC009S0764W0121.wav")
    parser.add_argument("--uttid", default="BAC009S0764W0121")
    parser.add_argument("--device", default="npu", help="npu, cpu, or cuda")
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=0)
    parser.add_argument("--softmax_smoothing", type=float, default=1.25)
    parser.add_argument("--aed_length_penalty", type=float, default=0.6)
    parser.add_argument("--eos_penalty", type=float, default=1.0)
    args = parser.parse_args()

    model = FireRedAsr.from_pretrained("aed", args.model_dir)
    results = model.transcribe(
        [args.uttid],
        [args.wav_path],
        {
            "device": args.device,
            "beam_size": args.beam_size,
            "nbest": args.nbest,
            "decode_max_len": args.decode_max_len,
            "softmax_smoothing": args.softmax_smoothing,
            "aed_length_penalty": args.aed_length_penalty,
            "eos_penalty": args.eos_penalty,
        },
    )
    print(results)


if __name__ == "__main__":
    main()
