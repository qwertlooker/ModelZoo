#!/usr/bin/env python3
"""MOSS-TTSD-v0.5 CPU/NPU inference entry.

This script is maintained by the ModelZoo adaptation and is not part of an
upstream patch set.  It uses the official Hugging Face remote-code model and
codec snapshots for MOSS-TTSD-v0.5, defaults to ``--device npu``, and leaves the
actual accelerator index to ``ASCEND_RT_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torchaudio
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor

MODEL_REPO = "OpenMOSS-Team/MOSS-TTSD-v0.5"
CODEC_REPO = "OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf"
MODEL_REVISION = "8527b9136b6afefe2252ae597cecea2e80e7ebeb"
CODEC_REVISION = "c884072fd69ed00b72cd0d43355c06341c4f51a6"


def resolve_device(device_name: str) -> torch.device:
    """Create a torch.device after registering the requested backend."""
    if device_name not in {"npu", "cpu", "cuda"}:
        raise ValueError("--device must be one of: npu, cpu, cuda")
    if device_name == "npu":
        import torch_npu  # noqa: F401  # Registers the NPU backend with torch.

        if not torch.npu.is_available():
            raise RuntimeError("NPU device requested, but torch.npu.is_available() is false")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    return torch.device(device_name)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError("--dtype must be one of: float32, float16, bfloat16")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["text"]
            if "prompt_audio" in item and "prompt_text" in item:
                item["prompt_audio"]
                item["prompt_text"]
            elif "prompt_audio_speaker1" in item and "prompt_text_speaker1" in item:
                item["prompt_audio_speaker1"]
                item["prompt_text_speaker1"]
            else:
                raise KeyError(
                    f"{path}:{line_no} must contain text plus either "
                    "prompt_audio/prompt_text or prompt_audio_speaker1/prompt_text_speaker1"
                )
            items.append(item)
    if not items:
        raise ValueError(f"empty input_jsonl: {path}")
    return items


def build_single_item(args: argparse.Namespace) -> dict[str, Any]:
    if args.text is None or args.prompt_audio is None or args.prompt_text is None:
        raise ValueError("single-sample mode requires --text, --prompt_audio, and --prompt_text")
    item = {
        "text": args.text,
        "prompt_audio": args.prompt_audio,
        "prompt_text": args.prompt_text,
    }
    if args.base_path is not None:
        item["base_path"] = args.base_path
    return item


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_snapshot(path_or_repo: str, revision: str, local_files_only: bool) -> str:
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return str(path)
    return snapshot_download(path_or_repo, revision=revision, local_files_only=local_files_only)


def load_processor_and_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any, int]:
    dtype = resolve_dtype(args.dtype)
    model_source = resolve_snapshot(args.model_path, args.model_revision, args.local_files_only)
    codec_source = resolve_snapshot(args.codec_path, args.codec_revision, args.local_files_only)
    processor = AutoProcessor.from_pretrained(
        model_source,
        codec_path=codec_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    processor.audio_tokenizer.eval()

    model = AutoModel.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()

    sample_rate = int(processor.audio_tokenizer.config.output_sample_rate)
    return processor, model, sample_rate


def move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def save_audio_fragments(
    audios: list[list[torch.Tensor]],
    output_dir: Path,
    sample_rate: int,
    offset: int,
) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    for local_idx, fragments in enumerate(audios):
        sample_idx = offset + local_idx
        sample_outputs: list[str] = []
        sample_seconds = 0.0
        for fragment_idx, fragment in enumerate(fragments):
            audio = fragment.detach().cpu().to(torch.float32)
            out_path = output_dir / f"sample_{sample_idx:04d}_{fragment_idx:02d}.wav"
            torchaudio.save(str(out_path), audio, sample_rate)
            sample_outputs.append(str(out_path))
            sample_seconds += float(audio.shape[-1]) / float(sample_rate)
        saved.append(
            {
                "sample_index": sample_idx,
                "audio_files": sample_outputs,
                "generated_audio_seconds": sample_seconds,
            }
        )
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MOSS-TTSD-v0.5 on CPU/NPU")
    parser.add_argument("--model_path", default=MODEL_REPO, help="HF repo id or local MOSS-TTSD-v0.5 snapshot")
    parser.add_argument("--codec_path", default=CODEC_REPO, help="HF repo id or local XY tokenizer snapshot")
    parser.add_argument("--model_revision", default=MODEL_REVISION, help="MOSS-TTSD-v0.5 commit/revision to load")
    parser.add_argument("--codec_revision", default=CODEC_REVISION, help="XY tokenizer codec commit/revision to load")
    parser.add_argument("--input_jsonl", "--jsonl", dest="input_jsonl", help="JSONL input file")
    parser.add_argument("--text", help="Single-sample generated dialogue text")
    parser.add_argument("--prompt_audio", help="Single-sample shared prompt audio path")
    parser.add_argument("--prompt_text", help="Single-sample prompt transcript with [S1]/[S2] tags")
    parser.add_argument("--base_path", help="Optional base path used by the processor for relative audio paths")
    parser.add_argument("--output_dir", "--save_dir", dest="output_dir", default="MOSS-TTSD-v0.5/outputs")
    parser.add_argument("--device", default="npu", choices=["npu", "cpu", "cuda"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=15000)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_normalize", action="store_true", help="Use the official processor text normalization path")
    parser.add_argument("--silence_duration", type=float, default=0.0)
    parser.add_argument("--local_files_only", action="store_true", help="Require local model/codec snapshots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    if args.input_jsonl:
        items = read_jsonl(Path(args.input_jsonl).expanduser())
    else:
        items = [build_single_item(args)]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, model, sample_rate = load_processor_and_model(args, device)

    manifest_rows: list[dict[str, Any]] = []
    total_audio_seconds = 0.0
    synchronize_device(device)
    started = time.perf_counter()

    generated_count = 0
    for batch_items in iter_batches(items, args.batch_size):
        inputs = processor(
            batch_items,
            return_tensors="pt",
            padding=True,
            use_normalize=args.text_normalize,
            silence_duration=args.silence_duration,
        )
        inputs = move_inputs_to_device(inputs, device)
        with torch.inference_mode():
            token_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        texts, audios = processor.batch_decode(token_ids, skip_special_tokens=True)
        saved = save_audio_fragments(audios, output_dir, sample_rate, offset=generated_count)
        for local_idx, row in enumerate(saved):
            row["generated_text"] = texts[local_idx]
            row["input_text"] = batch_items[local_idx]["text"]
            row["model_path"] = args.model_path
            row["model_revision"] = args.model_revision
            row["codec_path"] = args.codec_path
            row["codec_revision"] = args.codec_revision
            manifest_rows.append(row)
            total_audio_seconds += float(row["generated_audio_seconds"])
        generated_count += len(batch_items)

    synchronize_device(device)
    elapsed = time.perf_counter() - started

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "codec_path": args.codec_path,
        "codec_revision": args.codec_revision,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "batch_size": args.batch_size,
        "sample_rate": sample_rate,
        "num_samples": len(items),
        "elapsed_seconds": elapsed,
        "generated_audio_seconds": total_audio_seconds,
        "rtf": elapsed / total_audio_seconds if total_audio_seconds > 0 else None,
        "rtfx": total_audio_seconds / elapsed if elapsed > 0 else None,
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "output_manifest": str(manifest_path),
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
