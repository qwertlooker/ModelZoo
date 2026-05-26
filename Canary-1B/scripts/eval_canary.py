#!/usr/bin/env python3
"""Evaluate Canary-1B from already prepared JSONL manifests.

Keep this script intentionally small: data preparation is handled by
prepare_eval_data.py; inference uses the same NeMo model.transcribe() mechanism
as Canary-1B/infer.py; this file only adds metric calculation and result dumps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from whisper.normalizers import EnglishTextNormalizer

import nemo
import torch
from jiwer import wer
from nemo.collections.asr.models import EncDecMultiTaskModel
from sacrebleu import corpus_bleu

DEFAULT_MANIFESTS = [
    "Canary-1B/eval_data/librispeech_test_clean/manifest_asr_en.jsonl",
    "Canary-1B/eval_data/mls_test_german/manifest_asr_de.jsonl",
    "Canary-1B/eval_data/mls_test_spanish/manifest_asr_es.jsonl",
    "Canary-1B/eval_data/mls_test_french/manifest_asr_fr.jsonl",
    "Canary-1B/eval_data/fleurs/en-de/manifest_ast_en_de.jsonl",
    "Canary-1B/eval_data/fleurs/en-es/manifest_ast_en_es.jsonl",
    "Canary-1B/eval_data/fleurs/en-fr/manifest_ast_en_fr.jsonl",
    "Canary-1B/eval_data/fleurs/de-en/manifest_ast_de_en.jsonl",
    "Canary-1B/eval_data/fleurs/es-en/manifest_ast_es_en.jsonl",
    "Canary-1B/eval_data/fleurs/fr-en/manifest_ast_fr_en.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Canary-1B using prepared manifests")
    parser.add_argument("--model", required=True, help="Local .nemo file, model directory, or HF model id")
    parser.add_argument("--manifest", nargs="+", help="Prepared JSONL manifest(s). Defaults to all standard manifests.")
    parser.add_argument("--device", default="npu", choices=["npu", "cpu", "cuda"])
    parser.add_argument("--output_dir", default="Canary-1B/eval_results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        raise ValueError(f"empty manifest: {path}")
    return items


def tag_from_manifest(path: Path, items: list[dict[str, Any]]) -> str:
    first = items[0]
    task = first.get("taskname", "task")
    src = first.get("source_lang", "src")
    tgt = first.get("target_lang", "tgt")
    parent = path.parent.name.replace("-", "_")
    if task == "asr":
        return f"asr_{parent}_{src}"
    return f"ast_{src}_{tgt}"


def resolve_device(device_name: str):
    if device_name == "npu":
        import torch_npu  # noqa: F401
    return torch.device(device_name)


def load_model(args: argparse.Namespace):
    device = resolve_device(args.device)
    model_path = Path(args.model).expanduser()
    if model_path.is_file() and model_path.suffix == ".nemo":
        model = EncDecMultiTaskModel.restore_from(str(model_path), map_location=device)
    elif model_path.is_dir() and (model_path / "canary-1b.nemo").is_file():
        model = EncDecMultiTaskModel.restore_from(str(model_path / "canary-1b.nemo"), map_location=device)
    else:
        model = EncDecMultiTaskModel.from_pretrained(args.model, map_location=device)
    model.eval()
    model.to(device)
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = args.beam_size
    model.change_decoding_strategy(decode_cfg)
    return model


def extract_text(item: Any) -> str:
    """Return the expected NeMo transcription text field."""
    return str(item.text)


_WER_NORMALIZER = EnglishTextNormalizer()


def normalize_for_wer(text: str) -> str:
    """Normalize ASR text with the official Whisper EnglishTextNormalizer."""
    return _WER_NORMALIZER(text)


def compute_metrics(taskname: str, references: list[str], hypotheses: list[str]) -> dict[str, float]:
    if taskname == "asr":
        refs = [normalize_for_wer(x) for x in references]
        hyps = [normalize_for_wer(x) for x in hypotheses]
        value = float(wer(refs, hyps))
        return {"wer": value, "wer_percent": value * 100.0}
    return {"bleu": float(corpus_bleu(hypotheses, [references]).score)}


def env_report(args: argparse.Namespace) -> dict[str, Any]:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }
    report["torch"] = torch.__version__
    report["nemo"] = nemo.__version__
    return report


def evaluate_manifest(model: Any, manifest: Path, args: argparse.Namespace) -> dict[str, Any]:
    items = read_manifest(manifest)
    tag = tag_from_manifest(manifest, items)
    started = time.time()
    outputs = model.transcribe(audio=str(manifest), batch_size=args.batch_size)
    elapsed = time.time() - started

    hypotheses = [extract_text(x) for x in outputs]
    references = [str(x.get("answer", "")) for x in items]
    taskname = str(items[0].get("taskname", "asr"))
    metrics: dict[str, Any] = compute_metrics(taskname, references, hypotheses)
    total_audio = sum(float(x.get("duration", 0.0)) for x in items)
    metrics.update(
        {
            "tag": tag,
            "manifest": str(manifest),
            "num_samples": len(items),
            "audio_seconds": total_audio,
            "elapsed_seconds": elapsed,
            "rtf": elapsed / total_audio if total_audio > 0 else None,
        }
    )

    output_dir = Path(args.output_dir)
    pred_path = output_dir / f"{tag}.tsv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sample_id", "audio_path", "duration", "reference", "hypothesis"])
        for item, hyp in zip(items, hypotheses):
            writer.writerow(
                [
                    item.get("sample_id", ""),
                    item.get("audio_filepath", ""),
                    item.get("duration", ""),
                    item.get("answer", ""),
                    hyp,
                ]
            )

    with (output_dir / f"{tag}.metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    args = parse_args()
    manifests = [Path(x) for x in (args.manifest or DEFAULT_MANIFESTS)]
    missing = [str(x) for x in manifests if not x.is_file()]
    if missing:
        raise FileNotFoundError("Missing manifest(s). Run prepare_eval_data.py first or pass --manifest. Missing: " + ", ".join(missing))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_env.json").open("w", encoding="utf-8") as f:
        json.dump(env_report(args), f, ensure_ascii=False, indent=2)

    model = load_model(args)
    all_metrics = [evaluate_manifest(model, manifest, args) for manifest in manifests]
    with (output_dir / "summary.metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
