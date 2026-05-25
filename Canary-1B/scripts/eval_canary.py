#!/usr/bin/env python3
"""Evaluate Canary-1B ASR/AST on LibriSpeech and FLEURS subsets.

The script downloads/prepares small deterministic subsets, runs NeMo
EncDecMultiTaskModel.transcribe(), and writes metrics plus per-sample outputs.

Examples:
  # 30 min LibriSpeech test-clean ASR on NPU
  ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
    --task asr --device npu --model Canary-1B/weights/canary-1b.nemo \
    --librispeech_minutes 30 --batch_size 1 --beam_size 5

  # FLEURS 50 samples per AST direction on NPU
  ASCEND_RT_VISIBLE_DEVICES=0 python Canary-1B/scripts/eval_canary.py \
    --task ast --device npu --model Canary-1B/weights/canary-1b.nemo \
    --fleurs_limit 50 --batch_size 1 --beam_size 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
from tqdm import tqdm

FLEURS_CONFIG = {
    "en": "en_us",
    "de": "de_de",
    "es": "es_419",
    "fr": "fr_fr",
}

DEFAULT_AST_DIRECTIONS = ["en-de", "en-es", "en-fr", "de-en", "es-en", "fr-en"]


@dataclass
class Sample:
    sample_id: str
    audio_path: str
    duration: float
    reference: str
    taskname: str
    source_lang: str
    target_lang: str
    pnc: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canary-1B LibriSpeech/FLEURS evaluator")
    parser.add_argument("--model", required=True, help="Local .nemo file, model directory, or HF model id")
    parser.add_argument("--device", default="npu", choices=["npu", "cpu", "cuda"])
    parser.add_argument("--task", default="all", choices=["asr", "ast", "all"])
    parser.add_argument("--data_dir", default="Canary-1B/eval_data")
    parser.add_argument("--output_dir", default="Canary-1B/eval_results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking subsets")
    parser.add_argument("--prepare_only", action="store_true", help="Only download/write manifests; skip inference")

    parser.add_argument("--librispeech_dataset", default="openslr/librispeech_asr")
    parser.add_argument("--librispeech_config", default="clean")
    parser.add_argument("--librispeech_split", default="test")
    parser.add_argument("--librispeech_minutes", type=float, default=30.0, help="0 means no minute cap")
    parser.add_argument("--librispeech_limit", type=int, default=0, help="0 means no item-count cap")
    parser.add_argument("--asr_pnc", default="no", choices=["yes", "no"], help="Use no for WER-style ASR eval")

    parser.add_argument("--fleurs_dataset", default="google/fleurs")
    parser.add_argument("--fleurs_split", default="test")
    parser.add_argument("--fleurs_limit", type=int, default=50, help="Samples per AST direction; 0 means full split")
    parser.add_argument("--ast_directions", default=",".join(DEFAULT_AST_DIRECTIONS))
    parser.add_argument("--ast_pnc", default="yes", choices=["yes", "no"], help="Use yes for BLEU with punctuation/case")
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "sample"


def write_wav(audio: dict[str, Any], path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = audio["array"]
    sr = int(audio["sampling_rate"])
    # Canary expects 16 kHz. HF LibriSpeech/FLEURS are normally 16 kHz; resample only if needed.
    if sr != 16000:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        sr = 16000
    sf.write(path, array, sr)
    return float(len(array)) / float(sr)


def dataset_iter(dataset: Any, shuffle: bool, seed: int) -> Any:
    if shuffle:
        return dataset.shuffle(seed=seed)
    return dataset


def prepare_librispeech(args: argparse.Namespace) -> tuple[Path, list[Sample]]:
    from datasets import load_dataset

    ds = load_dataset(args.librispeech_dataset, args.librispeech_config, split=args.librispeech_split)
    ds = dataset_iter(ds, args.shuffle, args.seed)
    out_dir = Path(args.data_dir) / "librispeech_test_clean"
    samples: list[Sample] = []
    max_seconds = args.librispeech_minutes * 60.0 if args.librispeech_minutes > 0 else math.inf
    total_seconds = 0.0

    for row in tqdm(ds, desc="prepare LibriSpeech"):
        if args.librispeech_limit and len(samples) >= args.librispeech_limit:
            break
        if total_seconds >= max_seconds:
            break
        sid = str(row.get("id") or row.get("file") or len(samples))
        wav_path = out_dir / "wav" / f"{safe_name(sid)}.wav"
        duration = write_wav(row["audio"], wav_path)
        total_seconds += duration
        samples.append(
            Sample(
                sample_id=sid,
                audio_path=str(wav_path),
                duration=duration,
                reference=str(row.get("text") or row.get("transcription") or ""),
                taskname="asr",
                source_lang="en",
                target_lang="en",
                pnc=args.asr_pnc,
            )
        )

    manifest = out_dir / "manifest_asr_en.jsonl"
    write_manifest(manifest, samples)
    return manifest, samples


def fleurs_text(row: dict[str, Any], pnc: str) -> str:
    if pnc == "yes":
        return str(row.get("raw_transcription") or row.get("transcription") or "")
    return str(row.get("transcription") or row.get("raw_transcription") or "")


def load_fleurs_by_lang(dataset_name: str, split: str, lang: str) -> Any:
    from datasets import load_dataset

    return load_dataset(dataset_name, FLEURS_CONFIG[lang], split=split)


def prepare_fleurs_direction(args: argparse.Namespace, src: str, tgt: str) -> tuple[Path, list[Sample]]:
    src_ds = load_fleurs_by_lang(args.fleurs_dataset, args.fleurs_split, src)
    tgt_ds = load_fleurs_by_lang(args.fleurs_dataset, args.fleurs_split, tgt)
    if args.shuffle:
        src_ds = src_ds.shuffle(seed=args.seed)
    tgt_by_id = {str(row["id"]): row for row in tgt_ds}
    out_dir = Path(args.data_dir) / "fleurs" / f"{src}-{tgt}"
    samples: list[Sample] = []

    for row in tqdm(src_ds, desc=f"prepare FLEURS {src}->{tgt}"):
        if args.fleurs_limit and len(samples) >= args.fleurs_limit:
            break
        sid = str(row["id"])
        if sid not in tgt_by_id:
            continue
        wav_path = out_dir / "wav" / f"{safe_name(sid)}.wav"
        duration = write_wav(row["audio"], wav_path)
        ref = fleurs_text(tgt_by_id[sid], args.ast_pnc)
        samples.append(
            Sample(
                sample_id=sid,
                audio_path=str(wav_path),
                duration=duration,
                reference=ref,
                taskname="ast",
                source_lang=src,
                target_lang=tgt,
                pnc=args.ast_pnc,
            )
        )

    manifest = out_dir / f"manifest_ast_{src}_{tgt}.jsonl"
    write_manifest(manifest, samples)
    return manifest, samples


def write_manifest(path: Path, samples: Iterable[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            item = {
                "audio_filepath": s.audio_path,
                "duration": s.duration,
                "answer": s.reference,
                "taskname": s.taskname,
                "source_lang": s.source_lang,
                "target_lang": s.target_lang,
                "pnc": s.pnc,
                "sample_id": s.sample_id,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def resolve_device(device_name: str):
    import torch

    if device_name == "npu":
        import torch_npu  # noqa: F401
    return torch.device(device_name)


def load_model(args: argparse.Namespace):
    from nemo.collections.asr.models import EncDecMultiTaskModel

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
    if hasattr(decode_cfg, "beam") and hasattr(decode_cfg.beam, "beam_size"):
        decode_cfg.beam.beam_size = args.beam_size
        model.change_decoding_strategy(decode_cfg)
    return model


def extract_text(item: Any) -> str:
    if hasattr(item, "text"):
        return str(item.text)
    if isinstance(item, dict) and "text" in item:
        return str(item["text"])
    return str(item)


def normalize_for_wer(text: str) -> str:
    try:
        from whisper.normalizers import EnglishTextNormalizer

        return EnglishTextNormalizer()(text)
    except Exception:
        # Fallback normalization: lower, remove punctuation, collapse whitespace.
        text = text.lower()
        text = re.sub(r"[^\w\s']+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def compute_metrics(taskname: str, references: list[str], hypotheses: list[str]) -> dict[str, Any]:
    if taskname == "asr":
        from jiwer import wer

        refs = [normalize_for_wer(x) for x in references]
        hyps = [normalize_for_wer(x) for x in hypotheses]
        return {"wer": float(wer(refs, hyps)), "wer_percent": float(wer(refs, hyps) * 100.0)}
    from sacrebleu import corpus_bleu

    bleu = corpus_bleu(hypotheses, [references]).score
    return {"bleu": float(bleu)}


def run_one(model: Any, manifest: Path, samples: list[Sample], args: argparse.Namespace, tag: str) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    outputs = model.transcribe(audio=str(manifest), batch_size=args.batch_size)
    elapsed = time.time() - started
    hypotheses = [extract_text(x) for x in outputs]
    references = [s.reference for s in samples]
    metrics = compute_metrics(samples[0].taskname if samples else "asr", references, hypotheses) if samples else {}
    total_audio = sum(s.duration for s in samples)
    metrics.update(
        {
            "tag": tag,
            "num_samples": len(samples),
            "audio_seconds": total_audio,
            "elapsed_seconds": elapsed,
            "rtf": elapsed / total_audio if total_audio > 0 else None,
        }
    )

    pred_path = output_dir / f"{tag}.tsv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sample_id", "audio_path", "duration", "reference", "hypothesis"])
        for s, hyp in zip(samples, hypotheses):
            writer.writerow([s.sample_id, s.audio_path, f"{s.duration:.6f}", s.reference, hyp])

    metrics_path = output_dir / f"{tag}.metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


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
    try:
        import torch

        report["torch"] = torch.__version__
    except Exception as exc:
        report["torch_error"] = str(exc)
    try:
        import nemo

        report["nemo"] = getattr(nemo, "__version__", None)
    except Exception as exc:
        report["nemo_error"] = str(exc)
    return report


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.output_dir) / "run_env.json").open("w", encoding="utf-8") as f:
        json.dump(env_report(args), f, ensure_ascii=False, indent=2)

    jobs: list[tuple[str, Path, list[Sample]]] = []
    if args.task in {"asr", "all"}:
        manifest, samples = prepare_librispeech(args)
        jobs.append(("asr_librispeech_test_clean_en", manifest, samples))
    if args.task in {"ast", "all"}:
        for direction in [x.strip() for x in args.ast_directions.split(",") if x.strip()]:
            src, tgt = direction.split("-", 1)
            if src not in FLEURS_CONFIG or tgt not in FLEURS_CONFIG:
                raise ValueError(f"Unsupported FLEURS direction: {direction}")
            manifest, samples = prepare_fleurs_direction(args, src, tgt)
            jobs.append((f"ast_fleurs_{src}_{tgt}", manifest, samples))

    if args.prepare_only:
        print(f"Prepared {len(jobs)} manifests under {args.data_dir}; inference skipped.")
        return

    model = load_model(args)
    all_metrics = [run_one(model, manifest, samples, args, tag) for tag, manifest, samples in jobs]
    with (Path(args.output_dir) / "summary.metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
