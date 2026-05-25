#!/usr/bin/env python3
"""Prepare LibriSpeech/FLEURS manifests for Canary-1B evaluation.

This script only downloads/converts data and writes JSONL manifests. It does not
load the model or run inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
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
class ManifestItem:
    audio_filepath: str
    duration: float
    answer: str
    taskname: str
    source_lang: str
    target_lang: str
    pnc: str
    sample_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Canary-1B LibriSpeech/FLEURS eval data")
    parser.add_argument("--task", default="all", choices=["asr", "ast", "all"])
    parser.add_argument("--data_dir", default="Canary-1B/eval_data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking subsets")

    parser.add_argument("--librispeech_dataset", default="openslr/librispeech_asr")
    parser.add_argument("--librispeech_config", default="clean")
    parser.add_argument("--librispeech_split", default="test")
    parser.add_argument("--librispeech_minutes", type=float, default=30.0, help="0 means full split/no minute cap")
    parser.add_argument("--librispeech_limit", type=int, default=0, help="0 means no item-count cap")
    parser.add_argument("--asr_pnc", default="no", choices=["yes", "no"])

    parser.add_argument("--fleurs_dataset", default="google/fleurs")
    parser.add_argument("--fleurs_split", default="test")
    parser.add_argument("--fleurs_limit", type=int, default=50, help="Samples per AST direction; 0 means full split")
    parser.add_argument("--ast_directions", default=",".join(DEFAULT_AST_DIRECTIONS))
    parser.add_argument("--ast_pnc", default="yes", choices=["yes", "no"])
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "sample"


def write_wav(audio: dict[str, Any], path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = audio["array"]
    sr = int(audio["sampling_rate"])
    if sr != 16000:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        sr = 16000
    sf.write(path, array, sr)
    return float(len(array)) / float(sr)


def write_manifest(path: Path, items: Iterable[ManifestItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} items: {path}")


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"wrote metadata: {meta_path}")


def maybe_shuffle(ds: Any, shuffle: bool, seed: int) -> Any:
    return ds.shuffle(seed=seed) if shuffle else ds


def prepare_librispeech(args: argparse.Namespace) -> Path:
    from datasets import load_dataset

    print(
        f"loading LibriSpeech dataset={args.librispeech_dataset} "
        f"config={args.librispeech_config} split={args.librispeech_split}"
    )
    ds = load_dataset(args.librispeech_dataset, args.librispeech_config, split=args.librispeech_split)
    ds = maybe_shuffle(ds, args.shuffle, args.seed)
    out_dir = Path(args.data_dir) / "librispeech_test_clean"
    max_seconds = args.librispeech_minutes * 60.0 if args.librispeech_minutes > 0 else math.inf
    total_seconds = 0.0
    items: list[ManifestItem] = []

    for row in tqdm(ds, desc="prepare LibriSpeech"):
        if args.librispeech_limit and len(items) >= args.librispeech_limit:
            break
        if total_seconds >= max_seconds:
            break
        sid = str(row.get("id") or row.get("file") or len(items))
        wav_path = out_dir / "wav" / f"{safe_name(sid)}.wav"
        duration = write_wav(row["audio"], wav_path)
        total_seconds += duration
        items.append(
            ManifestItem(
                audio_filepath=str(wav_path),
                duration=duration,
                answer=str(row.get("text") or row.get("transcription") or ""),
                taskname="asr",
                source_lang="en",
                target_lang="en",
                pnc=args.asr_pnc,
                sample_id=sid,
            )
        )

    manifest = out_dir / "manifest_asr_en.jsonl"
    write_manifest(manifest, items)
    write_metadata(
        manifest,
        {
            "task": "asr",
            "dataset": args.librispeech_dataset,
            "config": args.librispeech_config,
            "split": args.librispeech_split,
            "minutes_limit": args.librispeech_minutes,
            "item_limit": args.librispeech_limit,
            "num_items": len(items),
            "total_audio_seconds": sum(item.duration for item in items),
        },
    )
    return manifest


def fleurs_text(row: dict[str, Any], pnc: str) -> str:
    if pnc == "yes":
        return str(row.get("raw_transcription") or row.get("transcription") or "")
    return str(row.get("transcription") or row.get("raw_transcription") or "")


def load_fleurs(dataset_name: str, split: str, lang: str) -> Any:
    from datasets import load_dataset

    config = FLEURS_CONFIG[lang]
    print(f"loading FLEURS dataset={dataset_name} config={config} split={split}")
    return load_dataset(dataset_name, config, split=split)


def prepare_fleurs_direction(args: argparse.Namespace, src: str, tgt: str) -> Path:
    src_ds = maybe_shuffle(load_fleurs(args.fleurs_dataset, args.fleurs_split, src), args.shuffle, args.seed)
    tgt_ds = load_fleurs(args.fleurs_dataset, args.fleurs_split, tgt)
    tgt_by_id = {str(row["id"]): row for row in tgt_ds}
    out_dir = Path(args.data_dir) / "fleurs" / f"{src}-{tgt}"
    items: list[ManifestItem] = []

    for row in tqdm(src_ds, desc=f"prepare FLEURS {src}->{tgt}"):
        if args.fleurs_limit and len(items) >= args.fleurs_limit:
            break
        sid = str(row["id"])
        if sid not in tgt_by_id:
            continue
        wav_path = out_dir / "wav" / f"{safe_name(sid)}.wav"
        duration = write_wav(row["audio"], wav_path)
        items.append(
            ManifestItem(
                audio_filepath=str(wav_path),
                duration=duration,
                answer=fleurs_text(tgt_by_id[sid], args.ast_pnc),
                taskname="ast",
                source_lang=src,
                target_lang=tgt,
                pnc=args.ast_pnc,
                sample_id=sid,
            )
        )

    manifest = out_dir / f"manifest_ast_{src}_{tgt}.jsonl"
    write_manifest(manifest, items)
    write_metadata(
        manifest,
        {
            "task": "ast",
            "dataset": args.fleurs_dataset,
            "source_config": FLEURS_CONFIG[src],
            "target_config": FLEURS_CONFIG[tgt],
            "split": args.fleurs_split,
            "direction": f"{src}-{tgt}",
            "item_limit": args.fleurs_limit,
            "num_items": len(items),
            "total_audio_seconds": sum(item.duration for item in items),
        },
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifests: list[Path] = []
    if args.task in {"asr", "all"}:
        manifests.append(prepare_librispeech(args))
    if args.task in {"ast", "all"}:
        for direction in [x.strip() for x in args.ast_directions.split(",") if x.strip()]:
            src, tgt = direction.split("-", 1)
            if src not in FLEURS_CONFIG or tgt not in FLEURS_CONFIG:
                raise ValueError(f"Unsupported FLEURS direction: {direction}")
            manifests.append(prepare_fleurs_direction(args, src, tgt))
    print("\nPrepared manifests:")
    for manifest in manifests:
        print(f"  {manifest}")


if __name__ == "__main__":
    main()
