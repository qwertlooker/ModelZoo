#!/usr/bin/env python3
"""Prepare MLS/LibriSpeech/FLEURS manifests for Canary-1B evaluation.

This script only downloads/converts data and writes JSONL manifests. It does not
load the model or run inference.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
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
    parser = argparse.ArgumentParser(description="Prepare Canary-1B MLS/LibriSpeech/FLEURS eval data")
    parser.add_argument("--task", default="all", choices=["asr", "ast", "all"])
    parser.add_argument("--data_dir", default="Canary-1B/eval_data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking subsets")

    parser.add_argument("--asr_dataset", default="facebook/multilingual_librispeech")
    parser.add_argument(
        "--asr_configs",
        default="german,spanish,french",
        help="Comma-separated facebook/multilingual_librispeech configs for ASR manifests.",
    )
    parser.add_argument("--asr_split", default="test")
    parser.add_argument("--asr_parquet_dir", default="", help=(
        "Optional local/download directory for facebook/multilingual_librispeech split parquet files. "
        "Files are stored as <dir>/<config>/<split>-00000-of-00001.parquet and reused if present."
    ))
    parser.add_argument("--asr_minutes", type=float, default=30.0, help="0 means full split/no minute cap")
    parser.add_argument("--asr_limit", type=int, default=0, help="0 means no item-count cap")
    parser.add_argument("--asr_pnc", default="no", choices=["yes", "no"])
    parser.add_argument(
        "--include_librispeech_test_clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also prepare OpenSLR LibriSpeech test-clean for Open ASR Leaderboard-style performance tests.",
    )
    parser.add_argument("--librispeech_dataset", default="openslr/librispeech_asr")
    parser.add_argument("--librispeech_config", default="clean")
    parser.add_argument("--librispeech_split", default="test")
    parser.add_argument("--librispeech_dir", default="", help=(
        "Optional local/download directory for OpenSLR LibriSpeech archives. "
        "For test-clean, the script reuses <dir>/LibriSpeech/test-clean if present, "
        "otherwise downloads <dir>/test-clean.tar.gz and extracts it."
    ))
    parser.add_argument(
        "--librispeech_minutes",
        type=float,
        default=None,
        help="LibriSpeech minute cap. Defaults to --asr_minutes; 0 means full split/no cap.",
    )
    parser.add_argument(
        "--librispeech_limit",
        type=int,
        default=None,
        help="LibriSpeech item cap. Defaults to --asr_limit; 0 means no item-count cap.",
    )

    parser.add_argument("--fleurs_dataset", default="google/fleurs")
    parser.add_argument("--fleurs_split", default="test")
    parser.add_argument("--fleurs_parquet_dir", default="", help=(
        "Optional local/download directory for google/fleurs split parquet files. "
        "Files are stored as <dir>/<config>/<split>-00000-of-00001.parquet and reused if present."
    ))
    parser.add_argument("--offline", action="store_true", help="Do not download missing local dataset files")
    parser.add_argument("--fleurs_limit", type=int, default=50, help="Samples per AST direction; 0 means full split")
    parser.add_argument("--ast_directions", default=",".join(DEFAULT_AST_DIRECTIONS))
    parser.add_argument("--ast_pnc", default="yes", choices=["yes", "no"])
    parser.add_argument(
        "--audio_workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help=(
            "Parallel workers for decoding/writing wav files. Set to 1 for serial behavior. "
            "Defaults to min(8, CPU count)."
        ),
    )
    parser.add_argument(
        "--force_rewrite_audio",
        action="store_true",
        help="Rewrite wav files even when they already exist. By default existing wav files are reused.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "sample"


def wav_duration(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def write_wav(audio: dict[str, Any], path: Path, force: bool = False) -> float:
    """Write a datasets Audio value to 16 kHz wav and return duration.

    HF dataset builders usually return {array, sampling_rate}; direct parquet
    loading may return {bytes, path}. Support both so FLEURS can be loaded from
    its test parquet only, without downloading train parquet files. Existing wav
    files are reused by default, which makes repeated manifest preparation much
    faster.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return wav_duration(path)

    if "array" in audio and audio["array"] is not None:
        array = audio["array"]
        sr = int(audio["sampling_rate"])
    elif "bytes" in audio and audio["bytes"] is not None:
        array, sr = sf.read(io.BytesIO(audio["bytes"]))
    elif "path" in audio and audio["path"]:
        array, sr = sf.read(audio["path"])
    else:
        raise ValueError(f"Unsupported audio field: {audio.keys()}")

    if sr != 16000:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        sr = 16000
    sf.write(path, array, sr)
    return float(len(array)) / float(sr)


def map_audio_jobs(jobs: list[tuple[dict[str, Any], Path]], workers: int, force: bool) -> list[float]:
    if workers <= 1 or len(jobs) <= 1:
        return [write_wav(audio, path, force=force) for audio, path in jobs]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda job: write_wav(job[0], job[1], force=force), jobs))


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


def dataset_has_column(ds: Any, column: str) -> bool:
    """Return whether a map-style or iterable HF dataset exposes a column."""
    column_names = getattr(ds, "column_names", None)
    if column_names is not None:
        return column in column_names
    features = getattr(ds, "features", None)
    return bool(features is not None and column in features)


def disable_audio_decode(ds: Any, column: str = "audio") -> Any:
    """Keep HF datasets Audio values encoded so torchcodec is not required.

    Recent versions of ``datasets`` decode Audio columns through torchcodec when
    rows are iterated. On Ascend/NPU environments torchcodec is often not
    available or not usable. This script writes audio with soundfile anyway, so
    keep the original bytes/path and decode locally in ``write_wav`` instead.
    """
    if not dataset_has_column(ds, column):
        return ds
    from datasets import Audio

    return ds.cast_column(column, Audio(decode=False))


def remove_existing_columns(ds: Any, columns: Iterable[str]) -> Any:
    existing = [column for column in columns if dataset_has_column(ds, column)]
    return ds.remove_columns(existing) if existing else ds


MLS_LANG = {
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "dutch": "nl",
    "italian": "it",
    "portuguese": "pt",
    "polish": "pl",
}


def split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def source_lang_for_mls_config(config: str) -> str:
    key = config.lower().replace("_", "-")
    if key not in MLS_LANG:
        raise ValueError(
            f"Unsupported multilingual_librispeech config for Canary ASR manifest: {config!r}. "
            f"Known configs: {', '.join(sorted(MLS_LANG))}."
        )
    return MLS_LANG[key]


def librispeech_openslr_subset(config: str, split: str) -> str:
    config = config.lower().replace("_", "-")
    split = split.lower().replace("_", "-")
    if split in {"test-clean", "test-other", "dev-clean", "dev-other", "train-clean-100", "train-clean-360", "train-other-500"}:
        return split
    if split == "validation":
        split = "dev"
    if split in {"test", "dev"} and config in {"clean", "other"}:
        return f"{split}-{config}"
    if split == "train" and config == "clean":
        return "train-clean-100"
    if split == "train" and config == "other":
        return "train-other-500"
    raise ValueError(
        f"Unsupported local LibriSpeech config/split: config={config!r}, split={split!r}. "
        "Use --librispeech_split test-clean/dev-clean/test-other/... or a supported config+split pair."
    )


def librispeech_subset_dir(librispeech_dir: str | Path, subset: str) -> Path:
    return Path(librispeech_dir) / "LibriSpeech" / subset


def safe_extract_tar(archive: Path, dest_dir: Path) -> None:
    dest_dir = dest_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest_dir / member.name).resolve()
            if not str(target).startswith(str(dest_dir)):
                raise ValueError(f"Refusing to extract path outside destination: {member.name}")
        tar.extractall(dest_dir)


def download_librispeech_subset(config: str, split: str, librispeech_dir: str | Path, offline: bool = False) -> Path:
    subset = librispeech_openslr_subset(config, split)
    root = Path(librispeech_dir)
    subset_dir = librispeech_subset_dir(root, subset)
    if subset_dir.exists():
        print(f"using existing LibriSpeech directory: {subset_dir}")
        return subset_dir

    archive = root / f"{subset}.tar.gz"
    if not archive.exists():
        if offline:
            raise FileNotFoundError(
                f"Offline mode enabled and LibriSpeech data is missing: {subset_dir} or {archive}"
            )
        from urllib.request import urlretrieve

        root.mkdir(parents=True, exist_ok=True)
        tmp_archive = archive.with_suffix(archive.suffix + ".tmp")
        url = f"https://www.openslr.org/resources/12/{subset}.tar.gz"
        print(f"downloading LibriSpeech {subset} to {archive}: {url}")
        urlretrieve(url, tmp_archive)
        tmp_archive.replace(archive)
    else:
        print(f"using existing LibriSpeech archive: {archive}")

    print(f"extracting LibriSpeech archive: {archive}")
    safe_extract_tar(archive, root)
    if not subset_dir.exists():
        raise FileNotFoundError(f"Expected extracted LibriSpeech directory not found: {subset_dir}")
    return subset_dir


def iter_librispeech_local_rows(subset_dir: Path) -> Iterable[dict[str, Any]]:
    for transcript in sorted(subset_dir.glob("*/*/*.trans.txt")):
        with transcript.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample_id, text = line.split(" ", 1)
                audio_path = transcript.parent / f"{sample_id}.flac"
                if not audio_path.exists():
                    raise FileNotFoundError(f"Missing LibriSpeech audio file: {audio_path}")
                yield {"id": sample_id, "audio": {"path": str(audio_path)}, "text": text}


def librispeech_minutes(args: argparse.Namespace) -> float:
    return args.asr_minutes if args.librispeech_minutes is None else args.librispeech_minutes


def librispeech_limit(args: argparse.Namespace) -> int:
    return args.asr_limit if args.librispeech_limit is None else args.librispeech_limit


def write_librispeech_manifest(args: argparse.Namespace, rows: Iterable[dict[str, Any]], dataset: str, config: str) -> Path:
    out_dir = Path(args.data_dir) / "librispeech_test_clean"
    minutes = librispeech_minutes(args)
    limit = librispeech_limit(args)
    max_seconds = minutes * 60.0 if minutes > 0 else math.inf
    total_seconds = 0.0
    items: list[ManifestItem] = []

    row_iter = iter(rows)
    chunk_size = max(1, args.audio_workers * 4)
    with tqdm(desc="prepare LibriSpeech") as pbar:
        while (not limit or len(items) < limit) and total_seconds < max_seconds:
            jobs: list[tuple[dict[str, Any], Path]] = []
            metas: list[tuple[dict[str, Any], str, Path]] = []
            for row in row_iter:
                if limit and len(items) + len(jobs) >= limit:
                    break
                sid = str(row.get("id") or row.get("file") or len(items) + len(jobs))
                wav_path = out_dir / "wav" / f"{safe_name(sid)}.wav"
                jobs.append((row["audio"], wav_path))
                metas.append((row, sid, wav_path))
                if len(jobs) >= chunk_size:
                    break
            if not jobs:
                break
            durations = map_audio_jobs(jobs, args.audio_workers, args.force_rewrite_audio)
            for (row, sid, wav_path), duration in zip(metas, durations):
                if total_seconds >= max_seconds:
                    break
                total_seconds += duration
                items.append(
                    ManifestItem(
                        audio_filepath=str(wav_path),
                        duration=duration,
                        answer=str(row.get("text") or row.get("transcript") or row.get("transcription") or ""),
                        taskname="asr",
                        source_lang="en",
                        target_lang="en",
                        pnc=args.asr_pnc,
                        sample_id=sid,
                    )
                )
                pbar.update(1)

    manifest = out_dir / "manifest_asr_en.jsonl"
    write_manifest(manifest, items)
    write_metadata(
        manifest,
        {
            "task": "asr",
            "dataset": dataset,
            "config": config,
            "split": args.librispeech_split,
            "minutes_limit": minutes,
            "item_limit": limit,
            "librispeech_dir": args.librispeech_dir,
            "offline": args.offline,
            "purpose": "performance / Open ASR Leaderboard-style LibriSpeech test-clean",
            "num_items": len(items),
            "total_audio_seconds": sum(item.duration for item in items),
        },
    )
    return manifest


def prepare_librispeech_local(args: argparse.Namespace) -> Path:
    subset_dir = download_librispeech_subset(
        args.librispeech_config, args.librispeech_split, args.librispeech_dir, args.offline
    )
    rows = list(iter_librispeech_local_rows(subset_dir))
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    return write_librispeech_manifest(args, rows, dataset=str(subset_dir), config=args.librispeech_config)


def prepare_librispeech(args: argparse.Namespace) -> Path:
    if args.librispeech_dir:
        return prepare_librispeech_local(args)

    from datasets import load_dataset

    print(
        f"loading LibriSpeech dataset={args.librispeech_dataset} "
        f"config={args.librispeech_config} split={args.librispeech_split}"
    )
    ds = load_dataset(args.librispeech_dataset, args.librispeech_config, split=args.librispeech_split)
    ds = disable_audio_decode(ds)
    ds = maybe_shuffle(ds, args.shuffle, args.seed)
    return write_librispeech_manifest(args, ds, dataset=args.librispeech_dataset, config=args.librispeech_config)



def mls_parquet_url(config: str, split: str) -> str:
    return (
        "https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/"
        f"{config}/{split}-00000-of-00001.parquet"
    )


def mls_parquet_path(parquet_dir: str | Path, config: str, split: str) -> Path:
    return Path(parquet_dir) / config / f"{split}-00000-of-00001.parquet"


def download_mls_parquet(config: str, split: str, parquet_dir: str | Path = "", offline: bool = False) -> Path:
    if parquet_dir:
        local_file = mls_parquet_path(parquet_dir, config, split)
        if local_file.exists():
            print(f"using existing MLS parquet: {local_file}")
            return local_file
        if offline:
            raise FileNotFoundError(f"Offline mode enabled and MLS parquet is missing: {local_file}")

        from urllib.request import urlretrieve

        local_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = local_file.with_suffix(local_file.suffix + ".tmp")
        url = mls_parquet_url(config, split)
        print(f"downloading MLS parquet to {local_file}: {url}")
        urlretrieve(url, tmp_file)
        tmp_file.replace(local_file)
        return local_file

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="facebook/multilingual_librispeech",
            repo_type="dataset",
            filename=f"{config}/{split}-00000-of-00001.parquet",
            local_files_only=offline,
        )
    )


def write_asr_manifest(
    args: argparse.Namespace, rows: Iterable[dict[str, Any]], dataset: str, config: str, split: str
) -> Path:
    lang = source_lang_for_mls_config(config) if dataset == "facebook/multilingual_librispeech" else "en"
    safe_config = safe_name(config.lower().replace("_", "-"))
    out_dir = Path(args.data_dir) / f"mls_{split}_{safe_config}"
    max_seconds = args.asr_minutes * 60.0 if args.asr_minutes > 0 else math.inf
    total_seconds = 0.0
    items: list[ManifestItem] = []

    row_iter = iter(rows)
    chunk_size = max(1, args.audio_workers * 4)
    with tqdm(desc=f"prepare ASR {config}/{split}") as pbar:
        while (not args.asr_limit or len(items) < args.asr_limit) and total_seconds < max_seconds:
            jobs: list[tuple[dict[str, Any], Path]] = []
            metas: list[tuple[dict[str, Any], str, Path]] = []
            for row in row_iter:
                if args.asr_limit and len(items) + len(jobs) >= args.asr_limit:
                    break
                sid = str(row.get("id") or row.get("file") or len(items) + len(jobs))
                wav_path = out_dir / "wav" / f"{safe_name(sid)}.wav"
                jobs.append((row["audio"], wav_path))
                metas.append((row, sid, wav_path))
                if len(jobs) >= chunk_size:
                    break
            if not jobs:
                break
            durations = map_audio_jobs(jobs, args.audio_workers, args.force_rewrite_audio)
            for (row, sid, wav_path), duration in zip(metas, durations):
                if total_seconds >= max_seconds:
                    break
                total_seconds += duration
                items.append(
                    ManifestItem(
                        audio_filepath=str(wav_path),
                        duration=duration,
                        answer=str(row.get("text") or row.get("transcript") or row.get("transcription") or ""),
                        taskname="asr",
                        source_lang=lang,
                        target_lang=lang,
                        pnc=args.asr_pnc,
                        sample_id=sid,
                    )
                )
                pbar.update(1)

    manifest = out_dir / f"manifest_asr_{lang}.jsonl"
    write_manifest(manifest, items)
    write_metadata(
        manifest,
        {
            "task": "asr",
            "dataset": dataset,
            "config": config,
            "split": split,
            "minutes_limit": args.asr_minutes,
            "item_limit": args.asr_limit,
            "asr_parquet_dir": args.asr_parquet_dir,
            "offline": args.offline,
            "num_items": len(items),
            "total_audio_seconds": sum(item.duration for item in items),
        },
    )
    return manifest


def prepare_asr_config(args: argparse.Namespace, config: str) -> Path:
    from datasets import load_dataset

    print(f"loading ASR dataset={args.asr_dataset} config={config} split={args.asr_split}")
    if args.asr_dataset == "facebook/multilingual_librispeech":
        # Load the requested split parquet directly. This avoids pulling the old
        # dataset loading script/README and keeps ASR preparation scoped to the
        # official MLS split file being evaluated.
        if args.asr_parquet_dir:
            local_file = download_mls_parquet(config, args.asr_split, args.asr_parquet_dir, args.offline)
            print(f"loading local MLS parquet: {local_file}")
            ds = load_dataset("parquet", data_files={args.asr_split: str(local_file)}, split=args.asr_split, streaming=True)
        else:
            data_file = mls_parquet_url(config, args.asr_split)
            print(f"loading MLS split-only parquet: {data_file}")
            try:
                ds = load_dataset("parquet", data_files={args.asr_split: data_file}, split=args.asr_split, streaming=True)
            except FileNotFoundError as exc:
                print(f"direct HTTPS parquet load failed: {exc}")
                local_file = download_mls_parquet(config, args.asr_split, offline=args.offline)
                print(f"loading local MLS parquet: {local_file}")
                ds = load_dataset("parquet", data_files={args.asr_split: str(local_file)}, split=args.asr_split, streaming=True)
    else:
        ds = load_dataset(args.asr_dataset, config, split=args.asr_split, streaming=True)
    ds = disable_audio_decode(ds)
    ds = maybe_shuffle(ds, args.shuffle, args.seed)
    return write_asr_manifest(args, ds, dataset=args.asr_dataset, config=config, split=args.asr_split)


def prepare_asr(args: argparse.Namespace) -> list[Path]:
    manifests = [prepare_asr_config(args, config) for config in split_csv(args.asr_configs)]
    if args.include_librispeech_test_clean:
        manifests.append(prepare_librispeech(args))
    return manifests


def fleurs_text(row: dict[str, Any], pnc: str) -> str:
    if pnc == "yes":
        return str(row.get("raw_transcription") or row.get("transcription") or "")
    return str(row.get("transcription") or row.get("raw_transcription") or "")


def fleurs_parquet_url(config: str, split: str) -> str:
    return (
        "https://huggingface.co/datasets/google/fleurs/resolve/main/"
        f"parquet-data/{config}/{split}-00000-of-00001.parquet"
    )


def fleurs_parquet_path(parquet_dir: str | Path, config: str, split: str) -> Path:
    return Path(parquet_dir) / config / f"{split}-00000-of-00001.parquet"


def download_fleurs_parquet(config: str, split: str, parquet_dir: str | Path = "", offline: bool = False) -> Path:
    if parquet_dir:
        local_file = fleurs_parquet_path(parquet_dir, config, split)
        if local_file.exists():
            print(f"using existing FLEURS parquet: {local_file}")
            return local_file
        if offline:
            raise FileNotFoundError(f"Offline mode enabled and FLEURS parquet is missing: {local_file}")

        from urllib.request import urlretrieve

        local_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = local_file.with_suffix(local_file.suffix + ".tmp")
        url = fleurs_parquet_url(config, split)
        print(f"downloading FLEURS parquet to {local_file}: {url}")
        urlretrieve(url, tmp_file)
        tmp_file.replace(local_file)
        return local_file

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="google/fleurs",
            repo_type="dataset",
            filename=f"parquet-data/{config}/{split}-00000-of-00001.parquet",
            local_files_only=offline,
        )
    )


def load_fleurs(dataset_name: str, split: str, lang: str, fleurs_parquet_dir: str = "", offline: bool = False) -> Any:
    from datasets import load_dataset

    config = FLEURS_CONFIG[lang]
    print(f"loading FLEURS dataset={dataset_name} config={config} split={split}")
    if dataset_name == "google/fleurs":
        # Important: load the requested split parquet directly.
        # load_dataset("google/fleurs", config, split="test") may still fetch
        # train/validation parquet files while preparing the builder cache.
        # This path restricts network/cache access to test-*.parquet only.
        # Use an HTTPS URL instead of hf://. Some environments (notably newer
        # Python/httpx stacks on Windows) pass hf:// through to httpx, which then
        # raises "Request URL is missing an 'http://' or 'https://' protocol.".
        if fleurs_parquet_dir:
            local_file = download_fleurs_parquet(config, split, fleurs_parquet_dir, offline)
            print(f"loading local FLEURS parquet: {local_file}")
            ds = load_dataset("parquet", data_files={split: str(local_file)}, split=split, streaming=True)
        else:
            data_file = fleurs_parquet_url(config, split)
            print(f"loading FLEURS split-only parquet: {data_file}")
            try:
                ds = load_dataset("parquet", data_files={split: data_file}, split=split, streaming=True)
            except FileNotFoundError as exc:
                # Some datasets/httpx/fsspec combinations on Windows fail URL
                # pattern resolution for Hugging Face HTTPS files even though the
                # same URL is reachable in a browser. Download exactly this split
                # file through huggingface_hub, then load the local parquet.
                print(f"direct HTTPS parquet load failed: {exc}")
                local_file = download_fleurs_parquet(config, split, offline=offline)
                print(f"loading local FLEURS parquet: {local_file}")
                ds = load_dataset("parquet", data_files={split: str(local_file)}, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, config, split=split)
    return disable_audio_decode(ds)


def prepare_fleurs_direction(args: argparse.Namespace, src: str, tgt: str) -> Path:
    src_ds = maybe_shuffle(
        load_fleurs(args.fleurs_dataset, args.fleurs_split, src, args.fleurs_parquet_dir, args.offline),
        args.shuffle,
        args.seed,
    )
    tgt_ds = load_fleurs(args.fleurs_dataset, args.fleurs_split, tgt, args.fleurs_parquet_dir, args.offline)
    # Target side only supplies text. Drop encoded audio bytes before building
    # the lookup table to avoid unnecessary memory use and any audio decoding.
    tgt_ds = remove_existing_columns(tgt_ds, ["audio"])
    tgt_by_id = {str(row["id"]): row for row in tgt_ds}
    out_dir = Path(args.data_dir) / "fleurs" / f"{src}-{tgt}"
    items: list[ManifestItem] = []

    row_iter = iter(src_ds)
    chunk_size = max(1, args.audio_workers * 4)
    with tqdm(desc=f"prepare FLEURS {src}->{tgt}") as pbar:
        while not args.fleurs_limit or len(items) < args.fleurs_limit:
            jobs: list[tuple[dict[str, Any], Path]] = []
            metas: list[tuple[str, Path]] = []
            for row in row_iter:
                if args.fleurs_limit and len(items) + len(jobs) >= args.fleurs_limit:
                    break
                sid = str(row["id"])
                if sid not in tgt_by_id:
                    continue
                wav_path = out_dir / "wav" / f"{safe_name(sid)}.wav"
                jobs.append((row["audio"], wav_path))
                metas.append((sid, wav_path))
                if len(jobs) >= chunk_size:
                    break
            if not jobs:
                break
            durations = map_audio_jobs(jobs, args.audio_workers, args.force_rewrite_audio)
            for (sid, wav_path), duration in zip(metas, durations):
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
                pbar.update(1)

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
            "fleurs_parquet_dir": args.fleurs_parquet_dir,
            "offline": args.offline,
            "num_items": len(items),
            "total_audio_seconds": sum(item.duration for item in items),
        },
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifests: list[Path] = []
    if args.task in {"asr", "all"}:
        manifests.extend(prepare_asr(args))
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
