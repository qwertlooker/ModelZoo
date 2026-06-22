#!/usr/bin/env python3
"""Prepare deterministic MOSS-TTSD JSONL subsets and evaluator manifests."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


TTSD_EVAL_COMMIT = "dea13b98529dc16dcfb5fe45779ad63ac9238337"
TTSD_EVAL_ARCHIVE_SHA256 = (
    "49ed8338f3e5323c5ffcff01f3480a9c245937256d9197d792c973cba5603e17"
)
TTSD_EVAL_MANIFEST_SHA256 = {
    "zh": "2c9cafed6eaea093e3dbdbc30dba0d3e87b91b4d1be9925ae97d7e8ce41a2dc4",
    "en": "e779ed7c9ece3d0d0c0364bfd235fcbb591e17b543dd693b37e265f1cffb4d4d",
}
WESPEAKER_COMMIT = "c92349a14d6b426808c4e09b8b12e076864dfc11"
WESPEAKER_ARCHIVE_SHA256 = (
    "ad0873d380acaa7f4256ff37d40217ee31e4955b26a45064a13a14998cc89d16"
)
WESPEAKER_MODEL_SHA256 = (
    "5aeee438ca23c0ca6e341bab6c6bf7f465497e1dc323bb1bc1074d6a0c778b11"
)
MMS_FA_SHA256 = (
    "20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680"
)
MMS_FA_SIZE = 1262047414
WHISPER_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
WHISPER_MODEL_SHA256 = (
    "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"
)
WHISPER_MODEL_SIZE = 3087130976
EVALUATOR_VERSIONS = {
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "soundfile": "0.13.1",
    "uroman": "1.3.1.1",
    "transformers": "4.57.6",
    "tqdm": "4.67.1",
    "jiwer": "4.0.0",
    "zhon": "2.1.1",
    "requests": "2.32.5",
    "s3prl": "0.4.18",
    "openai-whisper": "20250625",
    "peft": "0.18.0",
    "huggingface-hub": "0.36.0",
    "accelerate": "1.12.0",
    "safetensors": "0.6.2",
    "kaldiio": "2.18.1",
    "silero-vad": "6.2.1",
    "onnxruntime": "1.23.2",
}
EVALUATOR_VERSIONS_NPU = {
    **EVALUATOR_VERSIONS,
    "torch": "2.9.0",
    "torchaudio": "2.9.0",
}
TTSD_EVAL_NPU_PATCH = "patches/0002-adapt-ttsd-eval-to-npu.patch"
TTSD_EVAL_NPU_PATCH_SHA256 = (
    "5dd9c5ab357d64e5d43543821ee3324f32b9c1210bb4ba63e9fe9dcaa7438607"
)
TTSD_EVAL_NPU_PATCHED_SHA256 = {
    "tools/align.py": (
        "722028e9a7adbc90dfad3eb74cb1ab307cd6919bbc2969134f52175c0c2c49f2"
    ),
    "tools/run_similarity.py": (
        "65ccfb613a5248f9f40efe9a09fadaa2ebcf4aa05df6a9331bf7a619fdc6dc66"
    ),
    "wer/whisper_asr.py": (
        "b7a62bf6504ddf8a9fdf3c86f66ca4488608caf10ae5456901d4b275bf917194"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MOSS-TTSD evaluation data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset = subparsers.add_parser("subset")
    subset.add_argument("--input_jsonl", required=True)
    subset.add_argument("--output_jsonl", required=True)
    subset.add_argument("--limit", type=int, required=True)
    subset.add_argument("--dataset", required=True)
    subset.add_argument("--split", required=True)

    attach = subparsers.add_parser("attach-output")
    attach.add_argument("--input_jsonl", required=True)
    attach.add_argument("--output_jsonl", required=True)
    attach.add_argument("--output_dir", required=True)
    attach.add_argument(
        "--path_root",
        help="Directory used to validate relative prompt-audio paths.",
    )

    verify = subparsers.add_parser(
        "verify-ttsd-eval",
        help="Verify the pinned TTSD-eval source, data, models, and environment.",
    )
    verify.add_argument("--eval_root", required=True)
    verify.add_argument(
        "--scope",
        choices=("source-data", "full"),
        default="full",
        help="Use source-data before installing models; full is the acceptance gate.",
    )
    verify.add_argument(
        "--expected_device",
        choices=("cpu", "cuda", "npu"),
        help="Required with --scope full; verifies the evaluator device profile.",
    )
    verify.add_argument(
        "--report",
        help="Optional JSON path for the verification evidence.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "text" not in row:
                raise ValueError(f"{path}:{line_number}: missing text")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def subset(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    rows = read_rows(input_path)[: args.limit]
    write_rows(output_path, rows)
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "source": str(input_path),
        "source_sha256": sha256(input_path),
        "sample_count": len(rows),
        "manifest_sha256": sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def attach_output(args: argparse.Namespace) -> None:
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    path_root = Path(args.path_root).resolve() if args.path_root else Path.cwd()
    rows = read_rows(input_path)
    attached = []
    for index, row in enumerate(rows):
        audio_path = output_dir / f"output_{index}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        result = dict(row)
        result["output_audio"] = str(audio_path)
        for field in ("prompt_audio_speaker1", "prompt_audio_speaker2"):
            if field not in result:
                raise ValueError(f"{input_path}: missing {field}")
            prompt_path = Path(result[field]).expanduser()
            if not prompt_path.is_absolute():
                prompt_path = path_root / prompt_path
            if not prompt_path.is_file():
                raise FileNotFoundError(prompt_path)
        attached.append(result)
    write_rows(output_path, attached)
    metadata = {
        "source": str(input_path),
        "source_sha256": sha256(input_path),
        "output_dir": str(output_dir),
        "path_root": str(path_root),
        "sample_count": len(attached),
        "manifest_sha256": sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def check_file(
    path: Path,
    label: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> dict:
    require_file(path, label)
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{label}: expected {expected_size} bytes, got {size}")
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"{label}: expected SHA256 {expected_sha256}, got {digest}"
        )
    return {"path": str(path), "size": size, "sha256": digest}


def git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def check_source(eval_root: Path, expected_device: str | None = None) -> dict:
    required = (
        "README.md",
        "requirements.txt",
        "tools/align.py",
        "tools/split.py",
        "tools/run_similarity.py",
        "wer/whisper_asr.py",
        "wer/run_wer.py",
    )
    for relative in required:
        require_file(eval_root / relative, "TTSD-eval source file")
    commit = git_output(eval_root, "rev-parse", "HEAD")
    if commit != TTSD_EVAL_COMMIT:
        raise ValueError(
            f"TTSD-eval commit: expected {TTSD_EVAL_COMMIT}, got {commit}"
        )
    tracked_diff = git_output(eval_root, "status", "--short", "--untracked-files=no")
    if expected_device == "npu":
        modified = set()
        for line in tracked_diff.splitlines():
            line = line.strip()
            if not line:
                continue
            modified.add(line[2:].strip().split(" -> ")[-1])
        expected_modified = set(TTSD_EVAL_NPU_PATCHED_SHA256.keys())
        if modified != expected_modified:
            raise ValueError(
                f"TTSD-eval NPU profile: expected modified files "
                f"{sorted(expected_modified)}, got {sorted(modified)}"
            )
        for relative, expected_hash in TTSD_EVAL_NPU_PATCHED_SHA256.items():
            actual_hash = sha256(eval_root / relative)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"{relative}: expected patched SHA256 {expected_hash}, "
                    f"got {actual_hash}"
                )
        return {
            "commit": commit,
            "tracked_files_clean": False,
            "npu_patch_applied": True,
            "npu_patch": TTSD_EVAL_NPU_PATCH,
            "npu_patch_sha256": TTSD_EVAL_NPU_PATCH_SHA256,
        }
    if tracked_diff:
        raise ValueError(f"TTSD-eval tracked files are modified:\n{tracked_diff}")
    return {"commit": commit, "tracked_files_clean": True}


def check_testset(eval_root: Path) -> dict:
    archive = check_file(
        eval_root / "model/downloads/testset.zip",
        "TTSD-eval testset archive",
        expected_sha256=TTSD_EVAL_ARCHIVE_SHA256,
        expected_size=71138324,
    )
    testset_root = eval_root / "testset"
    languages = {}
    referenced_prompts = set()
    required_fields = (
        "text",
        "prompt_audio_speaker1",
        "prompt_text_speaker1",
        "prompt_audio_speaker2",
        "prompt_text_speaker2",
    )
    for language in ("zh", "en"):
        manifest = testset_root / f"ttsd_eval_{language}.jsonl"
        rows = read_rows(manifest)
        if len(rows) != 50:
            raise ValueError(f"{manifest}: expected 50 rows, got {len(rows)}")
        manifest_sha256 = sha256(manifest)
        if manifest_sha256 != TTSD_EVAL_MANIFEST_SHA256[language]:
            raise ValueError(
                f"{manifest}: expected SHA256 "
                f"{TTSD_EVAL_MANIFEST_SHA256[language]}, got {manifest_sha256}"
            )
        for index, row in enumerate(rows, 1):
            missing = [field for field in required_fields if not row.get(field)]
            if missing:
                raise ValueError(f"{manifest}:{index}: missing {missing}")
            if "[S1]" not in row["text"] or "[S2]" not in row["text"]:
                raise ValueError(f"{manifest}:{index}: missing [S1]/[S2] tags")
            for field in ("prompt_audio_speaker1", "prompt_audio_speaker2"):
                prompt = testset_root / row[field]
                require_file(prompt, f"{manifest}:{index}:{field}")
                referenced_prompts.add(prompt.resolve())
        languages[language] = {
            "manifest": str(manifest),
            "sample_count": len(rows),
            "sha256": manifest_sha256,
        }
    if len(referenced_prompts) != 200:
        raise ValueError(
            f"TTSD-eval prompt set: expected 200 files, got {len(referenced_prompts)}"
        )
    return {
        "archive": archive,
        "languages": languages,
        "referenced_prompt_count": len(referenced_prompts),
    }


def check_models(eval_root: Path) -> dict:
    model_root = eval_root / "model"
    wespeaker_archive = check_file(
        model_root / "downloads/voxblink2_samresnet100_ft.zip",
        "WeSpeaker archive",
        expected_sha256=WESPEAKER_ARCHIVE_SHA256,
        expected_size=186890839,
    )
    wespeaker_model = check_file(
        model_root / "voxblink2_samresnet100_ft/avg_model.pt",
        "WeSpeaker model",
        expected_sha256=WESPEAKER_MODEL_SHA256,
        expected_size=201318407,
    )
    check_file(
        model_root / "voxblink2_samresnet100_ft/config.yaml",
        "WeSpeaker config",
        expected_sha256=(
            "f57b7f3784c804d03ca8acd8d23471e85fe05fcf3e3fb1f0c984d461bfb7f589"
        ),
        expected_size=1575,
    )
    mms_fa = check_file(
        model_root / "checkpoints/model.pt",
        "MMS-FA checkpoint",
        expected_sha256=MMS_FA_SHA256,
        expected_size=MMS_FA_SIZE,
    )
    whisper_root = model_root / "whisper-large-v3"
    revision_path = require_file(
        whisper_root / "REVISION",
        "Whisper revision marker",
    )
    revision = revision_path.read_text(encoding="utf-8").strip()
    if revision != WHISPER_REVISION:
        raise ValueError(
            f"Whisper revision: expected {WHISPER_REVISION}, got {revision}"
        )
    for relative in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        require_file(whisper_root / relative, "Whisper model file")
    whisper_model = check_file(
        whisper_root / "model.safetensors",
        "Whisper model",
        expected_sha256=WHISPER_MODEL_SHA256,
        expected_size=WHISPER_MODEL_SIZE,
    )
    return {
        "wespeaker_archive": wespeaker_archive,
        "wespeaker_model": wespeaker_model,
        "mms_fa": mms_fa,
        "whisper_revision": revision,
        "whisper_model": whisper_model,
    }


def check_environment(expected_device: str) -> dict:
    if sys.version_info[:2] != (3, 11):
        raise ValueError(
            f"evaluator Python: expected 3.11, got "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    version_table = (
        EVALUATOR_VERSIONS_NPU if expected_device == "npu" else EVALUATOR_VERSIONS
    )
    versions = {}
    for distribution, expected in version_table.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Missing evaluator dependency: {distribution}") from error
        comparable = actual.split("+", 1)[0] if distribution in {
            "torch",
            "torchaudio",
        } else actual
        if comparable != expected:
            raise ValueError(
                f"{distribution}: expected version {expected}, got {actual}"
            )
        versions[distribution] = actual
    try:
        direct_url = importlib.metadata.distribution("wespeaker").read_text(
            "direct_url.json"
        )
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Missing evaluator dependency: wespeaker") from error
    if not direct_url:
        raise ValueError("wespeaker: missing direct_url.json; install from fixed Git commit")
    vcs_info = json.loads(direct_url).get("vcs_info", {})
    commit = vcs_info.get("commit_id")
    if commit != WESPEAKER_COMMIT:
        raise ValueError(
            f"wespeaker commit: expected {WESPEAKER_COMMIT}, got {commit}"
        )
    modules = {
        "torch": "torch",
        "torchaudio": "torchaudio",
        "soundfile": "soundfile",
        "uroman": "uroman",
        "transformers": "transformers",
        "jiwer": "jiwer",
        "wespeaker": "wespeaker",
        "whisper": "whisper",
        "peft": "peft",
    }
    imported = []
    for label, module in modules.items():
        try:
            importlib.import_module(module)
        except Exception as error:
            raise RuntimeError(f"Failed to import evaluator module: {label}") from error
        imported.append(label)
    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    cuda_runtime = getattr(torch.version, "cuda", None)
    npu_available = False
    npu_device_count = 0
    if expected_device == "npu":
        try:
            import torch_npu  # noqa: F401
            npu_available = bool(torch.npu.is_available())
            npu_device_count = int(torch.npu.device_count())
        except ImportError as error:
            raise RuntimeError(
                "NPU evaluator requested, but torch_npu is not installed"
            ) from error
        if not npu_available or npu_device_count < 1:
            raise RuntimeError(
                "NPU evaluator requested, but no NPU device is available"
            )
    elif expected_device == "cuda" and (not cuda_available or cuda_device_count < 1):
        raise RuntimeError("CUDA evaluator requested, but no CUDA device is available")
    if expected_device == "cuda" and not cuda_runtime:
        raise RuntimeError("CUDA evaluator requested, but torch is not a CUDA wheel")
    if expected_device == "cpu" and (
        cuda_available or cuda_runtime or "+cpu" not in versions["torch"]
    ):
        raise RuntimeError(
            "CPU evaluator requested, but this is not an isolated CPU wheel/profile; "
            "use the documented CPU wheel/profile"
        )
    return {
        "python": sys.version.split()[0],
        "device_profile": expected_device,
        "versions": versions,
        "wespeaker_commit": commit,
        "imports": imported,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_runtime": cuda_runtime,
        "npu_available": npu_available,
        "npu_device_count": npu_device_count,
    }


def verify_ttsd_eval(args: argparse.Namespace) -> None:
    eval_root = Path(args.eval_root).resolve()
    report = {
        "eval_root": str(eval_root),
        "scope": args.scope,
        "source": check_source(eval_root, args.expected_device),
        "testset": check_testset(eval_root),
    }
    if args.scope == "full":
        if not args.expected_device:
            raise ValueError("--expected_device is required with --scope full")
        report["models"] = check_models(eval_root)
        report["environment"] = check_environment(args.expected_device)
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["report"] = str(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "subset":
        subset(args)
    elif args.command == "attach-output":
        attach_output(args)
    else:
        verify_ttsd_eval(args)


if __name__ == "__main__":
    main()
