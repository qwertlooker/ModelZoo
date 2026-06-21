import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


def resolve_providers(device):
    if device == "cpu":
        return ["CPUExecutionProvider"]

    provider = "CANNExecutionProvider"
    if provider not in ort.get_available_providers():
        raise RuntimeError(
            "CANNExecutionProvider is unavailable. Install an ONNX Runtime CANN "
            "build matching the installed CANN version."
        )
    return [provider]


class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path, device):
        providers = resolve_providers(device)
        self.primary_session = ort.InferenceSession(
            primary_model_path, providers=providers
        )
        self.p808_session = ort.InferenceSession(
            p808_model_path, providers=providers
        )

    @staticmethod
    def audio_melspec(
        audio,
        n_mels=120,
        frame_size=320,
        hop_length=160,
        sample_rate=SAMPLING_RATE,
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        return ((librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40).T

    @staticmethod
    def polynomial_correction(sig, bak, ovr, personalized):
        if personalized:
            p_ovr = np.poly1d(
                [-0.00533021, 0.005101, 1.18058466, -0.11236046]
            )
            p_sig = np.poly1d(
                [-0.01019296, 0.02751166, 1.19576786, -0.24348726]
            )
            p_bak = np.poly1d(
                [-0.04976499, 0.44276479, -0.1644611, 0.96883132]
            )
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def __call__(self, audio_path, personalized):
        audio, input_rate = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if input_rate != SAMPLING_RATE:
            audio = librosa.resample(
                audio, orig_sr=input_rate, target_sr=SAMPLING_RATE
            ).astype(np.float32)

        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        if actual_audio_len == 0:
            raise ValueError(f"Empty audio file: {audio_path}")
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        raw_sig = []
        raw_bak = []
        raw_ovrl = []
        corrected_sig = []
        corrected_bak = []
        corrected_ovrl = []
        p808_scores = []

        for index in range(num_hops):
            start = index * SAMPLING_RATE
            end = int((index + INPUT_LENGTH) * SAMPLING_RATE)
            segment = audio[start:end]
            if len(segment) < len_samples:
                continue

            primary_input = segment[np.newaxis, :].astype(np.float32)
            p808_input = self.audio_melspec(segment[:-160])[
                np.newaxis, :, :
            ].astype(np.float32)
            p808 = self.p808_session.run(
                None, {"input_1": p808_input}
            )[0][0][0]
            sig, bak, ovrl = self.primary_session.run(
                None, {"input_1": primary_input}
            )[0][0]
            sig_c, bak_c, ovrl_c = self.polynomial_correction(
                sig, bak, ovrl, personalized
            )

            raw_sig.append(sig)
            raw_bak.append(bak)
            raw_ovrl.append(ovrl)
            corrected_sig.append(sig_c)
            corrected_bak.append(bak_c)
            corrected_ovrl.append(ovrl_c)
            p808_scores.append(p808)

        return {
            "filename": str(audio_path),
            "len_in_sec": actual_audio_len / SAMPLING_RATE,
            "sr": SAMPLING_RATE,
            "num_hops": num_hops,
            "OVRL_raw": np.mean(raw_ovrl),
            "SIG_raw": np.mean(raw_sig),
            "BAK_raw": np.mean(raw_bak),
            "OVRL": np.mean(corrected_ovrl),
            "SIG": np.mean(corrected_sig),
            "BAK": np.mean(corrected_bak),
            "P808_MOS": np.mean(p808_scores),
        }


def collect_audio(paths):
    clips = []
    for value in paths:
        path = Path(value)
        if path.is_file():
            if path.suffix.lower() != ".wav":
                raise ValueError(f"Only WAV input is supported: {path}")
            clips.append(path)
        elif path.is_dir():
            clips.extend(
                Path(item)
                for item in glob.glob(
                    os.path.join(path, "**", "*.wav"), recursive=True
                )
            )
        else:
            raise FileNotFoundError(path)
    clips = sorted(set(clips))
    if not clips:
        raise ValueError("No WAV files found.")
    return clips


def read_manifest(path):
    clips = []
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "audio_path" not in row:
                raise ValueError(
                    f"{manifest_path}:{line_number}: missing audio_path"
                )
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            if audio_path.suffix.lower() != ".wav":
                raise ValueError(f"Only WAV input is supported: {audio_path}")
            clips.append(audio_path.resolve())
    if not clips:
        raise ValueError(f"Empty manifest: {manifest_path}")
    if len(clips) != len(set(clips)):
        raise ValueError(f"Duplicate audio paths in manifest: {manifest_path}")
    return clips


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="DNSMOS P.835 inference")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--audio",
        nargs="+",
        help="WAV files or directories searched recursively.",
    )
    inputs.add_argument(
        "--manifest",
        help="JSONL manifest containing one audio_path field per row.",
    )
    parser.add_argument(
        "--model_root",
        default="weights",
        help="Directory containing DNSMOS/ and pDNSMOS/ model folders.",
    )
    parser.add_argument(
        "--device", choices=["npu", "cpu"], default="npu"
    )
    parser.add_argument("--personalized", action="store_true")
    parser.add_argument("--output_csv", default="results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    model_root = Path(args.model_root)
    p808_model = model_root / "DNSMOS" / "model_v8.onnx"
    primary_dir = "pDNSMOS" if args.personalized else "DNSMOS"
    primary_model = model_root / primary_dir / "sig_bak_ovr.onnx"
    for model_path in (primary_model, p808_model):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

    clips = read_manifest(args.manifest) if args.manifest else collect_audio(args.audio)
    scorer = ComputeScore(
        str(primary_model), str(p808_model), device=args.device
    )
    start = time.perf_counter()
    rows = [scorer(path, args.personalized) for path in clips]
    elapsed = time.perf_counter() - start

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    audio_seconds = sum(row["len_in_sec"] for row in rows)
    print(f"provider={scorer.primary_session.get_providers()[0]}")
    print(f"files={len(rows)} audio_seconds={audio_seconds:.3f}")
    print(f"elapsed_seconds={elapsed:.3f} rtf={elapsed / audio_seconds:.6f}")
    print(f"output={output_path}")
    metadata = {
        "command": " ".join(sys.argv),
        "device": args.device,
        "provider": scorer.primary_session.get_providers()[0],
        "personalized": args.personalized,
        "manifest": args.manifest,
        "files": len(rows),
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed,
        "rtf": elapsed / audio_seconds,
        "primary_model": str(primary_model),
        "primary_model_sha256": sha256(primary_model),
        "p808_model": str(p808_model),
        "p808_model_sha256": sha256(p808_model),
        "python": sys.version,
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
