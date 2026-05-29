#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-Canary-1B/test_data}"
OUT_WAV="${OUT_DIR}/dummy_1s_16k.wav"
META="${OUT_WAV}.meta.json"
mkdir -p "${OUT_DIR}"
# Network-free smoke-test input for CPU/NPU pipeline validation. This is not an
# ASR accuracy sample; it is only used to verify model loading, audio decoding,
# device placement, and transcription call wiring.
if [[ -f "${OUT_WAV}" ]]; then
  echo "using existing Canary test wav: ${OUT_WAV}"
else
  python3 - <<'PY' "${OUT_WAV}"
import math
import struct
import sys
import wave

path = sys.argv[1]
framerate = 16000
with wave.open(path, "w") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(framerate)
    for i in range(framerate):
        sample = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * i / framerate))
        wav.writeframes(struct.pack("<h", sample))
print(path)
PY
fi
python3 - <<'PY' "${OUT_WAV}" "${META}"
import json
import sys
import wave
from pathlib import Path

wav_path = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
with wave.open(str(wav_path)) as wav:
    data = {
        "dataset": "synthetic",
        "source": "generated 440Hz sine wave",
        "audio_filepath": str(wav_path),
        "sample_rate": wav.getframerate(),
        "channels": wav.getnchannels(),
        "frames": wav.getnframes(),
        "duration": wav.getnframes() / float(wav.getframerate()),
        "offline_safe": True,
    }
meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"wrote metadata: {meta_path}")
PY
