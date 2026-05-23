#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-Canary-1B/test_data}"
mkdir -p "${OUT_DIR}"
# Network-free smoke-test input for CPU/NPU pipeline validation. This is not an
# ASR accuracy sample; it is only used to verify model loading, audio decoding,
# device placement, and transcription call wiring.
python3 - <<'PY' "${OUT_DIR}/dummy_1s_16k.wav"
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
