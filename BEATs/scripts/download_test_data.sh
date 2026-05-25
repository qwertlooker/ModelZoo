#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-BEATs/test_data}"
OUT_WAV="${OUT_DIR}/dummy_1s_16k.wav"
mkdir -p "${OUT_DIR}"
python3 - "${OUT_WAV}" <<'PY'
import math, struct, sys, wave
path = sys.argv[1]
sr = 16000
with wave.open(path, 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)
    frames = bytearray()
    for i in range(sr):
        v = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * i / sr))
        frames += struct.pack('<h', v)
    f.writeframes(frames)
print(path)
PY
