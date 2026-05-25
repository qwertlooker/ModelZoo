#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-FireRedASR-AED/test_data}"
SRC_DIR="FireRedASR-AED/upstream/examples/wav"
mkdir -p "${OUT_DIR}"
if [[ -d "${SRC_DIR}" ]]; then
  cp "${SRC_DIR}"/*.wav "${OUT_DIR}"/
  cp "${SRC_DIR}"/wav.scp "${SRC_DIR}"/text "${OUT_DIR}"/ 2>/dev/null || true
  echo "Copied official example wavs to ${OUT_DIR}"
else
  python3 - "${OUT_DIR}/dummy_1s_16k.wav" <<'PY'
import math, struct, sys, wave
path = sys.argv[1]
sr = 16000
with wave.open(path, 'w') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
    frames = bytearray()
    for i in range(sr):
        frames += struct.pack('<h', int(0.2 * 32767 * math.sin(2 * math.pi * 440 * i / sr)))
    f.writeframes(frames)
print(path)
PY
fi
