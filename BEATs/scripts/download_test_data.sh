#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-BEATs/test_data}"
OUT_WAV="${OUT_DIR}/dummy_1s_16k.wav"
META="${OUT_DIR}/dummy_1s_16k.wav.meta.json"
mkdir -p "${OUT_DIR}"
if [[ -f "${OUT_WAV}" ]]; then
  echo "using existing BEATs test wav: ${OUT_WAV}"
else
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
fi
python3 - "${OUT_WAV}" "${META}" <<'PY'
import json, sys, wave
from pathlib import Path
wav = Path(sys.argv[1])
meta = Path(sys.argv[2])
with wave.open(str(wav)) as f:
    info = {
        "dataset": "synthetic",
        "source": "generated 440Hz sine wave",
        "audio_filepath": str(wav),
        "sample_rate": f.getframerate(),
        "channels": f.getnchannels(),
        "frames": f.getnframes(),
        "duration": f.getnframes() / float(f.getframerate()),
        "offline_safe": True,
    }
meta.write_text(json.dumps(info, indent=2), encoding="utf-8")
print(f"wrote metadata: {meta}")
PY
