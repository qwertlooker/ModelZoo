#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-FireRedASR-AED/test_data}"
SRC_DIR="${FIRERED_SAMPLE_SRC_DIR:-FireRedASR-AED/upstream/examples/wav}"
OFFLINE="${OFFLINE:-0}"
ALLOW_DUMMY="${ALLOW_DUMMY:-1}"
META="${OUT_DIR}/sample_data.meta.json"
mkdir -p "${OUT_DIR}"
if [[ -f "${OUT_DIR}/wav.scp" && -f "${OUT_DIR}/text" ]]; then
  echo "using existing FireRedASR test manifest: ${OUT_DIR}/wav.scp"
elif [[ -d "${SRC_DIR}" ]]; then
  cp "${SRC_DIR}"/*.wav "${OUT_DIR}"/
  cp "${SRC_DIR}"/text "${OUT_DIR}"/ 2>/dev/null || true
  : > "${OUT_DIR}/wav.scp"
  for wav in "${OUT_DIR}"/*.wav; do
    utt="$(basename "${wav}" .wav)"
    printf '%s %s\n' "${utt}" "${wav}" >> "${OUT_DIR}/wav.scp"
  done
  echo "Copied official example wavs to ${OUT_DIR}"
else
  if [[ "${OFFLINE}" == "1" && "${ALLOW_DUMMY}" != "1" ]]; then
    echo "Offline mode enabled and official FireRedASR examples are missing: ${SRC_DIR}" >&2
    exit 1
  fi
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
  printf 'dummy_1s_16k %s\n' "${OUT_DIR}/dummy_1s_16k.wav" > "${OUT_DIR}/wav.scp"
  printf 'dummy_1s_16k DUMMY SAMPLE\n' > "${OUT_DIR}/text"
fi
python3 - "${OUT_DIR}" "${META}" "${SRC_DIR}" "${OFFLINE}" <<'PY'
import json, sys, wave
from pathlib import Path
out = Path(sys.argv[1])
meta = Path(sys.argv[2])
src = sys.argv[3]
offline = sys.argv[4] == "1"
items = []
for wav in sorted(out.glob("*.wav")):
    with wave.open(str(wav)) as f:
        items.append({
            "audio_filepath": str(wav),
            "sample_rate": f.getframerate(),
            "channels": f.getnchannels(),
            "frames": f.getnframes(),
            "duration": f.getnframes() / float(f.getframerate()),
        })
meta.write_text(json.dumps({
    "dataset": "FireRedASR official examples" if src else "synthetic",
    "source_dir": src,
    "local_dir": str(out),
    "num_items": len(items),
    "items": items,
    "offline": offline,
    "offline_safe": True,
}, indent=2), encoding="utf-8")
print(f"wrote metadata: {meta}")
PY
