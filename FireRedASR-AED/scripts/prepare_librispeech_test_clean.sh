#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-FireRedASR-AED/eval_data/librispeech_raw}"
OFFLINE="${OFFLINE:-0}"
SUBSET="${LIBRISPEECH_SUBSET:-test-clean}"
URL="${LIBRISPEECH_URL:-https://www.openslr.org/resources/12/${SUBSET}.tar.gz}"
ARCHIVE="${DATA_DIR}/${SUBSET}.tar.gz"
SUBSET_DIR="${DATA_DIR}/LibriSpeech/${SUBSET}"
OUT_DIR="${2:-FireRedASR-AED/eval_data/librispeech_${SUBSET}}"
WAV_SCP="${OUT_DIR}/wav.scp"
TEXT="${OUT_DIR}/text"
META="${OUT_DIR}/manifest.meta.json"

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

if [[ ! -d "${SUBSET_DIR}" ]]; then
  if [[ ! -f "${ARCHIVE}" ]]; then
    if [[ "${OFFLINE}" == "1" ]]; then
      echo "Offline mode enabled and LibriSpeech data is missing: ${SUBSET_DIR} or ${ARCHIVE}" >&2
      exit 1
    fi
    tmp="${ARCHIVE}.tmp"
    echo "downloading LibriSpeech ${SUBSET} to ${ARCHIVE}: ${URL}"
    curl -L --fail --retry 5 --retry-delay 3 -o "${tmp}" "${URL}"
    mv "${tmp}" "${ARCHIVE}"
  else
    echo "using existing LibriSpeech archive: ${ARCHIVE}"
  fi
  echo "extracting LibriSpeech archive: ${ARCHIVE}"
  python3 - "${ARCHIVE}" "${DATA_DIR}" <<'PY'
import sys, tarfile
from pathlib import Path
archive = Path(sys.argv[1]).resolve()
dest = Path(sys.argv[2]).resolve()
with tarfile.open(archive, "r:gz") as tar:
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise SystemExit(f"Refusing to extract path outside destination: {member.name}")
    tar.extractall(dest)
PY
else
  echo "using existing LibriSpeech directory: ${SUBSET_DIR}"
fi

python3 - "${SUBSET_DIR}" "${WAV_SCP}" "${TEXT}" "${META}" "${URL}" "${OFFLINE}" <<'PY'
import json, sys
from pathlib import Path
subset = Path(sys.argv[1])
wav_scp = Path(sys.argv[2])
text = Path(sys.argv[3])
meta = Path(sys.argv[4])
url = sys.argv[5]
offline = sys.argv[6] == "1"
rows = []
for trans in sorted(subset.glob("*/*/*.trans.txt")):
    for line in trans.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt, transcript = line.split(" ", 1)
        audio = trans.parent / f"{utt}.flac"
        if not audio.exists():
            raise SystemExit(f"Missing LibriSpeech audio: {audio}")
        rows.append((utt, audio, transcript))
wav_scp.parent.mkdir(parents=True, exist_ok=True)
wav_scp.write_text("".join(f"{utt} {audio}\n" for utt, audio, _ in rows), encoding="utf-8")
text.write_text("".join(f"{utt} {transcript}\n" for utt, _, transcript in rows), encoding="utf-8")
meta.write_text(json.dumps({
    "dataset": "LibriSpeech",
    "subset": subset.name,
    "source_url": url,
    "local_dir": str(subset),
    "wav_scp": str(wav_scp),
    "text": str(text),
    "num_items": len(rows),
    "offline": offline,
}, indent=2), encoding="utf-8")
print(f"wrote {len(rows)} utterances: {wav_scp}")
print(f"wrote text: {text}")
print(f"wrote metadata: {meta}")
PY
