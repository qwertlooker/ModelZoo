#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-BEATs/eval_data/esc50}"
OFFLINE="${OFFLINE:-0}"
URL="${ESC50_URL:-https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip}"
ARCHIVE="${DATA_DIR}/ESC-50-master.zip"
EXTRACT_DIR="${DATA_DIR}/ESC-50-master"
MANIFEST="${DATA_DIR}/esc50_manifest.csv"
META="${MANIFEST}.meta.json"

mkdir -p "${DATA_DIR}"

if [[ -f "${MANIFEST}" && -d "${EXTRACT_DIR}/audio" ]]; then
  echo "using existing ESC-50 manifest: ${MANIFEST}"
  exit 0
fi

if [[ ! -d "${EXTRACT_DIR}/audio" ]]; then
  if [[ ! -f "${ARCHIVE}" ]]; then
    if [[ "${OFFLINE}" == "1" ]]; then
      echo "Offline mode enabled and ESC-50 data is missing: ${EXTRACT_DIR}/audio or ${ARCHIVE}" >&2
      exit 1
    fi
    tmp="${ARCHIVE}.tmp"
    echo "downloading ESC-50 to ${ARCHIVE}: ${URL}"
    curl -L --fail --retry 5 --retry-delay 3 -o "${tmp}" "${URL}"
    mv "${tmp}" "${ARCHIVE}"
  else
    echo "using existing ESC-50 archive: ${ARCHIVE}"
  fi
  echo "extracting ESC-50 archive: ${ARCHIVE}"
  python3 - "${ARCHIVE}" "${DATA_DIR}" <<'PY'
import sys, zipfile
from pathlib import Path
archive = Path(sys.argv[1]).resolve()
dest = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive) as zf:
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise SystemExit(f"Refusing to extract path outside destination: {info.filename}")
    zf.extractall(dest)
PY
fi

python3 - "${EXTRACT_DIR}" "${MANIFEST}" "${META}" "${URL}" "${OFFLINE}" <<'PY'
import csv, json, sys, wave
from pathlib import Path
root = Path(sys.argv[1])
out = Path(sys.argv[2])
meta = Path(sys.argv[3])
url = sys.argv[4]
offline = sys.argv[5] == "1"
source_csv = root / "meta" / "esc50.csv"
if not source_csv.exists():
    raise SystemExit(f"Missing ESC-50 metadata: {source_csv}")
rows = []
total_seconds = 0.0
with source_csv.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        audio = root / "audio" / row["filename"]
        if not audio.exists():
            raise SystemExit(f"Missing ESC-50 audio: {audio}")
        with wave.open(str(audio)) as wav:
            duration = wav.getnframes() / float(wav.getframerate())
        total_seconds += duration
        rows.append({
            "audio_filepath": str(audio),
            "label": row["category"],
            "target": row["target"],
            "duration": f"{duration:.6f}",
            "sample_id": Path(row["filename"]).stem,
            "split": f"fold{row['fold']}",
        })
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["audio_filepath", "label", "target", "duration", "sample_id", "split"])
    writer.writeheader()
    writer.writerows(rows)
meta.write_text(json.dumps({
    "dataset": "ESC-50",
    "source_url": url,
    "local_dir": str(root),
    "manifest": str(out),
    "num_items": len(rows),
    "total_audio_seconds": total_seconds,
    "offline": offline,
}, indent=2), encoding="utf-8")
print(f"wrote {len(rows)} items: {out}")
print(f"wrote metadata: {meta}")
PY
