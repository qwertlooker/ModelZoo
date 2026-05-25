#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L}"
HF_REPO="${FIRERED_HF_REPO:-fireredteam/FireRedASR-AED-L}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-api.gitee.com}"
HF_HOME="${HF_HOME:-~/.cache/gitee-ai}"
METHOD="${FIRERED_DOWNLOAD_METHOD:-hf_hub}"
mkdir -p "${OUT_DIR}"
if [[ "${METHOD}" == "hf_hub" ]]; then
  HF_REPO="${HF_REPO}" HF_ENDPOINT="${HF_ENDPOINT}" HF_HOME="${HF_HOME}" OUT_DIR="${OUT_DIR}" python3 - <<'PY'
import os
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit('Missing dependency: pip install huggingface_hub') from exc
os.environ['HF_HOME'] = os.path.expanduser(os.environ['HF_HOME'])
out_dir = os.path.expanduser(os.environ['OUT_DIR'])
Path(out_dir).mkdir(parents=True, exist_ok=True)
snapshot_download(os.environ['HF_REPO'], local_dir=out_dir)
PY
elif [[ "${METHOD}" == "modelscope" ]]; then
  python3 - <<'PY'
raise SystemExit('ModelScope download is documented, but this script uses hf_hub by default. Install/use modelscope CLI manually if preferred.')
PY
else
  echo "Unsupported FIRERED_DOWNLOAD_METHOD=${METHOD}; use hf_hub" >&2
  exit 1
fi
find "${OUT_DIR}" -maxdepth 1 -type f -print | sort
