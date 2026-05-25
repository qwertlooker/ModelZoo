#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-FireRedASR-AED/upstream/pretrained_models/FireRedASR-AED-L}"
HF_REPO="${FIRERED_HF_REPO:-fireredteam/FireRedASR-AED-L}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-api.gitee.com}"
HF_HOME="${HF_HOME:-~/.cache/gitee-ai}"
METHOD="${FIRERED_DOWNLOAD_METHOD:-hf_hub}"
CHECK_ONLY="${FIRERED_CHECK_ONLY:-0}"
mkdir -p "${OUT_DIR}"
if [[ "${METHOD}" == "hf_hub" ]]; then
  HF_REPO="${HF_REPO}" HF_ENDPOINT="${HF_ENDPOINT}" HF_HOME="${HF_HOME}" OUT_DIR="${OUT_DIR}" CHECK_ONLY="${CHECK_ONLY}" python3 - <<'PY'
import os
import urllib.request
from pathlib import Path
try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as exc:
    raise SystemExit('Missing dependency: pip install huggingface_hub') from exc
os.environ['HF_HOME'] = os.path.expanduser(os.environ['HF_HOME'])
out_dir = os.path.expanduser(os.environ['OUT_DIR'])
Path(out_dir).mkdir(parents=True, exist_ok=True)
repo = os.environ['HF_REPO']
endpoint = os.environ['HF_ENDPOINT'].rstrip("/")
required = ["cmvn.ark", "config.yaml", "dict.txt", "model.pth.tar", "train_bpe1000.model"]
api = HfApi(endpoint=endpoint)
info = api.model_info(repo)
files = {s.rfilename for s in info.siblings}
missing = [name for name in required if name not in files]
if missing:
    raise SystemExit(f"Missing required files in {repo}: {missing}")
print(f"Verified repo metadata: {repo} @ {info.sha}")
for name in required:
    url = f"{endpoint}/{repo}/resolve/main/{name}"
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        size = resp.headers.get("X-Linked-Size") or resp.headers.get("Content-Length") or "unknown"
        print(f"Verified file URL: {name} status={resp.status} size={size}")
if os.environ.get("CHECK_ONLY") == "1":
    raise SystemExit(0)
snapshot_download(repo, local_dir=out_dir)
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
