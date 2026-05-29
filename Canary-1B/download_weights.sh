#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-Canary-1B/weights/canary-1b}"
MODEL_FILE="${OUT_DIR}/canary-1b.nemo"
HF_REPO="${CANARY_HF_REPO:-nvidia/canary-1b}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-api.gitee.com}"
HF_HOME="${HF_HOME:-~/.cache/gitee-ai}"
URL="${CANARY_WEIGHT_URL:-https://hf-mirror.com/nvidia/canary-1b/resolve/main/canary-1b.nemo}"
EXPECTED_SHA256="b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a"
mkdir -p "${OUT_DIR}"

echo "Output: ${MODEL_FILE}"

if [[ -f "${MODEL_FILE}" ]]; then
  echo "Found existing file, skip download: ${MODEL_FILE}"
else
  # Default to huggingface_hub + Gitee HF endpoint because it works in more
  # restricted CN environments.  To force the old curl path:
  #   CANARY_DOWNLOAD_METHOD=curl ./Canary-1B/download_weights.sh ...
  METHOD="${CANARY_DOWNLOAD_METHOD:-hf_hub}"
  if [[ -n "${CANARY_WEIGHT_URL:-}" && -z "${CANARY_DOWNLOAD_METHOD:-}" ]]; then
    METHOD="curl"
  fi

  if [[ "${METHOD}" == "hf_hub" ]]; then
    echo "Downloading Canary-1B weights with huggingface_hub"
    echo "Repo: ${HF_REPO}"
    echo "HF_ENDPOINT: ${HF_ENDPOINT}"
    echo "HF_HOME: ${HF_HOME}"
    HF_REPO="${HF_REPO}" HF_ENDPOINT="${HF_ENDPOINT}" HF_HOME="${HF_HOME}" OUT_DIR="${OUT_DIR}" \
      python3 - <<'PY'
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install it with:\n"
        "  pip install huggingface_hub"
    ) from exc

hf_home = os.path.expanduser(os.environ["HF_HOME"])
out_dir = os.path.expanduser(os.environ["OUT_DIR"])
os.environ["HF_HOME"] = hf_home
Path(hf_home).mkdir(parents=True, exist_ok=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)

snapshot_download(
    os.environ["HF_REPO"],
    allow_patterns=["canary-1b.nemo"],
    local_dir=out_dir,
)
PY
  elif [[ "${METHOD}" == "curl" ]]; then
    echo "Downloading Canary-1B weights from: ${URL}"
    # In some proxy environments hf-mirror.com works only with direct connection.
    if [[ "${CANARY_DIRECT_DOWNLOAD:-1}" == "1" ]]; then
      env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
        curl -k -L --fail --retry 10 --retry-delay 5 -C - -o "${MODEL_FILE}" "${URL}"
    else
      curl -k -L --fail --retry 10 --retry-delay 5 -C - -o "${MODEL_FILE}" "${URL}"
    fi
  else
    echo "Unsupported CANARY_DOWNLOAD_METHOD=${METHOD}; use hf_hub or curl" >&2
    exit 1
  fi
fi

actual_sha256="$(sha256sum "${MODEL_FILE}" | awk '{print $1}')"
echo "${actual_sha256}  ${MODEL_FILE}" | tee "${MODEL_FILE}.sha256"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "SHA256 mismatch: expected ${EXPECTED_SHA256}, got ${actual_sha256}" >&2
  exit 1
fi

echo "Downloaded and verified Canary-1B weights: ${MODEL_FILE}"
