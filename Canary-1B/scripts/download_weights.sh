#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-Canary-1B/weights/canary-1b-hfmirror}"
MODEL_FILE="${OUT_DIR}/canary-1b.nemo"
URL="${CANARY_WEIGHT_URL:-https://hf-mirror.com/nvidia/canary-1b/resolve/main/canary-1b.nemo}"
EXPECTED_SHA256="b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a"
mkdir -p "${OUT_DIR}"

echo "Downloading Canary-1B weights from: ${URL}"
echo "Output: ${MODEL_FILE}"

# In some proxy environments hf-mirror.com works only with direct connection.
if [[ "${CANARY_DIRECT_DOWNLOAD:-1}" == "1" ]]; then
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    curl -k -L --fail --retry 10 --retry-delay 5 -C - -o "${MODEL_FILE}" "${URL}"
else
  curl -k -L --fail --retry 10 --retry-delay 5 -C - -o "${MODEL_FILE}" "${URL}"
fi

actual_sha256="$(sha256sum "${MODEL_FILE}" | awk '{print $1}')"
echo "${actual_sha256}  ${MODEL_FILE}" | tee "${MODEL_FILE}.sha256"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "SHA256 mismatch: expected ${EXPECTED_SHA256}, got ${actual_sha256}" >&2
  exit 1
fi

echo "Downloaded and verified Canary-1B weights: ${MODEL_FILE}"
