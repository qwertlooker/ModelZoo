#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-BEATs/weights}"
OUT_FILE="${2:-${OUT_DIR}/BEATs_finetuned.pt}"
mkdir -p "${OUT_DIR}"
cat <<'MSG'
BEATs 官方权重托管在 microsoft/unilm 的 beats/README.md 中列出的 OneDrive 链接。
由于 OneDrive 分享链接经常需要浏览器确认/重定向，本脚本默认不猜测具体权重。

用法二选一：
  1) 浏览器下载官方 fine-tuned checkpoint 后放到 BEATs/weights/，例如：
     BEATs/weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
  2) 如已有可直接下载的 URL，执行：
     BEATS_WEIGHT_URL=<direct-url> ./BEATs/scripts/download_weights.sh BEATs/weights BEATs/weights/model.pt
MSG
if [[ -n "${BEATS_WEIGHT_URL:-}" ]]; then
  echo "Downloading: ${BEATS_WEIGHT_URL}"
  curl -L --fail --retry 5 --retry-delay 3 -o "${OUT_FILE}" "${BEATS_WEIGHT_URL}"
  sha256sum "${OUT_FILE}" | tee "${OUT_FILE}.sha256"
  echo "Saved: ${OUT_FILE}"
else
  exit 0
fi
