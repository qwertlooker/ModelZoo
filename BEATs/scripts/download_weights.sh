#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-BEATs/weights}"
OUT_FILE="${2:-${OUT_DIR}/BEATs_finetuned.pt}"
CHECK_ONLY="${BEATS_CHECK_ONLY:-0}"
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
if [[ "${CHECK_ONLY}" == "1" ]]; then
  python3 - <<'PY'
import pathlib
import re
import urllib.error
import urllib.request

readme = pathlib.Path("BEATs/upstream/beats/README.md")
links = re.findall(r"https://1drv\.ms/[^\)\s]+", readme.read_text(encoding="utf-8"))
if not links:
    raise SystemExit("No official OneDrive links found in BEATs/upstream/beats/README.md")
print(f"Found {len(links)} official OneDrive links in upstream README")
for url in links[:3]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as resp:
            print(f"Checked OneDrive link: status={resp.status} url={url}")
    except urllib.error.HTTPError as exc:
        print(f"Checked OneDrive link: status={exc.code} url={url}")
    except Exception as exc:
        print(f"Checked OneDrive link: error={type(exc).__name__}: {exc} url={url}")
print("Note: OneDrive may reject non-browser direct downloads; use BEATS_WEIGHT_URL for a verified direct URL.")
PY
  exit 0
fi
if [[ -n "${BEATS_WEIGHT_URL:-}" ]]; then
  echo "Downloading: ${BEATS_WEIGHT_URL}"
  curl -L --fail --retry 5 --retry-delay 3 -o "${OUT_FILE}" "${BEATS_WEIGHT_URL}"
  sha256sum "${OUT_FILE}" | tee "${OUT_FILE}.sha256"
  echo "Saved: ${OUT_FILE}"
else
  exit 0
fi
