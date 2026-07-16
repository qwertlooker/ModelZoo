#!/usr/bin/env bash
set -euo pipefail

REVISION="2c6fda8a16e57d8a6fe1019412092d09a0363850"
MODEL_ID="XiaofengAlg/MechVL-4B-RL"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:?usage: download_weights.sh TARGET_DIR}"
CHECK_ONLY="${MODEL_CHECK_ONLY:-0}"

mkdir -p "${TARGET_DIR}"

while read -r expected file; do
    url="https://huggingface.co/${MODEL_ID}/resolve/${REVISION}/${file}?download=true"
    target="${TARGET_DIR}/${file}"
    if [[ "${CHECK_ONLY}" == "1" ]]; then
        curl --location --head --fail --retry 3 --connect-timeout 10 --max-time 60 \
            --max-redirs 10 "${url}" >/dev/null
        printf 'reachable: %s\n' "${file}"
        continue
    fi

    if [[ -f "${target}" ]] && printf '%s  %s\n' "${expected}" "${target}" | sha256sum --check --status; then
        printf 'verified existing: %s\n' "${file}"
        continue
    fi

    temporary="${target}.tmp"
    if [[ -f "${temporary}" ]]; then
        printf 'resuming partial download: %s\n' "${file}"
    fi
    curl --location --fail --retry 3 --continue-at - --output "${temporary}" "${url}"
    if ! printf '%s  %s\n' "${expected}" "${temporary}" | sha256sum --check --status; then
        rm -f "${temporary}"
        printf 'SHA256 mismatch; removed partial file: %s\n' "${file}" >&2
        exit 1
    fi
    mv "${temporary}" "${target}"
    printf 'downloaded and verified: %s\n' "${file}"
done < "${SCRIPT_DIR}/weights.sha256"

if [[ "${CHECK_ONLY}" != "1" ]]; then
    (
        cd "${TARGET_DIR}"
        sha256sum --check "${SCRIPT_DIR}/weights.sha256"
    )
fi
