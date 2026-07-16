#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:?usage: serve.sh MODEL_DIR}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MechVL-4B-RL}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

for value in "${PORT}" "${MAX_MODEL_LEN}" "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}" "${TENSOR_PARALLEL_SIZE}"; do
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'expected a positive integer, got: %s\n' "${value}" >&2
        exit 2
    fi
done

if [[ ! -d "${MODEL_DIR}" ]]; then
    printf 'model directory does not exist: %s\n' "${MODEL_DIR}" >&2
    exit 2
fi

(
    cd "${MODEL_DIR}"
    sha256sum --check "${SCRIPT_DIR}/weights.sha256"
)

python3 -c 'import torch, torch_npu; assert torch.npu.is_available(); print(torch.__version__, torch_npu.__version__, torch.randn(1).to("npu").device)'

exec vllm serve "${MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --dtype bfloat16 \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
