#!/usr/bin/env bash
# 一键批量测试 MechVL-4B-RL（对接本目录 serve.sh 部署的 vLLM-Ascend 服务）。
#
# 仿照 cad_prompt--openrouter/batch_analyze_mechvl.ps1 的用法：
#   - 默认用 tests/fixtures 下的自包含图片与 prompt
#   - 也可用 IMAGES_DIR / PROMPT_FILE 指向 runtime/MechVQA 真实数据
#   - SERVE=1 时自动后台拉起 serve.sh 并等待 /v1/models 就绪
#   - MOCK=1 不连接服务，用 mock 表格校验端到端管线
#
# 常用：
#   ./batch_test.sh                       # 连接已运行的 127.0.0.1:8000 服务
#   SERVE=1 MODEL_DIR=runtime/weights/MechVL-4B-RL ./batch_test.sh
#   MOCK=1 ./batch_test.sh                # 无 NPU 时校验管线
#   DRY_RUN=1 ./batch_test.sh             # 只打印计划
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-MechVL-4B-RL}"
API_KEY="${VQA_TARGET_API_KEY:-EMPTY}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
TIMEOUT="${TIMEOUT:-600}"

IMAGES_DIR="${IMAGES_DIR:-${SCRIPT_DIR}/tests/fixtures/images}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/tests/fixtures/prompt.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runtime/batch_results}"

SERVE="${SERVE:-0}"
MODEL_DIR="${MODEL_DIR:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

MOCK="${MOCK:-0}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_RAW="${KEEP_RAW:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

serve_pid=""
cleanup() {
    if [[ -n "${serve_pid}" ]] && kill -0 "${serve_pid}" 2>/dev/null; then
        echo "[batch_test] 停止后台服务 PID=${serve_pid}"
        kill "${serve_pid}" 2>/dev/null || true
        wait "${serve_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# 1. 校验自包含测试图片与提示词（tests/fixtures/images 下为已上库的真实图纸）。
if [[ ! -d "${IMAGES_DIR}" ]] || [[ -z "$(find "${IMAGES_DIR}" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.gif' \) -print -quit 2>/dev/null || true)" ]]; then
    echo "[ERROR] 图片目录为空或不存在：${IMAGES_DIR}" >&2
    exit 1
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "[ERROR] 提示词文件不存在：${PROMPT_FILE}" >&2
    exit 1
fi

# 2. 可选：后台拉起 vLLM-Ascend 服务（与 serve.sh 部署一致）。
if [[ "${SERVE}" == "1" ]]; then
    if [[ -z "${MODEL_DIR}" ]]; then
        echo "[ERROR] SERVE=1 时必须提供 MODEL_DIR（如 runtime/weights/MechVL-4B-RL）" >&2
        exit 2
    fi
    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "[ERROR] 模型目录不存在：${MODEL_DIR}" >&2
        exit 2
    fi
    echo "[batch_test] 启动 vLLM-Ascend 服务：${MODEL_DIR}"
    PORT="${PORT}" HOST="${HOST}" SERVED_MODEL_NAME="${MODEL}" \
        ./serve.sh "${MODEL_DIR}" >/tmp/mechvl_serve.log 2>&1 &
    serve_pid=$!
    echo "[batch_test] serve PID=${serve_pid}，日志 /tmp/mechvl_serve.log"
    # 轮询 /v1/models 最多 ~10 分钟
    for _ in $(seq 1 120); do
        if ! kill -0 "${serve_pid}" 2>/dev/null; then
            echo "[ERROR] serve 进程已退出，日志尾部：" >&2
            tail -n 50 /tmp/mechvl_serve.log >&2 || true
            exit 1
        fi
        if curl --fail --silent --max-time 5 "${BASE_URL%/v1}/v1/models" >/dev/null 2>&1 \
            || curl --fail --silent --max-time 5 "${BASE_URL}/models" >/dev/null 2>&1; then
            echo "[batch_test] 服务就绪：${BASE_URL}"
            break
        fi
        sleep 5
    done
    if ! curl --fail --silent --max-time 5 "${BASE_URL}/models" >/dev/null 2>&1; then
        echo "[ERROR] 服务未在超时内就绪：${BASE_URL}" >&2
        tail -n 50 /tmp/mechvl_serve.log >&2 || true
        exit 1
    fi
fi

# 3. 组装 batch_test.py 参数。
args=(
    --images-dir "${IMAGES_DIR}"
    --prompt-file "${PROMPT_FILE}"
    --output-dir "${OUTPUT_DIR}"
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --api-key "${API_KEY}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --timeout "${TIMEOUT}"
)
[[ "${KEEP_RAW}" == "1" ]] && args+=(--keep-raw)
[[ "${STOP_ON_FAILURE}" == "1" ]] && args+=(--stop-on-failure)
[[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)
[[ "${MOCK}" == "1" ]] && args+=(--mock)

echo "[batch_test] 运行：${PYTHON} batch_test.py ${args[*]}"
"${PYTHON}" batch_test.py "${args[@]}"
rc=$?

echo "[batch_test] 完成，退出码=${rc}；结果目录：${OUTPUT_DIR}"
echo "[batch_test] 总账：${OUTPUT_DIR}/mechvl_test_overall.md"
exit "${rc}"
