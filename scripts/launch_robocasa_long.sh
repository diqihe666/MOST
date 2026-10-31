#!/usr/bin/env bash
# Copyright (C) 2026 MOST Contributors.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_PORT="${BASE_PORT:-10086}"
SERVER_ADDR="${SERVER_ADDR:-127.0.0.1}"
PYTHON="${PYTHON:-python}"
NUM_PORTS="${1:-}"
LOG_PATH="${2:-}"
MODEL_PATH="${3:-${MODEL_PATH:-}}"
shift $(( $# >= 3 ? 3 : $# ))

if [[ -z "${NUM_PORTS}" || -z "${LOG_PATH}" || -z "${MODEL_PATH}" ]]; then
    echo "Usage: bash scripts/launch_robocasa_long.sh <num_ports> <log_path> [model_path]" >&2
    echo "       [extra evaluator args...]" >&2
    echo "Example: bash scripts/launch_robocasa_long.sh 8 ./eval_results/robocasa_long /path/to/checkpoint" >&2
    exit 1
fi
if ! [[ "${NUM_PORTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_ports must be a positive integer." >&2
    exit 1
fi

# Set CONDA_ENV when the client should activate a dedicated simulator environment.
if [[ -n "${CONDA_ENV:-}" ]]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
QUEUE_DIR="${QUEUE_DIR:-${LOG_PATH}/scheduler/${RUN_ID}}"
SPLIT="${SPLIT:-target}"
TASK_SET="${TASK_SET:-robocasa-long}"
NUM_TRIALS="${NUM_TRIALS:-30}"
REPLAN_STEPS="${REPLAN_STEPS:-16}"
OBS_HISTORY="${OBS_HISTORY:-4}"
OBS_INTERVAL="${OBS_INTERVAL:-2}"
ABSOLUTE_ROTATION_SCALE="${ABSOLUTE_ROTATION_SCALE:-1.0}"
ABSOLUTE_ROTATION_GAIN="${ABSOLUTE_ROTATION_GAIN:-1.0}"
ABSOLUTE_ROTATION_MAX_STEP_RADIANS="${ABSOLUTE_ROTATION_MAX_STEP_RADIANS:-0.5}"
SEED="${SEED:-7}"
CROP_RATIO="${CROP_RATIO:-0.95}"
HORIZON="${HORIZON:-8000}"
ROBOT="${ROBOT:-PandaOmron}"
CONTROL_FREQ="${CONTROL_FREQ:-20}"
CAMERA_WIDTH="${CAMERA_WIDTH:-256}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-256}"
VIDEO_STRIDE="${VIDEO_STRIDE:-2}"
VIDEO_FPS="${VIDEO_FPS:-10}"
ROBOT_TYPE="${ROBOT_TYPE:-robocasa_long_absrot}"

# Optional comma-separated task list shared with the Long training launcher.
# When unset, evaluation uses the three default RoboCasa-Long tasks.
TASK_ARGS=()
if [[ -n "${LONG_TASKS:-}" ]]; then
    IFS=',' read -r -a LONG_TASK_ARRAY <<< "${LONG_TASKS}"
    for task in "${LONG_TASK_ARRAY[@]}"; do
        task="${task#"${task%%[![:space:]]*}"}"
        task="${task%"${task##*[![:space:]]}"}"
        [[ -n "${task}" ]] || continue
        TASK_ARGS+=(--task-name "${task}")
    done
    if (( ${#TASK_ARGS[@]} == 0 )); then
        echo "LONG_TASKS must contain at least one task name" >&2
        exit 2
    fi
fi

"${PYTHON}" -u "${REPO_ROOT}/eval_robocasa_long/dynamic_eval.py" init \
    --queue-dir "${QUEUE_DIR}" -- \
    "$@" \
    "${TASK_ARGS[@]}" \
    --model-path "${MODEL_PATH}" \
    --robot-type "${ROBOT_TYPE}" \
    --server-addr "${SERVER_ADDR}" \
    --split "${SPLIT}" \
    --task-set "${TASK_SET}" \
    --num-trials "${NUM_TRIALS}" \
    --replan-steps "${REPLAN_STEPS}" \
    --obs-history "${OBS_HISTORY}" \
    --obs-interval "${OBS_INTERVAL}" \
    --absolute-rotation-scale "${ABSOLUTE_ROTATION_SCALE}" \
    --absolute-rotation-gain "${ABSOLUTE_ROTATION_GAIN}" \
    --absolute-rotation-max-step-radians "${ABSOLUTE_ROTATION_MAX_STEP_RADIANS}" \
    --seed "${SEED}" \
    --crop-ratio "${CROP_RATIO}" \
    --horizon "${HORIZON}" \
    --robot "${ROBOT}" \
    --control-freq "${CONTROL_FREQ}" \
    --camera-width "${CAMERA_WIDTH}" \
    --camera-height "${CAMERA_HEIGHT}" \
    --video-stride "${VIDEO_STRIDE}" \
    --video-fps "${VIDEO_FPS}" \
    --save-root-dir "${LOG_PATH}" \
    --run-id "${RUN_ID}"

worker_pids=()
cleanup_workers() {
    if (( ${#worker_pids[@]} > 0 )); then
        kill "${worker_pids[@]}" 2>/dev/null || true
        wait "${worker_pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((index = 0; index < NUM_PORTS; index++)); do
    port=$((BASE_PORT + index))
    "${PYTHON}" -u "${REPO_ROOT}/eval_robocasa_long/dynamic_eval.py" worker \
        --queue-dir "${QUEUE_DIR}" \
        --worker-id "gpu-${index}" \
        --server-addr "${SERVER_ADDR}" \
        --server-port "${port}" \
        >"${QUEUE_DIR}/logs/worker-${index}.log" 2>&1 &
    worker_pids+=("$!")
done

worker_status=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        worker_status=1
    fi
done

if ! "${PYTHON}" -u "${REPO_ROOT}/eval_robocasa_long/dynamic_eval.py" merge \
    --queue-dir "${QUEUE_DIR}"; then
    worker_status=1
fi

if (( worker_status != 0 )); then
    echo "Evaluation did not complete. Inspect ${QUEUE_DIR}/logs and ${QUEUE_DIR}/errors." >&2
    exit "${worker_status}"
fi

echo "RoboCasa-Long evaluation complete. Results: ${LOG_PATH}/${RUN_ID}"
