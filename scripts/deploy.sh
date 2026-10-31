#!/usr/bin/env bash
# Copyright (C) 2026 MOST Contributors.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/deploy.sh <model_path> <num_ports> <num_gpus>" >&2
  exit 2
fi

MODEL_PATH="$1"
NUM_PORTS="$2"
NUM_GPUS="$3"
BASE_PORT="${BASE_PORT:-10086}"
SESSION_NAME="${SESSION_NAME:-most-servers}"
CONDA_ENV="${CONDA_ENV:-most}"
PYTHON="${PYTHON:-python}"

[[ "${NUM_PORTS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "num_ports must be a positive integer" >&2
  exit 2
}
[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "num_gpus must be a positive integer" >&2
  exit 2
}
[[ "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "BASE_PORT must be a positive integer" >&2
  exit 2
}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

printf -v MODEL_ARG '%q' "${MODEL_PATH}"
tmux new-session -d -s "${SESSION_NAME}"

for ((index = 0; index < NUM_PORTS; index++)); do
  port=$((BASE_PORT + index))
  gpu_id=$((index % NUM_GPUS))
  window_name="server-$(printf '%02d' "${index}")"

  if (( index == 0 )); then
    tmux rename-window -t "${SESSION_NAME}:0" "${window_name}"
  else
    tmux new-window -d -t "${SESSION_NAME}" -n "${window_name}"
  fi

  server_command="CUDA_VISIBLE_DEVICES=${gpu_id} TOKENIZERS_PARALLELISM=false ${PYTHON} -u deploy/server.py --model ${MODEL_ARG} --port ${port}"
  if [[ -n "${CONDA_ENV}" ]]; then
    printf -v ENV_ARG '%q' "${CONDA_ENV}"
    server_command="conda run --no-capture-output -n ${ENV_ARG} ${server_command}"
  fi
  tmux send-keys -t "${SESSION_NAME}:${window_name}" "${server_command}" Enter
done

echo "Started ${NUM_PORTS} server(s) in tmux session '${SESSION_NAME}'."
echo "Ports: ${BASE_PORT}-$((BASE_PORT + NUM_PORTS - 1)); GPUs: ${NUM_GPUS}."
echo "Inspect with: tmux attach -t ${SESSION_NAME}"
