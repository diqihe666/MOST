#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_robocasa365_target.sh \
    <target10|target100> \
    <atomic|composite_seen|composite_unseen> \
    [max_steps] [num_gpus]

Required environment variables:
  MODEL_PATH             compatible base model directory
  ROBOCASA_TARGET_ROOT   dataset root containing atomic/ and composite/

Optional environment variables:
  MOST_PRETRAINED_CKPT   converted initialization checkpoint
  MOST_VLM_CONFIG_PATH   converted nested VLM config directory
  BATCH_SIZE             per-device batch size (default: 16)
  NUM_WORKERS            workers per process (default: 8)
  ACCUMULATE_GRAD_BATCHES (default: 1)
  SAVE_INTERVAL          checkpoint interval (default: 10000)
  USE_WANDB              true or false (default: false)
  RUN_NAME               output run name
EOF
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage
  exit 2
fi

FRACTION="$1"
SPLIT="$2"
MAX_STEPS="${3:-60000}"
GPU_COUNT="${4:-8}"

case "${FRACTION}" in
  target10|target100) ;;
  *) echo "Invalid target fraction: ${FRACTION}" >&2; usage; exit 2 ;;
esac
case "${SPLIT}" in
  atomic|composite_seen|composite_unseen) ;;
  *) echo "Invalid task split: ${SPLIT}" >&2; usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to the compatible base model directory}"
: "${ROBOCASA_TARGET_ROOT:?Set ROBOCASA_TARGET_ROOT to the dataset root}"

MOST_PRETRAINED_CKPT="${MOST_PRETRAINED_CKPT:-${REPO_ROOT}/pretrained_ckpt/robocasa365/model_states.pt}"
MOST_VLM_CONFIG_PATH="${MOST_VLM_CONFIG_PATH:-${REPO_ROOT}/pretrained_ckpt/robocasa365/vlm_config}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-most-robocasa365}"
RUN_NAME="${RUN_NAME:-most-${FRACTION}-${SPLIT}-s${MAX_STEPS}}"
RUN_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}/${RUN_NAME}"

[[ -f "${MOST_PRETRAINED_CKPT}" ]] || {
  echo "Missing checkpoint: ${MOST_PRETRAINED_CKPT}" >&2
  exit 1
}
[[ -f "${MOST_VLM_CONFIG_PATH}/config.json" ]] || {
  echo "Missing VLM config: ${MOST_VLM_CONFIG_PATH}/config.json" >&2
  exit 1
}
[[ -d "${ROBOCASA_TARGET_ROOT}" ]] || {
  echo "Missing dataset root: ${ROBOCASA_TARGET_ROOT}" >&2
  exit 1
}

mkdir -p "${RUN_ROOT}"
export MOST_VLM_CONFIG_PATH
export RESOURCE_GPU="${GPU_COUNT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
export TOKENIZERS_PARALLELISM=false

echo "Training MOST on RoboCasa365: ${FRACTION}/${SPLIT}, run=${RUN_NAME}"

bash scripts/train.sh \
  "data=robocasa365_${FRACTION}_${SPLIT}" \
  model=most \
  "model.params.pretrained=${MOST_PRETRAINED_CKPT}" \
  "trainer.max_steps=${MAX_STEPS}" \
  "trainer.ckpt_path=${RESUME_CKPT:-null}" \
  "trainer.save_interval=${SAVE_INTERVAL}" \
  "trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}" \
  "trainer.default_root_dir=${RUN_ROOT}" \
  "trainer.project=${WANDB_PROJECT}" \
  "trainer.exp_name=${RUN_NAME}" \
  "+trainer.use_wandb=${USE_WANDB}" \
  "data.params.train_datasets.batch_size=${BATCH_SIZE}" \
  "data.params.train_datasets.num_workers=${NUM_WORKERS}"
