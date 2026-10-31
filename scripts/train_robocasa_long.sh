#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to the compatible base model directory}"
: "${ROBOCASA_TARGET_ROOT:?Set ROBOCASA_TARGET_ROOT to the dataset root}"

MOST_PRETRAINED_CKPT="${MOST_PRETRAINED_CKPT:-${REPO_ROOT}/pretrained_ckpt/robocasa365/model_states.pt}"
MOST_VLM_CONFIG_PATH="${MOST_VLM_CONFIG_PATH:-${REPO_ROOT}/pretrained_ckpt/robocasa365/vlm_config}"
LONG_TASKS="${LONG_TASKS:-PackFruit,PrepareBreakfast,CleanAndStoreAfterMeal}"
LONG_EPISODES_PER_BATCH="${LONG_EPISODES_PER_BATCH:-2}"
LONG_ACTION_STATS_PATH="${LONG_ACTION_STATS_PATH:-null}"
LONG_MIN_ROTATION_STD="${LONG_MIN_ROTATION_STD:-0.05}"
LONG_ROBOT_TYPE="${LONG_ROBOT_TYPE:-robocasa_long_absrot}"

MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
GPU_COUNT="${GPU_COUNT:-8}"
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-most-robocasa-long}"
MEMORY_GRAD_CKPT="${MEMORY_GRAD_CKPT:-true}"
RUN_NAME="${RUN_NAME:-most-robocasa-long-e${LONG_EPISODES_PER_BATCH}-s${MAX_STEPS}}"
RUN_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}/${RUN_NAME}"
LOG_FILE="${LOG_ROOT:-${REPO_ROOT}/logs}/${RUN_NAME}.log"

[[ "${LONG_EPISODES_PER_BATCH}" =~ ^[1-9][0-9]*$ ]] || {
  echo "LONG_EPISODES_PER_BATCH must be a positive integer" >&2
  exit 2
}
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
}
(( BATCH_SIZE % LONG_EPISODES_PER_BATCH == 0 )) || {
  echo "BATCH_SIZE must be divisible by LONG_EPISODES_PER_BATCH" >&2
  exit 2
}

IFS=',' read -r -a RAW_LONG_TASKS <<< "${LONG_TASKS}"
LONG_TASK_LIST=()
for task in "${RAW_LONG_TASKS[@]}"; do
  task="${task#"${task%%[![:space:]]*}"}"
  task="${task%"${task##*[![:space:]]}"}"
  [[ -n "${task}" ]] || continue
  [[ "${task}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "Invalid RoboCasa-Long task name: ${task}" >&2
    exit 2
  }
  LONG_TASK_LIST+=("${task}")
done
(( ${#LONG_TASK_LIST[@]} > 0 )) || {
  echo "LONG_TASKS must contain at least one task name" >&2
  exit 2
}
LONG_TASKS_HYDRA="$(IFS=,; echo "${LONG_TASK_LIST[*]}")"

[[ -f "${MOST_PRETRAINED_CKPT}" ]] || {
  echo "Missing checkpoint: ${MOST_PRETRAINED_CKPT}" >&2
  exit 1
}
[[ -f "${MOST_VLM_CONFIG_PATH}/config.json" ]] || {
  echo "Missing VLM config: ${MOST_VLM_CONFIG_PATH}/config.json" >&2
  exit 1
}
for task in "${LONG_TASK_LIST[@]}"; do
  [[ -d "${ROBOCASA_TARGET_ROOT}/long/${task}" ]] || {
    echo "Missing RoboCasa-Long task: ${ROBOCASA_TARGET_ROOT}/long/${task}" >&2
    exit 1
  }
done

mkdir -p "${RUN_ROOT}" "$(dirname -- "${LOG_FILE}")"
export MOST_VLM_CONFIG_PATH
export RESOURCE_GPU="${GPU_COUNT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
export TOKENIZERS_PARALLELISM=false
export ROBOCASA_LONG_SOURCE_FPS="${ROBOCASA_LONG_SOURCE_FPS:-20.0}"

if [[ -z "${LONG_ACTION_STATS_PATH}" || "${LONG_ACTION_STATS_PATH}" == "null" ]]; then
  LONG_ACTION_STATS_PATH="${RUN_ROOT}/robocasa_long_absrot_stats.json"
fi
STATS_TASK_ARGS=()
for task in "${LONG_TASK_LIST[@]}"; do
  STATS_TASK_ARGS+=(--task "${task}")
done
"${PYTHON:-python}" -u tools/compute_robocasa_long_absrot_stats.py \
  --data-root "${ROBOCASA_TARGET_ROOT}" \
  --base-model "${MODEL_PATH}" \
  --base-robot-type robocasa365 \
  --action-length 16 \
  --min-rotation-std "${LONG_MIN_ROTATION_STD}" \
  --output "${LONG_ACTION_STATS_PATH}" \
  "${STATS_TASK_ARGS[@]}"

echo "Training MOST on pure RoboCasa-Long data: tasks=${LONG_TASKS_HYDRA}, run=${RUN_NAME}"

bash scripts/train.sh \
  data=robocasa_long \
  model=most \
  "model.params.pretrained=${MOST_PRETRAINED_CKPT}" \
  "model.params.model.memory_gradient_checkpointing=${MEMORY_GRAD_CKPT}" \
  "trainer.max_steps=${MAX_STEPS}" \
  "trainer.ckpt_path=${RESUME_CKPT:-null}" \
  "trainer.save_interval=${SAVE_INTERVAL}" \
  "trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}" \
  "trainer.default_root_dir=${RUN_ROOT}" \
  "trainer.project=${WANDB_PROJECT}" \
  "trainer.exp_name=${RUN_NAME}" \
  "+trainer.use_wandb=${USE_WANDB}" \
  "data.params.train_datasets.long_tasks=[${LONG_TASKS_HYDRA}]" \
  "data.params.train_datasets.long_episodes_per_batch=${LONG_EPISODES_PER_BATCH}" \
  "data.params.train_datasets.long_action_stats_path=${LONG_ACTION_STATS_PATH}" \
  "data.params.train_datasets.long_robot_type=${LONG_ROBOT_TYPE}" \
  "data.params.train_datasets.batch_size=${BATCH_SIZE}" \
  "data.params.train_datasets.num_workers=${NUM_WORKERS}" \
  2>&1 | tee "${LOG_FILE}"
