#!/usr/bin/env bash
set -euo pipefail

# Convert a compatible RoboCasa365 Hugging Face checkpoint into the training
# format and export the nested Qwen3-VL configuration for offline construction.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to the compatible base model directory}"

OUTPUT_DIR="${MOST_PRETRAINED_DIR:-${REPO_ROOT}/pretrained_ckpt/robocasa365}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON:-python}" tools/weight_convert.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_filename model_states.pt

cat <<MSG

Prepared MOST initialization:
  checkpoint: ${OUTPUT_DIR}/model_states.pt
  VLM config: ${OUTPUT_DIR}/vlm_config

Export these before training:
  export MOST_PRETRAINED_CKPT="${OUTPUT_DIR}/model_states.pt"
  export MOST_VLM_CONFIG_PATH="${OUTPUT_DIR}/vlm_config"
MSG
