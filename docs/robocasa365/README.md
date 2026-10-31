# MOST on RoboCasa365

[Back to the main README](../../README.md)

This guide covers RoboCasa365 data preparation, MOST training, and distributed
evaluation. Install the policy and simulator environments from the
[main README](../../README.md) before continuing.

## Data

Download and prepare RoboCasa365 by following the
[official RoboCasa dataset documentation](https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md).
MOST uses the standard LeRobot episode layout and expects the selected task
directories under one root:

```text
$ROBOCASA_TARGET_ROOT/
  atomic/<TaskName>/
  composite/<TaskName>/
```

Each task directory contains episode Parquet files, three synchronized camera
videos, metadata, and memory-supervision labels. Target subsets are selected
with `meta/target10.txt` or `meta/target100.txt`. Memory-enabled training
expects one label file per episode:

```text
<task-directory>/labels/episode_000000.json
```

Set the dataset root:

```bash
export ROBOCASA_TARGET_ROOT=/path/to/robocasa-target-data
```

Validate a subset before starting a long run:

```bash
python tools/validate_robocasa365_target.py \
  --data-root "$ROBOCASA_TARGET_ROOT" \
  --split composite_seen \
  --episode-ids-file meta/target10.txt
```

Replace `meta/target10.txt` with `meta/target100.txt` for the target100 split.

## Required paths

Prepare the common initialization checkpoint as described in the main README,
then export the following variables:

```bash
export MODEL_PATH=/path/to/compatible-robocasa365-base-model
export ROBOCASA_TARGET_ROOT=/path/to/robocasa-target-data
export MOST_PRETRAINED_CKPT="$PWD/pretrained_ckpt/robocasa365/model_states.pt"
export MOST_VLM_CONFIG_PATH="$PWD/pretrained_ckpt/robocasa365/vlm_config"
export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"
export MOST_MEM_DINO_CKPT="$PWD/third_party/dinov2/dinov2_vits14_pretrain.pth"
```

## Training

The launcher accepts one target fraction and one task split:

```bash
bash scripts/train_robocasa365_target.sh \
  <target10|target100> \
  <atomic|composite_seen|composite_unseen> \
  [max_steps] [num_gpus]
```

Example: target10 composite-seen, 60,000 steps, eight GPUs:

```bash
conda activate most

RUN_NAME=most-target10-composite-seen \
BATCH_SIZE=16 \
NUM_WORKERS=8 \
SAVE_INTERVAL=10000 \
USE_WANDB=false \
bash scripts/train_robocasa365_target.sh \
  target10 composite_seen 60000 8
```

Checkpoints are written to `outputs/<RUN_NAME>/`. Resume from a full
Lightning/DeepSpeed checkpoint directory with:

```bash
RESUME_CKPT=/path/to/epoch=0-step=20000.ckpt \
RUN_NAME=most-target10-composite-seen \
bash scripts/train_robocasa365_target.sh \
  target10 composite_seen 60000 8
```

## Checkpoint export and model servers

Export the trained checkpoint and start the inference servers by following
Sections 5 and 6 of the [main README](../../README.md). Use the resulting
Hugging Face directory as `EXPORT_DIR` below.

## Evaluation

Start the model servers first, then activate the simulator environment in a
second terminal.

Smoke test one task with two episodes:

```bash
RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"

BASE_PORT=10086 \
TASK_SET=composite_seen \
SPLIT=target \
NUM_TRIALS=2 \
CONDA_ENV=robocasa \
RUN_ID="$RUN_ID" \
bash scripts/launch_robocasa365.sh \
  1 \
  eval_results/robocasa365-smoke \
  "$EXPORT_DIR" \
  --task-name DeliverStraw
```

Run the full composite-seen evaluation with eight workers and 30 episodes per
task:

```bash
RUN_ID="most-t10-cs-$(date +%Y%m%d-%H%M%S)"

BASE_PORT=10086 \
TASK_SET=composite_seen \
SPLIT=target \
NUM_TRIALS=30 \
CONDA_ENV=robocasa \
RUN_ID="$RUN_ID" \
SEED=7 \
bash scripts/launch_robocasa365.sh \
  8 \
  eval_results/most-t10-cs \
  "$EXPORT_DIR"
```

See the [main README](../../README.md#7-results-and-logs) for the shared result
layout and troubleshooting notes.
