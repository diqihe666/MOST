# MOST on RoboCasa-Long

[Back to the main README](../../README.md)

This guide covers the RoboCasa-Long dataset, MOST training, Long-specific
checkpoint export, simulator integration, and distributed evaluation. Install
the policy and shared RoboCasa environments from the
[main README](../../README.md) before continuing.

RoboCasa-Long training in this repository uses only Long data. It does not
construct or read a RoboCasa365 Composite dataset.

## Components

| Component | Link |
| --- | --- |
| Dataset | [diqihe666/robocasa_long](https://huggingface.co/datasets/diqihe666/robocasa_long) |
| Simulator task code | [most-memory/RoboCasa-Long](https://github.com/most-memory/RoboCasa-Long) |

The simulator repository is a planned release and is not required for policy
training from the released offline dataset. It is required for simulator
evaluation once public.

## Dataset download

Install the Hugging Face CLI in the policy environment:

```bash
conda activate most
pip install -U huggingface_hub
```

Download the released dataset into the `long/` group expected by the training
loader:

```bash
export ROBOCASA_TARGET_ROOT=/path/to/most-data
mkdir -p "$ROBOCASA_TARGET_ROOT/long"

hf download diqihe666/robocasa_long \
  --repo-type dataset \
  --local-dir "$ROBOCASA_TARGET_ROOT/long"
```

The resulting layout must place each task directly below `long/`:

```text
$ROBOCASA_TARGET_ROOT/
  long/
    PackFruit/
      data/
      videos/
      meta/
      labels/
    <OtherTask>/
      ...
```

Each task contains episode Parquet files, three synchronized camera videos,
metadata, and memory labels such as:

```text
<task-directory>/labels/episode_000000.json
```

If a downloaded snapshot contains an additional wrapper directory, move or
link the task directories so they match the layout above before training.

## Required paths

Prepare the common initialization checkpoint as described in the main README,
then export:

```bash
export MODEL_PATH=/path/to/compatible-robocasa365-base-model
export ROBOCASA_TARGET_ROOT=/path/to/most-data
export MOST_PRETRAINED_CKPT="$PWD/pretrained_ckpt/robocasa365/model_states.pt"
export MOST_VLM_CONFIG_PATH="$PWD/pretrained_ckpt/robocasa365/vlm_config"
export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"
export MOST_MEM_DINO_CKPT="$PWD/third_party/dinov2/dinov2_vits14_pretrain.pth"
```

## Training

Example: train on `PackFruit` for 20,000 steps with eight GPUs:

```bash
conda activate most

LONG_TASKS=PackFruit \
LONG_EPISODES_PER_BATCH=2 \
BATCH_SIZE=16 \
NUM_WORKERS=8 \
MAX_STEPS=20000 \
SAVE_INTERVAL=5000 \
GPU_COUNT=8 \
RUN_NAME=most-long-packfruit-e2-20k \
bash scripts/train_robocasa_long.sh
```

`LONG_TASKS` accepts a comma-separated list. `LONG_EPISODES_PER_BATCH` must
divide `BATCH_SIZE`; `2` is the recommended starting point. Samples in a local
batch are grouped by episode and `frame_index % memory_stride` so their sparse
memory histories can share decoding work.

Resume from a full Lightning/DeepSpeed checkpoint directory:

```bash
RESUME_CKPT=/path/to/epoch=0-step=10000.ckpt \
RUN_NAME=most-long-packfruit-e2-20k \
bash scripts/train_robocasa_long.sh
```

### Absolute-rotation training

The default action representation is `delta`. To use the supported absolute
rotation-vector representation:

```bash
LONG_ROTATION_ACTION_MODE=absolute_rotvec \
bash scripts/train_robocasa_long.sh
```

The launcher computes and stores the required rotation statistics in the run
directory.

## Checkpoint export

For a standard `delta` checkpoint, use the common exporter documented in the
[main README](../../README.md#5-export-a-checkpoint).

For a checkpoint trained with `absolute_rotvec`, use:

```bash
BASE_MODEL=/path/to/compatible-robocasa365-base-model
CHECKPOINT=/path/to/epoch=0-step=20000.ckpt
EXPORT_DIR="$PWD/exports/most-long-step20000"

python tools/export_hf_checkpoint_robocasa_long.py \
  --base-model "$BASE_MODEL" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXPORT_DIR" \
  --stats /path/to/robocasa_long_absrot_stats.json
```

## Simulator environment

Start with the shared RoboCasa environment from the
[main README](../../README.md#2-shared-robocasa-simulator-environment). When
the RoboCasa-Long simulator repository is public, install its Long task
definitions into that environment:

```bash
conda activate robocasa
git clone https://github.com/most-memory/RoboCasa-Long.git third_party/robocasa-long
pip install -e third_party/robocasa-long
```

The installed package must register the selected Long tasks with RoboCasa
before evaluation starts.

## Evaluation

Start the model servers as described in the main README before launching the
simulator workers.

Smoke test `PackFruit` with three episodes:

```bash
RUN_ID="long-smoke-$(date +%Y%m%d-%H%M%S)"

BASE_PORT=10086 \
LONG_TASKS=PackFruit \
NUM_TRIALS=3 \
CONDA_ENV=robocasa \
RUN_ID="$RUN_ID" \
SEED=7 \
bash scripts/launch_robocasa_long.sh \
  3 \
  eval_results/long-smoke \
  "$EXPORT_DIR" \
  --save-videos
```

Run 30 episodes with eight workers:

```bash
RUN_ID="long-packfruit-$(date +%Y%m%d-%H%M%S)"

BASE_PORT=10086 \
LONG_TASKS=PackFruit \
NUM_TRIALS=30 \
CONDA_ENV=robocasa \
RUN_ID="$RUN_ID" \
SEED=7 \
bash scripts/launch_robocasa_long.sh \
  8 \
  eval_results/long-packfruit \
  "$EXPORT_DIR" \
  --save-videos
```

Without `LONG_TASKS`, the launcher evaluates the default tasks:
`PackTwoLunches`, `PrepareBreakfast`, and `CleanAndStoreAfterMeal`.

See the [main README](../../README.md#7-results-and-logs) for the shared result
layout and troubleshooting notes.
