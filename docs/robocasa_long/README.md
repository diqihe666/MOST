# MOST on RoboCasa-Long

[Back to the main README](../../README.md)

This guide starts after the common MOST RoboCasa installation and XR-1
initialization in the [main README](../../README.md). Follow the sections in
order to verify the Long tasks, download the dataset, train MOST, export a
checkpoint, start model servers, and run RoboCasa-Long evaluation.

Run every command from the root of the cloned MOST repository unless stated
otherwise. RoboCasa-Long training uses only Long data and never constructs or
reads a RoboCasa365 Composite dataset.

## 1. Verify the MOST RoboCasa environment

The common setup in the main README installs the complete
[MOST RoboCasa environment](https://github.com/diqihe666/MOST-RoboCasa) at
`third_party/most-robocasa`. This fork is required for RoboCasa-Long: it
contains the three Long task definitions, their fixed scenes, semantic
progress trackers, the shared metric utilities, and the Gym wrapper interface
used by evaluation.

Confirm that Python imports this checkout rather than another RoboCasa package:

```bash
cd /path/to/MOST-vla
conda activate robocasa
python -c "import robocasa; print(robocasa.__file__)"
```

The printed path must point into `third_party/most-robocasa/robocasa`.

Run its startup test before downloading data or starting a model server:

```bash
cd /path/to/MOST-vla
conda activate robocasa
python third_party/most-robocasa/tests/test_robocasa_long.py
```

The test creates and resets `PackFruit`, `PrepareBreakfast`, and
`CleanAndStoreAfterMeal`. All three tasks should report `[PASS]`.

Also verify that the Gym wrapper returns `SR`, `OPS`, `UPS`, `NCT`, and the
per-step progress curve used by the evaluator:

```bash
python third_party/most-robocasa/tests/test_gym_long_metrics.py
```

This shared check covers the 32 instrumented RoboCasa365 tasks and the three
RoboCasa-Long tasks, and should finish with `Passed: 35/35`.

## 2. Download the RoboCasa-Long dataset

The training data and memory annotations are released at
[diqihe666/robocasa_long](https://huggingface.co/datasets/diqihe666/robocasa_long).

Install the Hugging Face CLI in the policy environment and download the data
into the `long/` group expected by the loader:

```bash
conda activate most
pip install -U huggingface_hub

export ROBOCASA_TARGET_ROOT=/path/to/most-data
mkdir -p "$ROBOCASA_TARGET_ROOT/long"

hf download diqihe666/robocasa_long \
  --repo-type dataset \
  --local-dir "$ROBOCASA_TARGET_ROOT/long"
```

The task directories must be directly below `long/`:

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
metadata, and per-episode memory labels:

```text
<task-directory>/labels/episode_000000.json
```

If the downloaded snapshot contains an additional wrapper directory, move or
link the task directories so they match the layout above.

## 3. Verify the policy initialization

Activate the policy environment and verify the files prepared in the main
README:

```bash
cd /path/to/MOST-vla
conda activate most

export MODEL_PATH="$PWD/pretrained_ckpt/xr1-robocasa365-hf"
export MOST_PRETRAINED_CKPT="$PWD/pretrained_ckpt/robocasa365/model_states.pt"
export MOST_VLM_CONFIG_PATH="$PWD/pretrained_ckpt/robocasa365/vlm_config"
export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"
export MOST_MEM_DINO_CKPT="$PWD/third_party/dinov2/dinov2_vits14_pretrain.pth"

test -f "$MODEL_PATH/config.json"
test -f "$MOST_PRETRAINED_CKPT"
test -f "$MOST_VLM_CONFIG_PATH/config.json"
test -f "$MOST_MEM_DINO_CKPT"
test -d "$ROBOCASA_TARGET_ROOT/long/PackFruit"
```

Keep these variables exported in every terminal used for training or
checkpoint export.

## 4. Train MOST on RoboCasa-Long

Example: train on `PackFruit` for 20,000 steps with eight GPUs:

```bash
conda activate most

export MODEL_PATH="$PWD/pretrained_ckpt/xr1-robocasa365-hf"
export ROBOCASA_TARGET_ROOT=/path/to/most-data
export MOST_PRETRAINED_CKPT="$PWD/pretrained_ckpt/robocasa365/model_states.pt"
export MOST_VLM_CONFIG_PATH="$PWD/pretrained_ckpt/robocasa365/vlm_config"
export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"
export MOST_MEM_DINO_CKPT="$PWD/third_party/dinov2/dinov2_vits14_pretrain.pth"

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

The launcher computes the task-specific action statistics required by the
RoboCasa-Long policy and stores them with the run automatically.

Training writes checkpoints, the resolved configuration, and logs to:

```text
outputs/most-long-packfruit-e2-20k/
logs/most-long-packfruit-e2-20k.log
```

To resume the same run, pass the full Lightning/DeepSpeed checkpoint directory:

```bash
RESUME_CKPT="$PWD/outputs/most-long-packfruit-e2-20k/epoch=0-step=10000.ckpt" \
RUN_NAME=most-long-packfruit-e2-20k \
bash scripts/train_robocasa_long.sh
```

## 5. Export the trained checkpoint

```bash
conda activate most

export MODEL_PATH="$PWD/pretrained_ckpt/xr1-robocasa365-hf"
CHECKPOINT="$PWD/outputs/most-long-packfruit-e2-20k/epoch=0-step=20000.ckpt"
EXPORT_DIR="$PWD/exports/most-long-packfruit-e2-step20000"

python tools/export_hf_checkpoint_robocasa_long.py \
  --base-model "$MODEL_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXPORT_DIR"
```

The Long exporter reads the action-statistics path from the saved training
configuration and packages the matching processor entry into the export.

Verify the exported model contract:

```bash
test -f "$EXPORT_DIR/config.json"
grep -E '"memory_enabled"|"memory_interface"|"memory_stride"' \
  "$EXPORT_DIR/config.json"
```

## 6. Start the model servers

Start eight model-server processes on ports `10086` through `10093`:

```bash
conda activate most
export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"

BASE_PORT=10086 \
SESSION_NAME=most-long-packfruit \
CONDA_ENV=most \
bash scripts/deploy.sh "$EXPORT_DIR" 8 8
```

Inspect the server session:

```bash
tmux attach -t most-long-packfruit
```

Detach without stopping the servers with `Ctrl+B`, then `D`.

## 7. Run RoboCasa-Long evaluation

Leave the model servers running. Open another terminal at the MOST repository
root. The launcher activates the `robocasa` environment through `CONDA_ENV`.

First run a three-episode `PackFruit` smoke test:

```bash
EXPORT_DIR="$PWD/exports/most-long-packfruit-e2-step20000"
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

After the smoke test succeeds, run 30 episodes with eight workers:

```bash
EXPORT_DIR="$PWD/exports/most-long-packfruit-e2-step20000"
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
`PackFruit`, `PrepareBreakfast`, and `CleanAndStoreAfterMeal`.

The merged summary is written to:

```text
eval_results/long-packfruit/<RUN_ID>/summary.json
```

When evaluation is complete, stop the model servers:

```bash
tmux kill-session -t most-long-packfruit
```
