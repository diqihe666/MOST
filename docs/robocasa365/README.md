# MOST on RoboCasa365

[Back to the main README](../../README.md)

This guide starts after the common MOST RoboCasa installation and XR-1
initialization in the [main README](../../README.md). Follow the sections in
order to verify the simulator metrics, prepare data, train MOST, export a
checkpoint, start model servers, and run RoboCasa365 evaluation.

Run every command from the root of the cloned MOST repository unless stated
otherwise.

## 1. Verify the common setup

RoboCasa365 evaluation must use the
[MOST RoboCasa environment](https://github.com/diqihe666/MOST-RoboCasa), not
an unmodified upstream RoboCasa installation. The fork supplies the semantic
progress curves and long-horizon metrics consumed by the evaluator for all 32
Composite Seen and Composite Unseen tasks.

First activate the simulator environment and verify the imported package and
all metric interfaces:

```bash
cd /path/to/MOST-vla
conda activate robocasa

python -c "import robocasa; print(robocasa.__file__)"
python third_party/most-robocasa/tests/test_gym_long_metrics.py
```

The package path must point into `third_party/most-robocasa/robocasa`, and the
test must finish with `Passed: 35/35`. Resolve any other imported RoboCasa copy
before running evaluation.

Next activate the policy environment and verify the shared files prepared in
the main README:

```bash
cd /path/to/MOST
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
```

Keep these variables exported in every terminal used for training or
checkpoint export.

## 2. Prepare RoboCasa365 data

Download and prepare RoboCasa365 by following the
[official RoboCasa dataset guide](https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md).
The official guide is the source of truth for dataset download options,
available task splits, and RoboCasa-specific assets.

After preparation, set one root containing the task groups used by MOST:

```bash
export ROBOCASA_TARGET_ROOT=/path/to/robocasa365-data
```

The relevant layout is:

```text
$ROBOCASA_TARGET_ROOT/
  atomic/<TaskName>/
  composite/<TaskName>/
```

Each selected task directory must contain episode Parquet files, three camera
videos, metadata, target selectors, and memory labels:

```text
<task-directory>/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<camera-key>/episode_000000.mp4
  meta/target10.txt
  meta/target100.txt
  labels/episode_000000.json
```

Validate a small subset before training. This example checks at most two
target10 episodes from `DeliverStraw`:

```bash
python tools/validate_robocasa365_target.py \
  --data-root "$ROBOCASA_TARGET_ROOT" \
  --split composite_seen \
  --tasks DeliverStraw \
  --episode-ids-file meta/target10.txt \
  --max-episodes-per-task 2
```

Do not continue until the validator finishes without errors.

## 3. Train MOST on RoboCasa365

The launcher accepts a target fraction, a task split, the maximum number of
training steps, and the number of local GPUs:

```bash
bash scripts/train_robocasa365_target.sh \
  <target10|target100> \
  <atomic|composite_seen|composite_unseen> \
  [max_steps] [num_gpus]
```

Example: train target10 composite-seen for 60,000 steps on eight GPUs:

```bash
conda activate most

export MODEL_PATH="$PWD/pretrained_ckpt/xr1-robocasa365-hf"
export ROBOCASA_TARGET_ROOT=/path/to/robocasa365-data
export MOST_PRETRAINED_CKPT="$PWD/pretrained_ckpt/robocasa365/model_states.pt"
export MOST_VLM_CONFIG_PATH="$PWD/pretrained_ckpt/robocasa365/vlm_config"
export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"
export MOST_MEM_DINO_CKPT="$PWD/third_party/dinov2/dinov2_vits14_pretrain.pth"

RUN_NAME=most-target10-composite-seen \
BATCH_SIZE=16 \
NUM_WORKERS=8 \
SAVE_INTERVAL=10000 \
USE_WANDB=false \
bash scripts/train_robocasa365_target.sh \
  target10 composite_seen 60000 8
```

Training writes checkpoints and the resolved configuration to:

```text
outputs/most-target10-composite-seen/
```

To resume the same run, pass the full Lightning/DeepSpeed checkpoint directory:

```bash
RESUME_CKPT="$PWD/outputs/most-target10-composite-seen/epoch=0-step=20000.ckpt" \
RUN_NAME=most-target10-composite-seen \
bash scripts/train_robocasa365_target.sh \
  target10 composite_seen 60000 8
```

## 4. Export the trained checkpoint

Select the checkpoint created by training. Replace the example step number if
you want to export a different checkpoint:

```bash
conda activate most

export MODEL_PATH="$PWD/pretrained_ckpt/xr1-robocasa365-hf"
CHECKPOINT="$PWD/outputs/most-target10-composite-seen/epoch=0-step=60000.ckpt"
EXPORT_DIR="$PWD/exports/most-target10-composite-seen-step60000"

python tools/export_hf_checkpoint.py \
  --base-model "$MODEL_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXPORT_DIR"
```

Verify the exported model contract:

```bash
test -f "$EXPORT_DIR/config.json"
grep -E '"memory_enabled"|"memory_interface"|"memory_stride"' \
  "$EXPORT_DIR/config.json"
```

The exported directory is the model path used by both the inference servers
and the evaluation clients.

## 5. Start the model servers

Start eight model-server processes on ports `10086` through `10093`:

```bash
conda activate most
export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"

BASE_PORT=10086 \
SESSION_NAME=most-r365-t10-cs \
CONDA_ENV=most \
bash scripts/deploy.sh "$EXPORT_DIR" 8 8
```

Inspect the server session:

```bash
tmux attach -t most-r365-t10-cs
```

Detach without stopping the servers with `Ctrl+B`, then `D`.

## 6. Run RoboCasa365 evaluation

Leave the model servers running. Open another terminal at the MOST repository
root. The launcher activates the `robocasa` environment through `CONDA_ENV`.

First run a two-episode smoke test on one task:

```bash
EXPORT_DIR="$PWD/exports/most-target10-composite-seen-step60000"
RUN_ID="r365-smoke-$(date +%Y%m%d-%H%M%S)"

BASE_PORT=10086 \
TASK_SET=composite_seen \
SPLIT=target \
NUM_TRIALS=2 \
CONDA_ENV=robocasa \
RUN_ID="$RUN_ID" \
SEED=7 \
bash scripts/launch_robocasa365.sh \
  1 \
  eval_results/robocasa365-smoke \
  "$EXPORT_DIR" \
  --task-name DeliverStraw \
  --save-videos
```

After the smoke test succeeds, run the full composite-seen evaluation with
eight workers and 30 episodes per task:

```bash
EXPORT_DIR="$PWD/exports/most-target10-composite-seen-step60000"
RUN_ID="r365-t10-cs-$(date +%Y%m%d-%H%M%S)"

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
  "$EXPORT_DIR" \
  --save-videos
```

The merged summary is written to:

```text
eval_results/most-t10-cs/<RUN_ID>/summary.json
```

When evaluation is complete, stop the model servers:

```bash
tmux kill-session -t most-r365-t10-cs
```
