# MOST

MOST is a memory-augmented vision-language-action training and evaluation
codebase for **RoboCasa365** and **RoboCasa-Long**. The repository contains one
final model configuration, a pure RoboCasa-Long training path, Hugging Face
checkpoint export, socket-based deployment, and distributed evaluation.

This release intentionally excludes unrelated benchmarks, real-robot adapters,
experimental memory variants, baseline-only exporters, and mixed
Long/Composite training.

## Supported workflows

- RoboCasa365 post-training on `target10` or `target100`, with atomic,
  composite-seen, or composite-unseen tasks.
- Pure RoboCasa-Long post-training with memory-aligned episode sampling.
- Checkpoint export to a self-contained Hugging Face directory.
- Multi-GPU inference servers managed through `tmux`.
- RoboCasa365 and RoboCasa-Long smoke tests and full evaluations.

## Repository layout

```text
configs/                 Hydra model, data, and trainer configurations
deploy/                  Socket inference server and client
eval_robocasa365/        RoboCasa365 evaluator and worker scheduler
eval_robocasa_long/      Direct RoboSuite RoboCasa-Long evaluator
most/                    MOST Python package
scripts/                 Preparation, training, deployment, and evaluation scripts
tests/                   Dataset, action-contract, and Long-sampling tests
tools/                   Conversion, validation, export, and inspection utilities
```

## Requirements

- Linux
- NVIDIA GPU with a recent CUDA toolkit
- Conda or another Python environment manager
- Python 3.10 or newer; Python 3.12 is recommended
- `tmux` for multi-server evaluation
- A compatible RoboCasa365 base model in Hugging Face format
- RoboCasa365 / RoboCasa-Long data and memory-supervision labels
- A local DINOv2 checkout and ViT-S/14 checkpoint

The code is pinned to PyTorch 2.8 and Transformers 4.57.1. Other versions have
not been validated.

## 1. Training environment

Create the model-training and deployment environment:

```bash
conda create -n most python=3.12 -y
conda activate most

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install flash-attn --no-build-isolation
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

Install DINOv2 and download the ViT-S/14 checkpoint:

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
wget -P third_party/dinov2 \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth

export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"
export MOST_MEM_DINO_CKPT="$PWD/third_party/dinov2/dinov2_vits14_pretrain.pth"
```

## 2. RoboCasa simulator environment

Evaluation runs in a separate environment from the model server. Install the
RoboCasa and RoboSuite versions used to create your benchmark data. The runtime
must register the requested tasks and provide `robocasa.utils.prm_metrics`.

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa

git clone https://github.com/ARISE-Initiative/robosuite.git third_party/robosuite
pip install -e third_party/robosuite

git clone https://github.com/robocasa/robocasa.git third_party/robocasa
pip install -e third_party/robocasa
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets

pip install transformers==4.57.1 "imageio[ffmpeg]" tqdm scipy
```

RoboCasa-Long custom tasks must also be registered in this environment before
launching evaluation.

## 3. Required paths

Set the base model and dataset locations:

```bash
export MODEL_PATH=/path/to/compatible-robocasa365-base-model
export ROBOCASA_TARGET_ROOT=/path/to/robocasa-target-data
```

The expected dataset layout is:

```text
$ROBOCASA_TARGET_ROOT/
  atomic/<TaskName>/
  composite/<TaskName>/
  long/<TaskName>/
```

Each selected task contains episode Parquet files, three synchronized camera
videos, metadata, and memory labels. RoboCasa365 target splits additionally use
`meta/target10.txt` or `meta/target100.txt`. Memory-enabled training expects:

```text
<task-directory>/labels/episode_000000.json
```

Use the validator before starting a long run:

```bash
python tools/validate_robocasa365_target.py \
  --data-root "$ROBOCASA_TARGET_ROOT" \
  --split composite_seen \
  --episode-ids-file meta/target10.txt
```

Replace `meta/target10.txt` with `meta/target100.txt` for the target100 split.

## 4. Prepare the initialization checkpoint

Convert the compatible Hugging Face base model into the Lightning/DeepSpeed
initialization format:

```bash
conda activate most
export MODEL_PATH=/path/to/compatible-robocasa365-base-model

bash scripts/prepare_robocasa365_target.sh

export MOST_PRETRAINED_CKPT="$PWD/pretrained_ckpt/robocasa365/model_states.pt"
export MOST_VLM_CONFIG_PATH="$PWD/pretrained_ckpt/robocasa365/vlm_config"
```

## 5. Train on RoboCasa365

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

export MODEL_PATH=/path/to/compatible-robocasa365-base-model
export ROBOCASA_TARGET_ROOT=/path/to/robocasa-target-data
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

Checkpoints are written to `outputs/<RUN_NAME>/`. Resume from a full Lightning
or DeepSpeed checkpoint directory with:

```bash
RESUME_CKPT=/path/to/epoch=0-step=20000.ckpt \
RUN_NAME=most-target10-composite-seen \
bash scripts/train_robocasa365_target.sh \
  target10 composite_seen 60000 8
```

## 6. Train on pure RoboCasa-Long data

The Long pipeline never constructs or reads a Composite dataset. Samples in a
local batch are grouped by episode and `frame_index % memory_stride` so their
sparse memory histories share decode work.

Example: train only on `PackFruit` for 20,000 steps:

```bash
conda activate most

export MODEL_PATH=/path/to/compatible-robocasa365-base-model
export ROBOCASA_TARGET_ROOT=/path/to/robocasa-target-data
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
divide `BATCH_SIZE`; `2` is the recommended starting point.

The default action representation is `delta`. To train the supported absolute
rotation representation, set:

```bash
LONG_ROTATION_ACTION_MODE=absolute_rotvec \
bash scripts/train_robocasa_long.sh
```

The launcher computes and stores the required rotation statistics in the run
directory.

## 7. Export a checkpoint

Set the full checkpoint directory or model-state file:

```bash
conda activate most

BASE_MODEL=/path/to/compatible-robocasa365-base-model
CHECKPOINT=/path/to/epoch=0-step=60000.ckpt
EXPORT_DIR="$PWD/exports/most-step60000"

python tools/export_hf_checkpoint.py \
  --base-model "$BASE_MODEL" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXPORT_DIR"
```

The exporter performs a strict load after removing training-only choice heads.
It writes the memory contract into `config.json` and copies the custom remote
code required by `AutoModel.from_pretrained(..., trust_remote_code=True)`.

For a Long checkpoint trained with `absolute_rotvec`, use the Long wrapper:

```bash
python tools/export_hf_checkpoint_robocasa_long.py \
  --base-model "$BASE_MODEL" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXPORT_DIR" \
  --stats /path/to/robocasa_long_absrot_stats.json
```

Verify the exported contract:

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads((Path("exports/most-step60000") / "config.json").read_text())
for key in ("memory_enabled", "memory_interface", "memory_stride"):
    print(f"{key}: {config.get(key)}")
PY
```

Expected values include `memory_enabled: True` and `memory_interface: mot`.

## 8. Start inference servers

Start eight server processes on ports `10086` through `10093`:

```bash
conda activate most
export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"

BASE_PORT=10086 \
SESSION_NAME=most-target10-cs \
CONDA_ENV=most \
bash scripts/deploy.sh "$EXPORT_DIR" 8 8
```

Inspect the servers with:

```bash
tmux attach -t most-target10-cs
```

Detach without stopping them with `Ctrl+B`, then `D`. The deployment launcher
refuses to overwrite an existing tmux session; choose another `SESSION_NAME` or
stop the existing session explicitly.

## 9. RoboCasa365 evaluation

Start the model servers first. Then open another terminal for the simulator.

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

## 10. RoboCasa-Long evaluation

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

Without `LONG_TASKS`, the launcher evaluates the default Long tasks:
`PackTwoLunches`, `PrepareBreakfast`, and `CleanAndStoreAfterMeal`.

## 11. Results and logs

Merged evaluation output is stored under:

```text
<log_path>/<RUN_ID>/summary.json
<log_path>/<RUN_ID>/<TaskName>/stats.json
<log_path>/scheduler/<RUN_ID>/logs/worker-*.log
<log_path>/scheduler/<RUN_ID>/errors/
```

If a client reports a socket disconnect, inspect the matching model-server tmux
window first; the server traceback normally contains the root cause.

## 12. Tests

```bash
conda activate most
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest -q
```

Dataset integration tests require the dependencies in `requirements.txt`.
Simulator rollouts additionally require the separate RoboCasa environment.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
