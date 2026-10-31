<h1 align="center">
  MOST: Memory of Object States and Task Progress<br>
  for Long-Horizon Mobile Manipulation
</h1>

<p align="center">
  Diqi He<sup>1,2</sup>, Jinbang Guo<sup>2</sup>, Yijun Shen<sup>2,3</sup>,
  Zhiwei Yu<sup>2,*</sup>,<br>
  Huan Yang<sup>2</sup>, Dingwen Zhang<sup>1,*</sup>, Junwei Han<sup>4</sup>,
  Zhongyuan Wang<sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>Northwestern Polytechnical University &nbsp;&nbsp;
  <sup>2</sup>Beijing Academy of Artificial Intelligence<br>
  <sup>3</sup>East China Normal University &nbsp;&nbsp;
  <sup>4</sup>Chongqing University of Posts and Telecommunications
</p>

<p align="center">
  <sup>*</sup>Corresponding authors
</p>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb?logo=githubpages&logoColor=white)](https://most-memory.github.io/)
[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-b31b1b?logo=arxiv&logoColor=white)](./)
[![Data](https://img.shields.io/badge/Data-RoboCasa--Long-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/diqihe666/robocasa_long)
[![Code](https://img.shields.io/badge/Code-RoboCasa--Long_Simulator-181717?logo=github&logoColor=white)](https://github.com/most-memory/RoboCasa-Long)

</div>

---

This is the official implementation of **MOST: Memory of Object States and
Task Progress for Long-Horizon Mobile Manipulation**.

## Overview

MOST is a memory-augmented vision-language-action framework for long-horizon
mobile manipulation. It structures policy-internal memory around object-state
changes and task progress, allowing historical context to directly condition
continuous action generation without an external planner or retrieval stage.

This repository provides the policy training, checkpoint conversion,
deployment, and evaluation code for **RoboCasa365** and **RoboCasa-Long**.
RoboCasa-Long data and simulator construction are maintained as separate
release components.

## Documentation

Benchmark-specific instructions are separated so that each workflow can be
followed independently:

| Benchmark | Data, training, and evaluation guide |
| --- | --- |
| RoboCasa365 | [RoboCasa365 Guide](docs/robocasa365/README.md) |
| RoboCasa-Long | [RoboCasa-Long Guide](docs/robocasa_long/README.md) |

## Release components

MOST is organized into two code repositories and one dataset release:

| Component | Purpose | Link |
| --- | --- | --- |
| MOST policy code | Training, checkpoint export, deployment, and evaluation | This repository |
| RoboCasa-Long simulator | Long-horizon task definitions and simulator environment construction | [RoboCasa-Long Simulator](https://github.com/most-memory/RoboCasa-Long) |
| RoboCasa-Long dataset | Demonstrations and memory annotations used by the Long benchmark | [Hugging Face Dataset](https://huggingface.co/datasets/diqihe666/robocasa_long) |

The simulator repository is a planned release. Until it becomes public, use
the standard RoboCasa environment for RoboCasa365 and refer to the
RoboCasa-Long guide for the expected integration point.

## Repository layout

```text
configs/                 Hydra model, data, and trainer configurations
deploy/                  Socket inference server and client
docs/                    Benchmark-specific setup and command guides
eval_robocasa365/        RoboCasa365 evaluator and worker scheduler
eval_robocasa_long/      RoboCasa-Long evaluator and worker scheduler
most/                    MOST Python package
scripts/                 Preparation, training, deployment, and evaluation scripts
tests/                   Dataset, action-contract, and Long-sampling tests
tools/                   Conversion, validation, export, and inspection utilities
```

## Requirements

- Linux
- NVIDIA GPU with a recent CUDA toolkit
- Conda or another Python environment manager
- Python 3.10 or newer; Python 3.12 is recommended for policy training
- `tmux` for multi-server evaluation
- A compatible RoboCasa365 base model in Hugging Face format
- A local DINOv2 checkout and ViT-S/14 checkpoint

The policy environment is pinned to PyTorch 2.8 and Transformers 4.57.1.
Other versions have not been validated.

## 1. Policy training environment

Create the environment used for training, checkpoint export, and model
serving:

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

## 2. Shared RoboCasa simulator environment

RoboCasa365 and RoboCasa-Long evaluations share the same base RoboCasa and
RoboSuite environment. Keep it separate from the policy environment because
the simulator and model server have different dependency requirements.

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

RoboCasa-Long evaluation additionally requires the task definitions from the
separate [RoboCasa-Long simulator repository](https://github.com/most-memory/RoboCasa-Long)
once that repository is public.

## 3. Common paths

Set the compatible base model path before preparing or training either
benchmark:

```bash
export MODEL_PATH=/path/to/compatible-robocasa365-base-model
```

Dataset-specific `ROBOCASA_TARGET_ROOT` instructions are provided in the
[RoboCasa365 guide](docs/robocasa365/README.md) and
[RoboCasa-Long guide](docs/robocasa_long/README.md).

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

## 5. Export a checkpoint

Set the compatible base model, training checkpoint, and destination:

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

The exporter performs a strict load after removing training-only choice heads,
writes the memory contract into `config.json`, and copies the remote code
required by `AutoModel.from_pretrained(..., trust_remote_code=True)`.

RoboCasa-Long checkpoints trained with `absolute_rotvec` require the wrapper
described in the [RoboCasa-Long guide](docs/robocasa_long/README.md).

## 6. Start inference servers

Start eight model-server processes on ports `10086` through `10093`:

```bash
conda activate most
export MOST_MEM_DINO_REPO="$PWD/third_party/dinov2"

BASE_PORT=10086 \
SESSION_NAME=most-servers \
CONDA_ENV=most \
bash scripts/deploy.sh "$EXPORT_DIR" 8 8
```

Inspect the servers with:

```bash
tmux attach -t most-servers
```

Detach without stopping them with `Ctrl+B`, then `D`. Use a unique
`SESSION_NAME` for each checkpoint.

## 7. Results and logs

Both evaluators use the same result layout:

```text
<log_path>/<RUN_ID>/summary.json
<log_path>/<RUN_ID>/<TaskName>/stats.json
<log_path>/scheduler/<RUN_ID>/logs/worker-*.log
<log_path>/scheduler/<RUN_ID>/errors/
```

If a client reports a socket disconnect, inspect the matching model-server
`tmux` window first; the server traceback normally contains the root cause.

## 8. Tests

```bash
conda activate most
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest -q
```

Dataset integration tests require the dependencies in `requirements.txt`.
Simulator rollouts additionally require the separate RoboCasa environment.

## To-Do List

- [x] Release RoboCasa365 training and evaluation code
- [x] Release RoboCasa-Long training and evaluation pipeline
- [ ] Release pretrained MOST checkpoints
- [ ] Release the RoboCasa-Long simulation environment code

## Acknowledgements

MOST is built upon several excellent open-source projects, including
[RoboCasa](https://github.com/robocasa/robocasa),
[robosuite](https://github.com/ARISE-Initiative/robosuite),
[DINOv2](https://github.com/facebookresearch/dinov2), and
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL). We thank their authors and
contributors for making their work publicly available. Additional attribution
for the upstream VLA implementation is provided in [NOTICE](NOTICE).

## Citation

If you find this project useful, please cite:

```bibtex
@article{he2026most,
  title   = {MOST: Memory of Object States and Task Progress
             for Long-Horizon Mobile Manipulation},
  author  = {He, Diqi and Guo, Jinbang and Shen, Yijun and
             Yu, Zhiwei and Yang, Huan and Zhang, Dingwen and
             Han, Junwei and Wang, Zhongyuan},
  year    = {2026},
  note    = {Under review}
}
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
