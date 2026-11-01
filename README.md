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

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb?logo=githubpages&logoColor=white)](https://most-vla.github.io/)
[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-b31b1b?logo=arxiv&logoColor=white)](./)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97_Data-RoboCasa--Long-ffcc4d)](https://huggingface.co/datasets/diqihe666/robocasa_long)
[![Simulator](https://img.shields.io/badge/Simulator-MOST--RoboCasa-181717?logo=github&logoColor=white)](https://github.com/diqihe666/MOST-RoboCasa)

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
The unified simulator and metric implementation, together with the
RoboCasa-Long data, are maintained as separate release components.

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
| MOST RoboCasa environment | Unified RoboCasa365 metrics, RoboCasa-Long tasks, and simulator environment | [MOST RoboCasa Environment](https://github.com/diqihe666/MOST-RoboCasa) |
| RoboCasa-Long dataset | Demonstrations and memory annotations used by the Long benchmark | [Hugging Face Dataset](https://huggingface.co/datasets/diqihe666/robocasa_long) |

Both benchmark evaluators require the MOST RoboCasa environment below. Install
it once, then follow the benchmark-specific data, training, and inference
guide.

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
git clone https://github.com/diqihe666/MOST-vla.git
cd MOST-vla

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

## 2. Shared MOST RoboCasa environment

RoboCasa365 and RoboCasa-Long evaluations both require the
[MOST RoboCasa environment](https://github.com/diqihe666/MOST-RoboCasa).
This complete RoboCasa fork preserves the upstream environments, adds semantic
progress metrics to 32 RoboCasa365 Composite tasks, and adds the three
RoboCasa-Long tasks. The MOST evaluators consume these metrics directly, so do
not substitute an unmodified upstream RoboCasa installation.

Keep the simulator environment separate from the policy environment because
the simulator and model server have different dependency requirements.

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa

git clone https://github.com/ARISE-Initiative/robosuite.git third_party/robosuite
pip install -e third_party/robosuite

git clone https://github.com/diqihe666/MOST-RoboCasa.git \
  third_party/most-robocasa
pip install -e third_party/most-robocasa
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets

pip install transformers==4.57.1 "imageio[ffmpeg]" tqdm scipy
```

Confirm that Python resolves this fork rather than another RoboCasa checkout:

```bash
python -c "import robocasa; print(robocasa.__file__)"
```

The printed path must be inside `third_party/most-robocasa/robocasa`. Then run
the complete environment and metric-interface check:

```bash
python third_party/most-robocasa/tests/test_gym_long_metrics.py
```

A successful run reports `Passed: 35/35`. See the
[MOST RoboCasa environment repository](https://github.com/diqihe666/MOST-RoboCasa)
for the supported task list and metric definitions.

## 3. Initialize MOST from XR-1

MOST is post-trained from the
[Xiaomi-Robotics-1 RoboCasa365 checkpoint](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365),
referred to as XR-1 below. The release contains the pretrained model weights,
custom Transformers code, tokenizer and processor files, and RoboCasa365
action-normalization statistics. Both the RoboCasa365 and RoboCasa-Long
training pipelines use this checkpoint as their initialization.

### 3.1 Download the XR-1 checkpoint

Install the Hugging Face CLI and download the complete repository snapshot:

```bash
conda activate most
pip install -U huggingface_hub

export XR1_MODEL_ID=XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365
export MODEL_PATH="$PWD/pretrained_ckpt/xr1-robocasa365-hf"

hf download "$XR1_MODEL_ID" \
  --local-dir "$MODEL_PATH"
```

`MODEL_PATH` must point to the downloaded Hugging Face model directory, not to
an individual `.safetensors` file. Confirm that the model and processor files
are present:

```bash
test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/processor_config.json"
```

### 3.2 Convert XR-1 for MOST training

The MOST trainer uses a Lightning/DeepSpeed initialization file rather than
loading the Hugging Face shards directly. Convert the downloaded XR-1 model:

```bash
conda activate most
export MODEL_PATH="$PWD/pretrained_ckpt/xr1-robocasa365-hf"

bash scripts/prepare_robocasa365_target.sh
```

The conversion produces:

```text
pretrained_ckpt/robocasa365/model_states.pt
pretrained_ckpt/robocasa365/vlm_config/config.json
```

Export their locations before training:

```bash
export MOST_PRETRAINED_CKPT="$PWD/pretrained_ckpt/robocasa365/model_states.pt"
export MOST_VLM_CONFIG_PATH="$PWD/pretrained_ckpt/robocasa365/vlm_config"
```

`MOST_PRETRAINED_CKPT` initializes the MOST policy from the converted XR-1
weights. `MOST_VLM_CONFIG_PATH` supplies the exact nested Qwen3-VL architecture
configuration used by XR-1. Keep `MODEL_PATH` exported as well: the data
pipeline loads the XR-1 processor and action statistics from that directory.

The benchmark launchers use these default output paths automatically, but the
explicit variables make custom locations and cluster jobs easier to audit.
Dataset-specific `ROBOCASA_TARGET_ROOT` instructions are provided in the
[RoboCasa365 guide](docs/robocasa365/README.md) and
[RoboCasa-Long guide](docs/robocasa_long/README.md).

## Training and Inference

Choose a benchmark guide below for its complete data download, training,
checkpoint export, model-server, and inference workflow:

- [RoboCasa365: Data, Training, and Inference](docs/robocasa365/README.md)
- [RoboCasa-Long: Data, Training, and Inference](docs/robocasa_long/README.md)

## To-Do List

- [x] Release RoboCasa365 training and evaluation code
- [x] Release RoboCasa-Long training and evaluation pipeline
- [x] Release the MOST RoboCasa simulation environment code
- [ ] Release pretrained MOST checkpoints

## Acknowledgements

MOST is initialized from the official
[Xiaomi-Robotics-1 (XR-1) RoboCasa365 checkpoint](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365).
The project also builds upon several excellent open-source projects, including
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
