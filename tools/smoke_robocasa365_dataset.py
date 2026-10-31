#!/usr/bin/env python3
"""Decode and collate one RoboCasa365 target sample before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmengine import Config

from most.data.collate.custom_collate import CustomCollate
from most.data.datasets.robocasa365_dataset import RoboCasa365Dataset


def shape(value):
    return (
        tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-group", choices=["atomic", "composite"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-ids-file", default="meta/target10.txt")
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--prompt-mode", choices=["training", "inference"], default="training"
    )
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    params = Config(
        dict(
            dataset_type="robocasa365",
            max_steps=1,
            train_datasets=dict(
                dataset_type="robocasa365",
                data_root=str(args.data_root),
                task_group=args.task_group,
                tasks=[args.task],
                episode_ids_file=args.episode_ids_file,
                processor_path=str(args.processor_path),
                robot_type="robocasa365",
                use_checkpoint_processor=True,
                action_length=16,
                obs_history=4,
                obs_interval=2,
                crop_ratio=0.95,
                augment=not args.no_augment,
                strict=True,
                prompt_mode=args.prompt_mode,
                batch_size=1,
                num_workers=0,
            ),
        )
    )

    dataset = RoboCasa365Dataset(params)
    item = dataset[args.frame_index]
    print("Dataset sample:")
    print("  meta:", item["meta"])
    print(
        "  state:",
        shape(item["state"]),
        "finite=",
        bool(torch.isfinite(item["state"]).all()),
    )
    print(
        "  action:",
        shape(item["action"]),
        "finite=",
        bool(torch.isfinite(item["action"]).all()),
    )
    print("  temporal_action_mask:", shape(item["temporal_action_mask"]))
    print("  action actual length:", item["vlm_action_actual_length"])

    collate = CustomCollate(params)
    batch = collate([item])
    print("\nCollated batch:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(
                f"  {key}: shape={tuple(value.shape)} dtype={value.dtype} "
                f"finite={bool(torch.isfinite(value).all()) if value.is_floating_point() else 'n/a'}"
            )
        else:
            print(f"  {key}: {value}")

    assert batch["state"].shape[-2:] == (4, 60)
    assert batch["action"].shape[-2:] == (16, 60)
    assert batch["action_mask"].shape[-2:] == (16, 60)
    assert batch["pixel_values_videos"].numel() > 0
    print("\nRoboCasa365 dataset + checkpoint processor smoke test passed.")


if __name__ == "__main__":
    main()
