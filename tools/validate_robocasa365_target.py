#!/usr/bin/env python3
"""Validate fixed RoboCasa365 target selectors and episode files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from decord import VideoReader

from most.data.datasets.robocasa365_dataset import (
    CAMERA_KEYS,
    ROBOCASA_ACTION_DIM,
    ROBOCASA_STATE_DIM,
    _find_episode_file,
    _find_video_file,
    _instruction_from_metadata,
    _parse_episode_ids,
)


ATOMIC_TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]
COMPOSITE_SEEN_TASKS = [
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WashLettuce",
]
COMPOSITE_UNSEEN_TASKS = [
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CategorizeCondiments",
    "CuttingToolSelection",
    "GarnishPancake",
    "GatherTableware",
    "HeatKebabSandwich",
    "MakeIceLemonade",
    "PanTransfer",
    "PortionHotDogs",
    "RecycleBottlesByType",
    "SeparateFreezerRack",
    "WaffleReheat",
    "WashFruitColander",
    "WeighIngredients",
]
TASK_SPLITS = {
    "atomic": ("atomic", ATOMIC_TASKS),
    "composite_seen": ("composite", COMPOSITE_SEEN_TASKS),
    "composite_unseen": ("composite", COMPOSITE_UNSEEN_TASKS),
}


def _resolve_tasks(args: argparse.Namespace) -> tuple[str, list[str]]:
    group, defaults = TASK_SPLITS[args.split]
    return group, list(args.tasks) if args.tasks else list(defaults)


def validate_episode(
    task_dir: Path, episode_id: int, check_video_lengths: bool
) -> dict:
    parquet_path = _find_episode_file(task_dir, "data", episode_id, ".parquet")
    table = pq.read_table(
        parquet_path,
        columns=[
            "frame_index",
            "episode_index",
            "observation.state",
            "action",
            "annotation.human.task_description",
        ],
    )
    frame_count = int(table.num_rows)
    if frame_count <= 0:
        raise ValueError("empty parquet")

    frame_index = np.asarray(
        table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64
    )
    if not np.array_equal(frame_index, np.arange(frame_count, dtype=np.int64)):
        raise ValueError("frame_index is not contiguous from zero")

    episode_index = np.asarray(
        table["episode_index"].combine_chunks().to_pylist(), dtype=np.int64
    ).reshape(-1)
    if episode_index.size != frame_count or np.any(episode_index != episode_id):
        raise ValueError(
            f"episode_index column does not equal selector id {episode_id}"
        )

    states = np.stack(table["observation.state"].combine_chunks().to_pylist())
    actions = np.stack(table["action"].combine_chunks().to_pylist())
    if states.shape != (frame_count, ROBOCASA_STATE_DIM):
        raise ValueError(f"state shape is {states.shape}, expected [{frame_count},16]")
    if actions.shape != (frame_count, ROBOCASA_ACTION_DIM):
        raise ValueError(
            f"action shape is {actions.shape}, expected [{frame_count},12]"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("state/action contains NaN or Inf")

    instruction = ""
    if "annotation.human.task_description" in table.column_names:
        instructions = (
            table["annotation.human.task_description"].combine_chunks().to_pylist()
        )
        instruction = next(
            (
                str(value).strip()
                for value in instructions
                if isinstance(value, str) and value.strip()
            ),
            "",
        )

    if not instruction:
        instruction = _instruction_from_metadata(task_dir, episode_id)

    if not instruction:
        raise ValueError("missing task instruction in parquet and meta/tasks.jsonl")

    video_lengths = {}
    for key in CAMERA_KEYS:
        path = _find_video_file(task_dir, key, episode_id)
        if check_video_lengths:
            length = len(VideoReader(str(path), num_threads=1))
            video_lengths[key] = int(length)
            if length < frame_count:
                raise ValueError(
                    f"{key} video has {length} frames, parquet has {frame_count}"
                )

    return {
        "episode_id": episode_id,
        "frames": frame_count,
        "instruction": instruction,
        "video_lengths": video_lengths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=sorted(TASK_SPLITS), default="composite_seen"
    )
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--episode-ids-file", default="meta/target10.txt")
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--skip-video-lengths", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    task_group, tasks = _resolve_tasks(args)
    report = {
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "task_group": task_group,
        "episode_ids_file": args.episode_ids_file,
        "tasks": {},
        "errors": [],
    }

    total_episodes = 0
    total_frames = 0
    for task in tasks:
        task_dir = args.data_root / task_group / task
        try:
            selector = task_dir / args.episode_ids_file
            ids = _parse_episode_ids(selector)
            selected = (
                ids[: args.max_episodes_per_task] if args.max_episodes_per_task else ids
            )
            task_records = []
            for episode_id in selected:
                record = validate_episode(
                    task_dir,
                    episode_id,
                    check_video_lengths=not args.skip_video_lengths,
                )
                task_records.append(record)
                total_episodes += 1
                total_frames += record["frames"]
            report["tasks"][task] = {
                "selector_count": len(ids),
                "validated_count": len(task_records),
                "frames": sum(item["frames"] for item in task_records),
            }
            print(
                f"[OK] {task}: selector={len(ids)} validated={len(task_records)} "
                f"frames={report['tasks'][task]['frames']}"
            )
        except Exception as exc:
            message = f"{task}: {type(exc).__name__}: {exc}"
            report["errors"].append(message)
            print(f"[ERROR] {message}")

    report["total_episodes"] = total_episodes
    report["total_frames"] = total_frames
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")

    print(
        f"Summary: tasks={len(tasks)} episodes={total_episodes} frames={total_frames} "
        f"errors={len(report['errors'])}"
    )
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
