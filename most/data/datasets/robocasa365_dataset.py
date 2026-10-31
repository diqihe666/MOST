# Copyright (C) 2026 MOST Contributors.
"""RoboCasa / RoboCasa365 episode dataset for MOST post-training.

The loader consumes the episode-based LeRobot v2.1 layout produced by the
RoboCasa exporter used in this project::

    <data_root>/<atomic|composite>/<TaskName>/
      data/chunk-XXX/episode_XXXXXX.parquet
      videos/chunk-XXX/<camera-key>/episode_XXXXXX.mp4
      meta/target10.txt or meta/target100.txt

Custom task groups can live beside ``atomic`` and ``composite``.  RoboCasa-Long
uses the following layout in this project::

    <data_root>/long/<TaskName>/
      data/chunk-XXX/episode_XXXXXX.parquet
      videos/chunk-XXX/<camera-key>/episode_XXXXXX.mp4
      meta/episodes.jsonl
      labels/episode_XXXXXX.json

For standard RoboCasa365 training, ``episode_ids_file`` selects target10/100
episodes exactly as before.  For RoboCasa-Long, set ``episode_ids_file: null``
to train on every episode in each selected task.  Episode ids are read from
``meta/episodes.jsonl`` when available, with a parquet filename scan as a
fallback.

The dataset reproduces
MOST's public RoboCasa365 evaluator input contract:

* three camera videos: left, right, wrist;
* four history frames with configurable interval (default 2);
* raw 16D LeRobot state converted to the EE-first 14D axis-angle state;
* state padded to the MOST unified 60D state space;
* raw 12D RoboCasa action padded to the MOST unified 60D action space;
* 30-step action chunks clamped at the episode boundary.

Action and state normalization are intentionally performed by CustomCollate
using the processor shipped with the public RoboCasa365 checkpoint.  This keeps
fine-tuning numerically aligned with public checkpoint inference.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
import json
import os
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
import decord
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

STATE_DIM = 60
ACTION_DIM = 60
ROBOCASA_STATE_DIM = 16
ROBOCASA_ACTION_DIM = 12

CAMERA_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)

_EPISODE_ID_RE = re.compile(r"^(?:episode_)?(\d+)$")
_EPISODE_FILE_RE = re.compile(r"^episode_(\d+)\.parquet$")


@dataclass(frozen=True)
class EpisodeRecord:
    task_name: str
    task_dir: Path
    episode_id: int
    parquet_path: Path
    video_paths: tuple[Path, Path, Path]
    label_path: Path | None
    num_frames: int
    instruction: str


@dataclass(frozen=True)
class EpisodeArrays:
    state: np.ndarray
    action: np.ndarray
    instruction: str


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def robocasa_action_exporter_to_policy(action: np.ndarray) -> np.ndarray:
    """Convert dataset exporter-order 12D actions to MOST policy order.

    Exporter order:
        [base_motion(4), control_mode(1), ee_position(3),
         ee_rotation(3), gripper_close(1)]

    MOST / RoboCasa ``convert_action`` policy order:
        [ee_position(3), ee_rotation(3), gripper_close(1),
         base_motion(4), control_mode(1)]
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != ROBOCASA_ACTION_DIM:
        raise ValueError(
            f"Expected action last dimension {ROBOCASA_ACTION_DIM}, got {action.shape}"
        )

    policy_action = np.empty_like(action, dtype=np.float32)
    policy_action[..., 0:3] = action[..., 5:8]
    policy_action[..., 3:6] = action[..., 8:11]
    policy_action[..., 6:7] = action[..., 11:12]
    policy_action[..., 7:11] = action[..., 0:4]
    policy_action[..., 11:12] = action[..., 4:5]
    return policy_action


def robocasa_state_16_to_14(state: np.ndarray) -> np.ndarray:
    """Convert exporter order to the public MOST evaluator state order.

    Input order (16D):
        base pos 3, base quat xyzw 4, EE relative pos 3,
        EE relative quat xyzw 4, gripper qpos 2.

    Output order (14D):
        EE relative pos 3, EE axis-angle 3, gripper qpos 2,
        base pos 3, base axis-angle 3.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (ROBOCASA_STATE_DIM,):
        raise ValueError(f"Expected 16D RoboCasa state, got {state.shape}")
    converted = np.concatenate(
        [
            state[7:10],
            quat_xyzw_to_axis_angle(state[10:14]),
            state[14:16],
            state[0:3],
            quat_xyzw_to_axis_angle(state[3:7]),
        ]
    ).astype(np.float32)
    if converted.shape != (14,):
        raise AssertionError(f"Internal state conversion produced {converted.shape}")
    return converted


def _parse_episode_ids(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Episode selector does not exist: {path}")
    ids: list[int] = []
    seen: set[int] = set()
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        match = _EPISODE_ID_RE.fullmatch(value)
        if match is None:
            raise ValueError(f"{path}:{line_no}: invalid episode id {raw!r}")
        episode_id = int(match.group(1))
        if episode_id in seen:
            raise ValueError(f"{path}:{line_no}: duplicate episode id {episode_id}")
        seen.add(episode_id)
        ids.append(episode_id)
    if not ids:
        raise ValueError(f"Episode selector is empty: {path}")
    return ids


def _all_episode_ids(task_dir: Path) -> list[int]:
    """Return every exported episode id for one task in deterministic order."""
    metadata_path = task_dir / "meta" / "episodes.jsonl"
    ids: list[int] = []
    seen: set[int] = set()
    if metadata_path.is_file():
        for record in _read_jsonl(metadata_path):
            value = record.get("episode_index", record.get("episode_id"))
            if value is None:
                continue
            try:
                episode_id = int(str(value).replace("episode_", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid episode id {value!r} in {metadata_path}"
                ) from exc
            if episode_id in seen:
                raise ValueError(
                    f"Duplicate episode id {episode_id} in {metadata_path}"
                )
            seen.add(episode_id)
            ids.append(episode_id)
        if ids:
            return sorted(ids)

    data_dir = task_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot discover episodes: neither {metadata_path} nor {data_dir} exists"
        )
    for parquet_path in sorted(data_dir.glob("**/episode_*.parquet")):
        match = _EPISODE_FILE_RE.fullmatch(parquet_path.name)
        if match is None:
            continue
        episode_id = int(match.group(1))
        if episode_id in seen:
            raise ValueError(
                f"Multiple parquet files found for episode {episode_id} under {data_dir}"
            )
        seen.add(episode_id)
        ids.append(episode_id)
    if not ids:
        raise ValueError(f"No episodes found under {task_dir}")
    return sorted(ids)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_no}, got {type(value)}")
        records.append(value)
    return records


def _instruction_from_metadata(task_dir: Path, episode_id: int) -> str:
    """Read the canonical instruction for one LeRobot episode.

    RoboCasa365 stores the episode-to-language mapping in
    ``meta/episodes.jsonl`` as, for example::

        {"episode_index": 0, "tasks": ["Pick up ..."], "length": 542}

    ``meta/tasks.jsonl`` is the global task vocabulary and is not used to infer
    an episode instruction when ``episodes.jsonl`` already provides it.
    """
    metadata_path = task_dir / "meta" / "episodes.jsonl"
    for record in _read_jsonl(metadata_path):
        record_id = record.get("episode_index", record.get("episode_id"))
        try:
            record_id = int(str(record_id).replace("episode_", ""))
        except (TypeError, ValueError):
            continue
        if record_id != episode_id:
            continue

        tasks = record.get("tasks")
        if isinstance(tasks, str):
            tasks = [tasks]
        if isinstance(tasks, (list, tuple)):
            instructions = [
                str(value).strip()
                for value in tasks
                if isinstance(value, str) and value.strip()
            ]
            instructions = list(dict.fromkeys(instructions))
            if len(instructions) == 1:
                return instructions[0]
            if len(instructions) > 1:
                raise ValueError(
                    f"Episode {episode_id} in {metadata_path} has multiple "
                    f"distinct task instructions: {instructions}"
                )

        # Backward-compatible fallbacks for non-standard exports.
        for key in (
            "annotation.human.task_description",
            "task_description",
            "instruction",
            "task",
        ):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        raise ValueError(
            f"Episode {episode_id} exists in {metadata_path} but has no usable "
            "instruction in its 'tasks' field"
        )

    raise KeyError(f"Episode {episode_id} is missing from {metadata_path}")


def _find_episode_file(
    task_dir: Path, parent: str, episode_id: int, suffix: str
) -> Path:
    chunk = episode_id // 1000
    candidates = [
        task_dir / parent / f"chunk-{chunk:03d}" / f"episode_{episode_id:06d}{suffix}",
        task_dir / parent / f"episode_{episode_id:06d}{suffix}",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted((task_dir / parent).glob(f"**/episode_{episode_id:06d}{suffix}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files found for episode {episode_id} under {task_dir / parent}: {matches}"
        )
    raise FileNotFoundError(
        f"Missing episode {episode_id} under {task_dir / parent}; tried {candidates}"
    )


def _find_video_file(task_dir: Path, camera_key: str, episode_id: int) -> Path:
    chunk = episode_id // 1000
    candidates = [
        task_dir
        / "videos"
        / f"chunk-{chunk:03d}"
        / camera_key
        / f"episode_{episode_id:06d}.mp4",
        task_dir / "videos" / camera_key / f"episode_{episode_id:06d}.mp4",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(
        (task_dir / "videos").glob(f"**/{camera_key}/episode_{episode_id:06d}.mp4")
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple videos found for {camera_key}, episode {episode_id}: {matches}"
        )
    raise FileNotFoundError(
        f"Missing {camera_key} video for episode {episode_id} under {task_dir / 'videos'}"
    )


def _column_to_numpy(table: Any, name: str, dtype: np.dtype) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"Parquet is missing required column {name!r}")
    values = table[name].combine_chunks().to_pylist()
    try:
        return np.stack(values).astype(dtype, copy=False)
    except Exception as exc:
        raise ValueError(f"Cannot stack parquet column {name!r}") from exc


class RoboCasa365Dataset(Dataset):
    def __init__(self, params: Any):
        data = params["train_datasets"]
        self.action_length = int(
            data.get("action_length", params.get("action_length", 16))
        )
        self.obs_history = int(data.get("obs_history", 4))
        self.obs_interval = int(data.get("obs_interval", 2))
        # Long-term memory cadence is configured independently from MOST short
        # history.  RoboCasa-Long keeps memory_stride=16 so training density
        # matches replan=16 at evaluation.
        self.memory_enabled = bool(data.get("memory_enabled", False))
        # Long episodes contain heavily overlapping memory prefixes inside a
        # block-shuffled batch.  When enabled, the collate/model path keeps the
        # full logical memory sequence but shares identical *past* RGB
        # timesteps across samples before DINO.  Current timesteps are never
        # shared so the original DINO gradient path is preserved.
        self.memory_dedup_past = bool(data.get("memory_dedup_past", False))
        self.memory_stride = int(data.get("memory_stride", self.action_length))
        self.memory_future_steps = int(data.get("memory_future_steps", 16))
        self.memory_num_subtask_classes = int(
            data.get("memory_num_subtask_classes", 256)
        )
        self.memory_num_state_classes = int(data.get("memory_num_state_classes", 128))
        self.memory_num_object_categories = int(
            data.get("memory_num_object_categories", 256)
        )
        self.memory_max_instance_slots = int(data.get("memory_max_instance_slots", 32))
        self.source_video_fps = float(data.get("source_video_fps", 20.0))
        self.crop_ratio = float(data.get("crop_ratio", 0.95))
        self.augment = bool(data.get("augment", True))
        self.batch_size = int(data.get("batch_size", 1))
        self.strict = bool(data.get("strict", True))

        self.prompt_mode = str(data.get("prompt_mode", "training")).lower()
        if self.prompt_mode not in {"training", "inference"}:
            raise ValueError(
                f"prompt_mode must be 'training' or 'inference', got {self.prompt_mode!r}"
            )
        episode_ids_file = data.get("episode_ids_file", "meta/target10.txt")
        self.episode_ids_file = (
            None if episode_ids_file is None else str(episode_ids_file).strip()
        )
        if self.episode_ids_file == "":
            self.episode_ids_file = None
        self.data_root = Path(str(data["data_root"])).expanduser().resolve()
        self.task_group = str(data["task_group"])
        self.tasks = [str(task) for task in data["tasks"]]
        if not self.tasks:
            raise ValueError("RoboCasa task list cannot be empty")
        if self.task_group not in {"atomic", "composite", "long"}:
            raise ValueError(
                f"task_group must be atomic, composite, or long, got {self.task_group!r}"
            )
        self.expected_episodes_per_task = data.get("expected_episodes_per_task")
        if self.expected_episodes_per_task is not None:
            self.expected_episodes_per_task = int(self.expected_episodes_per_task)

        # Keep the original RoboCasa365 lru_cache path untouched.  Only long
        # videos use a smaller per-worker cache so three 3-4 minute camera
        # decoders do not multiply across many persistent workers.
        self.video_reader_cache_size = int(data.get("video_reader_cache_size", 3))
        self.episode_cache_size = int(data.get("episode_cache_size", 4))
        # Long batches contain many samples from the same local block. Cache
        # decoded *raw* frames before augmentation so repeated memory-prefix
        # requests do not seek/decode the same MP4 frames again.  The cache is
        # worker-local (Dataset instances live inside DataLoader workers), and
        # augmentation still runs independently after the cache lookup.
        self.raw_frame_cache_size = int(data.get("raw_frame_cache_size", 768))
        if (
            self.video_reader_cache_size <= 0
            or self.episode_cache_size <= 0
            or self.raw_frame_cache_size < 0
        ):
            raise ValueError(
                "video_reader_cache_size/episode_cache_size must be positive and "
                "raw_frame_cache_size must be non-negative"
            )
        self._long_video_reader_cache: OrderedDict[str, VideoReader] = OrderedDict()
        self._long_episode_cache: OrderedDict[str, EpisodeArrays] = OrderedDict()
        self._long_raw_frame_cache: OrderedDict[tuple[str, int], np.ndarray] = (
            OrderedDict()
        )
        # RoboCasa-Long samples in one DataLoader batch share most of their
        # sparse memory prefix. PyTorch can fetch a whole batch through
        # ``__getitems__``; while that batch is being assembled, keep a
        # temporary cache containing the union of decoded frames. Unlike the
        # persistent LRU above, this cache is never evicted mid-batch.
        self.batch_decode_prewarm = bool(
            data.get("batch_decode_prewarm", self.task_group == "long")
        )
        self._long_batch_frame_cache: dict[tuple[str, int], np.ndarray] | None = None
        if self.action_length <= 0 or self.obs_history <= 0 or self.obs_interval <= 0:
            raise ValueError(
                "action_length, obs_history, and obs_interval must be positive"
            )
        if self.memory_enabled and (
            self.memory_stride <= 0 or self.memory_future_steps < 0
        ):
            raise ValueError(
                "memory_stride must be positive and memory_future_steps non-negative"
            )
        if self.source_video_fps <= 0:
            raise ValueError("source_video_fps must be positive")
        self.video_fps = self.source_video_fps / self.obs_interval

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        max_steps = int(params.get("max_steps", 1000))
        self.max_samples = max_steps * self.batch_size * world_size
        self.episodes = self._discover_episodes()
        self.memory_subtask_to_idx: dict[str, int] = {}
        self.memory_state_to_idx: dict[str, int] = {}
        self.memory_category_to_idx: dict[str, int] = {}
        if self.memory_enabled:
            self._build_memory_label_vocab()
        self.cumulative_frames = np.cumsum(
            np.asarray(
                [episode.num_frames for episode in self.episodes], dtype=np.int64
            )
        )
        self.raw_samples = int(self.cumulative_frames[-1])
        if self.raw_samples <= 0:
            raise ValueError("Selected RoboCasa365 episodes contain no frames")
        if self.task_group == "long":
            logger.info(
                "RoboCasa-Long dataset: tasks=%d episodes=%d raw_frames=%d train_samples=%d",
                len(self.tasks),
                len(self.episodes),
                self.raw_samples,
                self.max_samples,
            )
        else:
            logger.info(
                "RoboCasa365Dataset: tasks=%d episodes=%d raw_frames=%d train_samples=%d",
                len(self.tasks),
                len(self.episodes),
                self.raw_samples,
                self.max_samples,
            )

    def __len__(self) -> int:
        return self.max_samples

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        """Batch-aware fetch used by PyTorch DataLoader auto-collation.

        For RoboCasa-Long, pre-decode the union of all MOST-history and
        Memory-Expert-history frames requested by this local batch.  Long
        samples share large sparse prefixes, so doing one Decord ``get_batch``
        per (episode, camera) avoids repeatedly seeking through the same 3-4
        minute MP4 for every sample.

        The decoded cache stores raw uint8 RGB only.  ``__getitem__`` still
        executes the original per-sample crop/color augmentation, so this is
        an I/O optimization only and does not change the training samples.
        """
        if (
            self.task_group != "long"
            or not self.batch_decode_prewarm
            or self.raw_frame_cache_size == 0
            or not indices
        ):
            return [self.__getitem__(index) for index in indices]

        requirements: dict[str, set[int]] = {}
        for index in indices:
            raw_index = int(index) % self.raw_samples
            episode_position = bisect.bisect_right(self.cumulative_frames, raw_index)
            previous = (
                0
                if episode_position == 0
                else int(self.cumulative_frames[episode_position - 1])
            )
            frame_index = raw_index - previous
            episode = self.episodes[episode_position]
            history_indices = self._history_indices(frame_index)
            memory_history_indices = (
                self._memory_history_indices(frame_index) if self.memory_enabled else []
            )
            visual_indices = set(history_indices) | set(memory_history_indices)
            for path in episode.video_paths:
                requirements.setdefault(str(path), set()).update(visual_indices)

        # A worker handles one batch at a time, so a dataset-local temporary
        # cache is safe here. Keep it alive until every __getitem__ below has
        # applied its own augmentation.
        self._long_batch_frame_cache = {}
        try:
            for video_path, requested in requirements.items():
                self._prewarm_long_batch_frames(video_path, tuple(sorted(requested)))
            return [self.__getitem__(index) for index in indices]
        finally:
            self._long_batch_frame_cache = None

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_index = int(index) % self.raw_samples
        episode_position = bisect.bisect_right(self.cumulative_frames, raw_index)
        previous = (
            0
            if episode_position == 0
            else int(self.cumulative_frames[episode_position - 1])
        )
        frame_index = raw_index - previous
        episode = self.episodes[episode_position]
        arrays = (
            self._read_long_episode(str(episode.parquet_path))
            if self.task_group == "long"
            else self._read_episode(str(episode.parquet_path))
        )
        if frame_index >= arrays.state.shape[0]:
            raise IndexError(
                f"Resolved frame {frame_index} outside episode {episode.episode_id} "
                f"length {arrays.state.shape[0]}"
            )

        history_indices = self._history_indices(frame_index)
        memory_history_indices = (
            self._memory_history_indices(frame_index) if self.memory_enabled else []
        )
        state = np.zeros((self.obs_history, STATE_DIM), dtype=np.float32)
        for output_index, source_index in enumerate(history_indices):
            converted = robocasa_state_16_to_14(arrays.state[source_index])
            state[output_index, : converted.shape[0]] = converted

        action_indices = np.minimum(
            frame_index + np.arange(self.action_length, dtype=np.int64),
            arrays.action.shape[0] - 1,
        )
        steps = min(self.action_length, arrays.action.shape[0] - frame_index)
        raw_action = arrays.action[action_indices]
        if raw_action.shape != (self.action_length, ROBOCASA_ACTION_DIM):
            raise ValueError(
                f"Unexpected action window shape {raw_action.shape} in {episode.parquet_path}"
            )
        policy_action = robocasa_action_exporter_to_policy(raw_action)
        action = np.zeros((self.action_length, ACTION_DIM), dtype=np.float32)
        action[:, :ROBOCASA_ACTION_DIM] = policy_action
        temporal_mask = np.zeros(self.action_length, dtype=np.int32)
        temporal_mask[:steps] = 1

        # Decode the union once so the normal MOST branch and sparse H branch
        # share exactly the same crop/color augmentation for overlapping frames.
        all_visual_indices = sorted(set(history_indices) | set(memory_history_indices))
        decoded = {
            camera_key: self._decode_video(str(path), tuple(all_visual_indices))
            for camera_key, path in zip(CAMERA_KEYS, episode.video_paths)
        }
        decoded = self._transform_videos(decoded)
        by_camera = {
            camera_key: {
                frame: image
                for frame, image in zip(all_visual_indices, decoded[camera_key])
            }
            for camera_key in CAMERA_KEYS
        }
        videos = {
            camera_key: [by_camera[camera_key][frame] for frame in history_indices]
            for camera_key in CAMERA_KEYS
        }
        instruction = arrays.instruction or episode.instruction
        if not instruction:
            raise ValueError(
                f"No task instruction found for {episode.task_name} episode {episode.episode_id}"
            )

        result = {
            "messages": self._messages(videos, instruction, steps, self.prompt_mode),
            "action": torch.from_numpy(action),
            "temporal_action_mask": torch.from_numpy(temporal_mask),
            "state": torch.from_numpy(state),
            # The videos above are already sampled at obs_interval from the
            # original 20 Hz RoboCasa stream. Qwen3-VL needs the effective
            # clip FPS only to construct timestamps; it must not resample.
            "video_fps": self.video_fps,
            "vlm_action_actual_length": steps,
            "meta": {
                "task_name": episode.task_name,
                "episode_id": episode.episode_id,
                "frame_index": frame_index,
            },
        }
        if self.memory_enabled:
            memory_images = [
                [by_camera[camera_key][frame] for camera_key in CAMERA_KEYS]
                for frame in memory_history_indices
            ]
            result.update(
                self._memory_payload(
                    episode=episode,
                    frame_index=frame_index,
                    history_indices=memory_history_indices,
                    images=memory_images,
                    instruction=instruction,
                )
            )
        return result

    def _memory_history_indices(self, frame_index: int) -> list[int]:
        frame_index = int(frame_index)
        indices = list(range(frame_index, -1, -self.memory_stride))
        indices.reverse()
        return indices

    @staticmethod
    @lru_cache(maxsize=256)
    def _read_memory_label(label_path: str) -> dict[str, Any]:
        payload = json.loads(Path(label_path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if len(payload) != 1 or not isinstance(payload[0], dict):
                raise ValueError(f"Expected one label object in {label_path}")
            payload = payload[0]
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {label_path}")
        info = payload.get("label_info", payload)
        actions = info.get("action_config", info.get("subtasks", []))
        states = info.get("state_config", info.get("states", []))
        if not isinstance(actions, list) or not isinstance(states, list):
            raise ValueError(f"Invalid action_config/state_config in {label_path}")
        actions = sorted(actions, key=lambda item: int(item.get("start_frame", 0)))
        return {"actions": actions, "states": states}

    def _build_memory_label_vocab(self) -> None:
        subtasks: set[str] = set()
        states: set[str] = set()
        categories: set[str] = set()
        max_instance = -1
        for episode in self.episodes:
            if episode.label_path is None:
                raise FileNotFoundError(
                    f"Memory enabled but label is missing for {episode.task_name}/{episode.episode_id}"
                )
            label = self._read_memory_label(str(episode.label_path))
            for action in label["actions"]:
                text = str(action.get("action_text", "")).strip()
                if text:
                    subtasks.add(text)
            for obj in label["states"]:
                category = str(obj.get("category", "")).strip()
                if category:
                    categories.add(category)
                max_instance = max(max_instance, int(obj.get("instance_id", 0)))
                for segment in obj.get("state", []):
                    text = str(segment.get("state_text", "")).strip()
                    if text:
                        states.add(text)
        self.memory_subtask_to_idx = {
            text: i for i, text in enumerate(sorted(subtasks))
        }
        self.memory_state_to_idx = {text: i for i, text in enumerate(sorted(states))}
        self.memory_category_to_idx = {
            text: i for i, text in enumerate(sorted(categories))
        }
        if len(self.memory_subtask_to_idx) > self.memory_num_subtask_classes:
            raise ValueError(
                f"Need {len(self.memory_subtask_to_idx)} subtask classes but model/data config allows "
                f"{self.memory_num_subtask_classes}"
            )
        if len(self.memory_state_to_idx) > self.memory_num_state_classes:
            raise ValueError(
                f"Need {len(self.memory_state_to_idx)} state classes but config allows "
                f"{self.memory_num_state_classes}"
            )
        if len(self.memory_category_to_idx) > self.memory_num_object_categories:
            raise ValueError(
                f"Need {len(self.memory_category_to_idx)} object categories but config allows "
                f"{self.memory_num_object_categories}"
            )
        if max_instance >= self.memory_max_instance_slots:
            raise ValueError(
                f"Observed instance_id={max_instance}, but memory_max_instance_slots="
                f"{self.memory_max_instance_slots}"
            )
        logger.info(
            "Memory labels: subtasks=%d states=%d categories=%d max_instance=%d",
            len(self.memory_subtask_to_idx),
            len(self.memory_state_to_idx),
            len(self.memory_category_to_idx),
            max_instance,
        )

    def _memory_payload(
        self,
        episode: EpisodeRecord,
        frame_index: int,
        history_indices: list[int],
        images: list[list[Image.Image]],
        instruction: str,
    ) -> dict[str, Any]:
        if episode.label_path is None:
            raise FileNotFoundError(
                "Memory supervision requires labels/episode_XXXXXX.json"
            )
        label = self._read_memory_label(str(episode.label_path))
        # Future H tokens live on the same memory grid.  Anchor them to the
        # latest actually observed memory frame rather than the arbitrary MOST
        # training frame, so train/eval temporal semantics are identical.
        memory_anchor = int(history_indices[-1])
        future_indices = [
            memory_anchor + self.memory_stride * step
            for step in range(1, self.memory_future_steps + 1)
        ]
        sequence_indices = list(history_indices) + future_indices

        subtask_labels = torch.full((len(sequence_indices),), -1, dtype=torch.long)
        for seq_index, raw_frame in enumerate(sequence_indices):
            if raw_frame >= episode.num_frames:
                continue
            for action in label["actions"]:
                if int(action["start_frame"]) <= raw_frame <= int(action["end_frame"]):
                    text = str(action["action_text"]).strip()
                    subtask_labels[seq_index] = self.memory_subtask_to_idx[text]
                    break

        objects = label["states"]
        num_objects = len(objects)
        category_ids = torch.full((num_objects,), -1, dtype=torch.long)
        instance_ids = torch.full((num_objects,), -1, dtype=torch.long)
        state_labels = torch.full(
            (len(sequence_indices), num_objects), -1, dtype=torch.long
        )
        for object_index, obj in enumerate(objects):
            category = str(obj["category"]).strip()
            category_ids[object_index] = self.memory_category_to_idx[category]
            instance_ids[object_index] = int(obj["instance_id"])
            for seq_index, raw_frame in enumerate(sequence_indices):
                if raw_frame >= episode.num_frames:
                    continue
                for segment in obj.get("state", []):
                    if (
                        int(segment["start_frame"])
                        <= raw_frame
                        <= int(segment["end_frame"])
                    ):
                        state_text = str(segment["state_text"]).strip()
                        state_labels[seq_index, object_index] = (
                            self.memory_state_to_idx[state_text]
                        )
                        break

        return {
            "memory_images": images,
            "memory_history_indices": torch.tensor(history_indices, dtype=torch.long),
            "memory_sequence_indices": torch.tensor(sequence_indices, dtype=torch.long),
            "memory_history_length": len(history_indices),
            "memory_future_steps": self.memory_future_steps,
            "memory_subtask_labels": subtask_labels,
            "memory_state_labels": state_labels,
            "memory_object_category_ids": category_ids,
            "memory_object_instance_ids": instance_ids,
            "memory_task_text": instruction,
            # This key is used only for optional batch-local past-memory
            # deduplication.  Include the task name because episode ids restart
            # from 000000 for every RoboCasa task.
            **(
                {"memory_episode_key": f"{episode.task_name}/{episode.episode_id}"}
                if self.memory_dedup_past
                else {}
            ),
            "memory_subtask_class_count": len(self.memory_subtask_to_idx),
            "memory_state_class_count": len(self.memory_state_to_idx),
        }

    def _discover_episodes(self) -> list[EpisodeRecord]:
        episodes: list[EpisodeRecord] = []
        for task_name in self.tasks:
            task_dir = self.data_root / self.task_group / task_name
            if not task_dir.is_dir():
                raise FileNotFoundError(f"Task directory does not exist: {task_dir}")
            if self.episode_ids_file is None:
                episode_ids = _all_episode_ids(task_dir)
            else:
                selector = task_dir / self.episode_ids_file
                episode_ids = _parse_episode_ids(selector)
            if (
                self.expected_episodes_per_task is not None
                and len(episode_ids) != self.expected_episodes_per_task
            ):
                raise ValueError(
                    f"Expected {self.expected_episodes_per_task} episodes for {task_name}, "
                    f"found {len(episode_ids)}: {episode_ids}"
                )
            for episode_id in episode_ids:
                parquet_path = _find_episode_file(
                    task_dir, "data", episode_id, ".parquet"
                )
                video_paths = tuple(
                    _find_video_file(task_dir, camera_key, episode_id)
                    for camera_key in CAMERA_KEYS
                )
                table = pq.read_table(
                    parquet_path,
                    columns=[
                        "frame_index",
                        "observation.state",
                        "action",
                    ],
                )
                num_frames = int(table.num_rows)
                if num_frames <= 0:
                    raise ValueError(f"Empty episode parquet: {parquet_path}")
                frame_values = np.asarray(
                    table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64
                ).reshape(-1)
                if not np.array_equal(
                    frame_values, np.arange(num_frames, dtype=np.int64)
                ):
                    raise ValueError(
                        f"Non-contiguous frame_index in {parquet_path}: "
                        f"first={frame_values[:5]} last={frame_values[-5:]}"
                    )
                states = _column_to_numpy(table, "observation.state", np.float32)
                actions = _column_to_numpy(table, "action", np.float32)
                if states.shape != (num_frames, ROBOCASA_STATE_DIM):
                    raise ValueError(
                        f"Expected state [{num_frames},16], got {states.shape}: {parquet_path}"
                    )
                if actions.shape != (num_frames, ROBOCASA_ACTION_DIM):
                    raise ValueError(
                        f"Expected action [{num_frames},12], got {actions.shape}: {parquet_path}"
                    )
                instruction = _instruction_from_metadata(task_dir, episode_id)
                label_path = task_dir / "labels" / f"episode_{episode_id:06d}.json"
                if self.memory_enabled and not label_path.is_file():
                    raise FileNotFoundError(
                        f"Memory supervision label does not exist: {label_path}"
                    )
                episodes.append(
                    EpisodeRecord(
                        task_name=task_name,
                        task_dir=task_dir,
                        episode_id=episode_id,
                        parquet_path=parquet_path,
                        video_paths=video_paths,
                        label_path=label_path if label_path.is_file() else None,
                        num_frames=num_frames,
                        instruction=instruction,
                    )
                )
        if not episodes:
            if self.task_group in {"atomic", "composite"}:
                raise ValueError("No RoboCasa365 episodes were selected")
            raise ValueError("No RoboCasa-Long episodes were selected")
        return episodes

    @staticmethod
    @lru_cache(maxsize=32)
    def _read_episode(parquet_path: str) -> EpisodeArrays:
        """Original RoboCasa365 parquet cache path."""
        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state",
                "action",
            ],
        )
        state = _column_to_numpy(table, "observation.state", np.float32)
        action = _column_to_numpy(table, "action", np.float32)
        return EpisodeArrays(state=state, action=action, instruction="")

    def _read_long_episode(self, parquet_path: str) -> EpisodeArrays:
        cached = self._long_episode_cache.pop(parquet_path, None)
        if cached is not None:
            self._long_episode_cache[parquet_path] = cached
            return cached

        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state",
                "action",
            ],
        )
        arrays = EpisodeArrays(
            state=_column_to_numpy(table, "observation.state", np.float32),
            action=_column_to_numpy(table, "action", np.float32),
            instruction="",
        )
        self._long_episode_cache[parquet_path] = arrays
        while len(self._long_episode_cache) > self.episode_cache_size:
            self._long_episode_cache.popitem(last=False)
        return arrays

    @staticmethod
    @lru_cache(maxsize=24)
    def _video_reader(video_path: str) -> VideoReader:
        """Original RoboCasa365 video-reader cache path."""
        return VideoReader(video_path, num_threads=1)

    def _long_video_reader(self, video_path: str) -> VideoReader:
        reader = self._long_video_reader_cache.pop(video_path, None)
        if reader is not None:
            self._long_video_reader_cache[video_path] = reader
            return reader

        reader = VideoReader(video_path, num_threads=1)
        self._long_video_reader_cache[video_path] = reader
        while len(self._long_video_reader_cache) > self.video_reader_cache_size:
            _, stale_reader = self._long_video_reader_cache.popitem(last=False)
            del stale_reader
        return reader

    def _prewarm_long_batch_frames(
        self, video_path: str, indices: tuple[int, ...]
    ) -> None:
        """Decode missing raw frames once for the active Long batch."""
        if not indices:
            return
        if self._long_batch_frame_cache is None:
            return

        reader = self._long_video_reader(video_path)
        if max(indices) >= len(reader):
            raise IndexError(
                f"Video {video_path} has {len(reader)} frames, requested {max(indices)}"
            )

        missing: list[int] = []
        for frame_index in indices:
            key = (video_path, int(frame_index))
            # Prefer a persistent-cache hit, but copy the reference into the
            # active batch cache so later LRU eviction cannot cause re-decode.
            cached = self._long_raw_frame_cache.pop(key, None)
            if cached is not None:
                self._long_raw_frame_cache[key] = cached
                self._long_batch_frame_cache[key] = cached
            else:
                missing.append(int(frame_index))

        if not missing:
            return

        unique_missing = list(dict.fromkeys(missing))
        decord.bridge.set_bridge("torch")
        decoded = reader.get_batch(unique_missing)
        if not isinstance(decoded, torch.Tensor):
            decoded = torch.as_tensor(decoded.asnumpy())
        decoded = decoded.cpu().contiguous()

        for frame_index, frame in zip(unique_missing, decoded):
            array = frame.numpy().copy()
            key = (video_path, int(frame_index))
            self._long_batch_frame_cache[key] = array
            self._long_raw_frame_cache[key] = array
            while len(self._long_raw_frame_cache) > self.raw_frame_cache_size:
                self._long_raw_frame_cache.popitem(last=False)

    def _decode_video(
        self, video_path: str, indices: tuple[int, ...]
    ) -> list[Image.Image]:
        """Decode MOST + Memory Expert indices in one Decord batch.

        Atomic/composite tasks keep the original cached VideoReader path.
        RoboCasa-Long additionally caches decoded raw uint8 frames *before*
        crop/color augmentation.  Block-local samples share almost the entire
        sparse memory prefix, so only newly requested frames need MP4 decoding;
        every sample still receives its own augmentation afterwards.
        """
        if self.task_group == "long":
            reader = self._long_video_reader(video_path)
        else:
            reader = self._video_reader(video_path)
        if not indices:
            raise ValueError("Video history indices cannot be empty")
        if max(indices) >= len(reader):
            raise IndexError(
                f"Video {video_path} has {len(reader)} frames, requested {max(indices)}"
            )

        # Preserve the original RoboCasa365 behavior exactly.
        if self.task_group != "long" or self.raw_frame_cache_size == 0:
            decord.bridge.set_bridge("torch")
            frames = reader.get_batch(list(indices))  # [T,H,W,C], uint8
            if not isinstance(frames, torch.Tensor):
                frames = torch.as_tensor(frames.asnumpy())
            frames = frames.cpu().contiguous()
            return [Image.fromarray(frame.numpy(), mode="RGB") for frame in frames]

        missing: list[int] = []
        for frame_index in indices:
            key = (video_path, int(frame_index))
            if (
                self._long_batch_frame_cache is not None
                and key in self._long_batch_frame_cache
            ):
                continue
            cached = self._long_raw_frame_cache.pop(key, None)
            if cached is None:
                missing.append(int(frame_index))
            else:
                # Refresh LRU order.
                self._long_raw_frame_cache[key] = cached

        if missing:
            # indices are unique/sorted in the caller, but keep this robust to
            # future changes without asking Decord for the same frame twice.
            unique_missing = list(dict.fromkeys(missing))
            decord.bridge.set_bridge("torch")
            decoded = reader.get_batch(unique_missing)
            if not isinstance(decoded, torch.Tensor):
                decoded = torch.as_tensor(decoded.asnumpy())
            decoded = decoded.cpu().contiguous()
            for frame_index, frame in zip(unique_missing, decoded):
                # Own the backing storage independently of the temporary Decord
                # batch. A uint8 ndarray is substantially cheaper than keeping
                # decoder tensors/contexts alive in the worker.
                array = frame.numpy().copy()
                key = (video_path, int(frame_index))
                self._long_raw_frame_cache[key] = array
                if self._long_batch_frame_cache is not None:
                    self._long_batch_frame_cache[key] = array
                while len(self._long_raw_frame_cache) > self.raw_frame_cache_size:
                    self._long_raw_frame_cache.popitem(last=False)

        # Image.fromarray may share ndarray storage. Copy the PIL image so LRU
        # eviction cannot invalidate an image while augmentation is using it.
        output: list[Image.Image] = []
        for frame_index in indices:
            key = (video_path, int(frame_index))
            array = (
                self._long_batch_frame_cache.get(key)
                if self._long_batch_frame_cache is not None
                else None
            )
            if array is None:
                array = self._long_raw_frame_cache[key]
            output.append(Image.fromarray(array, mode="RGB").copy())
        return output

    def _history_indices(self, frame_index: int) -> list[int]:
        return [
            max(0, frame_index - (self.obs_history - 1 - index) * self.obs_interval)
            for index in range(self.obs_history)
        ]

    def _transform_videos(
        self, videos: dict[str, list[Image.Image]]
    ) -> dict[str, list[Image.Image]]:
        ops = (
            (adjust_brightness, 1.0 + random.uniform(-32.0 / 255.0, 32.0 / 255.0)),
            (adjust_contrast, random.uniform(0.5, 1.5)),
            (adjust_saturation, random.uniform(0.5, 1.5)),
            (adjust_hue, 0.0),
        )
        flags = [self.augment and random.randint(0, 1) == 0 for _ in ops]
        output: dict[str, list[Image.Image]] = {}
        for camera_key, frames in videos.items():
            transformed = []
            for image in frames:
                image = self._center_crop(image)
                for use, (operation, value) in zip(flags, ops):
                    if use:
                        image = operation(image, value)
                transformed.append(image)
            output[camera_key] = transformed
        return output

    def _center_crop(self, image: Image.Image) -> Image.Image:
        if self.crop_ratio >= 1.0:
            return image
        width, height = image.size
        crop_width = max(1, int(width * self.crop_ratio))
        crop_height = max(1, int(height * self.crop_ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return image.resize((width, height), resampling)

    @staticmethod
    def _messages(
        videos: dict[str, list[Image.Image]],
        instruction: str,
        steps: int,
        prompt_mode: str = "training",
    ) -> list[dict[str, Any]]:
        history_length = len(videos[CAMERA_KEYS[0]])
        if history_length <= 0:
            raise ValueError("RoboCasa365 sample has no observation-history frames")
        for camera_key in CAMERA_KEYS[1:]:
            if len(videos[camera_key]) != history_length:
                raise ValueError(
                    "RoboCasa365 camera histories have inconsistent lengths: "
                    f"{CAMERA_KEYS[0]}={history_length}, "
                    f"{camera_key}={len(videos[camera_key])}"
                )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                    {"type": "text", "text": "\nRight camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                    {"type": "text", "text": "\nWrist camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                    {
                        "type": "text",
                        "text": (
                            "\n\nGenerate robot actions for the task:\n"
                            f"{instruction} /no_cot"
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]
        if prompt_mode == "inference":
            return messages
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Robot state: "
                            + "".join("<state>" for _ in range(history_length)),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "".join(f"<a_{index}>" for index in range(steps))
                            + "<score>",
                        }
                    ],
                },
            ]
        )
        return messages
