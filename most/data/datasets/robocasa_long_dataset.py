# Copyright (C) 2026 MOST Contributors.
"""RoboCasa-Long-only action targets.

The public RoboCasa365 dataset remains the source of images, state, memory
labels, and every non-rotation action dimension.  This subclass changes only
the meaning of policy action dimensions ``3:6`` when explicitly requested:

``delta``
    Preserve the public RoboCasa365 rotation-delta target.

``absolute_rotvec``
    Predict the end-effector orientation relative to the robot base, encoded
    as a three-dimensional local rotation vector around a fixed Long-dataset
    reference orientation.  More precisely, the label is
    ``log(R_goal @ R_reference.T)`` in radians.  ``action[t]`` is paired with
    the achieved orientation in ``state[t + 1]``.

Keeping the representation three-dimensional preserves the original 12-D
RoboCasa action layout.  A quaternion action would introduce a sign ambiguity
and shift all later action fields.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from most.data.datasets.robocasa365_dataset import (
    RoboCasa365Dataset,
)


ROTATION_ACTION_MODES = frozenset({"delta", "absolute_rotvec"})
ABSOLUTE_ROTVEC_REPRESENTATION = (
    "base_relative_eef_local_rotvec_radians_about_dataset_reference_from_state_t_plus_1"
)


def normalize_rotation_action_mode(value: Any) -> str:
    """Return the canonical Long rotation-action mode."""
    mode = str(value or "delta").strip().lower().replace("-", "_")
    aliases = {
        "delta": "delta",
        "relative": "delta",
        "absolute": "absolute_rotvec",
        "abs": "absolute_rotvec",
        "abs_rotvec": "absolute_rotvec",
        "absolute_rotvec": "absolute_rotvec",
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError(
            f"rotation_action_mode must be delta or absolute_rotvec, got {value!r}"
        ) from exc


def validate_rotation_reference_quaternion_xyzw(
    value: Any,
    *,
    source: str = "rotation reference",
) -> np.ndarray:
    """Strictly validate and return one unit XYZW reference quaternion."""
    reference = np.asarray(value, dtype=np.float64)
    if reference.shape != (4,):
        raise ValueError(
            f"{source} rotation_reference_quaternion_xyzw must have shape (4,), "
            f"got {reference.shape}"
        )
    if not np.all(np.isfinite(reference)):
        raise ValueError(
            f"{source} rotation_reference_quaternion_xyzw contains NaN/Inf"
        )
    norm = float(np.linalg.norm(reference))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(
            f"{source} rotation_reference_quaternion_xyzw must be unit length; "
            f"got norm={norm:.9g}"
        )
    return reference


def _quaternion_xyzw_to_rotvec_batch(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized-or-normalizable XYZW quaternions to local rotvecs."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"Expected quaternions [N,4], got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Quaternion contains NaN/Inf")
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norm[:, 0] < 1e-12):
        raise ValueError("Quaternion norm must be non-zero")
    normalized = value / norm
    # q and -q encode the same rotation.  Use the representative whose scalar
    # part is non-negative so log() follows the shortest branch.
    normalized[normalized[:, 3] < 0.0] *= -1.0
    xyz = normalized[:, :3]
    sin_half = np.linalg.norm(xyz, axis=1)
    angle = 2.0 * np.arctan2(
        sin_half,
        np.clip(normalized[:, 3], -1.0, 1.0),
    )
    output = np.zeros((len(value), 3), dtype=np.float64)
    moving = sin_half >= 1e-12
    output[moving] = xyz[moving] / sin_half[moving, None] * angle[moving, None]
    return output


def relative_rotation_vectors(
    goal_quaternion_xyzw: np.ndarray,
    reference_quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Return ``log(R_goal @ R_reference.T)`` for XYZW quaternions."""
    goal = np.asarray(goal_quaternion_xyzw, dtype=np.float64)
    if goal.ndim != 2 or goal.shape[1] != 4:
        raise ValueError(f"Expected goal quaternions [N,4], got {goal.shape}")
    if not np.all(np.isfinite(goal)):
        raise ValueError("Goal quaternion contains NaN/Inf")
    goal_norm = np.linalg.norm(goal, axis=1, keepdims=True)
    if np.any(goal_norm[:, 0] < 1e-12):
        raise ValueError("Goal quaternion norm must be non-zero")
    goal = goal / goal_norm
    reference = validate_rotation_reference_quaternion_xyzw(reference_quaternion_xyzw)
    inverse_reference = reference.copy()
    inverse_reference[:3] *= -1.0

    x1, y1, z1, w1 = goal.T
    x2, y2, z2, w2 = inverse_reference
    relative = np.column_stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )
    return _quaternion_xyzw_to_rotvec_batch(relative)


def absolute_rotation_targets(
    raw_state: np.ndarray,
    frame_index: int,
    action_length: int,
    reference_quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Build reference-centered ``q[t+k+1]`` EEF rotation-vector targets.

    The final target is clamped to the final observation because the exported
    episode contains no state after its last action.  The parent dataset's
    temporal mask still masks every padded action slot.
    """
    state = np.asarray(raw_state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 16:
        raise ValueError(f"Expected raw RoboCasa state [N,16], got {state.shape}")
    if state.shape[0] <= 0:
        raise ValueError("RoboCasa episode state cannot be empty")
    frame_index = int(frame_index)
    action_length = int(action_length)
    if not 0 <= frame_index < state.shape[0]:
        raise IndexError(frame_index)
    if action_length <= 0:
        raise ValueError("action_length must be positive")

    target_indices = np.minimum(
        frame_index + np.arange(action_length, dtype=np.int64) + 1,
        state.shape[0] - 1,
    )
    return relative_rotation_vectors(
        state[target_indices, 10:14],
        reference_quaternion_xyzw,
    ).astype(np.float32, copy=False)


class RoboCasaLongDataset(RoboCasa365Dataset):
    """Long-only wrapper with an optional absolute EEF orientation target."""

    def __init__(self, params: Any):
        data = params["train_datasets"]
        if str(data.get("task_group", "")) != "long":
            raise ValueError("RoboCasaLongDataset only supports task_group='long'")
        self.rotation_action_mode = normalize_rotation_action_mode(
            data.get("rotation_action_mode", "delta")
        )
        self.rotation_reference_quaternion_xyzw: np.ndarray | None = None
        if self.rotation_action_mode == "absolute_rotvec":
            raw_path = str(data.get("long_action_stats_path", "") or "").strip()
            if not raw_path or raw_path.lower() == "none":
                raise ValueError(
                    "absolute_rotvec training requires long_action_stats_path"
                )
            stats_path = Path(raw_path).expanduser().resolve()
            if not stats_path.is_file():
                raise FileNotFoundError(
                    f"RoboCasa-Long action statistics do not exist: {stats_path}"
                )
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Long action stats must be a JSON object: {stats_path}"
                )
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"Missing metadata object in {stats_path}")
            if metadata.get("rotation_action_mode") != "absolute_rotvec":
                raise ValueError(
                    f"{stats_path} is not an absolute_rotvec statistics file"
                )
            if metadata.get("representation") != ABSOLUTE_ROTVEC_REPRESENTATION:
                raise ValueError(
                    f"Unsupported rotation representation in {stats_path}: "
                    f"{metadata.get('representation')!r}"
                )
            if "rotation_reference_quaternion_xyzw" not in metadata:
                raise ValueError(
                    f"Missing metadata.rotation_reference_quaternion_xyzw in "
                    f"{stats_path}"
                )
            self.rotation_reference_quaternion_xyzw = (
                validate_rotation_reference_quaternion_xyzw(
                    metadata["rotation_reference_quaternion_xyzw"],
                    source=str(stats_path),
                )
            )
        super().__init__(params)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["meta"]["rotation_action_mode"] = self.rotation_action_mode
        if self.rotation_action_mode == "delta":
            return sample

        raw_index = int(index) % self.raw_samples
        episode_position = bisect.bisect_right(self.cumulative_frames, raw_index)
        previous = (
            0
            if episode_position == 0
            else int(self.cumulative_frames[episode_position - 1])
        )
        frame_index = raw_index - previous
        episode = self.episodes[episode_position]
        arrays = self._read_long_episode(str(episode.parquet_path))
        if self.rotation_reference_quaternion_xyzw is None:
            raise RuntimeError(
                "absolute_rotvec reference quaternion is not initialized"
            )
        target = absolute_rotation_targets(
            arrays.state,
            frame_index=frame_index,
            action_length=self.action_length,
            reference_quaternion_xyzw=self.rotation_reference_quaternion_xyzw,
        )
        sample["action"][:, 3:6] = torch.from_numpy(target)
        return sample
