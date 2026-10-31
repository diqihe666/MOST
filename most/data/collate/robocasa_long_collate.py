# Copyright (C) 2026 MOST Contributors.
"""Long-only action normalization for absolute EEF rotation targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from most.data.collate.custom_collate import CustomCollate
from most.data.datasets.robocasa_long_dataset import (
    ABSOLUTE_ROTVEC_REPRESENTATION,
    validate_rotation_reference_quaternion_xyzw,
)


DEFAULT_LONG_ROBOT_TYPE = "robocasa_long_absrot"


class RoboCasaLongCollate(CustomCollate):
    """Install Long absolute-rotation statistics without changing RoboCasa365.

    ``CustomCollate`` is first initialized with the public ``robocasa365``
    processor entry. This class then adds a new in-memory robot type whose
    state/action mask contract is identical but whose action mean/std were
    computed from the selected Long tasks.
    """

    def __init__(self, params: Any | None = None) -> None:
        params = params or {}
        data = params.get("train_datasets", params)
        base_robot_type = str(data.get("robot_type", "robocasa365"))
        super().__init__(params)

        raw_path = str(data.get("long_action_stats_path", "") or "").strip()
        if not raw_path or raw_path.lower() == "none":
            raise ValueError(
                "RoboCasa-Long training requires long_action_stats_path; "
                "train_robocasa_long.sh generates it automatically"
            )
        stats_path = Path(raw_path).expanduser().resolve()
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"RoboCasa-Long action statistics do not exist: {stats_path}"
            )
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Long action stats must be a JSON object: {stats_path}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing metadata object in {stats_path}")
        if metadata.get("rotation_action_mode") != "absolute_rotvec":
            raise ValueError(f"Invalid RoboCasa-Long statistics file: {stats_path}")
        if metadata.get("representation") != ABSOLUTE_ROTVEC_REPRESENTATION:
            raise ValueError(
                f"Unsupported rotation representation in {stats_path}: "
                f"{metadata.get('representation')!r}"
            )
        if "rotation_reference_quaternion_xyzw" not in metadata:
            raise ValueError(
                f"Missing metadata.rotation_reference_quaternion_xyzw in {stats_path}"
            )
        rotation_reference = validate_rotation_reference_quaternion_xyzw(
            metadata["rotation_reference_quaternion_xyzw"],
            source=str(stats_path),
        )
        if str(metadata.get("base_robot_type", base_robot_type)) != base_robot_type:
            raise ValueError(
                f"Stats base_robot_type={metadata.get('base_robot_type')!r} does "
                f"not match training robot_type={base_robot_type!r}"
            )

        mean = torch.as_tensor(payload["mean"], dtype=torch.float32)
        std = torch.as_tensor(payload["std"], dtype=torch.float32)
        action_length = int(data.get("action_length", 16))
        expected_shape = (action_length, 60)
        if tuple(mean.shape) != expected_shape or tuple(std.shape) != expected_shape:
            raise ValueError(
                f"Long action stats must have shape {expected_shape}; "
                f"got mean={tuple(mean.shape)} std={tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(f"Long action stats contain NaN/Inf: {stats_path}")
        if torch.any(std[:, 3:6] <= 1e-5):
            raise ValueError(
                "Long absolute-rotation std must exceed 1e-5 in action dims 3:6"
            )

        expected_tasks = {str(value) for value in data.get("long_tasks", ())}
        stats_tasks = {str(value) for value in metadata.get("tasks", ())}
        if expected_tasks and stats_tasks != expected_tasks:
            raise ValueError(
                f"Stats tasks {sorted(stats_tasks)} do not match configured Long "
                f"tasks {sorted(expected_tasks)}"
            )

        long_robot_type = str(
            data.get("long_robot_type", DEFAULT_LONG_ROBOT_TYPE)
        ).strip()
        if not long_robot_type:
            raise ValueError("long_robot_type cannot be empty")
        self.processor.action_config[long_robot_type] = {
            "mean": mean.unsqueeze(0),
            "std": std.unsqueeze(0),
        }
        self.robot_type = long_robot_type
        self.long_action_stats_path = stats_path
        self.rotation_reference_quaternion_xyzw = rotation_reference
