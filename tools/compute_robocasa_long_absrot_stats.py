#!/usr/bin/env python3
"""Compute Long-only action statistics for absolute EEF rotation vectors.

All non-rotation dimensions retain the public RoboCasa365 processor statistics.
Only action dimensions 3:6 are replaced by statistics for the achieved
``state[t+k+1]`` EEF orientation relative to the robot base.  Orientations are
encoded locally around the Markley mean of all selected Long-state
quaternions: ``log(R_goal @ R_reference.T)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoProcessor


ACTION_DIM = 60
RAW_STATE_DIM = 16
ROTATION_SLICE = slice(3, 6)
ABSOLUTE_ROTVEC_REPRESENTATION = (
    "base_relative_eef_local_rotvec_radians_about_dataset_reference_from_state_t_plus_1"
)


def normalize_quaternion_xyzw_batch(quaternion: np.ndarray) -> np.ndarray:
    """Validate and normalize a batch of XYZW quaternions."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"Expected quaternions [N,4], got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Quaternion contains NaN/Inf")
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norm[:, 0] < 1e-12):
        raise ValueError("Quaternion norm must be non-zero")
    return value / norm


def quaternion_xyzw_to_rotvec_batch(quaternion: np.ndarray) -> np.ndarray:
    """Canonical shortest-path XYZW quaternion to rotation-vector conversion."""
    normalized = normalize_quaternion_xyzw_batch(quaternion)
    normalized[normalized[:, 3] < 0] *= -1.0

    xyz = normalized[:, :3]
    sin_half = np.linalg.norm(xyz, axis=1)
    angle = 2.0 * np.arctan2(
        sin_half,
        np.clip(normalized[:, 3], -1.0, 1.0),
    )
    output = np.zeros((len(normalized), 3), dtype=np.float64)
    moving = sin_half >= 1e-12
    output[moving] = xyz[moving] / sin_half[moving, None] * angle[moving, None]
    return output


def markley_mean_quaternion_xyzw(scatter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the principal eigenvector of a quaternion scatter matrix."""
    matrix = np.asarray(scatter, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"Expected finite quaternion scatter [4,4], got {matrix.shape}"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    reference = eigenvectors[:, -1]
    # Pick a deterministic representative of the q/-q equivalence class.
    if reference[3] < 0.0:
        reference *= -1.0
    elif abs(reference[3]) < 1e-12:
        nonzero = np.flatnonzero(np.abs(reference) >= 1e-12)
        if len(nonzero) and reference[nonzero[0]] < 0.0:
            reference *= -1.0
    reference /= np.linalg.norm(reference)
    return reference, eigenvalues


def relative_rotation_vectors(
    goal_quaternion_xyzw: np.ndarray,
    reference_quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Return ``log(R_goal @ R_reference.T)`` for XYZW quaternions."""
    goal = normalize_quaternion_xyzw_batch(goal_quaternion_xyzw)
    reference = np.asarray(reference_quaternion_xyzw, dtype=np.float64)
    if reference.shape != (4,) or not np.all(np.isfinite(reference)):
        raise ValueError(
            f"Expected finite reference quaternion [4], got {reference.shape}"
        )
    reference_norm = float(np.linalg.norm(reference))
    if not np.isclose(reference_norm, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"Reference quaternion must be unit length, got {reference_norm}"
        )
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
    return quaternion_xyzw_to_rotvec_batch(relative)


def _processor_stats(
    model_path: str,
    robot_type: str,
    action_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_path = Path(model_path).expanduser()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=local_path.exists(),
    )
    if robot_type not in processor.action_config:
        raise KeyError(
            f"Robot type {robot_type!r} is absent from {model_path}; "
            f"available={list(processor.action_config)}"
        )

    def prepare(value: object, name: str) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != ACTION_DIM:
            raise ValueError(
                f"Processor {name} must end in {ACTION_DIM} dims, got {array.shape}"
            )
        if array.shape[0] == 1:
            array = np.repeat(array, action_length, axis=0)
        elif array.shape[0] >= action_length:
            array = array[:action_length]
        else:
            raise ValueError(
                f"Processor {name} has {array.shape[0]} horizons, fewer than "
                f"requested action_length={action_length}"
            )
        return array.copy()

    config = processor.action_config[robot_type]
    return prepare(config["mean"], "mean"), prepare(config["std"], "std")


def _episode_paths(data_root: Path, tasks: list[str]) -> list[Path]:
    paths: list[Path] = []
    for task in tasks:
        task_dir = data_root / "long" / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Long task directory does not exist: {task_dir}")
        task_paths = sorted((task_dir / "data").glob("**/episode_*.parquet"))
        if not task_paths:
            raise FileNotFoundError(
                f"No episode parquet files under {task_dir / 'data'}"
            )
        paths.extend(task_paths)
    return paths


def _episode_state(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["observation.state"])
    state = np.asarray(
        table["observation.state"].combine_chunks().to_pylist(),
        dtype=np.float32,
    )
    if state.ndim != 2 or state.shape[1] != RAW_STATE_DIM:
        raise ValueError(
            f"Expected state [N,{RAW_STATE_DIM}], got {state.shape}: {path}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"State contains NaN/Inf: {path}")
    if state.shape[0] == 0:
        raise ValueError(f"Episode contains no states: {path}")
    return state


def compute(
    data_root: Path,
    tasks: list[str],
    base_model: str,
    base_robot_type: str,
    action_length: int,
    min_rotation_std: float,
) -> dict:
    mean, std = _processor_stats(base_model, base_robot_type, action_length)
    paths = _episode_paths(data_root, tasks)

    # Markley / chordal quaternion mean.  The outer product makes this
    # invariant to the arbitrary q/-q sign stored by each episode.
    quaternion_scatter = np.zeros((4, 4), dtype=np.float64)
    num_frames = 0
    for path in paths:
        state = _episode_state(path)
        quaternion = normalize_quaternion_xyzw_batch(state[:, 10:14])
        quaternion_scatter += quaternion.T @ quaternion
        num_frames += len(quaternion)
    if num_frames <= 0:
        raise ValueError("Selected Long data contains no orientation states")
    rotation_reference, markley_eigenvalues = markley_mean_quaternion_xyzw(
        quaternion_scatter
    )

    sums = np.zeros((action_length, 3), dtype=np.float64)
    square_sums = np.zeros((action_length, 3), dtype=np.float64)
    counts = np.zeros(action_length, dtype=np.int64)
    adjacent_jumps: list[np.ndarray] = []

    for path in paths:
        state = _episode_state(path)
        rotvec = relative_rotation_vectors(
            state[:, 10:14],
            rotation_reference,
        )
        if len(rotvec) > 1:
            adjacent_jumps.append(np.linalg.norm(np.diff(rotvec, axis=0), axis=1))

        # Match the dataset exactly.  For horizon k, every valid action at
        # sample t uses state[min(t+k+1, N-1)].  The last state is therefore
        # included twice (the episode contains no post-final-action state).
        for horizon in range(action_length):
            valid_samples = len(rotvec) - horizon
            if valid_samples <= 0:
                continue
            indices = np.minimum(
                np.arange(valid_samples, dtype=np.int64) + horizon + 1,
                len(rotvec) - 1,
            )
            values = rotvec[indices]
            sums[horizon] += values.sum(axis=0)
            square_sums[horizon] += np.square(values).sum(axis=0)
            counts[horizon] += len(values)

    if np.any(counts == 0):
        raise ValueError(
            f"No valid rotation targets for horizons {np.flatnonzero(counts == 0).tolist()}"
        )
    rotation_mean = sums / counts[:, None]
    variance = np.maximum(square_sums / counts[:, None] - rotation_mean**2, 0.0)
    rotation_std_raw = np.sqrt(variance)
    rotation_std = np.maximum(rotation_std_raw, float(min_rotation_std))
    mean[:, ROTATION_SLICE] = rotation_mean.astype(np.float32)
    std[:, ROTATION_SLICE] = rotation_std.astype(np.float32)

    if adjacent_jumps:
        jumps = np.concatenate(adjacent_jumps)
        jump_summary = {
            "p50": float(np.quantile(jumps, 0.50)),
            "p95": float(np.quantile(jumps, 0.95)),
            "p99": float(np.quantile(jumps, 0.99)),
            "max": float(jumps.max(initial=0.0)),
        }
    else:
        jump_summary = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "metadata": {
            "rotation_action_mode": "absolute_rotvec",
            "representation": ABSOLUTE_ROTVEC_REPRESENTATION,
            "rotation_reference_quaternion_xyzw": rotation_reference.tolist(),
            "rotation_reference_method": "markley_chordal_mean_all_selected_long_states",
            "rotation_reference_markley_eigenvalues": markley_eigenvalues.tolist(),
            "base_robot_type": base_robot_type,
            "long_robot_type": "robocasa_long_absrot",
            "tasks": tasks,
            "action_length": action_length,
            "num_episodes": len(paths),
            "num_frames": int(num_frames),
            "min_rotation_std_radians": float(min_rotation_std),
            "rotation_mean": rotation_mean.tolist(),
            "rotation_std_before_floor": rotation_std_raw.tolist(),
            "rotation_std_after_floor": rotation_std.tolist(),
            "adjacent_rotvec_jump_radians": jump_summary,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-robot-type", default="robocasa365")
    parser.add_argument("--action-length", type=int, default=16)
    parser.add_argument("--min-rotation-std", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action_length <= 0:
        parser.error("--action-length must be positive")
    if args.min_rotation_std <= 1e-5:
        parser.error("--min-rotation-std must be greater than 1e-5")
    tasks = list(
        dict.fromkeys(str(task).strip() for task in args.tasks if str(task).strip())
    )
    if not tasks:
        parser.error("At least one non-empty --task is required")

    result = compute(
        args.data_root.expanduser().resolve(),
        tasks,
        args.base_model,
        args.base_robot_type,
        args.action_length,
        args.min_rotation_std,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote RoboCasa-Long absolute-rotation stats: {output}")
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
