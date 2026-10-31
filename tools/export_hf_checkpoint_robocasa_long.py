#!/usr/bin/env python3
"""Export a Joint checkpoint and attach Long absolute-rotation processor stats.

This is a Long-only wrapper around ``export_hf_checkpoint.py``.  Keeping the
post-processing here means the generic / RoboCasa365 export path is untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
ABSOLUTE_ROTVEC_REPRESENTATION = (
    "base_relative_eef_local_rotvec_radians_about_dataset_reference_from_state_t_plus_1"
)


def _rotation_reference_quaternion(metadata: dict, stats_source: Path) -> list[float]:
    raw = metadata.get("rotation_reference_quaternion_xyzw")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(
            "Absolute-rotation statistics are missing a 4D "
            "metadata.rotation_reference_quaternion_xyzw. Regenerate the "
            f"statistics with the reference-centered implementation: {stats_source}"
        )
    quaternion = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in quaternion):
        raise ValueError(
            "metadata.rotation_reference_quaternion_xyzw contains NaN/Inf: "
            f"{stats_source}"
        )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError(
            f"metadata.rotation_reference_quaternion_xyzw has zero norm: {stats_source}"
        )
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(
            "metadata.rotation_reference_quaternion_xyzw must be unit length, "
            f"got norm={norm:.8f}: {stats_source}"
        )
    quaternion = [value / norm for value in quaternion]
    # q and -q encode the same rotation. Keep exported metadata deterministic.
    if quaternion[3] < 0:
        quaternion = [-value for value in quaternion]
    return quaternion


def _training_data_config(output_dir: Path) -> tuple[dict, Path]:
    manifest_path = output_dir / "finetune_export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Generic export did not write {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = Path(manifest["training_config"]).expanduser().resolve()
    namespace = runpy.run_path(str(config_path))
    data_cfg = dict(
        namespace.get("data", {}).get("params", {}).get("train_datasets", {})
    )
    return data_cfg, manifest_path


def patch_processor(
    output_dir: Path,
    stats_path: Path | None,
    robot_type_override: str | None,
) -> None:
    data_cfg, manifest_path = _training_data_config(output_dir)
    if stats_path is None:
        raw_path = str(data_cfg.get("long_action_stats_path", "") or "").strip()
        if not raw_path or raw_path.lower() == "none":
            raise ValueError(
                "Saved training config has no long_action_stats_path; pass --stats"
            )
        stats_path = Path(raw_path)
    stats_source = stats_path.expanduser().resolve()
    if not stats_source.is_file():
        raise FileNotFoundError(
            f"Long absolute-rotation statistics do not exist: {stats_source}"
        )
    stats = json.loads(stats_source.read_text(encoding="utf-8"))
    metadata = dict(stats.get("metadata", {}) or {})
    if metadata.get("rotation_action_mode") != "absolute_rotvec":
        raise ValueError(f"Invalid absolute-rotation statistics: {stats_source}")
    if metadata.get("representation") != ABSOLUTE_ROTVEC_REPRESENTATION:
        raise ValueError(
            "Unsupported absolute-rotation representation in "
            f"{stats_source}: {metadata.get('representation')!r}"
        )
    rotation_reference_quaternion = _rotation_reference_quaternion(
        metadata, stats_source
    )

    robot_type = str(
        robot_type_override
        or data_cfg.get("long_robot_type")
        or metadata.get("long_robot_type")
        or "robocasa_long_absrot"
    ).strip()
    if not robot_type:
        raise ValueError("Long robot type cannot be empty")

    processor_path = output_dir / "preprocessor_config.json"
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    action_config = dict(processor.get("action_config", {}) or {})
    action_config[robot_type] = {
        "mean": stats["mean"],
        "std": stats["std"],
    }
    processor["action_config"] = action_config
    processor_path.write_text(json.dumps(processor, indent=2) + "\n", encoding="utf-8")

    stats_target = output_dir / "robocasa_long_absrot_stats.json"
    if stats_source != stats_target.resolve():
        shutil.copy2(stats_source, stats_target)

    model_config_path = output_dir / "config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_config["robocasa_long_rotation_action_mode"] = "absolute_rotvec"
    model_config["robocasa_long_rotation_representation"] = (
        ABSOLUTE_ROTVEC_REPRESENTATION
    )
    model_config["robocasa_long_robot_type"] = robot_type
    model_config["robocasa_long_rotation_reference_quaternion_xyzw"] = (
        rotation_reference_quaternion
    )
    model_config["robocasa_long_action_stats_file"] = stats_target.name
    model_config_path.write_text(
        json.dumps(model_config, indent=2) + "\n", encoding="utf-8"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["robocasa_long_absolute_rotation"] = {
        "rotation_action_mode": "absolute_rotvec",
        "rotation_representation": ABSOLUTE_ROTVEC_REPRESENTATION,
        "robot_type": robot_type,
        "stats_source": str(stats_source),
        "stats_file": stats_target.name,
        "rotation_reference_quaternion_xyzw": rotation_reference_quaternion,
        "tasks": metadata.get("tasks", []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Override long_action_stats_path saved in the training config.",
    )
    parser.add_argument("--long-robot-type", default=None)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(TOOLS_DIR / "export_hf_checkpoint.py"),
        "--base-model",
        str(args.base_model),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(args.output_dir),
        "--max-shard-size",
        str(args.max_shard_size),
    ]
    subprocess.run(command, check=True)
    output_dir = args.output_dir.expanduser().resolve()
    patch_processor(output_dir, args.stats, args.long_robot_type)
    print(f"Attached RoboCasa-Long absolute-rotation processor entry to {output_dir}")


if __name__ == "__main__":
    main()
