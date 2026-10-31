# Copyright (C) 2026 MOST Contributors.
"""Evaluate MOST on direct-RoboSuite RoboCasa-Long tasks."""

from __future__ import annotations

import argparse
import collections
import json
import logging
import runpy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from client import Client  # noqa: E402

from robocasa.utils.prm_metrics import (  # noqa: E402
    compute_prm_metrics,
    preprocess_progress_curve,
)


DEFAULT_MODEL_PATH = ""
ROBOT_TYPE = "robocasa_long_absrot"
STATE_DIM = 60
ACTION_DIM = 12
ABSOLUTE_ROTVEC_REPRESENTATION = (
    "base_relative_eef_local_rotvec_radians_about_dataset_reference_from_state_t_plus_1"
)
# Training decodes RoboCasa365 at 20 FPS and samples every 2 frames, so
# Qwen3-VL sees an effective 10 FPS clip. Keep evaluation timestamps identical.
QWEN_VIDEO_FPS = 10.0
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)

RAW_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)

RAW_STATE_KEYS = (
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
)

# RoboCasa-Long tasks live outside TASK_SET_REGISTRY in this project.
# Keep this canonical order fixed so task indices / episode seeds are stable
# between single-task and whole-suite evaluation.
ROBOCASA_LONG_TASKS = (
    "PackFruit",
    "PrepareBreakfast",
    "CleanAndStoreAfterMeal",
)

# Match the scenes used to construct the current Long demonstrations.
# A global --layout/--style pair can still override these for diagnostics.
ROBOCASA_LONG_DEFAULT_SCENES = {
    "PackFruit": (7, 7),
    "PrepareBreakfast": (4, 4),
    "CleanAndStoreAfterMeal": (2, 2),
}

ROBOCASA_LONG_TASK_SET_ALIASES = {
    "robocasa-long",
    "robocasa_long",
    "robocasalong",
    "long",
}


def _compact_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


_LONG_TASK_BY_COMPACT_NAME = {_compact_name(task): task for task in ROBOCASA_LONG_TASKS}


def resolve_long_task_name(name: str) -> str | None:
    """Resolve case/separator-insensitive RoboCasa-Long task names."""
    return _LONG_TASK_BY_COMPACT_NAME.get(_compact_name(name))


def registered_robosuite_task_names() -> tuple[str, ...]:
    """Return task names registered by the installed RoboCasa/RoboSuite build."""
    import robocasa  # noqa: F401 - registers custom tasks
    import robosuite

    registries: list[Any] = [
        getattr(robosuite, "ALL_ENVIRONMENTS", None),
        getattr(robosuite, "REGISTERED_ENVS", None),
    ]
    try:
        from robosuite.environments.base import REGISTERED_ENVS

        registries.append(REGISTERED_ENVS)
    except ImportError:
        pass

    names: set[str] = set()
    for registry in registries:
        if isinstance(registry, dict):
            names.update(str(value) for value in registry)
        elif isinstance(registry, (list, tuple, set, frozenset)):
            names.update(str(value) for value in registry)
    return tuple(sorted(names))


def resolve_requested_long_task(name: str) -> str:
    """Resolve built-ins and custom registered Long tasks without a fixed list."""
    raw = str(name).strip()
    if not raw:
        raise ValueError("--task-name cannot be empty")
    known = resolve_long_task_name(raw)
    if known is not None:
        return known

    registered = {
        _compact_name(task): task for task in registered_robosuite_task_names()
    }
    # If a custom environment is registered, this also makes inputs such as
    # pack-fruit / pack_fruit / PackFruit resolve to its canonical class name.
    return registered.get(_compact_name(raw), raw)


def is_long_task_set(name: str) -> bool:
    raw = str(name).strip().lower()
    return raw in ROBOCASA_LONG_TASK_SET_ALIASES or _compact_name(raw) == "robocasalong"


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)

    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def observation_to_state(observation: dict[str, Any]) -> np.ndarray:
    """Build the EE-first 14D state used to train the RoboCasa365 policy."""
    state = np.concatenate(
        [
            np.asarray(
                observation["state.end_effector_position_relative"], dtype=np.float32
            ).reshape(-1),
            quat_xyzw_to_axis_angle(
                observation["state.end_effector_rotation_relative"]
            ),
            np.asarray(observation["state.gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(observation["state.base_position"], dtype=np.float32).reshape(
                -1
            ),
            quat_xyzw_to_axis_angle(observation["state.base_rotation"]),
        ]
    ).astype(np.float32)
    if state.shape != (14,):
        raise ValueError(f"Expected a 14D RoboCasa365 state, got {state.shape}")
    return state


def collect_images(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.ascontiguousarray(observation[key], dtype=np.uint8)
        for key in CAMERA_KEYS
    }


def sample_history(
    history: collections.deque[np.ndarray], length: int, interval: int
) -> np.ndarray:
    items = list(history)
    if not items:
        raise ValueError("Observation history cannot be empty")
    indices = [
        max(0, len(items) - 1 - (length - 1 - index) * interval)
        for index in range(length)
    ]
    return np.ascontiguousarray(np.stack([items[index] for index in indices], axis=0))


def center_crop(image: np.ndarray, crop_ratio: float) -> Image.Image:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image

    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


def make_progress_curve_image(
    progress_curve: list[float] | np.ndarray,
    current_index: int,
    width: int,
    height: int = 260,
) -> np.ndarray:
    """Render PRM progress curve below the original robot video."""
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if progress_curve:
        values = np.asarray(progress_curve, dtype=np.float32)
        values = np.clip(values, 0.0, 1.0)

        margin_x = 40
        margin_y = 30
        plot_w = width - 2 * margin_x
        plot_h = height - 2 * margin_y

        points = []
        for i, value in enumerate(values):
            x = margin_x + int(i / max(1, len(values) - 1) * plot_w)
            y = margin_y + int((1.0 - float(value)) * plot_h)
            points.append((x, y))

        if len(points) > 1:
            draw.line(points, fill=(30, 90, 180), width=3)

        if current_index >= 0 and current_index < len(points):
            x, y = points[current_index]
            draw.ellipse(
                (x - 5, y - 5, x + 5, y + 5),
                fill=(220, 40, 40),
            )

        draw.text(
            (margin_x, 5),
            "Task Progress",
            fill=(0, 0, 0),
        )
        draw.text(
            (margin_x, height - 22),
            f"step {current_index + 1}/{len(values)}",
            fill=(0, 0, 0),
        )

    return np.asarray(canvas, dtype=np.uint8)


def compose_video_with_progress(
    video_frame: np.ndarray,
    progress_curve: list[float],
    frame_index: int,
) -> np.ndarray:
    """Stack robot visualization and progress curve vertically."""
    curve = make_progress_curve_image(
        progress_curve,
        frame_index,
        width=video_frame.shape[1],
    )
    return np.concatenate([video_frame, curve], axis=0)


def make_video_frame(observation: dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([observation[key] for key in CAMERA_KEYS], axis=1)
    )


def _find_training_config(model_path: str) -> Path | None:
    path = Path(model_path).expanduser().resolve()
    candidates = [path] if path.is_dir() else [path.parent]
    candidates.extend(list(candidates[0].parents)[:4])
    for directory in candidates:
        config = directory / "config.py"
        if config.is_file():
            return config
    return None


def _resolve_eval_model_info(model_path: str) -> tuple[str, bool, int, int, int]:
    # An exported HF directory must win over any config.py in its parents.
    # Otherwise a Long export nested under a training run silently falls back
    # to the base processor and loses its absolute-rotation robot statistics.
    path = Path(model_path).expanduser()
    hf_config = path / "config.json" if path.is_dir() else None
    hf_processor = path / "preprocessor_config.json" if path.is_dir() else None
    if (
        hf_config is not None
        and hf_config.is_file()
        and hf_processor is not None
        and hf_processor.is_file()
    ):
        payload = json.loads(hf_config.read_text(encoding="utf-8"))
        memory_enabled = bool(payload.get("memory_enabled", False))
        return (
            model_path,
            memory_enabled,
            int(payload.get("memory_image_size", 224)),
            int(payload.get("memory_task_max_tokens", 128)),
            int(payload.get("memory_stride", 32)),
        )

    # During training/debugging a run directory may still expose config.py.
    config_path = _find_training_config(model_path)
    if config_path is not None:
        namespace = runpy.run_path(str(config_path))
        data = namespace.get("data", {})
        train = data.get("params", {}).get("train_datasets", {})
        model_cfg = namespace.get("model", {}).get("params", {}).get("model", {})
        processor_path = str(train.get("processor_path", model_path))
        memory_enabled = bool(model_cfg.get("memory_enabled", False))
        return (
            processor_path,
            memory_enabled,
            int(train.get("memory_image_size", 224)),
            int(train.get("memory_task_max_tokens", 128)),
            int(train.get("memory_stride", 32)),
        )

    return model_path, False, 224, 128, 32


def _local_model_config(model_path: str) -> tuple[dict[str, Any] | None, Path | None]:
    """Read an exported local HF config without resolving remote model IDs."""
    path = Path(model_path).expanduser()
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.is_file():
        return None, None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model config must contain a JSON object: {config_path}")
    return payload, config_path


def _validated_reference_quaternion(value: Any, *, source: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(
            "robocasa_long_rotation_reference_quaternion_xyzw must be four "
            f"finite XYZW values in {source}, got {value!r}"
        )
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError(
            "robocasa_long_rotation_reference_quaternion_xyzw has zero norm "
            f"in {source}"
        )
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(
            "robocasa_long_rotation_reference_quaternion_xyzw must be unit "
            f"length in {source}, got norm={norm:.8f}"
        )
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion.astype(np.float64)


def validate_eval_model_contract(
    model_path: str,
    requested_robot_type: str,
) -> np.ndarray:
    """Validate the exported RoboCasa-Long action and processor contract."""
    payload, config_path = _local_model_config(model_path)
    if payload is None:
        raise ValueError(
            "RoboCasa-Long evaluation requires a local exported model directory "
            "whose config.json contains the Long action metadata. Export with "
            "tools/export_hf_checkpoint_robocasa_long.py."
        )

    assert config_path is not None
    mode_key = "robocasa_long_rotation_action_mode"
    representation_key = "robocasa_long_rotation_representation"
    robot_key = "robocasa_long_robot_type"
    reference_key = "robocasa_long_rotation_reference_quaternion_xyzw"
    model_mode_raw = payload.get(mode_key)
    model_robot_raw = payload.get(robot_key)
    reference_raw = payload.get(reference_key)
    if model_mode_raw != "absolute_rotvec":
        raise ValueError(
            f"Incomplete or unsupported Long metadata in {config_path}: "
            f"{mode_key}={model_mode_raw!r}. Re-export with "
            "tools/export_hf_checkpoint_robocasa_long.py."
        )

    if model_robot_raw is None or not str(model_robot_raw).strip():
        raise ValueError(
            f"Incomplete Long metadata in {config_path}: missing {robot_key}"
        )
    model_robot_type = str(model_robot_raw).strip()
    if str(requested_robot_type).strip() != model_robot_type:
        raise ValueError(
            "Evaluation robot type does not match the exported model: "
            f"requested={requested_robot_type!r}, model={model_robot_type!r} "
            f"({config_path})."
        )

    if payload.get(representation_key) != ABSOLUTE_ROTVEC_REPRESENTATION:
        raise ValueError(
            f"Incomplete or unsupported Long metadata in {config_path}: "
            f"{representation_key}={payload.get(representation_key)!r}. "
            "Re-export with tools/export_hf_checkpoint_robocasa_long.py."
        )
    processor_config = config_path.parent / "preprocessor_config.json"
    if not processor_config.is_file():
        raise ValueError(
            "RoboCasa-Long evaluation requires a complete local HF export; "
            f"missing {processor_config}."
        )
    if reference_raw is None:
        raise ValueError(
            f"Incomplete Long metadata in {config_path}: missing {reference_key}. "
            "Regenerate statistics and re-export the checkpoint."
        )
    return _validated_reference_quaternion(reference_raw, source=str(config_path))


class EvalClient:
    def __init__(
        self,
        model_path: str,
        host: str,
        port: int,
        robot_type: str,
        crop_ratio: float,
    ) -> None:
        self.rotation_reference_quaternion_xyzw = validate_eval_model_contract(
            model_path,
            requested_robot_type=robot_type,
        )
        (
            processor_path,
            self.memory_enabled,
            self.memory_image_size,
            self.memory_task_max_tokens,
            self.memory_stride,
        ) = _resolve_eval_model_info(model_path)
        self.client = Client(host=host, port=port, model_path=processor_path)
        self.processor = self.client.processor
        self.robot_type = robot_type
        self.crop_ratio = crop_ratio
        if self.memory_enabled:
            logging.info(
                "Joint MemLearn evaluation enabled: image=%d task_tokens=%d stride=%d",
                self.memory_image_size,
                self.memory_task_max_tokens,
                self.memory_stride,
            )

        robot_types = self.processor.list_robot_types()
        if robot_type not in robot_types:
            self.client.close()
            raise ValueError(
                f"Robot type {robot_type!r} is missing from the checkpoint. "
                f"Available robot types: {robot_types}."
            )
        logging.info(
            "Loaded processor from %s; robot types: %s", model_path, robot_types
        )
        if self.rotation_reference_quaternion_xyzw is not None:
            logging.info(
                "Absolute rotation reference quaternion (XYZW): %s",
                self.rotation_reference_quaternion_xyzw.tolist(),
            )

    def _build_messages(
        self, image_history: dict[str, np.ndarray], instruction: str
    ) -> list[dict[str, Any]]:
        videos = {
            key: [center_crop(frame, self.crop_ratio) for frame in image_history[key]]
            for key in CAMERA_KEYS
        }
        return [
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
                        "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

    def infer(
        self,
        state_history: np.ndarray,
        image_history: dict[str, np.ndarray],
        instruction: str,
        memory_frame: dict[str, np.ndarray] | None = None,
        memory_reset: bool = False,
    ) -> np.ndarray:
        state_history = np.asarray(state_history, dtype=np.float32)
        state = np.zeros((1, state_history.shape[0], STATE_DIM), dtype=np.float32)
        state[0, :, : state_history.shape[-1]] = state_history

        # Match the training collate exactly: the 4 already-sampled frames are
        # treated as a 10 FPS clip (20 FPS source / obs_interval=2).  Without
        # this metadata Qwen3-VL falls back to 24 FPS and constructs different
        # temporal position/timestamp tokens at evaluation time.
        history_length = int(state_history.shape[0])
        video_metadata = [
            {
                "total_num_frames": history_length,
                "fps": QWEN_VIDEO_FPS,
                "duration": history_length / QWEN_VIDEO_FPS,
                "frames_indices": list(range(history_length)),
            }
            for _ in CAMERA_KEYS
        ]

        inputs = self.processor.apply_chat_template(
            self._build_messages(image_history, instruction),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
            video_metadata=video_metadata,
            state=state,
            robot_type=self.robot_type,
        )
        request = dict(inputs)
        request["task_id"] = self.robot_type
        if self.memory_enabled:
            # Inference is incremental: only the newly sampled long-term
            # observation is sent to the server.  The Joint HF model caches
            # DINO+ViewFusion features from previous memory timesteps.
            # This avoids re-sending and re-encoding the complete RGB history
            # at every MOST replan.
            task = self.processor.tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=self.memory_task_max_tokens,
                add_special_tokens=True,
            )
            request.update(
                memory_reset=bool(memory_reset),
                memory_task_input_ids=task["input_ids"],
                memory_task_attention_mask=task["attention_mask"],
            )
            if memory_frame is not None:
                views = []
                resampling = getattr(Image, "Resampling", Image).BILINEAR
                for key in CAMERA_KEYS:
                    image = center_crop(memory_frame[key], self.crop_ratio)
                    image = image.resize(
                        (self.memory_image_size, self.memory_image_size), resampling
                    )
                    array = np.asarray(image, dtype=np.uint8)
                    views.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
                # [new_steps=1, views=3, C, H, W].  No historical RGB is
                # retained or transmitted by the evaluator.
                request["memory_frames"] = torch.stack(views, dim=0).unsqueeze(0)

        actions = self.client(**request)
        actions = actions[0, :, :ACTION_DIM].float().cpu().numpy()
        return np.asarray(actions, dtype=np.float32)

    def close(self) -> None:
        self.client.close()


def direct_observation_to_state(observation: dict[str, Any]) -> np.ndarray:
    """Map direct RoboSuite state to the same MOST 14D evaluator state used in training."""
    missing = [key for key in RAW_STATE_KEYS if key not in observation]
    if missing:
        raise KeyError(
            f"Missing raw RoboSuite observation keys {missing}. "
            f"Available keys: {sorted(observation.keys())}"
        )

    state = np.concatenate(
        [
            np.asarray(observation["robot0_base_to_eef_pos"], dtype=np.float32).reshape(
                -1
            ),
            quat_xyzw_to_axis_angle(observation["robot0_base_to_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(
                -1
            ),
            np.asarray(observation["robot0_base_pos"], dtype=np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(observation["robot0_base_quat"]),
        ]
    ).astype(np.float32, copy=False)

    if state.shape != (14,):
        raise ValueError(f"Expected MOST 14D state, got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("MOST state contains non-finite values")
    return state


def collect_direct_images(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Expose direct RoboSuite cameras with the same orientation/key names as the Gym wrapper."""
    images: dict[str, np.ndarray] = {}
    for mapped_name, camera_name in zip(CAMERA_KEYS, RAW_CAMERA_NAMES):
        raw_key = f"{camera_name}_image"
        if raw_key not in observation:
            raise KeyError(
                f"Missing camera key {raw_key!r}. "
                f"Available keys: {sorted(observation.keys())}"
            )
        image = np.asarray(observation[raw_key], dtype=np.uint8)
        # RoboCasaGymEnv.get_basic_observation() flips direct camera images.
        images[mapped_name] = np.ascontiguousarray(image[::-1, :, :])
    return images


def resolve_arm_rotation_controller(env: Any) -> Any:
    """Resolve and validate PandaOmron's live arm OSC controller."""
    robots = getattr(env, "robots", None)
    if not robots:
        raise RuntimeError("RoboSuite environment exposes no robots")
    controllers = getattr(robots[0], "part_controllers", None)
    if not isinstance(controllers, dict):
        raise RuntimeError("Robot exposes no part_controllers mapping")

    controller = None
    for key in ("right", "right_arm", "arm"):
        candidate = controllers.get(key)
        if candidate is not None:
            controller = candidate
            break
    if controller is None:
        candidates = [
            value
            for value in controllers.values()
            if hasattr(value, "goal_origin_to_eef_pose") and hasattr(value, "goal_ori")
        ]
        if len(candidates) == 1:
            controller = candidates[0]
    if controller is None:
        raise RuntimeError(
            "Could not identify the arm OSC controller; "
            f"part controller keys={sorted(controllers)}"
        )

    input_type = str(getattr(controller, "input_type", "")).lower()
    reference_frame = str(getattr(controller, "input_ref_frame", "")).lower()
    if input_type != "delta":
        raise RuntimeError(
            f"absolute_rotvec conversion expects OSC input_type='delta', got {input_type!r}"
        )
    if reference_frame != "base":
        raise RuntimeError(
            "absolute_rotvec targets are base-relative, so OSC "
            f"input_ref_frame must be 'base', got {reference_frame!r}"
        )
    return controller


def _current_site_orientation(
    observation: dict[str, Any],
    arm_controller: Any,
) -> np.ndarray:
    """Return the achieved grip-site orientation in the controller base frame."""
    site_key = "robot0_base_to_eef_quat_site"
    if site_key in observation:
        return Rotation.from_quat(
            np.asarray(observation[site_key], dtype=np.float64).reshape(4)
        ).as_matrix()
    update = getattr(arm_controller, "update", None)
    if callable(update):
        update(force=True)
    pose_fn = getattr(arm_controller, "goal_origin_to_eef_pose", None)
    if not callable(pose_fn):
        raise RuntimeError(
            "Arm controller cannot expose the achieved grip-site pose needed "
            "for absolute rotation control"
        )
    pose = np.asarray(pose_fn(), dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise RuntimeError(
            f"goal_origin_to_eef_pose() returned invalid shape/value {pose.shape}"
        )
    return pose[:3, :3]


def initialize_absolute_rotation_controller(
    observation: dict[str, Any],
    arm_controller: Any,
) -> None:
    """Initialize the desired OSC site orientation in its base frame after reset."""
    arm_controller.goal_ori = _current_site_orientation(
        observation, arm_controller
    ).copy()


def _controller_rotation_range(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(6, float(array), dtype=np.float64)
    array = array.reshape(-1)
    if array.size < 6:
        raise RuntimeError(
            f"OSC {name} must be scalar or contain at least 6 values, got {array.shape}"
        )
    result = array[3:6]
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"OSC {name} contains NaN/Inf")
    return result


def absolute_rotvec_to_controller_input(
    goal_rotvec: np.ndarray,
    observation: dict[str, Any],
    arm_controller: Any,
    *,
    base_mode: bool,
    rotation_reference_quaternion_xyzw: np.ndarray,
    scale: float = 1.0,
    gain: float = 1.0,
    max_step_radians: float = 0.5,
) -> np.ndarray:
    """Convert a base-relative absolute body orientation to live OSC input.

    HybridMobileBase updates OSC goals from the achieved pose in arm mode and
    from the previous desired goal in base mode.  The OSC controls the grip
    site while the dataset stores the EEF body, so base mode also needs the
    fixed body-to-site rotation offset.
    """
    current_quat = np.asarray(
        observation["robot0_base_to_eef_quat"], dtype=np.float64
    ).reshape(4)
    if not np.all(np.isfinite(current_quat)):
        raise ValueError("Current EEF quaternion contains NaN/Inf")
    current_body = Rotation.from_quat(current_quat).as_matrix()
    current_site = _current_site_orientation(observation, arm_controller)
    body_to_site = current_body.T @ current_site

    requested = np.asarray(goal_rotvec, dtype=np.float64).reshape(3) * float(scale)
    if not np.all(np.isfinite(requested)):
        raise ValueError("Absolute rotation target contains NaN/Inf")
    rotation_reference = Rotation.from_quat(
        _validated_reference_quaternion(
            rotation_reference_quaternion_xyzw,
            source="absolute-rotation controller conversion",
        )
    ).as_matrix()
    # The policy predicts the Lie-algebra offset around the fixed training
    # reference, not a global axis-angle vector: R_goal = Exp(v) @ R_ref.
    goal_body = Rotation.from_rotvec(requested).as_matrix() @ rotation_reference
    goal_site = goal_body @ body_to_site

    if base_mode:
        desired = getattr(arm_controller, "goal_ori", None)
        if desired is None:
            raise RuntimeError("OSC goal_ori is unset in base-control mode")
        reference = np.asarray(desired, dtype=np.float64)
        if reference.shape != (3, 3):
            raise RuntimeError(f"OSC goal_ori has invalid shape {reference.shape}")
    else:
        reference = current_site

    error = Rotation.from_matrix(goal_site @ reference.T).as_rotvec()
    error *= float(gain)

    output_min = _controller_rotation_range(
        getattr(arm_controller, "output_min", None), "output_min"
    )
    output_max = _controller_rotation_range(
        getattr(arm_controller, "output_max", None), "output_max"
    )
    input_min = _controller_rotation_range(
        getattr(arm_controller, "input_min", None), "input_min"
    )
    input_max = _controller_rotation_range(
        getattr(arm_controller, "input_max", None), "input_max"
    )
    symmetric_output_cap = float(np.min(np.minimum(output_max, -output_min)))
    if symmetric_output_cap <= 0:
        raise RuntimeError(
            f"OSC rotation output range must straddle zero: {output_min}, {output_max}"
        )
    step_cap = min(float(max_step_radians), symmetric_output_cap)
    error_norm = float(np.linalg.norm(error))
    if error_norm > step_cap:
        error *= step_cap / error_norm

    output_span = output_max - output_min
    input_span = input_max - input_min
    if np.any(output_span <= 0) or np.any(input_span <= 0):
        raise RuntimeError("OSC input/output scaling range is invalid")
    controller_input = (error - output_min) * input_span / output_span + input_min
    return np.clip(controller_input, input_min, input_max).astype(np.float32)


def policy_action_to_robosuite_action(
    policy_action: np.ndarray,
    *,
    current_observation: dict[str, Any],
    arm_controller: Any,
    absolute_rotation_reference_quaternion_xyzw: np.ndarray,
    absolute_rotation_scale: float = 1.0,
    absolute_rotation_gain: float = 1.0,
    absolute_rotation_max_step_radians: float = 0.5,
) -> np.ndarray:
    """Convert MOST policy order into the direct PandaOmron 12D controller vector.

    MOST policy order:
        eef_pos[0:3], eef_rot[3:6], gripper[6],
        base_xyz+torso[7:11], control_mode[11]

    Direct RoboSuite order is the same continuous layout, but gripper/base mode
    are discrete signed controller channels. This mirrors PandaOmronKeyConverter.unmap_action().
    """
    action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"Expected 12D policy action, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("Policy action contains non-finite values")

    env_action = action.copy()
    env_action[6] = -1.0 if float(action[6]) < 0.5 else 1.0
    env_action[11] = -1.0 if float(action[11]) < 0.5 else 1.0
    env_action[3:6] = absolute_rotvec_to_controller_input(
        action[3:6],
        current_observation,
        arm_controller,
        base_mode=bool(env_action[11] > 0),
        rotation_reference_quaternion_xyzw=(
            absolute_rotation_reference_quaternion_xyzw
        ),
        scale=absolute_rotation_scale,
        gain=absolute_rotation_gain,
        max_step_radians=absolute_rotation_max_step_radians,
    )
    return env_action


def create_long_env(env_name: str, args: argparse.Namespace, seed: int) -> Any:
    """Create one custom RoboCasa-Long episode with direct robosuite.make()."""
    import robocasa  # noqa: F401 - registers custom tasks
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    obj_instance_split = None if args.split == "none" else args.split
    config: dict[str, Any] = {
        "env_name": env_name,
        "robots": args.robot,
        "controller_configs": load_composite_controller_config(robot=args.robot),
        "translucent_robot": args.translucent_robot,
        "seed": int(seed),
        "horizon": int(args.horizon),
        "obj_instance_split": obj_instance_split,
    }
    if args.layout is not None:
        layout_id, style_id = int(args.layout), int(args.style)
        config["layout_and_style_ids"] = [[layout_id, style_id]]
    elif env_name in ROBOCASA_LONG_DEFAULT_SCENES:
        layout_id, style_id = ROBOCASA_LONG_DEFAULT_SCENES[env_name]
        config["layout_and_style_ids"] = [[layout_id, style_id]]
    else:
        # Custom Long environments may define their own default scene.  Do not
        # force one of the three legacy task scenes onto them.
        layout_id, style_id = None, None

    logging.info(
        "Creating direct RoboSuite env task=%s seed=%d split=%s layout=%s style=%s horizon=%d",
        env_name,
        seed,
        args.split,
        layout_id,
        style_id,
        args.horizon,
    )

    return robosuite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=None,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=list(RAW_CAMERA_NAMES),
        camera_widths=int(args.camera_width),
        camera_heights=int(args.camera_height),
        camera_depths=False,
        control_freq=int(args.control_freq),
        renderer="mjviewer",
    )


def extract_instruction(env: Any) -> str:
    meta: dict[str, Any] = {}
    if hasattr(env, "get_ep_meta"):
        try:
            value = env.get_ep_meta()
            if isinstance(value, dict):
                meta = value
        except Exception:
            pass
    if not meta:
        value = getattr(env, "_ep_meta", {})
        if isinstance(value, dict):
            meta = value

    instruction = str(meta.get("lang", "")).strip()
    if not instruction:
        raise RuntimeError(
            f"{env.__class__.__name__} did not provide a non-empty ep_meta['lang'] instruction"
        )
    return instruction


def read_long_metrics(env: Any) -> dict[str, Any]:
    if not hasattr(env, "get_long_horizon_metrics"):
        return {}
    metrics = env.get_long_horizon_metrics()
    if not isinstance(metrics, dict):
        raise TypeError(
            f"get_long_horizon_metrics() must return dict, got {type(metrics)}"
        )
    return dict(metrics)


def official_success(env: Any, info: dict[str, Any]) -> bool:
    # Preserve the task's original RoboCasa success criterion.
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    return bool(info.get("success", False))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def reset_env(env: Any, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return env.reset(seed=seed)
    except TypeError as error:
        raise RuntimeError(
            "This evaluator requires a RoboCasa version whose reset() accepts a seed."
        ) from error


def resolve_task_horizon(
    env: Any,
    env_name: str,
    requested_horizon: int | None,
    get_task_horizon: Any,
) -> int:
    """Resolve a rollout horizon for both registry and custom RoboCasa tasks."""
    if requested_horizon is not None:
        return int(requested_horizon)

    try:
        horizon = int(get_task_horizon(env_name))
        if horizon > 0:
            return horizon
    except (KeyError, ValueError, TypeError):
        pass

    candidates = [
        getattr(getattr(env, "unwrapped", env), "horizon", None),
        getattr(getattr(env, "spec", None), "max_episode_steps", None),
    ]
    for value in candidates:
        if value is not None and int(value) > 0:
            return int(value)

    raise KeyError(
        f"No horizon is registered for custom task {env_name!r}. "
        "Pass --horizon explicitly or register the task horizon in RoboCasa."
    )


def evaluate_task(
    env_name: str,
    task_index: int,
    args: argparse.Namespace,
    client: EvalClient,
    output_dir: Path,
    episode_indices: list[int] | None = None,
    show_progress: bool = True,
    write_task_stats: bool = True,
) -> dict[str, Any]:
    task_dir = output_dir / env_name
    task_dir.mkdir(parents=True, exist_ok=True)
    episode_results: list[dict[str, Any]] = []

    episodes = range(args.num_trials) if episode_indices is None else episode_indices
    for episode in tqdm(episodes, desc=env_name, disable=not show_progress):
        episode = int(episode)
        global_episode_index = task_index * args.num_trials + episode
        episode_seed = args.seed + global_episode_index
        env = create_long_env(env_name, args, episode_seed)
        video_writer = None
        temp_video_path: Path | None = None

        try:
            raw_observation = env.reset()
            if isinstance(raw_observation, tuple):
                raw_observation = raw_observation[0]
            if not isinstance(raw_observation, dict):
                raise TypeError(
                    f"Direct RoboSuite reset() returned {type(raw_observation)}, expected dict"
                )
            arm_controller = resolve_arm_rotation_controller(env)
            initialize_absolute_rotation_controller(raw_observation, arm_controller)

            instruction = extract_instruction(env)
            ep_meta = env.get_ep_meta() if hasattr(env, "get_ep_meta") else {}
            scene_layout = (
                int(ep_meta["layout_id"])
                if isinstance(ep_meta, dict) and "layout_id" in ep_meta
                else None
            )
            scene_style = (
                int(ep_meta["style_id"])
                if isinstance(ep_meta, dict) and "style_id" in ep_meta
                else None
            )
            queue_length = (args.obs_history - 1) * args.obs_interval + 1
            image_queues = {
                key: collections.deque(maxlen=queue_length) for key in CAMERA_KEYS
            }
            state_queue: collections.deque[np.ndarray] = collections.deque(
                maxlen=queue_length
            )

            current_images = collect_direct_images(raw_observation)
            for key, image in current_images.items():
                image_queues[key].append(image)
            state_queue.append(direct_observation_to_state(raw_observation))

            action_plan: collections.deque[np.ndarray] = collections.deque()
            capture_video = args.save_videos or args.save_failure_videos
            video_frame_count = 0
            if capture_video:
                # Long rollouts can last several minutes. Stream the normal
                # video directly to disk instead of keeping all RGB frames in
                # RAM. This is especially important with multiple eval workers.
                temp_video_path = task_dir / (
                    f".episode_{episode:03d}_seed_{episode_seed}.partial.mp4"
                )
                temp_video_path.unlink(missing_ok=True)
                video_writer = imageio.get_writer(
                    temp_video_path,
                    fps=args.video_fps,
                )
                video_writer.append_data(make_video_frame(current_images))
                video_frame_count += 1

            success = False
            latest_metrics: dict[str, Any] = {}
            progress_curve_history: list[float] = []
            steps = 0

            while steps < args.horizon:
                if not action_plan:
                    states = sample_history(
                        state_queue, args.obs_history, args.obs_interval
                    )
                    images = {
                        key: sample_history(queue, args.obs_history, args.obs_interval)
                        for key, queue in image_queues.items()
                    }

                    memory_frame = None
                    if client.memory_enabled and steps % client.memory_stride == 0:
                        memory_frame = current_images

                    action_chunk = client.infer(
                        states,
                        images,
                        instruction,
                        memory_frame=memory_frame,
                        memory_reset=bool(client.memory_enabled and steps == 0),
                    )
                    if len(action_chunk) < args.replan_steps:
                        raise RuntimeError(
                            f"Model returned {len(action_chunk)} actions, but "
                            f"--replan-steps={args.replan_steps}."
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                policy_action = np.asarray(action_plan.popleft(), dtype=np.float32)
                env_action = policy_action_to_robosuite_action(
                    policy_action,
                    current_observation=raw_observation,
                    arm_controller=arm_controller,
                    absolute_rotation_reference_quaternion_xyzw=(
                        client.rotation_reference_quaternion_xyzw
                    ),
                    absolute_rotation_scale=args.absolute_rotation_scale,
                    absolute_rotation_gain=args.absolute_rotation_gain,
                    absolute_rotation_max_step_radians=(
                        args.absolute_rotation_max_step_radians
                    ),
                )
                result = env.step(env_action)

                if len(result) == 4:
                    raw_observation, _reward, done, info = result
                    truncated = False
                elif len(result) == 5:
                    raw_observation, _reward, done, truncated, info = result
                else:
                    raise RuntimeError(
                        f"Expected direct RoboSuite step() to return 4 or 5 values, got {len(result)}"
                    )

                steps += 1
                current_images = collect_direct_images(raw_observation)
                for key, image in current_images.items():
                    image_queues[key].append(image)
                state_queue.append(direct_observation_to_state(raw_observation))

                latest_metrics = read_long_metrics(env)
                if "progress_curve" in latest_metrics:
                    progress_curve_history = list(latest_metrics["progress_curve"])

                success = official_success(env, info)

                if capture_video and (
                    steps % args.video_stride == 0 or success or done or truncated
                ):
                    assert video_writer is not None
                    video_writer.append_data(make_video_frame(current_images))
                    video_frame_count += 1

                if success or done or truncated:
                    break

            if progress_curve_history:
                progress_curve_history = list(progress_curve_history)
                if success:
                    progress_curve_history[-1] = 1.0
                latest_metrics = dict(latest_metrics)
                latest_metrics["progress_curve"] = progress_curve_history

            final_metrics = dict(latest_metrics)
            if progress_curve_history:
                final_metrics.update(
                    compute_prm_metrics(progress_curve_history, success=success)
                )
            final_metrics["NCT"] = (
                min(1.0, float(steps) / float(args.horizon)) if success else 1.0
            )

            # Finish the streaming writer before renaming / reading it.
            if video_writer is not None:
                video_writer.close()
                video_writer = None

            status = "success" if success else "failure"
            should_save_video = args.save_videos or (
                args.save_failure_videos and not success
            )

            if capture_video and temp_video_path is not None:
                if should_save_video and video_frame_count > 0:
                    video_path = task_dir / (
                        f"episode_{episode:03d}_seed_{episode_seed}_{status}.mp4"
                    )
                    temp_video_path.replace(video_path)
                    temp_video_path = None

                    if progress_curve_history:
                        processed_curve = preprocess_progress_curve(
                            progress_curve_history
                        ).tolist()
                        progress_path = task_dir / (
                            f"episode_{episode:03d}_seed_{episode_seed}_{status}_progress.mp4"
                        )

                        # Stream the derived progress visualization as well.
                        # This keeps RAM bounded even for 3-4 minute episodes.
                        reader = imageio.get_reader(video_path)
                        progress_writer = imageio.get_writer(
                            progress_path,
                            fps=args.video_fps,
                        )
                        try:
                            for frame_index, frame in enumerate(reader):
                                curve_idx = int(
                                    frame_index
                                    / max(video_frame_count - 1, 1)
                                    * max(len(processed_curve) - 1, 1)
                                )
                                progress_writer.append_data(
                                    compose_video_with_progress(
                                        np.asarray(frame, dtype=np.uint8),
                                        processed_curve,
                                        curve_idx,
                                    )
                                )
                        finally:
                            reader.close()
                            progress_writer.close()
                else:
                    temp_video_path.unlink(missing_ok=True)
                    temp_video_path = None

            episode_results.append(
                _jsonable(
                    {
                        "episode": episode,
                        "global_episode_index": global_episode_index,
                        "seed": episode_seed,
                        "success": success,
                        "steps": steps,
                        "instruction": instruction,
                        "layout_id": scene_layout,
                        "style_id": scene_style,
                        "metrics": final_metrics,
                    }
                )
            )
        finally:
            if video_writer is not None:
                try:
                    video_writer.close()
                except Exception:
                    pass
            if temp_video_path is not None:
                try:
                    temp_video_path.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                env.close()
            except Exception:
                pass

    successes = sum(int(item["success"]) for item in episode_results)
    stats = {
        "env_name": env_name,
        "split": args.split,
        "num_episodes": len(episode_results),
        "successes": successes,
        "success_rate": successes / len(episode_results) if episode_results else 0.0,
        "horizon": args.horizon,
        "episodes": episode_results,
    }
    if write_task_stats:
        with (task_dir / "stats.json").open("w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MOST on direct-RoboSuite RoboCasa-Long."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--robot-type", default=ROBOT_TYPE)
    parser.add_argument("--server-addr", default="localhost")
    parser.add_argument("--server-port", type=int, default=10086)
    parser.add_argument(
        "--task-set",
        default="robocasa-long",
        choices=("robocasa-long", "robocasa_long", "long"),
    )
    parser.add_argument(
        "--task-name",
        action="append",
        default=None,
        help=(
            "Evaluate one Long task; repeat to select multiple. Registered custom "
            "Long tasks are accepted. Omit for the three legacy Long tasks."
        ),
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=30)

    parser.add_argument("--robot", default="PandaOmron")
    parser.add_argument(
        "--split",
        default="target",
        choices=("train", "target", "test", "pretrain", "none"),
        help="Mapped directly to obj_instance_split.",
    )
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=7000)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--translucent-robot", action="store_true")

    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--obs-history", type=int, default=4)
    parser.add_argument("--obs-interval", type=int, default=2)
    parser.add_argument(
        "--absolute-rotation-scale",
        type=float,
        default=1.0,
        help="Multiplier from decoded absolute_rotvec values to radians.",
    )
    parser.add_argument(
        "--absolute-rotation-gain",
        type=float,
        default=1.0,
        help="Closed-loop orientation error gain before OSC range conversion.",
    )
    parser.add_argument(
        "--absolute-rotation-max-step-radians",
        type=float,
        default=0.5,
        help="Norm cap on one physical OSC orientation-error command.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-ratio", type=float, default=0.95)

    parser.add_argument("--save-root-dir", default="eval_results/robocasa_long")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--save-failure-videos", action="store_true")
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=10)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.model_path).strip():
        raise ValueError("--model-path is required")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("--max-tasks must be at least 1")
    if args.num_trials < 1:
        raise ValueError("--num-trials must be at least 1")
    if args.replan_steps < 1:
        raise ValueError("--replan-steps must be at least 1")
    if args.obs_history < 1 or args.obs_interval < 1:
        raise ValueError("--obs-history and --obs-interval must be at least 1")
    if (
        not np.isfinite(args.absolute_rotation_scale)
        or args.absolute_rotation_scale <= 0
        or not np.isfinite(args.absolute_rotation_gain)
        or args.absolute_rotation_gain <= 0
        or not np.isfinite(args.absolute_rotation_max_step_radians)
        or args.absolute_rotation_max_step_radians <= 0
    ):
        raise ValueError(
            "absolute rotation scale, gain, and max-step-radians must be finite "
            "and positive"
        )
    if args.horizon < 1 or args.control_freq < 1:
        raise ValueError("--horizon and --control-freq must be at least 1")
    if args.camera_width < 1 or args.camera_height < 1:
        raise ValueError("camera dimensions must be positive")
    if not 0 < args.crop_ratio <= 1:
        raise ValueError("--crop-ratio must be in (0, 1]")
    if args.video_stride < 1 or args.video_fps < 1:
        raise ValueError("--video-stride and --video-fps must be at least 1")
    if (args.layout is None) != (args.style is None):
        raise ValueError(
            "--layout and --style must be supplied together or both omitted"
        )
    reference = validate_eval_model_contract(
        args.model_path,
        requested_robot_type=args.robot_type,
    )
    args.rotation_reference_quaternion_xyzw = reference.tolist()


def select_tasks(args: argparse.Namespace) -> tuple[dict[str, int], list[str]]:
    task_to_index = {task: index for index, task in enumerate(ROBOCASA_LONG_TASKS)}
    if args.task_name:
        selected_tasks = []
        for name in args.task_name:
            resolved = resolve_requested_long_task(name)
            selected_tasks.append(resolved)
    else:
        selected_tasks = list(ROBOCASA_LONG_TASKS)

    if len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError("--task-name must not select the same task more than once")
    if args.max_tasks is not None:
        selected_tasks = selected_tasks[: args.max_tasks]
    next_index = len(task_to_index)
    for task in selected_tasks:
        if task not in task_to_index:
            task_to_index[task] = next_index
            next_index += 1
    return task_to_index, selected_tasks


def build_summary(
    args: argparse.Namespace,
    task_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    num_episodes = sum(stats["num_episodes"] for stats in task_stats.values())
    successes = sum(stats["successes"] for stats in task_stats.values())
    success_rates = [stats["success_rate"] for stats in task_stats.values()]

    # Collect per-episode metrics. Sequence-valued fields such as
    # progress_curve are kept in episode records but excluded from aggregation.
    all_metrics = []
    for stats in task_stats.values():
        for episode in stats.get("episodes", []):
            if episode.get("metrics"):
                all_metrics.append(episode["metrics"])

    metric_mean = {}
    metric_median = {}

    if all_metrics:
        metric_keys = sorted({key for metric in all_metrics for key in metric})

        for key in metric_keys:
            values = [metric[key] for metric in all_metrics if key in metric]

            if not values:
                continue

            # Skip trajectory / sequence / dict data.
            if not all(np.isscalar(v) for v in values):
                continue

            # Filter NaN/inf so conditioned metrics are aggregated only over
            # eligible episodes:
            #   FNS -> failed rollouts
            #   DRR -> rollouts with drawdown
            #   SQS -> successful rollouts
            numeric_values = np.asarray(values, dtype=np.float64)
            numeric_values = numeric_values[np.isfinite(numeric_values)]

            if numeric_values.size == 0:
                continue

            metric_mean[key] = float(np.mean(numeric_values))
            metric_median[key] = float(np.median(numeric_values))

    return {
        "model_path": args.model_path,
        "robot_type": args.robot_type,
        "split": args.split,
        "task_set": "robocasa-long",
        "num_tasks": len(task_stats),
        "num_episodes": num_episodes,
        "successes": successes,
        "episode_success_rate": successes / num_episodes if num_episodes else None,
        "mean_task_success_rate": float(np.mean(success_rates))
        if success_rates
        else None,
        # Preserve the original key as the mean aggregation for compatibility.
        "long_horizon_metrics": metric_mean,
        # Also report the median aggregation for direct comparison.
        "long_horizon_metrics_median": metric_median,
        "replan_steps": args.replan_steps,
        "obs_history": args.obs_history,
        "obs_interval": args.obs_interval,
        "absolute_rotation_scale": args.absolute_rotation_scale,
        "absolute_rotation_gain": args.absolute_rotation_gain,
        "absolute_rotation_max_step_radians": (args.absolute_rotation_max_step_radians),
        "rotation_reference_quaternion_xyzw": getattr(
            args, "rotation_reference_quaternion_xyzw", None
        ),
        "tasks": task_stats,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    validate_args(args)
    task_to_index, selected_tasks = select_tasks(args)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    args.run_id = run_id
    output_dir = Path(args.save_root_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing RoboCasa-Long evaluation outputs to %s", output_dir)

    client = EvalClient(
        model_path=args.model_path,
        host=args.server_addr,
        port=args.server_port,
        robot_type=args.robot_type,
        crop_ratio=args.crop_ratio,
    )
    task_stats = {}
    try:
        for env_name in selected_tasks:
            task_stats[env_name] = evaluate_task(
                env_name=env_name,
                task_index=task_to_index[env_name],
                args=args,
                client=client,
                output_dir=output_dir,
            )
    finally:
        client.close()

    summary = build_summary(args, task_stats)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    logging.info("Wrote summary to %s", summary_path)
    logging.info(
        "Success rate: %.2f%% (%d/%d)",
        100.0 * float(summary["episode_success_rate"] or 0.0),
        summary["successes"],
        summary["num_episodes"],
    )


if __name__ == "__main__":
    main()
