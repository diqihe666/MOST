# Copyright (C) 2026 MOST Contributors.
"""Evaluate MOST on RoboCasa365 through deploy/client.py."""

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
from tqdm import tqdm
from robocasa.utils.prm_metrics import compute_prm_metrics, preprocess_progress_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from client import Client  # noqa: E402


DEFAULT_MODEL_PATH = ""
ROBOT_TYPE = "robocasa365"
STATE_DIM = 60
ACTION_DIM = 12
# Training decodes RoboCasa365 at 20 FPS and samples every 2 frames, so
# Qwen3-VL sees an effective 10 FPS clip. Keep evaluation timestamps identical.
QWEN_VIDEO_FPS = 10.0
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)


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
    # During training/debugging a run directory may still expose config.py.
    # Normal evaluation uses the exported HF directory, whose config.json now
    # carries the Joint MemLearn input contract.
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

    path = Path(model_path).expanduser()
    hf_config = path / "config.json" if path.is_dir() else None
    if hf_config is not None and hf_config.is_file():
        payload = json.loads(hf_config.read_text(encoding="utf-8"))
        memory_enabled = bool(payload.get("memory_enabled", False))
        return (
            model_path,
            memory_enabled,
            int(payload.get("memory_image_size", 224)),
            int(payload.get("memory_task_max_tokens", 128)),
            int(payload.get("memory_stride", 32)),
        )

    return model_path, False, 224, 128, 32


class EvalClient:
    def __init__(
        self,
        model_path: str,
        host: str,
        port: int,
        robot_type: str,
        crop_ratio: float,
    ) -> None:
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
    gym: Any,
    get_task_horizon: Any,
    convert_action: Any,
    output_dir: Path,
    episode_indices: list[int] | None = None,
    show_progress: bool = True,
    write_task_stats: bool = True,
) -> dict[str, Any]:
    task_dir = output_dir / env_name
    task_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(f"robocasa/{env_name}", split=args.split, seed=args.seed)
    horizon = resolve_task_horizon(env, env_name, args.horizon, get_task_horizon)
    episode_results = []

    episodes = range(args.num_trials) if episode_indices is None else episode_indices
    try:
        for episode in tqdm(episodes, desc=env_name, disable=not show_progress):
            global_episode_index = task_index * args.num_trials + episode
            episode_seed = args.seed + global_episode_index
            observation, _ = reset_env(env, episode_seed)
            instruction = observation["annotation.human.task_description"]

            queue_length = (args.obs_history - 1) * args.obs_interval + 1
            image_queues = {
                key: collections.deque(maxlen=queue_length) for key in CAMERA_KEYS
            }
            state_queue: collections.deque[np.ndarray] = collections.deque(
                maxlen=queue_length
            )
            for key, image in collect_images(observation).items():
                image_queues[key].append(image)
            state_queue.append(observation_to_state(observation))

            action_plan: collections.deque[np.ndarray] = collections.deque()
            video_frames = []
            progress_curve_history = []
            capture_video = args.save_videos or args.save_failure_videos
            if capture_video:
                video_frames.append(make_video_frame(observation))

            success = False
            latest_metrics = {}
            steps = 0
            while steps < horizon:
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
                        # Only this newly sampled memory observation is sent.
                        # Older DINO+ViewFusion tokens live in the model-side
                        # inference cache for the duration of this episode.
                        memory_frame = collect_images(observation)
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
                            f"--replan-steps is {args.replan_steps}."
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                policy_action = np.asarray(action_plan.popleft(), dtype=np.float32)
                observation, _, done, truncated, info = env.step(
                    convert_action(policy_action)
                )
                steps += 1

                for key, image in collect_images(observation).items():
                    image_queues[key].append(image)
                state_queue.append(observation_to_state(observation))

                success = bool(info.get("success", False))
                if "metrics" in info:
                    latest_metrics = info["metrics"]
                    if "progress_curve" in latest_metrics:
                        progress_curve_history = list(latest_metrics["progress_curve"])
                if capture_video and (
                    steps % args.video_stride == 0 or success or done or truncated
                ):
                    video_frames.append(make_video_frame(observation))
                if success or done or truncated:
                    break

            # Keep the semantic progress curve consistent with the official
            # RoboCasa success signal. If the episode is officially successful,
            # force only the LAST progress value to 1.0. The preceding curve is
            # left untouched so regressions and process quality remain visible.
            if progress_curve_history:
                progress_curve_history = list(progress_curve_history)
                if success:
                    progress_curve_history[-1] = 1.0

                latest_metrics = dict(latest_metrics)
                latest_metrics["progress_curve"] = list(progress_curve_history)

            status = "success" if success else "failure"
            if video_frames and (
                args.save_videos or (args.save_failure_videos and not success)
            ):
                # Always keep the original RoboCasa visualization unchanged.
                video_path = (
                    task_dir / f"episode_{episode:03d}_seed_{episode_seed}_{status}.mp4"
                )
                imageio.mimsave(
                    video_path,
                    video_frames,
                    fps=args.video_fps,
                )

                # Additional visualization:
                # Only composite progress curve videos when this episode
                # actually provides progress_curve (e.g. long-horizon tasks).
                # Atomic tasks or episodes without semantic progress do not
                # generate this extra file.
                if progress_curve_history:
                    # Visualize the preprocessed curve, matching the curve
                    # convention used by PRM metric computation.
                    processed_progress_curve = preprocess_progress_curve(
                        progress_curve_history
                    ).tolist()

                    composed_frames = []
                    total_video = len(video_frames)
                    total_curve = len(processed_progress_curve)
                    for i, frame in enumerate(video_frames):
                        curve_idx = int(
                            i / max(total_video - 1, 1) * max(total_curve - 1, 1)
                        )
                        composed_frames.append(
                            compose_video_with_progress(
                                frame,
                                processed_progress_curve,
                                curve_idx,
                            )
                        )

                    progress_video_path = (
                        task_dir
                        / f"episode_{episode:03d}_seed_{episode_seed}_{status}_progress.mp4"
                    )
                    imageio.mimsave(
                        progress_video_path,
                        composed_frames,
                        fps=args.video_fps,
                    )

            # Compute PRM metrics only after the trajectory is complete.
            # The environment provides the semantic progress curve; the
            # evaluator converts it into process metrics.
            final_metrics = dict(latest_metrics)
            if "progress_curve" in final_metrics:
                final_metrics.update(
                    compute_prm_metrics(
                        final_metrics["progress_curve"],
                        success=success,
                    )
                )

            # Normalized task time cost.
            # Success: fraction of horizon consumed.
            # Failure: full rollout budget cost.
            final_metrics["NCT"] = (
                min(1.0, float(steps) / float(horizon)) if success else 1.0
            )

            episode_results.append(
                {
                    "episode": episode,
                    "global_episode_index": global_episode_index,
                    "seed": episode_seed,
                    "success": success,
                    "steps": steps,
                    "metrics": final_metrics,
                }
            )
            successes = sum(int(item["success"]) for item in episode_results)
            stats = {
                "env_name": env_name,
                "split": args.split,
                "num_episodes": len(episode_results),
                "successes": successes,
                "success_rate": successes / len(episode_results),
                "horizon": horizon,
                "episodes": episode_results,
            }
            if write_task_stats:
                with (task_dir / "stats.json").open("w", encoding="utf-8") as file:
                    json.dump(stats, file, indent=2)

        return stats
    finally:
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MOST on RoboCasa365 or custom RoboCasa tasks."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--robot-type", default=ROBOT_TYPE)
    parser.add_argument("--server-addr", default="localhost")
    parser.add_argument("--server-port", type=int, default=10086)
    parser.add_argument("--split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--task-set", default="target50")
    parser.add_argument(
        "--allow-custom-tasks",
        action="store_true",
        help=(
            "Treat explicitly supplied --task-name values as custom RoboCasa env names "
            "instead of requiring membership in TASK_SET_REGISTRY."
        ),
    )
    parser.add_argument(
        "--task-name",
        action="append",
        default=None,
        help="Evaluate one task. Repeat this option to select multiple tasks.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit selected tasks for a smoke test.",
    )
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--obs-history", type=int, default=4)
    parser.add_argument("--obs-interval", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-ratio", type=float, default=0.95)
    parser.add_argument("--save-root-dir", default="eval_results/robocasa365")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--save-failure-videos", action="store_true")
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
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
    if not 0 < args.crop_ratio <= 1:
        raise ValueError("--crop-ratio must be in (0, 1]")
    if args.video_stride < 1:
        raise ValueError("--video-stride must be at least 1")
    if getattr(args, "allow_custom_tasks", False) and not args.task_name:
        raise ValueError("--allow-custom-tasks requires at least one --task-name")


def select_tasks(
    args: argparse.Namespace,
    task_set_registry: dict[str, Any],
) -> tuple[dict[str, int], list[str]]:
    if getattr(args, "allow_custom_tasks", False):
        selected_tasks = list(args.task_name)
        if len(selected_tasks) != len(set(selected_tasks)):
            raise ValueError("--task-name must not select the same task more than once")
        if args.max_tasks is not None:
            selected_tasks = selected_tasks[: args.max_tasks]
        return {
            task: index for index, task in enumerate(selected_tasks)
        }, selected_tasks

    if args.task_set not in task_set_registry:
        choices = ", ".join(sorted(task_set_registry))
        raise KeyError(
            f"Unknown task set {args.task_set!r}. Available task sets: {choices}"
        )

    all_tasks = list(task_set_registry[args.task_set])
    task_to_index = {task: index for index, task in enumerate(all_tasks)}
    selected_tasks = list(args.task_name) if args.task_name else all_tasks
    unknown_tasks = [task for task in selected_tasks if task not in task_to_index]
    if unknown_tasks:
        raise KeyError(f"Tasks are not in {args.task_set!r}: {unknown_tasks}")
    if len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError("--task-name must not select the same task more than once")
    if args.max_tasks is not None:
        selected_tasks = selected_tasks[: args.max_tasks]
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
        "task_set": args.task_set,
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
        "tasks": task_stats,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    validate_args(args)

    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    task_to_index, selected_tasks = select_tasks(args, TASK_SET_REGISTRY)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.save_root_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing evaluation outputs to %s", output_dir)

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
                gym=gym,
                get_task_horizon=get_task_horizon,
                convert_action=convert_action,
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
        100.0 * summary["episode_success_rate"],
        summary["successes"],
        summary["num_episodes"],
    )


if __name__ == "__main__":
    main()
