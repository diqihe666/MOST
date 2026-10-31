# Copyright (C) 2026 MOST Contributors.
"""Packed multimodal collation for RoboCasa365 and RoboCasa-Long."""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import pil_to_tensor, resize
from transformers import AutoProcessor


class CustomCollate:
    """Tokenize and pack VLA samples while preserving memory supervision."""

    def __init__(self, params: Any | None = None) -> None:
        params = params or {}
        data = params.get("train_datasets", params)
        self.max_length = int(os.environ.get("MAX_LENGTH", 20000))
        self.robot_type = str(data.get("robot_type", "robocasa365"))
        self.memory_image_size = int(data.get("memory_image_size", 224))
        self.memory_task_max_tokens = int(data.get("memory_task_max_tokens", 128))

        processor_path = str(data.get("processor_path", "")).strip()
        if not processor_path:
            raise ValueError("train_datasets.processor_path is required")
        local_path = Path(processor_path).expanduser()
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=local_path.exists(),
        )
        if self.robot_type not in self.processor.action_config:
            available = sorted(self.processor.action_config)
            raise KeyError(
                f"Robot type {self.robot_type!r} is unavailable; found {available}"
            )

        tokenizer = self.processor.tokenizer
        self.state_id = tokenizer.convert_tokens_to_ids("<state>")
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")

    def _position_ids(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build Qwen3-VL multimodal rotary positions for one packed sample."""
        input_ids = inputs["input_ids"]
        image_grid = inputs.get("image_grid_thw")
        video_grid = inputs.get("video_grid_thw")
        if image_grid is None and video_grid is None:
            return torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)

        if video_grid is not None:
            video_grid = torch.repeat_interleave(video_grid, video_grid[:, 0], dim=0)
            video_grid = video_grid.clone()
            video_grid[:, 0] = 1

        merge_size = int(self.processor.image_processor.merge_size)
        tokens = input_ids[0].tolist()
        positions: list[torch.Tensor] = []
        image_index = 0
        video_index = 0
        cursor = 0
        vision_starts = [
            index
            for index, token in enumerate(tokens[:-1])
            if token == self.vision_start_id
        ]

        for start in vision_starts:
            modality = tokens[start + 1]
            if modality == self.image_id and image_grid is not None:
                grid = image_grid[image_index]
                image_index += 1
            elif modality == self.video_id and video_grid is not None:
                grid = video_grid[video_index]
                video_index += 1
            else:
                continue

            token_index = tokens.index(modality, cursor)
            text_length = token_index - cursor
            base = int(positions[-1].max().item()) + 1 if positions else 0
            positions.append(torch.arange(text_length).view(1, -1).expand(3, -1) + base)
            time, height, width = (int(value) for value in grid)
            height //= merge_size
            width //= merge_size
            temporal = (
                torch.arange(time).view(-1, 1).expand(-1, height * width).flatten()
            )
            rows = torch.arange(height).view(1, -1, 1).expand(time, -1, width).flatten()
            columns = (
                torch.arange(width).view(1, 1, -1).expand(time, height, -1).flatten()
            )
            positions.append(
                torch.stack([temporal, rows, columns]) + text_length + base
            )
            cursor = token_index + time * height * width

        if cursor < len(tokens):
            base = int(positions[-1].max().item()) + 1 if positions else 0
            positions.append(
                torch.arange(len(tokens) - cursor).view(1, -1).expand(3, -1) + base
            )
        return torch.cat(positions, dim=1).view(3, 1, -1)

    def _normalize_action(
        self, item: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = torch.as_tensor(item["action"], dtype=torch.float32)
        config = self.processor.action_config[self.robot_type]
        mean = torch.as_tensor(config["mean"], dtype=torch.float32)
        std = torch.as_tensor(config["std"], dtype=torch.float32)
        if mean.ndim == 3 and mean.shape[0] == 1:
            mean = mean.squeeze(0)
        if std.ndim == 3 and std.shape[0] == 1:
            std = std.squeeze(0)
        if mean.ndim == 1:
            mean = mean.unsqueeze(0)
        if std.ndim == 1:
            std = std.unsqueeze(0)
        if mean.ndim != 2 or std.ndim != 2:
            raise ValueError(
                "Action statistics must be one- or two-dimensional after "
                f"removing the processor batch axis; got mean={tuple(mean.shape)}, "
                f"std={tuple(std.shape)}"
            )
        if mean.shape[1] != action.shape[1] or std.shape[1] != action.shape[1]:
            raise ValueError(
                f"Action statistics must have {action.shape[1]} columns; "
                f"got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
            )

        action_length = int(action.shape[0])
        if mean.shape[0] == 1:
            mean = mean.expand(action_length, -1)
        elif mean.shape[0] >= action_length:
            mean = mean[:action_length]
        else:
            raise ValueError(
                f"Action mean has {mean.shape[0]} steps, expected at least "
                f"{action_length}"
            )
        if std.shape[0] == 1:
            std = std.expand(action_length, -1)
        elif std.shape[0] >= action_length:
            std = std[:action_length]
        else:
            raise ValueError(
                f"Action std has {std.shape[0]} steps, expected at least "
                f"{action_length}"
            )
        dimension_mask = std > 1e-5
        temporal_mask = torch.as_tensor(
            item["temporal_action_mask"], dtype=torch.bool
        ).view(-1, 1)
        action_mask = dimension_mask & temporal_mask
        normalized = torch.zeros_like(action)
        normalized[action_mask] = (action - mean)[action_mask] / std[action_mask]
        return normalized, action_mask.float()

    def _tokenize_item(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        state = torch.as_tensor(item["state"], dtype=torch.float32)
        video_count = sum(
            content.get("type") == "video"
            for message in item["messages"]
            for content in message.get("content", ())
            if isinstance(content, dict)
        )
        history_length = int(state.shape[-2])
        video_fps = float(item.get("video_fps", 10.0))
        video_metadata = [
            {
                "total_num_frames": history_length,
                "fps": video_fps,
                "duration": history_length / video_fps,
                "frames_indices": list(range(history_length)),
            }
            for _ in range(video_count)
        ]
        inputs = self.processor.apply_chat_template(
            item["messages"],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state,
            robot_type=self.robot_type,
            do_resize=False,
            do_sample_frames=False,
            video_metadata=video_metadata,
        )
        inputs.pop("state", None)
        inputs.pop("action_mask", None)
        inputs["position_ids"] = self._position_ids(inputs)
        return inputs

    def _memory_frame(self, image: Any) -> torch.Tensor:
        frame = image if isinstance(image, torch.Tensor) else pil_to_tensor(image)
        if frame.ndim != 3:
            raise ValueError(f"Memory image must be CHW, got {tuple(frame.shape)}")
        if frame.shape[-2:] != (self.memory_image_size, self.memory_image_size):
            frame = resize(
                frame,
                [self.memory_image_size, self.memory_image_size],
                antialias=True,
            )
        if frame.dtype != torch.uint8:
            if frame.is_floating_point() and float(frame.max()) <= 1.0:
                frame = frame.mul(255.0)
            frame = frame.clamp(0, 255).to(torch.uint8)
        return frame

    @staticmethod
    def _pad_1d(values: list[torch.Tensor], padding_value: int = -1) -> torch.Tensor:
        return pad_sequence(values, batch_first=True, padding_value=padding_value)

    @staticmethod
    def _pad_2d(values: list[torch.Tensor], padding_value: int = -1) -> torch.Tensor:
        max_rows = max(int(value.shape[0]) for value in values)
        max_columns = max(int(value.shape[1]) for value in values)
        output = torch.full(
            (len(values), max_rows, max_columns),
            padding_value,
            dtype=values[0].dtype,
        )
        for index, value in enumerate(values):
            output[index, : value.shape[0], : value.shape[1]] = value
        return output

    def _collate_memory(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        enabled = ["memory_images" in payload for payload in payloads]
        if not any(enabled):
            return {}
        if not all(enabled):
            raise ValueError("A batch cannot mix memory and non-memory samples")

        packed_histories: list[torch.Tensor] = []
        history_lengths: list[int] = []
        for payload in payloads:
            history = torch.stack(
                [
                    torch.stack(
                        [self._memory_frame(image) for image in timestep], dim=0
                    )
                    for timestep in payload.pop("memory_images")
                ],
                dim=0,
            )
            if history.shape[1] != 3:
                raise ValueError("MOST expects exactly three memory camera views")
            packed_histories.append(history)
            history_lengths.append(int(history.shape[0]))

        lengths = torch.tensor(history_lengths, dtype=torch.long)
        offsets = torch.tensor(
            [0, *itertools.accumulate(history_lengths)], dtype=torch.long
        )
        task_tokens = self.processor.tokenizer(
            [str(payload.pop("memory_task_text")) for payload in payloads],
            padding=True,
            truncation=True,
            max_length=self.memory_task_max_tokens,
            return_tensors="pt",
        )

        subtask_counts = {
            int(payload.pop("memory_subtask_class_count")) for payload in payloads
        }
        state_counts = {
            int(payload.pop("memory_state_class_count")) for payload in payloads
        }
        if len(subtask_counts) != 1 or len(state_counts) != 1:
            raise ValueError("Memory label vocabularies must match within a batch")

        result = {
            "memory_frames": torch.cat(packed_histories, dim=0),
            "memory_history_offsets": offsets,
            "memory_history_lengths": lengths,
            "memory_task_input_ids": task_tokens["input_ids"],
            "memory_task_attention_mask": task_tokens["attention_mask"],
            "memory_subtask_labels": self._pad_1d(
                [payload.pop("memory_subtask_labels") for payload in payloads]
            ),
            "memory_state_labels": self._pad_2d(
                [payload.pop("memory_state_labels") for payload in payloads]
            ),
            "memory_object_category_ids": self._pad_1d(
                [payload.pop("memory_object_category_ids") for payload in payloads]
            ),
            "memory_object_instance_ids": self._pad_1d(
                [payload.pop("memory_object_instance_ids") for payload in payloads]
            ),
            "memory_subtask_class_count": torch.tensor(subtask_counts.pop()),
            "memory_state_class_count": torch.tensor(state_counts.pop()),
        }
        for key in ("memory_history_indices", "memory_sequence_indices"):
            if key in payloads[0]:
                result[key] = self._pad_1d([payload.pop(key) for payload in payloads])
        for payload in payloads:
            payload.pop("memory_history_length", None)
            payload.pop("memory_future_steps", None)
            payload.pop("memory_episode_key", None)
        return result

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        selected: list[tuple[dict[str, torch.Tensor], dict[str, Any], int]] = []
        total_length = 0
        for raw_item in batch:
            item = dict(raw_item)
            inputs = self._tokenize_item(item)
            length = int(inputs["input_ids"].shape[1])
            if total_length + length > self.max_length:
                continue

            state_tokens = (inputs["input_ids"][0] == self.state_id).nonzero(
                as_tuple=False
            )
            im_starts = (
                (inputs["input_ids"][0] == self.im_start_id)
                .nonzero(as_tuple=False)
                .flatten()
            )
            im_starts = (
                im_starts[im_starts < state_tokens[0, 0]]
                if state_tokens.numel()
                else im_starts[:0]
            )
            if im_starts.numel() == 0:
                raise ValueError("Cannot locate the action-conditioning turn")

            action, action_mask = self._normalize_action(item)
            actual_length = int(item["vlm_action_actual_length"])
            payload = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "messages",
                    "temporal_action_mask",
                    "video_fps",
                    "meta",
                    "vlm_action_actual_length",
                }
            }
            payload["action"] = action
            payload["action_mask"] = action_mask
            payload["state"] = torch.as_tensor(item["state"], dtype=torch.float32)
            payload["vlm_action_target"] = action[:actual_length]
            payload["vlm_action_mask"] = action_mask[:actual_length]
            payload["vlm_action_actual_length"] = actual_length
            selected.append((inputs, payload, int(im_starts[-1])))
            total_length += length

        if not selected:
            raise ValueError(
                f"No training sample fits within MAX_LENGTH={self.max_length}"
            )

        lengths = [inputs["input_ids"].shape[1] for inputs, _, _ in selected]
        offsets = [0, *itertools.accumulate(lengths)]
        result: dict[str, Any] = {
            "input_ids": torch.cat(
                [inputs["input_ids"] for inputs, _, _ in selected], dim=1
            ),
            "position_ids": torch.cat(
                [inputs["position_ids"] for inputs, _, _ in selected], dim=2
            ),
            "cu_seq_lens_q": torch.tensor(offsets, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(offsets, dtype=torch.int32),
            "max_length_q": max(lengths),
            "max_length_k": max(lengths),
            "action_segments": torch.tensor(
                list(zip(offsets[:-1], offsets[1:])), dtype=torch.long
            ),
            "action_vlm_condition_segments": torch.tensor(
                [
                    [offsets[index], offsets[index] + end]
                    for index, (_, _, end) in enumerate(selected)
                ],
                dtype=torch.long,
            ),
        }
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        ):
            values = [
                inputs[key] for inputs, _, _ in selected if inputs.get(key) is not None
            ]
            if values:
                result[key] = torch.cat(values)

        payloads = [payload for _, payload, _ in selected]
        result.update(self._collate_memory(payloads))
        targets = [payload.pop("vlm_action_target") for payload in payloads]
        masks = [payload.pop("vlm_action_mask") for payload in payloads]
        result.update(default_collate(payloads))
        result["vlm_action_target"] = torch.cat(targets)
        result["vlm_action_mask"] = torch.cat(masks)
        return result
