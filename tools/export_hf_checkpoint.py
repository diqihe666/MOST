#!/usr/bin/env python3
"""Export a MOST DeepSpeed checkpoint to Hugging Face format."""

from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
from pathlib import Path
from typing import Mapping

import torch
from transformers import AutoModel, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".ckpt")
TRAINING_ONLY_PREFIXES = (
    "state_projector_choice.",
    "action_projector_choice.",
    "score_projector_choice.",
    "vlm.model.action_embed.",
    "vlm.model.score_embed.",
)


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "checkpoint" / "mp_rank_00_model_states.pt",
        path / "checkpoint" / "zero_pp_rank_0_mp_rank_00_model_states.pt",
        path / "last.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt",
        path / "last.ckpt" / "checkpoint" / "zero_pp_rank_0_mp_rank_00_model_states.pt",
        path / "mp_rank_00_model_states.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    recursive = sorted(path.glob("**/*model_states.pt"))
    if len(recursive) == 1:
        return recursive[0]
    if not recursive:
        raise FileNotFoundError(f"No DeepSpeed model_states checkpoint under {path}")
    raise RuntimeError(f"Multiple possible checkpoints under {path}: {recursive}")


def find_training_config(checkpoint: Path) -> Path:
    start = checkpoint.parent
    for directory in (start, *list(start.parents)[:8]):
        candidate = directory / "config.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the run config.py above checkpoint {checkpoint}"
    )


def extract_state_dict(payload: Mapping) -> dict[str, torch.Tensor]:
    state = payload.get("module") or payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise KeyError(
            "Checkpoint contains neither a 'module' nor 'state_dict' mapping"
        )

    output = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ("_forward_module.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        if key.startswith("model."):
            key = key[len("model.") :]
        output[key] = value
    return output


def strip_training_only_state(
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    kept = {}
    removed = []
    for key, value in state.items():
        if key.startswith(TRAINING_ONLY_PREFIXES):
            removed.append(key)
        else:
            kept[key] = value
    return kept, removed


def copy_support_files(base_model: Path, output_dir: Path) -> None:
    if not base_model.is_dir():
        return
    for source in base_model.iterdir():
        if not source.is_file():
            continue
        name = source.name
        if source.suffix in WEIGHT_SUFFIXES or name.endswith(".index.json"):
            continue
        target = output_dir / name
        if not target.exists():
            shutil.copy2(source, target)


def load_joint_config(config_path: Path) -> tuple[dict, dict]:
    namespace = runpy.run_path(str(config_path))
    model_cfg = dict(namespace.get("model", {}).get("params", {}).get("model", {}))
    data_cfg = dict(
        namespace.get("data", {}).get("params", {}).get("train_datasets", {})
    )
    if not bool(model_cfg.get("memory_enabled", False)):
        raise ValueError(
            f"Checkpoint config {config_path} is not Joint MemLearn (memory_enabled != true)"
        )
    return model_cfg, data_cfg


def attach_joint_modules(model, model_cfg: dict) -> None:
    """Make the public HF model's state_dict match the joint training checkpoint."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from most.models.vla.most_policy import DiT, Projector
    from most.models.vla.memory import (
        PackedMemLearnVisual,
        TemporalMemoryEncoder,
    )

    hidden_size = int(model.config.dit_config.hidden_size)
    layer_num = int(model.config.dit_config.num_hidden_layers)
    memory_dim = int(model_cfg.get("memory_dim", 384))
    memory_interface = str(model_cfg.get("memory_interface", "final_h"))
    memory_action_heads = int(model_cfg.get("memory_action_heads", 8))

    model.dit = DiT(
        layer_num=layer_num,
        hidden_size=hidden_size,
        memory_enabled=True,
        memory_heads=memory_action_heads,
        memory_interface=memory_interface,
    )
    model.memory_visual = PackedMemLearnVisual(
        dim=memory_dim,
        num_views=3,
        fusion_heads=int(model_cfg.get("memory_fusion_heads", 8)),
        dino_repo=str(model_cfg.get("memory_dino_repo", "")),
        dino_checkpoint=str(model_cfg.get("memory_dino_checkpoint", "")),
        dino_model=str(model_cfg.get("memory_dino_model", "dinov2_vits14")),
        dino_chunk_size=int(model_cfg.get("memory_dino_chunk_size", 512)),
        dino_checkpointing=False,
        freeze_dino=False,
    )
    memory_layers = int(model_cfg.get("memory_layers", 6))
    if memory_interface == "mot" and memory_layers != layer_num:
        raise ValueError(
            "Memory Expert MoT export requires depth alignment: "
            f"memory_layers={memory_layers}, action_layers={layer_num}"
        )
    model.memory_encoder = TemporalMemoryEncoder(
        dim=memory_dim,
        num_layers=memory_layers,
        num_heads=int(model_cfg.get("memory_heads", 6)),
        block_size=int(model_cfg.get("memory_block_size", 8)),
        future_steps=int(model_cfg.get("memory_future_steps", 16)),
        task_input_dim=int(model.config.vlm_config.text_config.hidden_size),
        max_task_tokens=int(model_cfg.get("memory_task_max_tokens", 128)),
        num_subtask_classes=int(model_cfg.get("memory_num_subtask_classes", 1024)),
        num_state_classes=int(model_cfg.get("memory_num_state_classes", 256)),
        num_object_categories=int(model_cfg.get("memory_num_object_categories", 256)),
        max_instance_slots=int(model_cfg.get("memory_max_instance_slots", 32)),
        action_output_dim=hidden_size if memory_interface == "mot" else None,
        action_heads=memory_action_heads,
        gradient_checkpointing=False,
    )
    if memory_interface == "final_h":
        model.memory_action_projector = Projector(
            memory_dim, hidden_size, num_layers=1, bias=False
        )
        model.memory_action_type_embedding = torch.nn.Embedding(
            3, hidden_size, padding_idx=0
        )


def write_joint_hf_code(output_dir: Path) -> None:
    shutil.copy2(PROJECT_ROOT / "modeling_most.py", output_dir / "modeling_most.py")
    shutil.copy2(
        PROJECT_ROOT / "most" / "models" / "vla" / "memory.py",
        output_dir / "memory_most.py",
    )


def set_export_config(model, model_cfg: dict, data_cfg: dict) -> None:
    exported = {
        "memory_enabled": True,
        "memory_interface": str(model_cfg.get("memory_interface", "final_h")),
        "memory_dim": int(model_cfg.get("memory_dim", 384)),
        "memory_layers": int(model_cfg.get("memory_layers", 6)),
        "memory_heads": int(model_cfg.get("memory_heads", 6)),
        "memory_action_heads": int(model_cfg.get("memory_action_heads", 8)),
        "memory_fusion_heads": int(model_cfg.get("memory_fusion_heads", 8)),
        "memory_block_size": int(model_cfg.get("memory_block_size", 8)),
        "memory_future_steps": int(model_cfg.get("memory_future_steps", 16)),
        "memory_task_max_tokens": int(model_cfg.get("memory_task_max_tokens", 128)),
        "memory_num_subtask_classes": int(
            model_cfg.get("memory_num_subtask_classes", 1024)
        ),
        "memory_num_state_classes": int(model_cfg.get("memory_num_state_classes", 256)),
        "memory_num_object_categories": int(
            model_cfg.get("memory_num_object_categories", 256)
        ),
        "memory_max_instance_slots": int(
            model_cfg.get("memory_max_instance_slots", 32)
        ),
        "memory_dino_repo": "",
        "memory_dino_model": str(model_cfg.get("memory_dino_model", "dinov2_vits14")),
        "memory_dino_chunk_size": int(model_cfg.get("memory_dino_chunk_size", 512)),
        "memory_image_size": int(data_cfg.get("memory_image_size", 224)),
        "memory_stride": int(data_cfg.get("memory_stride", 32)),
    }
    for key, value in exported.items():
        setattr(model.config, key, value)
    auto_map = dict(getattr(model.config, "auto_map", {}) or {})
    auto_map["AutoModel"] = "modeling_most.MOSTForActionGeneration"
    model.config.auto_map = auto_map
    model.config.architectures = ["MOSTForActionGeneration"]


def patch_saved_config(output_dir: Path, model_cfg: dict, data_cfg: dict) -> None:
    """Force AutoModel to use the joint deployment class after save_pretrained.

    Transformers may rewrite auto_map while saving a trust_remote_code model, so
    patch the serialized config.json as the final source of truth.
    """
    path = output_dir / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["memory_enabled"] = True
    payload["memory_interface"] = str(model_cfg.get("memory_interface", "final_h"))
    payload["memory_dim"] = int(model_cfg.get("memory_dim", 384))
    payload["memory_layers"] = int(model_cfg.get("memory_layers", 6))
    payload["memory_heads"] = int(model_cfg.get("memory_heads", 6))
    payload["memory_action_heads"] = int(model_cfg.get("memory_action_heads", 8))
    payload["memory_fusion_heads"] = int(model_cfg.get("memory_fusion_heads", 8))
    payload["memory_block_size"] = int(model_cfg.get("memory_block_size", 8))
    payload["memory_future_steps"] = int(model_cfg.get("memory_future_steps", 16))
    payload["memory_task_max_tokens"] = int(
        model_cfg.get("memory_task_max_tokens", 128)
    )
    payload["memory_num_subtask_classes"] = int(
        model_cfg.get("memory_num_subtask_classes", 1024)
    )
    payload["memory_num_state_classes"] = int(
        model_cfg.get("memory_num_state_classes", 256)
    )
    payload["memory_num_object_categories"] = int(
        model_cfg.get("memory_num_object_categories", 256)
    )
    payload["memory_max_instance_slots"] = int(
        model_cfg.get("memory_max_instance_slots", 32)
    )
    payload["memory_dino_repo"] = ""
    payload["memory_dino_model"] = str(
        model_cfg.get("memory_dino_model", "dinov2_vits14")
    )
    payload["memory_dino_chunk_size"] = int(
        model_cfg.get("memory_dino_chunk_size", 512)
    )
    payload["memory_image_size"] = int(data_cfg.get("memory_image_size", 224))
    payload["memory_stride"] = int(data_cfg.get("memory_stride", 32))
    auto_map = dict(payload.get("auto_map", {}) or {})
    auto_map["AutoModel"] = "modeling_most.MOSTForActionGeneration"
    payload["auto_map"] = auto_map
    payload["architectures"] = ["MOSTForActionGeneration"]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="4GB")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    training_config = find_training_config(checkpoint_path)
    model_cfg, data_cfg = load_joint_config(training_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_support_files(args.base_model, args.output_dir)

    print(f"Loading base Hugging Face model: {args.base_model}")
    model = AutoModel.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=args.base_model.exists(),
    ).cpu()

    print("Attaching Joint MemLearn modules before loading checkpoint...")
    attach_joint_modules(model, model_cfg)

    print(f"Loading training checkpoint: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    raw_state = extract_state_dict(payload)
    state, removed = strip_training_only_state(raw_state)
    print(f"Ignoring {len(removed)} training-only choice-head tensors during export.")

    # This is deliberately strict.  A successful export must contain every
    # Joint MemLearn parameter; silently discarding memory weights is forbidden.
    info = model.load_state_dict(state, strict=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(
            f"Joint checkpoint mismatch: missing={info.missing_keys}, "
            f"unexpected={info.unexpected_keys}"
        )

    set_export_config(model, model_cfg, data_cfg)
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    processor = AutoProcessor.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=args.base_model.exists(),
    )
    processor.save_pretrained(args.output_dir)
    copy_support_files(args.base_model, args.output_dir)
    write_joint_hf_code(args.output_dir)
    patch_saved_config(args.output_dir, model_cfg, data_cfg)

    manifest = {
        "base_model": str(args.base_model.resolve()),
        "training_checkpoint": str(checkpoint_path.resolve()),
        "training_config": str(training_config.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "memory_enabled": True,
        "memory_interface": str(model_cfg.get("memory_interface", "final_h")),
        "memory_stride": int(data_cfg.get("memory_stride", 32)),
        "memory_future_steps": int(model_cfg.get("memory_future_steps", 16)),
        "filtered_training_only_keys": removed,
    }
    (args.output_dir / "finetune_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported Joint MemLearn HF checkpoint to {args.output_dir}")
    print("Serve this directory with deploy/server.py.")


if __name__ == "__main__":
    main()
