# Copyright (C) 2026 MOST Contributors.
"""Convert a Hugging Face MOST checkpoint for the Lightning/DeepSpeed trainer.

Besides ``model_states.pt``, this script extracts the exact nested Qwen3-VL
configuration used by the public checkpoint.  The trainer can therefore build
its architecture fully offline and does not need to fetch
``Qwen/Qwen3-VL-4B-Instruct`` from the internet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModel, Qwen3VLConfig


def _iter_config_candidates(model: torch.nn.Module) -> Iterable[Any]:
    seen: set[int] = set()
    for candidate in [getattr(model, "config", None)]:
        if candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            yield candidate
    for module in model.modules():
        candidate = getattr(module, "config", None)
        if candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            yield candidate


def _find_qwen3vl_config(model: torch.nn.Module) -> Qwen3VLConfig:
    for config in _iter_config_candidates(model):
        if isinstance(config, Qwen3VLConfig):
            return config
        if getattr(config, "model_type", None) == "qwen3_vl":
            return Qwen3VLConfig.from_dict(config.to_dict())

        # Some remote-code wrappers store the VLM configuration under a
        # nested attribute instead of exposing it as a module config.
        for name in ("vlm_config", "qwen_config", "backbone_config"):
            nested = getattr(config, name, None)
            if isinstance(nested, Qwen3VLConfig):
                return nested
            if getattr(nested, "model_type", None) == "qwen3_vl":
                return Qwen3VLConfig.from_dict(nested.to_dict())

        if hasattr(config, "to_dict"):
            payload = config.to_dict()
            for value in payload.values():
                if isinstance(value, dict) and value.get("model_type") == "qwen3_vl":
                    return Qwen3VLConfig.from_dict(value)

    raise RuntimeError(
        "Could not locate the nested Qwen3-VL configuration in the public MOST model. "
        "Do not start training with a guessed backbone configuration."
    )


def convert(model_path: str, output_dir: str, output_filename: str) -> None:
    local_model = Path(model_path).expanduser()
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=local_model.exists(),
    ).cpu()
    state_dict = {
        f"model.{name}": tensor for name, tensor in model.state_dict().items()
    }

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / output_filename
    torch.save({"module": state_dict}, output_path)

    qwen_config = _find_qwen3vl_config(model)
    vlm_config_dir = output_dir_path / "vlm_config"
    qwen_config.save_pretrained(vlm_config_dir)

    manifest = {
        "source_model": str(Path(model_path).expanduser().resolve()),
        "checkpoint": str(output_path.resolve()),
        "vlm_config": str(vlm_config_dir.resolve()),
        "num_state_tensors": len(state_dict),
        "dtype": "bfloat16",
    }
    (output_dir_path / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved checkpoint to {output_path}")
    print(f"Saved Qwen3-VL config to {vlm_config_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="pretrained_ckpt")
    parser.add_argument("--output_filename", default="model_states.pt")
    args = parser.parse_args()
    convert(args.model_path, args.output_dir, args.output_filename)


if __name__ == "__main__":
    main()
