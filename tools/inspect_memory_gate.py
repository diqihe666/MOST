#!/usr/bin/env python3
"""Inspect learned Action-Expert memory residual gates in a training checkpoint.

The Joint MemLearn action block uses the gate directly, without sigmoid/tanh::

    hidden_states = hidden_states + memory_gate * memory_cross_attn(...)

Therefore the scalar stored in each ``dit.layers.*.memory_gate`` parameter is
also the effective residual multiplier used during inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


GATE_SUFFIX = ".memory_gate"
LAYER_PATTERN = re.compile(r"(?:^|\.)dit\.layers\.(\d+)\.memory_gate$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the learned scalar memory residual gate from every Action "
            "Expert layer in an MOST-MemEx training checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Checkpoint file or a checkpoint directory containing "
            "mp_rank_00_model_states.pt"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optionally save the per-layer values and summary as JSON",
    )
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates = [
        path / "checkpoint" / "mp_rank_00_model_states.pt",
        path / "checkpoint" / "zero_pp_rank_0_mp_rank_00_model_states.pt",
        path / "last.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt",
        path / "last.ckpt" / "checkpoint" / "zero_pp_rank_0_mp_rank_00_model_states.pt",
        path / "mp_rank_00_model_states.pt",
        path / "zero_pp_rank_0_mp_rank_00_model_states.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    recursive = sorted(path.glob("**/*model_states.pt"))
    if len(recursive) == 1:
        return recursive[0]
    if not recursive:
        raise FileNotFoundError(
            f"No *model_states.pt checkpoint was found under {path}"
        )
    formatted = "\n  ".join(str(candidate) for candidate in recursive)
    raise RuntimeError(
        "Multiple checkpoint files were found. Pass one file explicitly:\n  "
        + formatted
    )


def load_checkpoint(path: Path) -> Any:
    def load_with_options(*, use_mmap: bool) -> Any:
        kwargs: dict[str, Any] = {"map_location": "cpu"}
        if use_mmap:
            kwargs["mmap"] = True
        try:
            return torch.load(path, weights_only=False, **kwargs)
        except TypeError:
            # Compatibility with older PyTorch versions that predate the
            # weights_only keyword.  An unsupported mmap keyword is handled by
            # the outer fallback below.
            return torch.load(path, **kwargs)

    # mmap avoids materializing the full multi-billion-parameter checkpoint in
    # RAM when the checkpoint uses PyTorch's zip serialization format.
    try:
        return load_with_options(use_mmap=True)
    except (RuntimeError, TypeError, ValueError) as mmap_error:
        print(f"mmap load unavailable ({mmap_error}); retrying without mmap")
        return load_with_options(use_mmap=False)


def extract_state_dict(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Checkpoint root must be a mapping, got {type(payload).__name__}"
        )

    for container_key in ("module", "state_dict", "model"):
        candidate = payload.get(container_key)
        if isinstance(candidate, Mapping):
            return candidate

    # Also support a state_dict saved directly with torch.save(state_dict, ...).
    if any(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise KeyError(
        "Checkpoint contains no 'module', 'state_dict', or 'model' parameter mapping"
    )


def collect_memory_gates(state_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    unrecognized: list[str] = []

    for raw_key, value in state_dict.items():
        key = str(raw_key)
        if not key.endswith(GATE_SUFFIX):
            continue
        match = LAYER_PATTERN.search(key)
        if match is None:
            unrecognized.append(key)
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} is not a tensor: {type(value).__name__}")
        if value.numel() != 1:
            raise ValueError(
                f"{key} must contain one scalar, got shape {tuple(value.shape)}"
            )
        records.append(
            {
                "layer": int(match.group(1)),
                "key": key,
                "value": float(value.detach().cpu().float().reshape(-1)[0].item()),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
        )

    if unrecognized:
        formatted = "\n  ".join(unrecognized)
        raise ValueError(
            f"Found memory_gate parameters outside dit.layers.<N>:\n  {formatted}"
        )
    if not records:
        examples = [str(key) for key in state_dict if "memory" in str(key).lower()][:10]
        hint = "\nNearby memory keys:\n  " + "\n  ".join(examples) if examples else ""
        raise KeyError(
            "No dit.layers.<N>.memory_gate parameters were found. "
            "Check that this is a trained memory-enabled checkpoint." + hint
        )

    records.sort(key=lambda item: item["layer"])
    layers = [record["layer"] for record in records]
    if len(layers) != len(set(layers)):
        raise ValueError(f"Duplicate memory gates found for layers: {layers}")
    expected = list(range(layers[-1] + 1))
    if layers != expected:
        raise ValueError(f"Memory gate layers are not contiguous: {layers}")
    return records


def make_report(checkpoint: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(record["value"]) for record in records]
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    summary = {
        "count": len(records),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std_population": math.sqrt(variance),
        "mean_abs": math.fsum(abs(value) for value in values) / len(values),
        "l2_norm": math.sqrt(math.fsum(value * value for value in values)),
        "zero_count": sum(value == 0.0 for value in values),
    }
    return {
        "checkpoint": str(checkpoint),
        "forward_formula": (
            "hidden_states = hidden_states + memory_gate * memory_cross_attn(...)"
        ),
        "gate_activation": "identity (no sigmoid/tanh)",
        "layers": records,
        "summary": summary,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"checkpoint: {report['checkpoint']}")
    print(f"formula:    {report['forward_formula']}")
    print(f"activation: {report['gate_activation']}")
    print()
    print(f"{'layer':>5}  {'gate':>16}  {'dtype':>10}  parameter")
    print(f"{'-' * 5}  {'-' * 16}  {'-' * 10}  {'-' * 40}")
    for record in report["layers"]:
        print(
            f"{record['layer']:5d}  {record['value']:+16.9e}  "
            f"{record['dtype']:>10}  {record['key']}"
        )

    summary = report["summary"]
    print()
    print(
        "summary: "
        f"count={summary['count']} "
        f"min={summary['min']:+.9e} "
        f"max={summary['max']:+.9e} "
        f"mean={summary['mean']:+.9e} "
        f"std={summary['std_population']:.9e} "
        f"mean_abs={summary['mean_abs']:.9e} "
        f"l2={summary['l2_norm']:.9e} "
        f"zeros={summary['zero_count']}"
    )


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    payload = load_checkpoint(checkpoint)
    state_dict = extract_state_dict(payload)
    records = collect_memory_gates(state_dict)
    report = make_report(checkpoint, records)
    print_report(report)

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
