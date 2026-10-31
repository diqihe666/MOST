# coding=utf-8
# Copyright (C) 2026 MOST Contributors.
"""Hugging Face deployment model for MOST with the memory pathway."""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, rotate_half

# The compatible base checkpoint exposes this legacy module and class name.
# Keep the compatibility detail private to the exported Hugging Face model.
from .modeling_mibot import MiBoTForActionGeneration as _BaseActionModel
from .memory_most import (
    ActionMemoryCrossAttention,
    ActionMemoryMoTAttention,
    PackedMemLearnVisual,
    TemporalMemoryEncoder,
)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x.repeat_interleave(repeats, dim=1)


def repeat_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    if x.shape[0] == batch_size:
        return x
    if batch_size % x.shape[0] != 0:
        raise ValueError(f"Cannot repeat batch of size {x.shape[0]} to {batch_size}")
    return x.repeat_interleave(batch_size // x.shape[0], dim=0)


def apply_rotary_pos_emb(query, key, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (
        rotate_half(key) * sin
    )


class JointAttention(nn.Module):
    """MOST action attention matching the post-training implementation."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(config.head_dim)
        self.num_heads = self.hidden_size // self.head_dim
        kv_heads = int(config.num_key_value_heads)
        self.kv_group = self.num_heads // kv_heads
        self.qkv_proj = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(self.head_dim)
        self.k_norm = Qwen2RMSNorm(self.head_dim)

    def forward(self, hidden_states, past_key_values, position_embeds, attn_mask):
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeds
        if cos.ndim == 4:
            cos, sin = cos[0], sin[0]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        cache_key, cache_value = past_key_values
        cache_key = repeat_kv(repeat_batch(cache_key, batch_size), self.kv_group)
        cache_value = repeat_kv(repeat_batch(cache_value, batch_size), self.kv_group)
        key = torch.cat([cache_key, key], dim=-2)
        value = torch.cat([cache_value, value], dim=-2)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        return self.o_proj(output)


class JointMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class JointDecoderLayer(nn.Module):
    """MOST decoder layer with a depth-aligned Memory Expert MoT interaction."""

    def __init__(self, config, memory_heads: int, memory_interface: str = "mot"):
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.attn = JointAttention(config)
        self.mlp = JointMLP(hidden_size)
        self.input_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.post_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.memory_interface = str(memory_interface)
        self.memory_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        if self.memory_interface == "mot":
            self.memory_cross_attn = ActionMemoryMoTAttention(
                dim=hidden_size, num_heads=memory_heads
            )
        elif self.memory_interface == "final_h":
            self.memory_cross_attn = ActionMemoryCrossAttention(
                dim=hidden_size, num_heads=memory_heads
            )
        else:
            raise ValueError(f"Unknown memory_interface={self.memory_interface!r}")
        self.memory_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states,
        past_key_values,
        position_embeds,
        t_embeds,
        attn_mask=None,
        memory_tokens=None,
        memory_kv=None,
        memory_pad=None,
    ):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_table[None] + t_embeds
        ).chunk(6, dim=1)
        residual = hidden_states
        hidden_states = modulate(
            self.input_layernorm(hidden_states), shift_attn, scale_attn
        )
        hidden_states = residual + gate_attn * self.attn(
            hidden_states, past_key_values, position_embeds, attn_mask
        )
        memory_query = self.memory_layernorm(hidden_states)
        if self.memory_interface == "mot" and memory_kv is not None:
            hidden_states = hidden_states + self.memory_gate * self.memory_cross_attn(
                memory_query, memory_kv, memory_pad
            )
        elif self.memory_interface == "final_h" and memory_tokens is not None:
            hidden_states = hidden_states + self.memory_gate * self.memory_cross_attn(
                memory_query, memory_tokens, memory_pad
            )
        residual = hidden_states
        hidden_states = modulate(
            self.post_layernorm(hidden_states), shift_mlp, scale_mlp
        )
        return residual + gate_mlp * self.mlp(hidden_states)


class JointDiT(nn.Module):
    """Drop-in MOST action expert consuming VLM KV then Memory Expert KV."""

    def __init__(self, config, memory_heads: int, memory_interface: str = "mot"):
        super().__init__()
        self.memory_interface = str(memory_interface)
        self.layers = nn.ModuleList(
            [
                JointDecoderLayer(
                    config,
                    memory_heads=memory_heads,
                    memory_interface=self.memory_interface,
                )
                for _ in range(int(config.num_hidden_layers))
            ]
        )
        self._memory_tokens: Optional[torch.Tensor] = None
        self._memory_pad: Optional[torch.Tensor] = None
        self._memory_layer_kvs = None

    def set_memory(self, memory_tokens, memory_pad):
        self._memory_tokens = memory_tokens
        self._memory_pad = memory_pad
        self._memory_layer_kvs = None

    def set_memory_kv(self, memory_layer_kvs, memory_pad):
        if memory_layer_kvs is not None and len(memory_layer_kvs) != len(self.layers):
            raise ValueError(
                f"Memory Expert returned {len(memory_layer_kvs)} K/V pairs for "
                f"{len(self.layers)} Action Expert layers"
            )
        self._memory_tokens = None
        self._memory_pad = memory_pad
        self._memory_layer_kvs = memory_layer_kvs

    def clear_memory(self):
        self._memory_tokens = None
        self._memory_pad = None
        self._memory_layer_kvs = None

    def forward(
        self, hidden_states, past_key_values, attn_mask, position_embeds, t_embeds
    ):
        start_index = len(past_key_values) - len(self.layers)
        if start_index < 0:
            raise ValueError("VLM cache has fewer layers than JointDiT")
        for i, layer in enumerate(self.layers):
            memory_kv = (
                None if self._memory_layer_kvs is None else self._memory_layer_kvs[i]
            )
            hidden_states = layer(
                hidden_states,
                past_key_values[start_index + i],
                position_embeds,
                t_embeds,
                attn_mask=attn_mask,
                memory_tokens=self._memory_tokens,
                memory_kv=memory_kv,
                memory_pad=self._memory_pad,
            )
        return hidden_states


class JointProjector(nn.Module):
    """One-layer projector with the same state-dict naming as training."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False))

    def forward(self, x):
        return self.layers(x)


class MOSTForActionGeneration(_BaseActionModel):
    """MOST with the jointly trained MemLearn memory branch."""

    def __init__(self, config):
        super().__init__(config)
        if not bool(getattr(config, "memory_enabled", False)):
            raise ValueError(
                "MOSTForActionGeneration requires config.memory_enabled=true"
            )

        self.memory_interface = str(getattr(config, "memory_interface", "final_h"))
        if self.memory_interface not in {"final_h", "mot"}:
            raise ValueError(f"Unknown memory_interface={self.memory_interface!r}")
        self.memory_dim = int(getattr(config, "memory_dim", 384))
        self.memory_future_steps = int(getattr(config, "memory_future_steps", 16))
        self.memory_task_max_tokens = int(
            getattr(config, "memory_task_max_tokens", 128)
        )
        memory_action_heads = int(getattr(config, "memory_action_heads", 8))

        # Replace only the action expert. The base VLM and projectors are kept.
        self.dit = JointDiT(
            config.dit_config,
            memory_heads=memory_action_heads,
            memory_interface=self.memory_interface,
        )

        dino_repo = os.environ.get("MOST_MEM_DINO_REPO") or str(
            getattr(config, "memory_dino_repo", "")
        )
        dino_chunk = int(
            os.environ.get(
                "DINO_CHUNK", str(getattr(config, "memory_dino_chunk_size", 512))
            )
        )
        self.memory_visual = PackedMemLearnVisual(
            dim=self.memory_dim,
            num_views=3,
            fusion_heads=int(getattr(config, "memory_fusion_heads", 8)),
            dino_repo=dino_repo,
            # DINO weights are inside the exported safetensors. Only its
            # architecture is constructed from the local DINOv2 repository.
            dino_checkpoint=None,
            dino_model=str(getattr(config, "memory_dino_model", "dinov2_vits14")),
            dino_chunk_size=dino_chunk,
            dino_checkpointing=False,
            freeze_dino=True,
        )
        memory_layers = int(getattr(config, "memory_layers", 6))
        if self.memory_interface == "mot" and memory_layers != len(self.dit.layers):
            raise ValueError(
                "Memory Expert MoT requires depth alignment with Action Expert: "
                f"memory_layers={memory_layers}, action_layers={len(self.dit.layers)}"
            )
        self.memory_encoder = TemporalMemoryEncoder(
            dim=self.memory_dim,
            num_layers=memory_layers,
            num_heads=int(getattr(config, "memory_heads", 6)),
            block_size=int(getattr(config, "memory_block_size", 8)),
            future_steps=self.memory_future_steps,
            task_input_dim=int(config.vlm_config.text_config.hidden_size),
            max_task_tokens=self.memory_task_max_tokens,
            num_subtask_classes=int(
                getattr(config, "memory_num_subtask_classes", 1024)
            ),
            num_state_classes=int(getattr(config, "memory_num_state_classes", 256)),
            num_object_categories=int(
                getattr(config, "memory_num_object_categories", 256)
            ),
            max_instance_slots=int(getattr(config, "memory_max_instance_slots", 32)),
            action_output_dim=(
                int(config.dit_config.hidden_size)
                if self.memory_interface == "mot"
                else None
            ),
            action_heads=memory_action_heads,
            gradient_checkpointing=False,
        )
        if self.memory_interface == "final_h":
            self.memory_action_projector = JointProjector(
                self.memory_dim, int(config.dit_config.hidden_size)
            )
            self.memory_action_type_embedding = nn.Embedding(
                3, int(config.dit_config.hidden_size), padding_idx=0
            )

        # Stateful block-causal inference cache. Completed temporal blocks keep
        # their per-layer K/V and final H; only the active block + future slots
        # are recomputed. Raw RGB and completed-block visual tokens are discarded.
        self._memory_inference_state = None

    @staticmethod
    def _signed_memory_time_embedding(
        history_lengths: torch.Tensor,
        seq_len: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if dim % 2 != 0:
            raise ValueError("memory action time embedding dim must be even")
        history_lengths = history_lengths.to(device=device, dtype=torch.long)
        positions = torch.arange(seq_len, device=device, dtype=torch.long)[None]
        relative = positions - (history_lengths[:, None] - 1)
        half = dim // 2
        inv = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / half
        )
        angles = relative.float()[..., None] * inv[None, None, :]
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(dtype)

    def _reset_memory_inference_cache(self) -> None:
        self._memory_inference_state = self.memory_encoder.init_inference_state()

    def _encode_new_memory_frames(
        self,
        memory_frames: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if memory_frames is None:
            return None
        if memory_frames.ndim != 5:
            raise ValueError(
                f"memory_frames must be [N,V,C,H,W], got {tuple(memory_frames.shape)}"
            )
        new_steps = int(memory_frames.shape[0])
        if new_steps <= 0:
            return None
        device = memory_frames.device
        offsets = torch.tensor([0, new_steps], dtype=torch.long, device=device)
        lengths = torch.tensor([new_steps], dtype=torch.long, device=device)
        return self.memory_visual(memory_frames, offsets, lengths)

    def _memory_forward_cached(
        self,
        memory_frames: Optional[torch.Tensor],
        memory_task_input_ids: torch.Tensor,
        memory_task_attention_mask: torch.Tensor,
        memory_reset: bool = False,
    ):
        if memory_reset or self._memory_inference_state is None:
            self._reset_memory_inference_cache()

        new_visual = self._encode_new_memory_frames(memory_frames)
        if new_visual is None and self._memory_inference_state.active_history is None:
            raise RuntimeError(
                "Joint memory inference cache is empty. The first request of "
                "an episode must set memory_reset=True and include memory_frames."
            )

        dtype = (
            new_visual.dtype
            if new_visual is not None
            else self._memory_inference_state.active_history.dtype
        )
        raw_task = self.vlm.model.get_input_embeddings()(memory_task_input_ids)
        task_tokens, task_pad = self.memory_encoder.task_condition(
            raw_task.to(dtype), memory_task_attention_mask
        )

        if self.memory_interface == "mot":
            (
                _h,
                memory_pad,
                _is_history,
                _is_future,
                _history_length,
                memory_layer_kvs,
            ) = self.memory_encoder.forward_incremental(
                self._memory_inference_state,
                new_history_tokens=new_visual,
                task_tokens=task_tokens,
                task_pad=task_pad,
                return_action_kv=True,
            )
            return None, memory_layer_kvs, memory_pad

        h, memory_pad, is_history, is_future, history_length = (
            self.memory_encoder.forward_incremental(
                self._memory_inference_state,
                new_history_tokens=new_visual,
                task_tokens=task_tokens,
                task_pad=task_pad,
            )
        )
        history_lengths = torch.tensor(
            [history_length], dtype=torch.long, device=h.device
        )
        action_memory = self.memory_action_projector(h)
        type_ids = torch.zeros_like(memory_pad, dtype=torch.long)
        type_ids[is_history] = 1
        type_ids[is_future] = 2
        action_memory = action_memory + self.memory_action_type_embedding(type_ids).to(
            action_memory.dtype
        )
        action_memory = action_memory + self._signed_memory_time_embedding(
            history_lengths,
            action_memory.shape[1],
            action_memory.shape[2],
            action_memory.dtype,
            action_memory.device,
        )
        return action_memory, None, memory_pad

    @torch.no_grad()
    def forward(self, state, action_mask, num_steps=5, **kwargs):
        required = (
            "memory_task_input_ids",
            "memory_task_attention_mask",
        )
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise KeyError(f"Joint MemLearn inference request is missing: {missing}")

        memory_task_input_ids = kwargs.pop("memory_task_input_ids")
        memory_task_attention_mask = kwargs.pop("memory_task_attention_mask")
        memory_frames = kwargs.pop("memory_frames", None)
        memory_reset = bool(kwargs.pop("memory_reset", False))

        # Drop legacy full-history metadata if an older client happens to send
        # it.  Incremental inference derives history length from the cache.
        kwargs.pop("memory_history_offsets", None)
        kwargs.pop("memory_history_lengths", None)

        memory_tokens, memory_layer_kvs, memory_pad = self._memory_forward_cached(
            memory_frames=memory_frames,
            memory_task_input_ids=memory_task_input_ids,
            memory_task_attention_mask=memory_task_attention_mask,
            memory_reset=memory_reset,
        )
        if self.memory_interface == "mot":
            self.dit.set_memory_kv(memory_layer_kvs, memory_pad)
        else:
            self.dit.set_memory(memory_tokens, memory_pad)
        try:
            return super().forward(
                state=state,
                action_mask=action_mask,
                num_steps=num_steps,
                **kwargs,
            )
        finally:
            self.dit.clear_memory()


__all__ = ["MOSTForActionGeneration"]
