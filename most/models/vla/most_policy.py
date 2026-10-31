# Copyright (C) 2026 MOST Contributors.
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributions import Beta
from transformers import Qwen3VLConfig
from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, rotate_half
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from most.models import MODEL_REGISTRY
from most.models.vlm.qwen3vl import (
    ACTION_END_ID,
    ACTION_START_ID,
    SCORE_ID,
    Qwen3VLForConditionalGeneration,
)
from most.utils.model_utils import auto_cast
from most.models.vla.memory import (
    ActionMemoryCrossAttention,
    ActionMemoryMoTAttention,
    PackedMemLearnVisual,
    TemporalMemoryEncoder,
)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def repeat_kv(x, repeats):
    return x.repeat_interleave(repeats, dim=1)


def repeat_batch(x, batch_size):
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


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, inter_dim=None, num_layers=1, bias=False):
        super().__init__()
        inter_dim = output_dim if inter_dim is None else inter_dim
        if num_layers == 1:
            layers = [nn.Linear(input_dim, output_dim, bias=bias)]
        else:
            layers = [nn.Linear(input_dim, inter_dim, bias=bias)]
            for _ in range(1, num_layers - 1):
                layers.extend(
                    [
                        nn.GELU(approximate="tanh"),
                        nn.Linear(inter_dim, inter_dim, bias=bias),
                    ]
                )
            layers.extend(
                [
                    nn.GELU(approximate="tanh"),
                    nn.Linear(inter_dim, output_dim, bias=bias),
                ]
            )
        self.layers = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, timestep):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / half
        )
        args = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(
            torch.bfloat16
        )
        return self.mlp(embedding)[:, None]


class Attention(nn.Module):
    def __init__(self, hidden_size=1024, head_dim=128, kv_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // kv_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(head_dim)
        self.k_norm = Qwen2RMSNorm(head_dim)

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


class MLP(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        memory_enabled=False,
        memory_heads=8,
        memory_interface="final_h",
    ):
        super().__init__()
        self.attn = Attention(hidden_size=hidden_size)
        self.mlp = MLP(hidden_size=hidden_size)
        self.input_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.post_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.memory_enabled = bool(memory_enabled)
        self.memory_interface = str(memory_interface)
        if self.memory_interface not in {"final_h", "mot"}:
            raise ValueError(f"Unknown memory_interface={self.memory_interface!r}")
        if self.memory_enabled:
            self.memory_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
            if self.memory_interface == "mot":
                # Memory Expert already emitted the layer-wise K/V.  Action
                # Expert computes only Q and consumes the corresponding cache.
                self.memory_cross_attn = ActionMemoryMoTAttention(
                    dim=hidden_size, num_heads=memory_heads
                )
            else:
                self.memory_cross_attn = ActionMemoryCrossAttention(
                    dim=hidden_size, num_heads=memory_heads
                )
            # Zero-init keeps the pretrained MOST policy exactly unchanged at
            # initialization while the Memory Expert learns from aux losses.
            self.memory_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states,
        past_key_values,
        position_embeds,
        timestep,
        attn_mask,
        memory_tokens=None,
        memory_kv=None,
        memory_pad=None,
    ):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_table[None] + timestep
        ).chunk(6, dim=1)
        residual = hidden_states
        hidden_states = modulate(
            self.input_layernorm(hidden_states), shift_attn, scale_attn
        )
        # Original MOST MoT interaction with the layer-aligned VLM cache.
        hidden_states = residual + gate_attn * self.attn(
            hidden_states, past_key_values, position_embeds, attn_mask
        )
        if self.memory_enabled:
            memory_query = self.memory_layernorm(hidden_states)
            if self.memory_interface == "mot" and memory_kv is not None:
                hidden_states = (
                    hidden_states
                    + self.memory_gate
                    * self.memory_cross_attn(memory_query, memory_kv, memory_pad)
                )
            elif self.memory_interface == "final_h" and memory_tokens is not None:
                hidden_states = (
                    hidden_states
                    + self.memory_gate
                    * self.memory_cross_attn(memory_query, memory_tokens, memory_pad)
                )
        residual = hidden_states
        hidden_states = modulate(
            self.post_layernorm(hidden_states), shift_mlp, scale_mlp
        )
        return residual + gate_mlp * self.mlp(hidden_states)


class DiT(nn.Module):
    def __init__(
        self,
        layer_num=36,
        hidden_size=1024,
        memory_enabled=False,
        memory_heads=8,
        memory_interface="final_h",
    ):
        super().__init__()
        self.layer_num = int(layer_num)
        self.memory_interface = str(memory_interface)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_size,
                    memory_enabled=memory_enabled,
                    memory_heads=memory_heads,
                    memory_interface=self.memory_interface,
                )
                for _ in range(self.layer_num)
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        hidden_states,
        past_key_values,
        attn_mask,
        position_embeds,
        timestep,
        memory_tokens=None,
        memory_layer_kvs=None,
        memory_pad=None,
    ):
        start = len(past_key_values) - self.layer_num
        if start < 0:
            raise ValueError("VLM cache has fewer layers than DiT")
        if memory_layer_kvs is not None and len(memory_layer_kvs) != self.layer_num:
            raise ValueError(
                f"Memory Expert returned {len(memory_layer_kvs)} layer K/V pairs, "
                f"but Action Expert has {self.layer_num} layers"
            )
        for index, layer in enumerate(self.layers):
            layer_memory_kv = (
                None if memory_layer_kvs is None else memory_layer_kvs[index]
            )
            hidden_states = layer(
                hidden_states,
                past_key_values[start + index],
                position_embeds,
                timestep,
                attn_mask,
                memory_tokens=memory_tokens,
                memory_kv=layer_memory_kv,
                memory_pad=memory_pad,
            )
        return hidden_states


@MODEL_REGISTRY.register_module()
class MOSTPolicy(nn.Module):
    def __init__(
        self,
        freq_coefficient=1.0,
        freq_excluded_dims=None,
        ffn_gradient_checkpointing=True,
        async_train=True,
        vlm_config_path="Qwen/Qwen3-VL-4B-Instruct",
        local_files_only=False,
        memory_enabled=False,
        memory_interface="final_h",
        memory_dim=384,
        memory_layers=6,
        memory_heads=6,
        memory_action_heads=8,
        memory_fusion_heads=8,
        memory_block_size=8,
        memory_future_steps=16,
        memory_task_max_tokens=128,
        memory_num_subtask_classes=1024,
        memory_num_state_classes=256,
        memory_num_object_categories=256,
        memory_max_instance_slots=32,
        memory_subtask_loss_weight=1.0,
        memory_state_loss_weight=1.0,
        memory_dino_repo="",
        memory_dino_checkpoint="",
        memory_dino_model="dinov2_vits14",
        memory_dino_chunk_size=512,
        memory_dino_checkpointing=True,
        memory_freeze_dino=False,
        memory_gradient_checkpointing=False,
    ):
        super().__init__()
        self.state_shape = (1, 60)
        self.action_shape = (30, 60)
        self.freq_coefficient = float(freq_coefficient)
        self.freq_excluded_dims = list(
            [17, 18, 19] if freq_excluded_dims is None else freq_excluded_dims
        )
        self.ffn_gradient_checkpointing = bool(ffn_gradient_checkpointing)
        self.async_train = bool(async_train)
        self.vlm_config_path = str(vlm_config_path)
        self.local_files_only = bool(local_files_only)
        self.memory_enabled = bool(memory_enabled)
        self.memory_interface = str(memory_interface)
        if self.memory_interface not in {"final_h", "mot"}:
            raise ValueError(f"Unknown memory_interface={self.memory_interface!r}")
        self.memory_dim = int(memory_dim)
        self.memory_layers = int(memory_layers)
        self.memory_heads = int(memory_heads)
        self.memory_action_heads = int(memory_action_heads)
        self.memory_fusion_heads = int(memory_fusion_heads)
        self.memory_block_size = int(memory_block_size)
        self.memory_future_steps = int(memory_future_steps)
        self.memory_task_max_tokens = int(memory_task_max_tokens)
        self.memory_num_subtask_classes = int(memory_num_subtask_classes)
        self.memory_num_state_classes = int(memory_num_state_classes)
        self.memory_num_object_categories = int(memory_num_object_categories)
        self.memory_max_instance_slots = int(memory_max_instance_slots)
        self.memory_subtask_loss_weight = float(memory_subtask_loss_weight)
        self.memory_state_loss_weight = float(memory_state_loss_weight)
        self.memory_dino_repo = str(memory_dino_repo)
        self.memory_dino_checkpoint = str(memory_dino_checkpoint)
        self.memory_dino_model = str(memory_dino_model)
        self.memory_dino_chunk_size = max(1, int(memory_dino_chunk_size))
        self.memory_dino_checkpointing = bool(memory_dino_checkpointing)
        self.memory_freeze_dino = bool(memory_freeze_dino)
        self.memory_gradient_checkpointing = bool(memory_gradient_checkpointing)
        self._memory_shape_logged = False
        self.training_repeat = 4
        self.num_steps = 5
        self.prefix_mask_prob = 0.5
        self.n_choices = 5
        self.beta = Beta(1.5, 1.0)
        self._build_model()

    def _build_model(self):
        config = Qwen3VLConfig.from_pretrained(
            self.vlm_config_path,
            local_files_only=self.local_files_only,
        )
        self.vlm = Qwen3VLForConditionalGeneration._from_config(
            config,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        ).train()
        self.vlm.model.get_input_embeddings().requires_grad_(False)
        self.vlm.model.visual.gradient_checkpointing_enable()
        if self.ffn_gradient_checkpointing:
            for layer in self.vlm.model.language_model.layers:
                mlp = layer.mlp
                original_forward = mlp.forward

                def checkpointed_forward(x, forward=original_forward, module=mlp):
                    if module.training and torch.is_grad_enabled():
                        return torch.utils.checkpoint.checkpoint(
                            forward, x, use_reentrant=False
                        )
                    return forward(x)

                mlp.forward = checkpointed_forward

        vlm_hidden_size = self.vlm.config.text_config.hidden_size
        self.state_projector_choice = Projector(
            self.state_shape[-1], vlm_hidden_size, num_layers=2
        )
        self.action_projector_choice = nn.Sequential(
            Projector(vlm_hidden_size, vlm_hidden_size, num_layers=4),
            Projector(vlm_hidden_size, self.action_shape[-1] * self.n_choices),
        )
        self.score_projector_choice = nn.Sequential(
            Projector(vlm_hidden_size, vlm_hidden_size, num_layers=4),
            Projector(vlm_hidden_size, self.n_choices),
        )
        self.dit = DiT(
            memory_enabled=self.memory_enabled,
            memory_heads=self.memory_action_heads,
            memory_interface=self.memory_interface,
        )
        if self.memory_enabled:
            self.memory_visual = PackedMemLearnVisual(
                dim=self.memory_dim,
                num_views=3,
                fusion_heads=self.memory_fusion_heads,
                dino_repo=self.memory_dino_repo,
                dino_checkpoint=self.memory_dino_checkpoint,
                dino_model=self.memory_dino_model,
                dino_chunk_size=self.memory_dino_chunk_size,
                dino_checkpointing=self.memory_dino_checkpointing,
                freeze_dino=self.memory_freeze_dino,
            )
            if (
                self.memory_interface == "mot"
                and self.memory_layers != self.dit.layer_num
            ):
                raise ValueError(
                    "Memory Expert MoT requires depth alignment with the Action Expert: "
                    f"memory_layers={self.memory_layers}, action_layers={self.dit.layer_num}"
                )
            self.memory_encoder = TemporalMemoryEncoder(
                dim=self.memory_dim,
                num_layers=self.memory_layers,
                num_heads=self.memory_heads,
                block_size=self.memory_block_size,
                future_steps=self.memory_future_steps,
                task_input_dim=vlm_hidden_size,
                max_task_tokens=self.memory_task_max_tokens,
                num_subtask_classes=self.memory_num_subtask_classes,
                num_state_classes=self.memory_num_state_classes,
                num_object_categories=self.memory_num_object_categories,
                max_instance_slots=self.memory_max_instance_slots,
                action_output_dim=1024 if self.memory_interface == "mot" else None,
                action_heads=self.memory_action_heads,
                gradient_checkpointing=self.memory_gradient_checkpointing,
            )
            if self.memory_interface == "final_h":
                # Legacy prototype path retained only for reproducing the first
                # H-effect experiments.  The formal model uses layer-wise MoT.
                self.memory_action_projector = Projector(
                    self.memory_dim, 1024, num_layers=1, bias=False
                )
                self.memory_action_type_embedding = nn.Embedding(3, 1024, padding_idx=0)
        self.state_projector = Projector(self.state_shape[-1], 1024, num_layers=2)
        self.action_projector = Projector(self.action_shape[-1], 1024, num_layers=2)
        self.action_output_layer = Projector(
            1024, self.action_shape[-1], inter_dim=1024, num_layers=2
        )
        self.t_embedder = TimestepEmbedder(1024)
        self.t_projector = Projector(1024, 6 * 1024, bias=True)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(self.vlm.config.text_config)
        self.sink = nn.Embedding(1, 1024)
        self.to(torch.bfloat16)

    @staticmethod
    def _unpad(past_key_values, position_ids, attention_mask, segments):
        if segments is None:
            return past_key_values, position_ids, attention_mask
        keys = torch.stack([layer[0] for layer in past_key_values])
        values = torch.stack([layer[1] for layer in past_key_values])
        lengths = segments[:, 1] - segments[:, 0]
        max_length = int(lengths.max().item())
        layer_num, _, head_num, _, head_dim = keys.shape
        batch_size = len(segments)
        new_keys = keys.new_zeros(layer_num, batch_size, head_num, max_length, head_dim)
        new_values = values.new_zeros(new_keys.shape)
        new_positions = position_ids.new_zeros(3, batch_size, max_length)
        attention_mask = torch.zeros(
            batch_size, max_length, device=keys.device, dtype=torch.bool
        )
        for index, (start, end) in enumerate(segments.tolist()):
            length = end - start
            new_keys[:, index, :, :length] = keys[:, 0, :, start:end]
            new_values[:, index, :, :length] = values[:, 0, :, start:end]
            new_positions[:, index, :length] = position_ids[:, 0, start:end]
            attention_mask[index, :length] = True
        return list(zip(new_keys, new_values)), new_positions, attention_mask

    def _repeat(self, x, dim=0):
        if not self.training:
            return x
        return x.repeat_interleave(self.training_repeat, dim=dim)

    def _random_mask_prefix(self, mask, prefix_length, state_length, keep_last=2):
        if prefix_length <= keep_last:
            return mask
        action_start = 1 + state_length
        prefix_end = action_start + prefix_length - keep_last
        suffix_start = action_start + prefix_length
        random_mask = (
            torch.rand(prefix_length - keep_last, device=mask.device)
            < self.prefix_mask_prob
        )
        mask = mask.clone()
        mask[:, suffix_start:, action_start:prefix_end] *= (~random_mask).int()
        return mask

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

    def _memory_forward(self, batch):
        if not self.memory_enabled:
            return None, None, None, {}

        core_required = (
            "memory_frames",
            "memory_history_offsets",
            "memory_history_lengths",
            "memory_task_input_ids",
            "memory_task_attention_mask",
        )
        missing = [name for name in core_required if name not in batch]
        if missing:
            raise KeyError(f"Memory-enabled MOST batch is missing fields: {missing}")

        frames = batch.pop("memory_frames")
        history_offsets = batch.pop("memory_history_offsets")
        history_lengths = batch.pop("memory_history_lengths")
        task_input_ids = batch.pop("memory_task_input_ids")
        task_attention_mask = batch.pop("memory_task_attention_mask")

        if self.training:
            train_required = (
                "memory_subtask_labels",
                "memory_state_labels",
                "memory_object_category_ids",
                "memory_object_instance_ids",
                "memory_subtask_class_count",
                "memory_state_class_count",
            )
            train_missing = [name for name in train_required if name not in batch]
            if train_missing:
                raise KeyError(
                    f"Memory-enabled MOST training batch is missing fields: {train_missing}"
                )
            subtask_labels = batch.pop("memory_subtask_labels")
            state_labels = batch.pop("memory_state_labels")
            category_ids = batch.pop("memory_object_category_ids")
            instance_ids = batch.pop("memory_object_instance_ids")
            subtask_class_count = int(batch.pop("memory_subtask_class_count").item())
            state_class_count = int(batch.pop("memory_state_class_count").item())
        else:
            subtask_labels = batch.pop("memory_subtask_labels", None)
            state_labels = batch.pop("memory_state_labels", None)
            category_ids = batch.pop("memory_object_category_ids", None)
            instance_ids = batch.pop("memory_object_instance_ids", None)
            raw_subtask_count = batch.pop("memory_subtask_class_count", None)
            raw_state_count = batch.pop("memory_state_class_count", None)
            subtask_class_count = (
                int(raw_subtask_count.item()) if raw_subtask_count is not None else None
            )
            state_class_count = (
                int(raw_state_count.item()) if raw_state_count is not None else None
            )
        batch.pop("memory_history_indices", None)
        batch.pop("memory_sequence_indices", None)
        frame_restore_indices = batch.pop("memory_frame_restore_indices", None)
        current_frame_indices = batch.pop("memory_current_frame_indices", None)

        history = self.memory_visual(
            frames,
            history_offsets,
            history_lengths,
            frame_restore_indices=frame_restore_indices,
            current_frame_indices=current_frame_indices,
        )
        if not self._memory_shape_logged:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            if rank == 0:
                logical_steps = int(history_lengths.sum().item())
                physical_steps = int(frames.shape[0])
                views = int(frames.shape[1])
                images = physical_steps * views
                current_images = int(history_lengths.numel()) * views
                saved_steps = max(0, logical_steps - physical_steps)
                print(
                    "[MOST+MemoryExpert] "
                    f"interface={self.memory_interface}, layers={self.memory_layers}, "
                    f"samples={history_lengths.numel()}, "
                    f"history_steps={logical_steps} "
                    f"(avg={history_lengths.float().mean().item():.1f}), "
                    f"physical_rgb_steps={physical_steps}, "
                    f"dedup_saved_steps={saved_steps}, "
                    f"dino_images={images}, "
                    f"past_no_grad_images={max(0, images - current_images)}, "
                    f"current_grad_images={current_images}, "
                    f"memory_rgb={tuple(frames.shape)}, "
                    f"dino_chunk={self.memory_dino_chunk_size}"
                )
            self._memory_shape_logged = True

        # Reuse MOST's frozen language embedding table for global task
        # conditioning; no second language model is introduced.
        with torch.no_grad():
            raw_task = self.vlm.model.get_input_embeddings()(task_input_ids)
        task_tokens, task_pad = self.memory_encoder.task_condition(
            raw_task.to(history.dtype), task_attention_mask
        )

        if self.memory_interface == "mot":
            h, memory_pad, _is_history, _is_future, memory_layer_kvs = (
                self.memory_encoder(
                    history,
                    history_lengths,
                    task_tokens,
                    task_pad,
                    return_action_kv=True,
                )
            )
            memory_tokens = None
        else:
            h, memory_pad, is_history, is_future = self.memory_encoder(
                history, history_lengths, task_tokens, task_pad
            )
            memory_layer_kvs = None
            action_memory = self.memory_action_projector(h)
            type_ids = torch.zeros_like(memory_pad, dtype=torch.long)
            type_ids[is_history] = 1
            type_ids[is_future] = 2
            action_memory = action_memory + self.memory_action_type_embedding(
                type_ids
            ).to(action_memory.dtype)
            action_memory = action_memory + self._signed_memory_time_embedding(
                history_lengths,
                action_memory.shape[1],
                action_memory.shape[2],
                action_memory.dtype,
                action_memory.device,
            )
            memory_tokens = action_memory.masked_fill(memory_pad[..., None], 0.0)

        if self.training:
            losses = self.memory_encoder.auxiliary_losses(
                h,
                subtask_labels=subtask_labels,
                state_labels=state_labels,
                object_category_ids=category_ids,
                object_instance_ids=instance_ids,
                subtask_class_count=subtask_class_count,
                state_class_count=state_class_count,
            )
        else:
            losses = {}
        return memory_tokens, memory_layer_kvs, memory_pad, losses

    def dit_forward(
        self,
        noisy_action,
        timestep,
        action_mask,
        state_embed,
        position_embeds,
        past_key_values,
        attn_mask,
        prefix_length,
        memory_tokens=None,
        memory_layer_kvs=None,
        memory_pad=None,
    ):
        timestep = self.t_projector(self.t_embedder(timestep[:, 0, 0] * 1000)).view(
            -1, 6, 1024
        )
        noisy_action = self.action_projector(noisy_action * action_mask)
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()
        hidden_states = self.dit(
            hidden_states,
            past_key_values,
            attn_mask,
            position_embeds,
            timestep,
            memory_tokens=memory_tokens,
            memory_layer_kvs=memory_layer_kvs,
            memory_pad=memory_pad,
        )
        output = self.action_output_layer(hidden_states[:, -noisy_action.shape[1] :])
        if prefix_length:
            output[:, :prefix_length] = 0.0
        return output

    @torch.no_grad()
    def _generate(self, noise, kwargs):
        sample = noise.clone()
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            timestep = (
                torch.ones(
                    (sample.shape[0], 1, 1), device=sample.device, dtype=sample.dtype
                )
                * step
                / self.num_steps
            )
            sample = sample + self.dit_forward(sample, timestep, **kwargs) * dt
        return sample

    @torch.no_grad()
    def generate(self, batch):
        return self.forward(batch, return_loss=False)

    @auto_cast
    def forward(self, batch, return_loss=False):
        segments = batch.pop("action_vlm_condition_segments", None)
        batch.pop("action_segments", None)
        action = batch.pop("action")
        action_mask = batch.pop("action_mask")
        state = batch.pop("state")
        memory_tokens, memory_layer_kvs, memory_pad, memory_losses = (
            self._memory_forward(batch)
        )
        if self.training:
            prefix_length = (
                random.randint(1, 6)
                if self.async_train and random.random() < 0.5
                else 0
            )
        else:
            prefix_length = batch.pop("prefix_length", 0)

        batch.pop("labels", None)
        if self.training:
            actual_lengths = batch.pop("vlm_action_actual_length")
            choice_target = batch.pop("vlm_action_target")
            choice_mask = batch.pop("vlm_action_mask")
            state_embeds = self.state_projector_choice(state.flatten(0, 1))
            vlm_outputs = self.vlm(
                **batch, state_embeds=state_embeds, use_cache=True, skip_logits=True
            )
            action_choice = self.action_projector_choice(
                vlm_outputs.hidden_states[
                    (batch["input_ids"] >= ACTION_START_ID)
                    & (batch["input_ids"] < ACTION_END_ID)
                ]
            )
            score_choice = self.score_projector_choice(
                vlm_outputs.hidden_states[batch["input_ids"] == SCORE_ID]
            )
        else:
            vlm_outputs = self.vlm(**batch, use_cache=True, skip_logits=True)

        past_key_values, position_ids, cache_mask = self._unpad(
            list(vlm_outputs.past_key_values),
            vlm_outputs.position_ids,
            vlm_outputs.attention_mask,
            segments,
        )
        batch_size, action_length, _ = action.shape
        state_length = state.shape[1]
        query_length = 1 + state_length + action_length
        prefix = action[:, :prefix_length]

        dit_position_ids = (
            torch.arange(query_length, device=action.device)
            .view(1, 1, -1)
            .repeat(3, batch_size, 1)
            + position_ids.max(dim=-1).values[..., None]
            + 1
        )
        if action_length > prefix_length:
            dit_position_ids[:, :, -(action_length - prefix_length) :] += 10
        cache_mask = cache_mask[:, None, :].expand(-1, query_length, -1)
        causal_mask = torch.tril(
            torch.ones(batch_size, query_length, query_length, device=action.device)
        )
        if self.training and prefix_length > 2:
            causal_mask = self._random_mask_prefix(
                causal_mask, prefix_length, state_length
            )
        attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None].bool()

        state_embed = self.state_projector(state)
        dit_position_ids = self._repeat(dit_position_ids, dim=1)
        action = self._repeat(action)
        action_mask = self._repeat(action_mask)
        prefix = self._repeat(prefix)
        state_embed = self._repeat(state_embed)
        attn_mask = self._repeat(attn_mask)
        position_embeds = self.rotary_emb(action, dit_position_ids)

        noise = torch.randn_like(action)
        kwargs = dict(
            action_mask=action_mask,
            state_embed=state_embed,
            position_embeds=position_embeds,
            past_key_values=past_key_values,
            attn_mask=attn_mask,
            prefix_length=prefix_length,
            memory_tokens=memory_tokens,
            memory_layer_kvs=memory_layer_kvs,
            memory_pad=memory_pad,
        )
        if self.training:
            timestep = (
                (1 - self.beta.sample((action.shape[0],)).to(action.device)) * 0.999
            ).to(action.dtype)
            timestep = timestep[:, None, None]
            noisy_action = (1 - timestep) * noise + timestep * action
            target = action - noise
            pred = self.dit_forward(
                torch.cat([prefix, noisy_action[:, prefix_length:]], dim=1),
                timestep,
                **kwargs,
            )[:, prefix_length:]
            target = target[:, prefix_length:]
            if prefix_length:
                prefix_pred = self._generate(
                    torch.cat([prefix, noise[:, prefix_length:]], dim=1), kwargs
                )
                weight = (
                    prefix_pred[:, prefix_length:] - action[:, prefix_length:]
                ).abs()
            else:
                weight = torch.ones_like(pred)
            action_mask = action_mask[:, prefix_length:]
        else:
            return self._generate(
                torch.cat([prefix, noise[:, prefix_length:]], dim=1), kwargs
            )

        if not return_loss:
            return pred

        loss_mse, loss_freq = self.compute_flow_loss(pred, target, action_mask, weight)
        loss_choice, loss_score = self.compute_choice_loss(
            action_choice, score_choice, choice_target, choice_mask, actual_lengths
        )
        loss = (
            1.0 * loss_mse
            + self.freq_coefficient * loss_freq
            + 0.5 * loss_choice
            + 0.5 * loss_score
        )
        if self.memory_enabled:
            loss = (
                loss
                + self.memory_subtask_loss_weight * memory_losses["loss_memory_subtask"]
                + self.memory_state_loss_weight * memory_losses["loss_memory_state"]
            )
        result = {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_freq": loss_freq,
            "loss_l1": loss_choice,
            "loss_score": loss_score,
        }
        if self.memory_enabled:
            result.update(memory_losses)
        return result

    def compute_flow_loss(self, pred, target, action_mask, weight):
        pred, target, weight = pred.float(), target.float(), weight.float()
        if not torch.any(action_mask):
            zero = (pred.sum() + target.sum()) * 0.0
            return zero, zero
        action_mask = action_mask.bool()
        with torch.no_grad():
            weight = weight.clone()
            weight[action_mask] /= weight[action_mask].mean()
            weight.clamp_(0.5, 5.0)
        loss_mse = (F.mse_loss(pred, target, reduction="none") * weight)[
            action_mask
        ].mean()
        freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
        valid_batch = action_mask[:, -1].any(dim=1)
        if not torch.any(valid_batch):
            return loss_mse, freq.sum() * 0.0
        freq_mask = action_mask[valid_batch, : freq.shape[1]].clone()
        dims = [dim for dim in self.freq_excluded_dims if dim < freq_mask.shape[-1]]
        freq_mask[:, :, dims] = False
        freq_weight = weight.mean(dim=(1, 2)).view(-1, 1, 1)
        loss_freq = (freq * freq_weight)[valid_batch][freq_mask].mean()
        return loss_mse, loss_freq

    def compute_choice_loss(
        self, action_pred, score_pred, target, mask, actual_lengths
    ):
        lengths = [int(length) for length in actual_lengths.detach().cpu().tolist()]
        if sum(lengths) != action_pred.shape[0] or len(lengths) != score_pred.shape[0]:
            raise ValueError("Choice token counts do not match targets")
        action_losses, score_losses = [], []
        start = 0
        for sample_index, length in enumerate(lengths):
            end = start + length
            predictions = (
                action_pred[start:end]
                .reshape(length, self.n_choices, -1)
                .transpose(0, 1)
            )
            sample_target = target[start:end].unsqueeze(0).repeat(self.n_choices, 1, 1)
            sample_mask = (
                mask[start:end].bool().unsqueeze(0).repeat(self.n_choices, 1, 1)
            )
            if not sample_mask.any():
                raise ValueError(
                    f"Choice target {sample_index} has no valid action values"
                )
            absolute_error = F.l1_loss(predictions, sample_target, reduction="none")
            choice_error = (
                absolute_error[sample_mask].reshape(self.n_choices, -1).mean(dim=-1)
            )
            choice_index = choice_error.argmin()
            action_losses.append(choice_error[choice_index])
            score_losses.append(
                ((score_pred[sample_index] - choice_error.detach()) ** 2).mean()
            )
            start = end
        return torch.stack(action_losses).mean(), torch.stack(score_losses).mean()
