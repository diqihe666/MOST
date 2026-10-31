# Copyright (C) 2026 MOST Contributors.
"""Joint MemLearn modules for MOST.

This is the single-stage variant used by MOST + MemLearn:

* MOST/Qwen3-VL keeps its original short-horizon 4-frame visual pathway.
* Long-horizon RGB is encoded by a separate, lightweight trainable DINOv2 ViT-S/14
  at 224x224, exactly in the spirit of the original MemLearn implementation.
* Three per-view CLS tokens are fused with the original learnable-query ViewFusion.
* A depth-aligned block-causal Memory Expert builds [history | future] tokens.
* In the formal MoT variant, each of its 36 layers emits an action-compatible
  K/V cache for the corresponding MOST Action Expert layer.
* The final memory representation receives explicit subtask/object-state
  supervision, while action gradients flow through all layer-wise MoT caches.

The large MOST visual encoder is therefore never rerun on the long history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DinoHistoryEncoder(nn.Module):
    """Trainable DINOv2 ViT-S/14 CLS encoder for packed memory RGB."""

    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str | None = None,
        model_name: str = "dinov2_vits14",
        output_dim: int = 384,
        chunk_size: int = 128,
        use_checkpoint: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.chunk_size = max(1, int(chunk_size))
        self.use_checkpoint = bool(use_checkpoint)
        self.freeze = bool(freeze)

        repo = Path(repo_path).expanduser()
        ckpt = Path(checkpoint_path).expanduser() if checkpoint_path else None
        if not repo.is_dir():
            raise FileNotFoundError(
                f"DINOv2 repo not found: {repo}. Set MOST_MEM_DINO_REPO."
            )
        if ckpt is not None and not ckpt.is_file():
            raise FileNotFoundError(
                f"DINOv2 checkpoint not found: {ckpt}. Set MOST_MEM_DINO_CKPT."
            )

        self.dino = torch.hub.load(
            str(repo), model_name, source="local", pretrained=False
        )
        # Training initializes DINO from its standalone checkpoint. Exported HF
        # models set checkpoint_path=None: the DINO parameters are then restored
        # directly from the exported model safetensors together with MOST.
        if ckpt is not None:
            state_dict = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            if (
                isinstance(state_dict, dict)
                and "model" in state_dict
                and isinstance(state_dict["model"], dict)
            ):
                state_dict = state_dict["model"]
            info = self.dino.load_state_dict(state_dict, strict=False)
            if info.missing_keys or info.unexpected_keys:
                raise RuntimeError(
                    "DINO checkpoint does not exactly match the configured backbone: "
                    f"missing={info.missing_keys}, unexpected={info.unexpected_keys}"
                )

        for name, parameter in self.dino.named_parameters():
            if "mask_token" in name:
                parameter.requires_grad_(False)
        if self.freeze:
            for parameter in self.dino.parameters():
                parameter.requires_grad_(False)
            self.dino.eval()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.dino.eval()
        return self

    def _forward_features(self, frames: torch.Tensor) -> torch.Tensor:
        return self.dino.forward_features(frames)["x_norm_clstoken"]

    def _encode_chunk(self, frames: torch.Tensor) -> torch.Tensor:
        # Dataset keeps memory RGB uint8. Convert only at the encoder boundary.
        if frames.dtype == torch.uint8:
            param_dtype = next(self.dino.parameters()).dtype
            frames = frames.to(dtype=param_dtype).div_(255.0)
        elif not frames.is_floating_point():
            frames = frames.float()
        else:
            frames = frames.to(dtype=next(self.dino.parameters()).dtype)

        mean = self.mean.to(device=frames.device, dtype=frames.dtype)
        std = self.std.to(device=frames.device, dtype=frames.dtype)
        frames = (frames - mean) / std

        if self.freeze:
            with torch.no_grad():
                return self._forward_features(frames)
        # Checkpointing only helps when this branch actually participates in
        # autograd. Historical frames are intentionally encoded under no_grad,
        # so wrapping those forwards in checkpoint() only adds Python/recompute
        # bookkeeping without saving useful activation memory.
        if torch.is_grad_enabled() and self.training and self.use_checkpoint:
            return checkpoint(self._forward_features, frames, use_reentrant=False)
        return self._forward_features(frames)

    def encode_flat(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode [N,C,224,224] -> [N,384]."""
        if frames.ndim != 4:
            raise ValueError(
                f"memory frames must be [N,C,H,W], got {tuple(frames.shape)}"
            )
        if frames.shape[0] == 0:
            return torch.empty(
                0,
                self.output_dim,
                device=frames.device,
                dtype=next(self.dino.parameters()).dtype,
            )
        outputs = []
        for start in range(0, frames.shape[0], self.chunk_size):
            outputs.append(self._encode_chunk(frames[start : start + self.chunk_size]))
        output = torch.cat(outputs, dim=0)
        if output.shape[-1] != self.output_dim:
            raise ValueError(
                f"DINO output dim is {output.shape[-1]}, configured memory dim is {self.output_dim}"
            )
        return output


class ViewFusion(nn.Module):
    """Original MemLearn learned-query fusion over per-view DINO CLS tokens."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("memory dim must be divisible by fusion heads")
        self.fusion_query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.fusion_query, std=0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self, view_tokens: torch.Tensor, view_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if view_tokens.ndim != 4:
            raise ValueError(
                f"view_tokens must be [B,T,V,D], got {tuple(view_tokens.shape)}"
            )
        batch, timesteps, views, dim = view_tokens.shape
        kv = self.norm_kv(view_tokens.reshape(batch * timesteps, views, dim))
        query = self.norm_q(self.fusion_query.expand(batch * timesteps, 1, dim))
        q = (
            self.q_proj(query)
            .view(batch * timesteps, 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(kv)
            .view(batch * timesteps, views, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(kv)
            .view(batch * timesteps, views, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if view_mask is not None:
            pad = (~view_mask.bool()).reshape(batch * timesteps, views)
            attn = attn.masked_fill(pad[:, None, None, :], torch.finfo(attn.dtype).min)
        weights = F.softmax(attn.float(), dim=-1).to(attn.dtype).nan_to_num(0.0)
        out = (weights @ v).transpose(1, 2).reshape(batch * timesteps, 1, dim)
        out = self.out_proj(out).squeeze(1)
        return self.norm_out(out).reshape(batch, timesteps, dim)


class PackedMemLearnVisual(nn.Module):
    """DINO + 3-view fusion for packed variable-length decision histories."""

    def __init__(
        self,
        dim: int = 384,
        num_views: int = 3,
        fusion_heads: int = 8,
        dino_repo: str = "",
        dino_checkpoint: str = "",
        dino_model: str = "dinov2_vits14",
        dino_chunk_size: int = 512,
        dino_checkpointing: bool = True,
        freeze_dino: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_views = int(num_views)
        self.vit = DinoHistoryEncoder(
            repo_path=dino_repo,
            checkpoint_path=dino_checkpoint,
            model_name=dino_model,
            output_dim=dim,
            chunk_size=dino_chunk_size,
            use_checkpoint=dino_checkpointing,
            freeze=freeze_dino,
        )
        self.view_fusion = ViewFusion(dim, num_heads=fusion_heads)

    def forward(
        self,
        packed_frames: torch.Tensor,
        history_offsets: torch.Tensor,
        history_lengths: torch.Tensor,
        frame_restore_indices: Optional[torch.Tensor] = None,
        current_frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
          packed_frames: [sum(H_i), V, C, 224, 224] uint8
          history_offsets: [B+1], offsets into packed_frames
          history_lengths: [B]
        Returns:
          padded fused observation tokens [B,max(H_i),D]
        """
        if frame_restore_indices is None:
            return self._forward_rank5_standard(
                packed_frames, history_offsets, history_lengths
            )
        return self._forward_rank5_dedup(
            packed_frames,
            history_offsets,
            history_lengths,
            frame_restore_indices=frame_restore_indices,
            current_frame_indices=current_frame_indices,
        )

    def _forward_rank5_standard(
        self,
        packed_frames: torch.Tensor,
        history_offsets: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if packed_frames.ndim != 5:
            raise ValueError(
                f"packed memory frames must be [N,V,C,H,W], got {tuple(packed_frames.shape)}"
            )
        total, views, channels, height, width = packed_frames.shape
        if views != self.num_views:
            raise ValueError(f"Expected {self.num_views} memory views, got {views}")
        history_offsets = history_offsets.to(packed_frames.device, dtype=torch.long)
        history_lengths = history_lengths.to(packed_frames.device, dtype=torch.long)
        batch = int(history_lengths.numel())
        if history_offsets.numel() != batch + 1:
            raise ValueError("memory_history_offsets must have shape [B+1]")
        if int(history_offsets[-1].item()) != total:
            raise ValueError(
                f"memory packed frame mismatch: offsets end at {int(history_offsets[-1])}, N={total}"
            )
        if bool((history_lengths <= 0).any()):
            raise ValueError("Every memory sample must have at least one history frame")

        # Only the newest memory timestep of each sample updates DINO.  Older
        # history is context: it still goes through the *current* DINO weights,
        # but its visual backbone forward is stop-gradient.  This preserves
        # direct H supervision of the visual encoder over training (every random
        # sample contributes its current frame), while avoiding long-horizon
        # visual BPTT over all historical frames.
        current_indices = history_offsets[1:] - 1  # [B], one current timestep/sample
        is_current = torch.zeros(total, dtype=torch.bool, device=packed_frames.device)
        is_current[current_indices] = True
        past_indices = torch.nonzero(~is_current, as_tuple=False).flatten()

        encoded_parts = []
        timestep_parts = []
        if past_indices.numel() > 0:
            past_frames = packed_frames.index_select(0, past_indices)
            past_flat = past_frames.reshape(-1, channels, height, width)
            with torch.no_grad():
                past_cls = self.vit.encode_flat(past_flat)
            encoded_parts.append(past_cls.view(-1, views, self.dim))
            timestep_parts.append(past_indices)

        current_frames = packed_frames.index_select(0, current_indices)
        current_flat = current_frames.reshape(-1, channels, height, width)
        current_cls = self.vit.encode_flat(current_flat)
        encoded_parts.append(current_cls.view(batch, views, self.dim))
        timestep_parts.append(current_indices)

        # Reconstruct the original packed timestep order without an in-place
        # scatter, so gradients from current_cls remain connected to DINO.
        packed_cls = torch.cat(encoded_parts, dim=0)
        packed_indices = torch.cat(timestep_parts, dim=0)
        restore_order = torch.argsort(packed_indices)
        cls = packed_cls.index_select(0, restore_order)

        # ViewFusion remains trainable for both detached historical CLS tokens
        # and gradient-carrying current CLS tokens.
        fused = self.view_fusion(cls.unsqueeze(1)).squeeze(1)  # [N,D]

        max_history = int(history_lengths.max().item())
        output = fused.new_zeros(batch, max_history, self.dim)
        # Small B loop only scatters already-computed fused tokens.
        for index in range(batch):
            start = int(history_offsets[index].item())
            end = int(history_offsets[index + 1].item())
            length = int(history_lengths[index].item())
            if end - start != length:
                raise ValueError("memory history offsets/lengths disagree")
            output[index, :length] = fused[start:end]
        return output

    def _forward_rank5_dedup(
        self,
        packed_frames: torch.Tensor,
        history_offsets: torch.Tensor,
        history_lengths: torch.Tensor,
        frame_restore_indices: Optional[torch.Tensor] = None,
        current_frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if packed_frames.ndim != 5:
            raise ValueError(
                f"packed memory frames must be [N,V,C,H,W], got {tuple(packed_frames.shape)}"
            )
        physical_total, views, channels, height, width = packed_frames.shape
        if views != self.num_views:
            raise ValueError(f"Expected {self.num_views} memory views, got {views}")
        history_offsets = history_offsets.to(packed_frames.device, dtype=torch.long)
        history_lengths = history_lengths.to(packed_frames.device, dtype=torch.long)
        batch = int(history_lengths.numel())
        if history_offsets.numel() != batch + 1:
            raise ValueError("memory_history_offsets must have shape [B+1]")
        logical_total = int(history_offsets[-1].item())
        if bool((history_lengths <= 0).any()):
            raise ValueError("Every memory sample must have at least one history frame")

        # Standard RoboCasa365 batches keep one RGB timestep per logical memory
        # timestep.  RoboCasa-Long may compact repeated *past* timesteps across
        # samples.  The restore map expands compact visual features back to the
        # original logical sequence before ViewFusion/Memory Expert, so token
        # density and train/eval temporal semantics are unchanged.
        if frame_restore_indices is None:
            if logical_total != physical_total:
                raise ValueError(
                    f"memory packed frame mismatch: offsets end at {logical_total}, "
                    f"N={physical_total}"
                )
            frame_restore_indices = torch.arange(
                physical_total, device=packed_frames.device, dtype=torch.long
            )
            physical_current_indices = history_offsets[1:] - 1
        else:
            frame_restore_indices = frame_restore_indices.to(
                packed_frames.device, dtype=torch.long
            )
            if frame_restore_indices.numel() != logical_total:
                raise ValueError(
                    "memory_frame_restore_indices must have one entry per logical "
                    f"history timestep: got {frame_restore_indices.numel()}, "
                    f"expected {logical_total}"
                )
            if frame_restore_indices.numel() and (
                int(frame_restore_indices.min().item()) < 0
                or int(frame_restore_indices.max().item()) >= physical_total
            ):
                raise ValueError(
                    "memory_frame_restore_indices points outside memory_frames"
                )
            if current_frame_indices is None:
                raise ValueError(
                    "Compacted memory_frames require memory_current_frame_indices"
                )
            physical_current_indices = current_frame_indices.to(
                packed_frames.device, dtype=torch.long
            )
            if physical_current_indices.numel() != batch:
                raise ValueError(
                    "memory_current_frame_indices must contain one entry per sample"
                )

        if physical_current_indices.numel() and (
            int(physical_current_indices.min().item()) < 0
            or int(physical_current_indices.max().item()) >= physical_total
        ):
            raise ValueError(
                "memory_current_frame_indices points outside memory_frames"
            )
        if torch.unique(physical_current_indices).numel() != batch:
            raise ValueError(
                "Current memory timesteps must occupy distinct physical RGB slots"
            )

        # Only the newest memory timestep of each sample updates DINO. Older
        # physical history slots are context and are encoded under no_grad.
        is_current = torch.zeros(
            physical_total, dtype=torch.bool, device=packed_frames.device
        )
        is_current[physical_current_indices] = True
        past_indices = torch.nonzero(~is_current, as_tuple=False).flatten()

        encoded_parts = []
        timestep_parts = []
        if past_indices.numel() > 0:
            past_frames = packed_frames.index_select(0, past_indices)
            past_flat = past_frames.reshape(-1, channels, height, width)
            with torch.no_grad():
                past_cls = self.vit.encode_flat(past_flat)
            encoded_parts.append(past_cls.view(-1, views, self.dim))
            timestep_parts.append(past_indices)

        current_frames = packed_frames.index_select(0, physical_current_indices)
        current_flat = current_frames.reshape(-1, channels, height, width)
        current_cls = self.vit.encode_flat(current_flat)
        encoded_parts.append(current_cls.view(batch, views, self.dim))
        timestep_parts.append(physical_current_indices)

        # Reconstruct compact physical timestep order without an in-place
        # scatter, so gradients from current_cls remain connected to DINO.
        packed_cls = torch.cat(encoded_parts, dim=0)
        packed_indices = torch.cat(timestep_parts, dim=0)
        physical_restore_order = torch.argsort(packed_indices)
        physical_cls = packed_cls.index_select(0, physical_restore_order)

        # Expand physical visual features back to the full logical history.
        # Repeated past entries intentionally share the same detached DINO CLS;
        # ViewFusion and the Memory Expert still run on every logical token.
        cls = physical_cls.index_select(0, frame_restore_indices)

        # ViewFusion remains trainable for both detached historical CLS tokens
        # and gradient-carrying current CLS tokens.
        fused = self.view_fusion(cls.unsqueeze(1)).squeeze(1)  # [N,D]

        max_history = int(history_lengths.max().item())
        output = fused.new_zeros(batch, max_history, self.dim)
        # Small B loop only scatters already-computed fused tokens.
        for index in range(batch):
            start = int(history_offsets[index].item())
            end = int(history_offsets[index + 1].item())
            length = int(history_lengths[index].item())
            if end - start != length:
                raise ValueError("memory history offsets/lengths disagree")
            output[index, :length] = fused[start:end]
        return output


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_len: int = 4096) -> None:
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._build(max_len)

    def _build(self, length: int) -> None:
        t = torch.arange(length, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freq = torch.outer(t, self.inv_freq)
        emb = torch.cat([freq, freq], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_pos = int(positions.max().item()) + 1 if positions.numel() else 0
        if max_pos > self.cos.shape[0]:
            self._build(max_pos)
        return self.cos.index_select(0, positions), self.sin.index_select(0, positions)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None].to(query)
    sin = sin[None, None].to(query)
    return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin


class MemorySelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_len=max_seq_len)

    def _project_qkv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = self.rope(positions)
        query, key = _apply_rope(query, key, cos, sin)
        return query, key, value

    def forward(self, x: torch.Tensor, forbidden: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        positions = torch.arange(length, device=x.device)
        query, key, value = self._project_qkv(x, positions)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(forbidden[:, None], torch.finfo(logits.dtype).min)
        weights = F.softmax(logits.float(), dim=-1).to(logits.dtype).nan_to_num(0.0)
        out = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, dim)
        return self.out(out)

    def forward_cached(
        self,
        x: torch.Tensor,
        cache_key: Optional[torch.Tensor],
        cache_value: Optional[torch.Tensor],
        cache_len: int,
        forbidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attend local tokens to committed block K/V plus the revisable local block.

        Only completed history blocks are supplied through ``cache_key/value``.
        The current history block and all future query slots are projected and
        recomputed on every call, preserving full attention inside the active block.
        """
        batch, length, dim = x.shape
        positions = torch.arange(cache_len, cache_len + length, device=x.device)
        query, local_key, local_value = self._project_qkv(x, positions)
        if cache_key is None:
            key = local_key
            value = local_value
        else:
            key = torch.cat([cache_key, local_key], dim=2)
            value = torch.cat([cache_value, local_value], dim=2)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(forbidden[:, None], torch.finfo(logits.dtype).min)
        weights = F.softmax(logits.float(), dim=-1).to(logits.dtype).nan_to_num(0.0)
        out = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, dim)
        return self.out(out), local_key.detach(), local_value.detach()


class MemoryTaskCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        task_tokens: Optional[torch.Tensor],
        task_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if task_tokens is None or task_tokens.shape[1] == 0:
            return torch.zeros_like(x)
        batch, q_len, dim = x.shape
        kv_len = task_tokens.shape[1]
        q = (
            self.q_proj(x)
            .view(batch, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(task_tokens)
            .view(batch, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(task_tokens)
            .view(batch, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if task_pad is not None:
            logits = logits.masked_fill(
                task_pad[:, None, None, :], torch.finfo(logits.dtype).min
            )
        weights = F.softmax(logits.float(), dim=-1).to(logits.dtype).nan_to_num(0.0)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, q_len, dim)
        return self.out(out)


class MemoryTransformerBlock(nn.Module):
    """One block of the long-horizon Memory Expert.

    Besides updating the compact 384-D memory stream, a block can expose an
    action-compatible K/V view of its *depth-aligned* output.  In MoT mode this
    gives Action Expert layer ``l`` the K/V produced from Memory Expert layer
    ``l`` without first collapsing the entire memory tower to one final H.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        action_output_dim: Optional[int] = None,
        action_heads: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MemorySelfAttention(dim, num_heads)
        self.norm_task = nn.LayerNorm(dim)
        self.task_attn = MemoryTaskCrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim),
        )

        self.action_output_dim = (
            None if action_output_dim is None else int(action_output_dim)
        )
        self.action_heads = int(action_heads)
        if self.action_output_dim is not None:
            if self.action_output_dim % self.action_heads != 0:
                raise ValueError("action_output_dim must be divisible by action_heads")
            self.action_head_dim = self.action_output_dim // self.action_heads
            self.action_norm = nn.LayerNorm(dim)
            self.action_k_proj = nn.Linear(dim, self.action_output_dim, bias=False)
            self.action_v_proj = nn.Linear(dim, self.action_output_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        forbidden: torch.Tensor,
        task_tokens: Optional[torch.Tensor],
        task_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), forbidden)
        x = x + self.task_attn(self.norm_task(x), task_tokens, task_pad)
        x = x + self.ff(self.norm2(x))
        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        cache_key: Optional[torch.Tensor],
        cache_value: Optional[torch.Tensor],
        cache_len: int,
        forbidden: torch.Tensor,
        task_tokens: Optional[torch.Tensor],
        task_pad: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_attn, local_key, local_value = self.self_attn.forward_cached(
            self.norm1(x),
            cache_key=cache_key,
            cache_value=cache_value,
            cache_len=cache_len,
            forbidden=forbidden,
        )
        x = x + self_attn
        x = x + self.task_attn(self.norm_task(x), task_tokens, task_pad)
        x = x + self.ff(self.norm2(x))
        return x, local_key, local_value

    def project_action_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project this layer's memory representation to Action Expert K/V.

        Memory Expert stays compact (e.g. 384-D) while the returned cache is in
        the 1024-D / 8-head Action Expert space.  This is the layer-wise MoT
        interface; it is not the Memory Expert's own self-attention K/V cache.
        """
        if self.action_output_dim is None:
            raise RuntimeError(
                "This memory block was created without an action K/V head"
            )
        x = self.action_norm(x)
        batch, length, _ = x.shape
        key = (
            self.action_k_proj(x)
            .view(batch, length, self.action_heads, self.action_head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        value = (
            self.action_v_proj(x)
            .view(batch, length, self.action_heads, self.action_head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        return key, value


def build_block_future_mask(
    pad: torch.Tensor,
    history_lengths: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """True means forbidden. History is block-causal; future sees all valid tokens."""
    batch, length = pad.shape
    pos = torch.arange(length, device=pad.device)
    qi = pos.view(1, length, 1)
    kj = pos.view(1, 1, length)
    hist_len = history_lengths.view(batch, 1, 1)
    q_hist = qi < hist_len
    k_hist = kj < hist_len
    q_block = qi // block_size
    k_block = kj // block_size
    hist_allowed = k_hist & (k_block <= q_block)
    future_allowed = ~pad[:, None, :]
    allowed = torch.where(q_hist, hist_allowed, future_allowed)
    allowed = allowed & (~pad[:, :, None]) & (~pad[:, None, :])
    return ~allowed


@dataclass
class MemoryTransformerKVCache:
    """Persistent cache for completed block-causal history blocks.

    ``keys/values`` are the Memory Expert's own per-layer self-attention cache.
    ``action_keys/action_values`` are the corresponding depth-aligned K/V views
    consumed by the MOST Action Expert.  Only completed history blocks are ever
    committed; the current block and future query slots remain revisable.
    """

    keys: Optional[list[torch.Tensor]] = None
    values: Optional[list[torch.Tensor]] = None
    action_keys: Optional[list[torch.Tensor]] = None
    action_values: Optional[list[torch.Tensor]] = None
    h: Optional[torch.Tensor] = None
    length: int = 0

    def append(
        self,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        h: torch.Tensor,
        action_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        new_action_keys = None
        new_action_values = None
        if action_kvs is not None:
            new_action_keys = [item[0].detach() for item in action_kvs]
            new_action_values = [item[1].detach() for item in action_kvs]

        if self.keys is None:
            self.keys = [key.detach() for key in keys]
            self.values = [value.detach() for value in values]
            self.h = h.detach()
            self.action_keys = new_action_keys
            self.action_values = new_action_values
        else:
            self.keys = [
                torch.cat([old, new.detach()], dim=2)
                for old, new in zip(self.keys, keys)
            ]
            self.values = [
                torch.cat([old, new.detach()], dim=2)
                for old, new in zip(self.values, values)
            ]
            self.h = torch.cat([self.h, h.detach()], dim=1)
            if new_action_keys is not None:
                if self.action_keys is None or self.action_values is None:
                    raise RuntimeError(
                        "Action K/V cache appeared after earlier blocks had none"
                    )
                self.action_keys = [
                    torch.cat([old, new], dim=2)
                    for old, new in zip(self.action_keys, new_action_keys)
                ]
                self.action_values = [
                    torch.cat([old, new], dim=2)
                    for old, new in zip(self.action_values, new_action_values)
                ]
        self.length = int(self.h.shape[1])


@dataclass
class TemporalMemoryInferenceState:
    """Exact online state for block-causal Memory Expert inference.

    ``committed`` contains only completed blocks. ``active_history`` contains
    the current revisable block and is recomputed in full whenever it grows.
    """

    committed: MemoryTransformerKVCache
    active_history: Optional[torch.Tensor] = None


class TemporalMemoryEncoder(nn.Module):
    """Block-causal long-horizon Memory Expert.

    In the formal MoT variant this tower is depth-aligned with the 36-layer MOST
    Action Expert.  Each memory block produces an action-compatible K/V cache
    from its own representation, so Action layer ``l`` consumes Memory layer
    ``l`` rather than all action layers attending to one terminal H.

    Training uses the exact full block-causal graph.  Inference uses exact
    completed-block caching: only completed history blocks persist, while the
    active block and all future query slots are recomputed on every replan.
    """

    def __init__(
        self,
        dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        block_size: int = 8,
        future_steps: int = 16,
        task_input_dim: int = 2560,
        max_task_tokens: int = 128,
        num_subtask_classes: int = 1024,
        num_state_classes: int = 256,
        num_object_categories: int = 256,
        max_instance_slots: int = 32,
        action_output_dim: Optional[int] = None,
        action_heads: int = 8,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.block_size = int(block_size)
        self.future_steps = int(future_steps)
        self.max_task_tokens = int(max_task_tokens)
        self.num_subtask_classes = int(num_subtask_classes)
        self.num_state_classes = int(num_state_classes)
        self.num_object_categories = int(num_object_categories)
        self.max_instance_slots = int(max_instance_slots)
        self.action_output_dim = (
            None if action_output_dim is None else int(action_output_dim)
        )
        self.action_heads = int(action_heads)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.history_type = nn.Parameter(torch.empty(1, 1, dim))
        self.future_type = nn.Parameter(torch.empty(1, 1, dim))
        self.future_query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.history_type, std=0.02)
        nn.init.normal_(self.future_type, std=0.02)
        nn.init.normal_(self.future_query, std=0.02)

        self.task_proj = nn.Linear(task_input_dim, dim, bias=False)
        self.task_pos = nn.Embedding(max_task_tokens, dim)
        self.blocks = nn.ModuleList(
            [
                MemoryTransformerBlock(
                    dim,
                    num_heads,
                    action_output_dim=self.action_output_dim,
                    action_heads=self.action_heads,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

        self.subtask_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, num_subtask_classes)
        )
        category_dim = 128
        instance_dim = 64
        object_dim = 256
        self.category_embedding = nn.Embedding(num_object_categories, category_dim)
        self.instance_embedding = nn.Embedding(max_instance_slots, instance_dim)
        self.object_projector = nn.Sequential(
            nn.Linear(category_dim + instance_dim, object_dim),
            nn.GELU(),
            nn.LayerNorm(object_dim),
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(dim + object_dim),
            nn.Linear(dim + object_dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_state_classes),
        )

    @property
    def produces_action_kv(self) -> bool:
        return self.action_output_dim is not None

    def task_condition(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if token_embeddings.shape[1] > self.max_task_tokens:
            token_embeddings = token_embeddings[:, : self.max_task_tokens]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_task_tokens]
        task = self.task_proj(token_embeddings)
        positions = torch.arange(task.shape[1], device=task.device)
        task = task + self.task_pos(positions)[None].to(task.dtype)
        task_pad = None if attention_mask is None else ~attention_mask.bool()
        return task, task_pad

    def _run_training_block(
        self,
        block: MemoryTransformerBlock,
        x: torch.Tensor,
        forbidden: torch.Tensor,
        task_tokens: torch.Tensor,
        task_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.training and torch.is_grad_enabled() and self.gradient_checkpointing:

            def block_forward(y: torch.Tensor) -> torch.Tensor:
                return block(y, forbidden, task_tokens, task_pad)

            return checkpoint(block_forward, x, use_reentrant=False)
        return block(x, forbidden, task_tokens, task_pad)

    def forward(
        self,
        history_tokens: torch.Tensor,
        history_lengths: torch.Tensor,
        task_tokens: torch.Tensor,
        task_pad: Optional[torch.Tensor],
        return_action_kv: bool = False,
    ):
        batch, max_history, dim = history_tokens.shape
        history_lengths = history_lengths.to(history_tokens.device, dtype=torch.long)
        seq_lengths = history_lengths + self.future_steps
        max_seq = int(seq_lengths.max().item())
        pos = torch.arange(max_seq, device=history_tokens.device)[None].expand(
            batch, -1
        )
        valid = pos < seq_lengths[:, None]
        is_history = valid & (pos < history_lengths[:, None])
        is_future = valid & (~is_history)
        pad = ~valid

        gather = pos.clamp(max=max(max_history - 1, 0))
        gathered = torch.gather(
            history_tokens, 1, gather[..., None].expand(-1, -1, dim)
        )
        history = gathered + self.history_type.to(gathered.dtype)
        future = self.future_query.expand(batch, max_seq, -1).to(gathered.dtype)
        future = future + self.future_type.to(gathered.dtype)
        x = torch.where(is_history[..., None], history, future)
        x = x * valid[..., None]

        forbidden = build_block_future_mask(pad, history_lengths, self.block_size)
        action_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            x = self._run_training_block(block, x, forbidden, task_tokens, task_pad)
            if return_action_kv:
                if not self.produces_action_kv:
                    raise RuntimeError(
                        "return_action_kv=True but action K/V heads are disabled"
                    )
                action_kvs.append(block.project_action_kv(x))
        x = self.norm(x)
        x = x * valid[..., None]
        result = (x, pad, is_history, is_future)
        if return_action_kv:
            return (*result, action_kvs)
        return result

    def init_inference_state(self) -> TemporalMemoryInferenceState:
        return TemporalMemoryInferenceState(committed=MemoryTransformerKVCache())

    def _cached_local_mask(
        self,
        local_is_history: torch.Tensor,
        cache_len: int,
    ) -> torch.Tensor:
        """Mask for [committed prefix | active block | future queries]."""
        batch, local_len = local_is_history.shape
        local_future = ~local_is_history
        local_forbidden = local_is_history[:, :, None] & local_future[:, None, :]
        if cache_len <= 0:
            return local_forbidden
        prefix = torch.zeros(
            batch,
            local_len,
            cache_len,
            dtype=torch.bool,
            device=local_is_history.device,
        )
        return torch.cat([prefix, local_forbidden], dim=-1)

    def _forward_cached_local(
        self,
        local_tokens: torch.Tensor,
        local_is_history: torch.Tensor,
        state: TemporalMemoryInferenceState,
        task_tokens: torch.Tensor,
        task_pad: Optional[torch.Tensor],
        return_local_kv: bool = False,
        return_action_kv: bool = False,
    ):
        cache = state.committed
        forbidden = self._cached_local_mask(local_is_history, cache.length)
        x = local_tokens
        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        action_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, block in enumerate(self.blocks):
            cache_key = None if cache.keys is None else cache.keys[layer_index]
            cache_value = None if cache.values is None else cache.values[layer_index]
            x, local_key, local_value = block.forward_cached(
                x,
                cache_key=cache_key,
                cache_value=cache_value,
                cache_len=cache.length,
                forbidden=forbidden,
                task_tokens=task_tokens,
                task_pad=task_pad,
            )
            new_keys.append(local_key)
            new_values.append(local_value)
            if return_action_kv:
                if not self.produces_action_kv:
                    raise RuntimeError("Action K/V heads are disabled")
                action_kvs.append(block.project_action_kv(x))
        x = self.norm(x)
        outputs = [x]
        if return_local_kv:
            outputs.extend([new_keys, new_values])
        if return_action_kv:
            outputs.append(action_kvs)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _commit_history_block(
        self,
        state: TemporalMemoryInferenceState,
        block_history: torch.Tensor,
        task_tokens: torch.Tensor,
        task_pad: Optional[torch.Tensor],
    ) -> None:
        """Commit one completed block; later blocks can never revise it."""
        local_tokens = block_history + self.history_type.to(block_history.dtype)
        local_is_history = torch.ones(
            block_history.shape[:2], dtype=torch.bool, device=block_history.device
        )
        if self.produces_action_kv:
            h, keys, values, action_kvs = self._forward_cached_local(
                local_tokens,
                local_is_history,
                state,
                task_tokens,
                task_pad,
                return_local_kv=True,
                return_action_kv=True,
            )
        else:
            h, keys, values = self._forward_cached_local(
                local_tokens,
                local_is_history,
                state,
                task_tokens,
                task_pad,
                return_local_kv=True,
                return_action_kv=False,
            )
            action_kvs = None
        state.committed.append(keys, values, h, action_kvs=action_kvs)

    def forward_incremental(
        self,
        state: TemporalMemoryInferenceState,
        new_history_tokens: Optional[torch.Tensor],
        task_tokens: torch.Tensor,
        task_pad: Optional[torch.Tensor],
        return_action_kv: bool = False,
    ):
        """Exact completed-block KV-cache inference.

        Persistent state contains only completed history blocks.  The active
        block is recomputed in full (block-internal attention is bidirectional),
        and the 16 future slots are always regenerated.  In MoT mode the method
        additionally returns one full [committed | active | future] Action K/V
        cache per Memory Expert layer.
        """
        if new_history_tokens is not None:
            if new_history_tokens.ndim == 2:
                new_history_tokens = new_history_tokens.unsqueeze(1)
            if new_history_tokens.ndim != 3:
                raise ValueError(
                    "new_history_tokens must be [B,N,D], got "
                    f"{tuple(new_history_tokens.shape)}"
                )
            if state.active_history is None:
                state.active_history = new_history_tokens
            else:
                state.active_history = torch.cat(
                    [state.active_history, new_history_tokens], dim=1
                )

        if state.active_history is None:
            raise RuntimeError("Memory inference state has no history token")

        # With exactly block_size active tokens the block remains revisable.  It
        # becomes immutable only once the first token of the next block arrives.
        while state.active_history.shape[1] > self.block_size:
            completed = state.active_history[:, : self.block_size]
            state.active_history = state.active_history[:, self.block_size :]
            self._commit_history_block(
                state, completed, task_tokens=task_tokens, task_pad=task_pad
            )

        batch, active_len, _ = state.active_history.shape
        history = state.active_history + self.history_type.to(
            state.active_history.dtype
        )
        future = self.future_query.expand(batch, self.future_steps, -1).to(
            history.dtype
        )
        future = future + self.future_type.to(history.dtype)
        local_tokens = torch.cat([history, future], dim=1)
        local_is_history = torch.zeros(
            batch,
            active_len + self.future_steps,
            dtype=torch.bool,
            device=history.device,
        )
        local_is_history[:, :active_len] = True

        if return_action_kv:
            h_local, local_action_kvs = self._forward_cached_local(
                local_tokens,
                local_is_history,
                state,
                task_tokens=task_tokens,
                task_pad=task_pad,
                return_action_kv=True,
            )
        else:
            h_local = self._forward_cached_local(
                local_tokens,
                local_is_history,
                state,
                task_tokens=task_tokens,
                task_pad=task_pad,
            )
            local_action_kvs = None

        if state.committed.h is None:
            h = h_local
        else:
            h = torch.cat([state.committed.h, h_local], dim=1)

        history_length = state.committed.length + active_len
        seq_len = history_length + self.future_steps
        pad = torch.zeros(batch, seq_len, dtype=torch.bool, device=h.device)
        is_history = torch.zeros_like(pad)
        is_history[:, :history_length] = True
        is_future = ~is_history
        result = (h, pad, is_history, is_future, history_length)

        if return_action_kv:
            if local_action_kvs is None:
                raise RuntimeError("Missing local Action K/V")
            full_action_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer_index, (local_k, local_v) in enumerate(local_action_kvs):
                if state.committed.action_keys is None:
                    full_action_kvs.append((local_k, local_v))
                else:
                    full_action_kvs.append(
                        (
                            torch.cat(
                                [state.committed.action_keys[layer_index], local_k],
                                dim=2,
                            ),
                            torch.cat(
                                [state.committed.action_values[layer_index], local_v],
                                dim=2,
                            ),
                        )
                    )
            return (*result, full_action_kvs)
        return result

    def auxiliary_losses(
        self,
        h: torch.Tensor,
        subtask_labels: Optional[torch.Tensor],
        state_labels: Optional[torch.Tensor],
        object_category_ids: Optional[torch.Tensor],
        object_instance_ids: Optional[torch.Tensor],
        subtask_class_count: Optional[int] = None,
        state_class_count: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        zero = h.sum() * 0.0
        subtask_loss = zero
        state_loss = zero
        valid_subtask = h.new_tensor(0.0)
        valid_state = h.new_tensor(0.0)

        if subtask_labels is not None:
            valid = subtask_labels >= 0
            if valid.any():
                logits = self.subtask_head(h[valid]).float()
                if subtask_class_count is not None:
                    logits = logits[:, : int(subtask_class_count)]
                subtask_loss = F.cross_entropy(logits, subtask_labels[valid].long())
                valid_subtask = valid.sum().to(h.dtype)
            else:
                subtask_loss = (
                    self.subtask_head(h.reshape(-1, h.shape[-1])[:1]).sum() * 0.0
                )

        if (
            state_labels is not None
            and object_category_ids is not None
            and object_instance_ids is not None
        ):
            valid = state_labels >= 0
            if valid.any():
                b_idx, s_idx, o_idx = torch.nonzero(valid, as_tuple=True)
                cat = object_category_ids[b_idx, o_idx].long()
                inst = object_instance_ids[b_idx, o_idx].long()
                if (cat < 0).any() or (cat >= self.num_object_categories).any():
                    raise ValueError(
                        "Object category index exceeds configured memory vocabulary"
                    )
                if (inst < 0).any() or (inst >= self.max_instance_slots).any():
                    raise ValueError(
                        "Object instance index exceeds configured memory slots"
                    )
                obj = torch.cat(
                    [self.category_embedding(cat), self.instance_embedding(inst)],
                    dim=-1,
                )
                obj = self.object_projector(obj)
                rows = torch.cat([h[b_idx, s_idx], obj.to(h.dtype)], dim=-1)
                logits = self.state_head(rows).float()
                if state_class_count is not None:
                    logits = logits[:, : int(state_class_count)]
                labels = state_labels[b_idx, s_idx, o_idx].long()
                state_loss = F.cross_entropy(logits, labels)
                valid_state = labels.numel() * h.new_ones(())
            else:
                dummy_cat = torch.zeros(1, dtype=torch.long, device=h.device)
                dummy_inst = torch.zeros(1, dtype=torch.long, device=h.device)
                obj = self.object_projector(
                    torch.cat(
                        [
                            self.category_embedding(dummy_cat),
                            self.instance_embedding(dummy_inst),
                        ],
                        dim=-1,
                    )
                )
                state_loss = (
                    self.state_head(
                        torch.cat(
                            [h.reshape(-1, h.shape[-1])[:1], obj.to(h.dtype)], dim=-1
                        )
                    ).sum()
                    * 0.0
                )

        return {
            "loss_memory_subtask": subtask_loss,
            "loss_memory_state": state_loss,
            "memory_valid_subtask": valid_subtask.detach(),
            "memory_valid_state": valid_state.detach(),
        }


class ActionMemoryCrossAttention(nn.Module):
    """Independent H cross-attention used after MOST's original mixed attention."""

    def __init__(self, dim: int = 1024, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _repeat_batch(
        x: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if x is None or x.shape[0] == batch_size:
            return x
        if batch_size % x.shape[0] != 0:
            raise ValueError(f"Cannot repeat memory batch {x.shape[0]} to {batch_size}")
        return x.repeat_interleave(batch_size // x.shape[0], dim=0)

    def project_memory_kv(
        self, memory: Optional[torch.Tensor]
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Project a fixed H sequence once for reuse across diffusion steps.

        The returned tensors intentionally keep the *memory* batch size.  If the
        action path expands the batch (for example by repeating candidates),
        ``forward_projected`` repeats the cached K/V lazily to match the query.
        """
        if memory is None or memory.shape[1] == 0:
            return None
        batch, kv_len, _ = memory.shape
        k = (
            self.k_proj(memory)
            .view(batch, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(memory)
            .view(batch, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        return k, v

    def forward_projected(
        self,
        query: torch.Tensor,
        memory_kv: Optional[tuple[torch.Tensor, torch.Tensor]],
        memory_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cross-attend to pre-projected memory K/V.

        This is mathematically identical to ``forward`` but avoids rerunning
        ``k_proj`` and ``v_proj`` on the same H tokens at every diffusion step.
        """
        if memory_kv is None:
            return torch.zeros_like(query)
        batch, q_len, _ = query.shape
        k, v = memory_kv
        k = self._repeat_batch(k, batch)
        v = self._repeat_batch(v, batch)
        memory_pad = self._repeat_batch(memory_pad, batch)
        q = (
            self.q_proj(query)
            .view(batch, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        allowed = None
        if memory_pad is not None:
            allowed = (~memory_pad.bool())[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(batch, q_len, self.dim)
        return self.out_proj(out)

    def forward(
        self,
        query: torch.Tensor,
        memory: Optional[torch.Tensor],
        memory_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.forward_projected(query, self.project_memory_kv(memory), memory_pad)


class ActionMemoryMoTAttention(nn.Module):
    """Action-side attention over layer-wise K/V emitted by Memory Expert.

    K/V projection lives inside the corresponding Memory Expert layer.  The
    Action Expert therefore computes only Q + attention + output projection,
    mirroring the way MOST consumes an already-built VLM cache.
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

    @staticmethod
    def _repeat_batch(
        x: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if x is None or x.shape[0] == batch_size:
            return x
        if batch_size % x.shape[0] != 0:
            raise ValueError(f"Cannot repeat memory batch {x.shape[0]} to {batch_size}")
        return x.repeat_interleave(batch_size // x.shape[0], dim=0)

    def forward(
        self,
        query: torch.Tensor,
        memory_kv: Optional[tuple[torch.Tensor, torch.Tensor]],
        memory_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if memory_kv is None:
            return torch.zeros_like(query)
        batch, q_len, _ = query.shape
        key, value = memory_kv
        key = self._repeat_batch(key, batch)
        value = self._repeat_batch(value, batch)
        memory_pad = self._repeat_batch(memory_pad, batch)
        if key.shape[1] != self.num_heads or key.shape[-1] != self.head_dim:
            raise ValueError(
                "Memory Expert K/V shape does not match Action Expert heads: "
                f"key={tuple(key.shape)}, expected heads={self.num_heads}, "
                f"head_dim={self.head_dim}"
            )
        q = (
            self.q_proj(query)
            .view(batch, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        allowed = None
        if memory_pad is not None:
            allowed = (~memory_pad.bool())[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, key, value, attn_mask=allowed, dropout_p=0.0
        )
        out = out.transpose(1, 2).contiguous().view(batch, q_len, self.dim)
        return self.out_proj(out)
