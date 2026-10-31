# Copyright (C) 2026 MOST Contributors.
"""Pure RoboCasa-Long training data module and memory-aligned sampler."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Iterator

import torch
from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import DataLoader, DistributedSampler

from most.data.collate.robocasa_long_collate import RoboCasaLongCollate
from most.data.datasets.robocasa_long_dataset import RoboCasaLongDataset


class LongMemoryAlignedDistributedSampler(DistributedSampler):
    """Build local batches from a small number of memory-aligned episodes.

    Samples from one chunk share ``frame_index % memory_stride``. Their sparse
    histories are nested prefixes, which improves RGB decoding and DINO feature
    reuse without mixing RoboCasa-Long with another dataset.
    """

    def __init__(
        self,
        dataset: RoboCasaLongDataset,
        samples_per_batch: int,
        episodes_per_batch: int,
        memory_stride: int,
        seed: int = 42,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if samples_per_batch <= 0:
            raise ValueError("samples_per_batch must be positive")
        if episodes_per_batch <= 0:
            raise ValueError("episodes_per_batch must be positive")
        if episodes_per_batch > samples_per_batch:
            raise ValueError("episodes_per_batch cannot exceed samples_per_batch")
        if samples_per_batch % episodes_per_batch:
            raise ValueError(
                "samples_per_batch must be divisible by episodes_per_batch"
            )
        if memory_stride <= 0:
            raise ValueError("memory_stride must be positive")

        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=True,
        )
        self.samples_per_batch = int(samples_per_batch)
        self.episodes_per_batch = int(episodes_per_batch)
        self.samples_per_episode = self.samples_per_batch // self.episodes_per_batch
        self.memory_stride = int(memory_stride)
        self.episode_chunks = self._build_episode_chunks(dataset)

        generator = torch.Generator().manual_seed(self.seed)
        available_batches = len(self._make_batches(generator))
        batches_per_rank = available_batches // self.num_replicas
        if batches_per_rank == 0:
            raise ValueError(
                "RoboCasa-Long does not contain enough aligned batches for all ranks"
            )
        self.num_samples = batches_per_rank * self.samples_per_batch
        self.total_size = self.num_samples * self.num_replicas

    def _build_episode_chunks(
        self, dataset: RoboCasaLongDataset
    ) -> list[tuple[int, tuple[int, ...]]]:
        chunks: list[tuple[int, tuple[int, ...]]] = []
        episode_start = 0
        for episode_id, raw_end in enumerate(dataset.cumulative_frames.tolist()):
            episode_end = int(raw_end)
            episode_length = episode_end - episode_start
            for residue in range(min(self.memory_stride, episode_length)):
                lane = list(
                    range(
                        episode_start + residue,
                        episode_end,
                        self.memory_stride,
                    )
                )
                if len(lane) < self.samples_per_episode:
                    continue

                full_end = (
                    len(lane) // self.samples_per_episode
                ) * self.samples_per_episode
                lane_chunks = [
                    tuple(lane[offset : offset + self.samples_per_episode])
                    for offset in range(0, full_end, self.samples_per_episode)
                ]
                tail = tuple(lane[-self.samples_per_episode :])
                if not lane_chunks or tail != lane_chunks[-1]:
                    lane_chunks.append(tail)
                chunks.extend((episode_id, chunk) for chunk in lane_chunks)
            episode_start = episode_end

        if not chunks:
            raise ValueError(
                "No complete memory-aligned chunk was found. Reduce batch size, "
                "reduce LONG_EPISODES_PER_BATCH, or add longer episodes."
            )
        if len({episode_id for episode_id, _ in chunks}) < self.episodes_per_batch:
            raise ValueError("Not enough eligible episodes for the requested batch")
        return chunks

    def _make_batches(self, generator: torch.Generator) -> list[tuple[int, ...]]:
        order = torch.randperm(
            len(self.episode_chunks), generator=generator, device="cpu"
        ).tolist()
        pending = deque(self.episode_chunks[index] for index in order)
        batches: list[tuple[int, ...]] = []

        while pending:
            selected: list[tuple[int, tuple[int, ...]]] = []
            selected_episodes: set[int] = set()
            deferred: list[tuple[int, tuple[int, ...]]] = []
            for _ in range(len(pending)):
                episode_id, chunk = pending.popleft()
                if episode_id in selected_episodes:
                    deferred.append((episode_id, chunk))
                    continue
                selected.append((episode_id, chunk))
                selected_episodes.add(episode_id)
                if len(selected) == self.episodes_per_batch:
                    break
            pending.extend(deferred)
            if len(selected) < self.episodes_per_batch:
                break

            batch: list[int] = []
            for _, chunk in selected:
                reverse = bool(torch.randint(0, 2, (1,), generator=generator).item())
                batch.extend(reversed(chunk) if reverse else chunk)
            batches.append(tuple(batch))
        return batches

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        batches = self._make_batches(generator)
        usable = (len(batches) // self.num_replicas) * self.num_replicas
        rank_batches = batches[:usable][self.rank :: self.num_replicas]
        for batch in rank_batches:
            yield from batch


@DATASETS.register_module()
class RoboCasaLongDataModule(LightningDataModule):
    """Train exclusively on selected RoboCasa-Long tasks."""

    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params = params
        data = params.train_datasets
        self.batch_size = int(data.get("batch_size", 16))
        self.num_workers = int(data.get("num_workers", 8))
        self.prefetch_factor = int(data.get("prefetch_factor", 2))
        self.persistent_workers = bool(
            data.get("persistent_workers", self.num_workers > 0)
        )
        self.pin_memory = bool(data.get("pin_memory", True))
        self.seed = int(data.get("seed", 42))
        self.memory_stride = int(data.get("memory_stride", 16))
        self.episodes_per_batch = int(data.get("long_episodes_per_batch", 2))
        self.long_tasks = [str(task) for task in data.get("long_tasks", ())]
        if not self.long_tasks:
            raise ValueError("long_tasks cannot be empty")
        self.collate_fn = RoboCasaLongCollate(params)
        self.train_set: RoboCasaLongDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit") or self.train_set is not None:
            return
        dataset_params = deepcopy(self.params)
        data = dataset_params.train_datasets
        data["dataset_type"] = "robocasa365"
        data["task_group"] = "long"
        data["tasks"] = self.long_tasks
        data["episode_ids_file"] = None
        data["expected_episodes_per_task"] = None
        self.train_set = RoboCasaLongDataset(dataset_params)

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")
        if self.train_set is None:
            raise RuntimeError("RoboCasa-Long dataset was not initialized")

        sampler = LongMemoryAlignedDistributedSampler(
            self.train_set,
            samples_per_batch=self.batch_size,
            episodes_per_batch=self.episodes_per_batch,
            memory_stride=self.memory_stride,
            seed=self.seed,
        )
        loader_options = {
            "dataset": self.train_set,
            "batch_size": self.batch_size,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "collate_fn": self.collate_fn,
            "persistent_workers": (
                self.persistent_workers if self.num_workers > 0 else False
            ),
            "pin_memory": self.pin_memory,
            "drop_last": True,
        }
        if self.num_workers > 0:
            loader_options["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**loader_options)
