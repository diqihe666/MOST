import numpy as np
import pytest

from most.data.datamodule.robocasa_long_datamodule import (
    LongMemoryAlignedDistributedSampler,
)


class _FakeLongDataset:
    def __init__(self, cumulative_frames):
        self.cumulative_frames = np.asarray(cumulative_frames, dtype=np.int64)

    def __len__(self):
        return int(self.cumulative_frames[-1])


@pytest.mark.parametrize("episodes_per_batch", [1, 2, 4])
def test_memory_aligned_sampler_controls_episodes_per_batch(episodes_per_batch):
    dataset = _FakeLongDataset([64, 128, 192, 256])
    sampler = LongMemoryAlignedDistributedSampler(
        dataset,
        samples_per_batch=8,
        episodes_per_batch=episodes_per_batch,
        memory_stride=4,
        seed=7,
        num_replicas=1,
        rank=0,
    )

    indices = list(sampler)

    assert len(indices) == len(sampler) == 256
    for offset in range(0, len(indices), 8):
        batch = indices[offset : offset + 8]
        episode_ids = {index // 64 for index in batch}
        assert len(episode_ids) == episodes_per_batch
        for episode_id in episode_ids:
            local = [
                index - episode_id * 64 for index in batch if index // 64 == episode_id
            ]
            assert len(local) == 8 // episodes_per_batch
            assert len({index % 4 for index in local}) == 1


def test_memory_aligned_sampler_keeps_episode_tail():
    dataset = _FakeLongDataset([19])
    sampler = LongMemoryAlignedDistributedSampler(
        dataset,
        samples_per_batch=3,
        episodes_per_batch=1,
        memory_stride=4,
        seed=7,
        num_replicas=1,
        rank=0,
    )

    assert any(18 in chunk for _, chunk in sampler.episode_chunks)


def test_batch_must_divide_evenly_across_episodes():
    dataset = _FakeLongDataset([64, 128])
    with pytest.raises(ValueError, match="divisible"):
        LongMemoryAlignedDistributedSampler(
            dataset,
            samples_per_batch=8,
            episodes_per_batch=3,
            memory_stride=4,
            num_replicas=1,
            rank=0,
        )
