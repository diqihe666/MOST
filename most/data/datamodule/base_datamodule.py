# Copyright (C) 2026 MOST Contributors.
"""RoboCasa365 Lightning data module."""

from copy import deepcopy

from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import DataLoader, DistributedSampler

from most.data.collate.custom_collate import CustomCollate
from most.data.datasets.robocasa365_dataset import RoboCasa365Dataset


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params = params
        data = params.train_datasets
        self.batch_size = int(data.get("batch_size", 16))
        self.num_workers = int(data.get("num_workers", 8))
        self.prefetch_factor = int(data.get("prefetch_factor", 4))
        self.persistent_workers = bool(
            data.get("persistent_workers", self.num_workers > 0)
        )
        self.pin_memory = bool(data.get("pin_memory", True))
        self.drop_last = bool(data.get("drop_last", True))
        self.seed = int(data.get("seed", 42))
        self.collate_fn = CustomCollate(params)
        self.train_set: RoboCasa365Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit") or self.train_set is not None:
            return
        self.train_set = RoboCasa365Dataset(deepcopy(self.params))

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")
        if self.train_set is None:
            raise RuntimeError("RoboCasa365 dataset was not initialized")

        sampler = DistributedSampler(self.train_set, shuffle=True, seed=self.seed)
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
            "drop_last": self.drop_last,
        }
        if self.num_workers > 0:
            loader_options["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**loader_options)
