# Copyright (C) 2026 MOST Contributors.
from importlib import import_module
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from transformers.utils import logging

from most.models import MODEL_REGISTRY

logger = logging.get_logger(__name__)


@MODEL_REGISTRY.register_module()
class BaseRunner(LightningModule):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self._pretrained: Optional[str] = params.get("pretrained")
        self._model: Dict[str, Any] = params.get("model")
        self._optimizer: Dict[str, Any] = params.get("optimizer")
        self._scheduler: Optional[Dict[str, Any]] = params.get("scheduler")
        self._allowed_missing_pretrained_prefixes = tuple(
            params.get("allowed_missing_pretrained_prefixes", [])
        )
        self._allowed_missing_pretrained_substrings = tuple(
            params.get("allowed_missing_pretrained_substrings", [])
        )

    def configure_model(self) -> None:
        self.model = MODEL_REGISTRY.build(self._model)

        if self._pretrained:
            ckpt = torch.load(
                self._pretrained,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            info = self.load_state_dict(ckpt["module"], strict=False)
            allowed_missing = [
                key
                for key in info.missing_keys
                if key.startswith(self._allowed_missing_pretrained_prefixes)
                or any(
                    token in key
                    for token in self._allowed_missing_pretrained_substrings
                )
            ]
            disallowed_missing = [
                key for key in info.missing_keys if key not in allowed_missing
            ]
            if disallowed_missing or info.unexpected_keys:
                raise RuntimeError(
                    "Checkpoint does not match the configured model: "
                    f"missing_keys={disallowed_missing}, "
                    f"unexpected_keys={info.unexpected_keys}"
                )
            if allowed_missing:
                logger.warning(
                    "Checkpoint omits explicitly allowed post-training modules; "
                    "keeping their initialized weights: %s",
                    allowed_missing,
                )
            logger.info(
                "Loaded pretrained model from %s; allowed_missing=%d",
                self._pretrained,
                len(allowed_missing),
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.build_optimizer(self._optimizer, self.named_parameters())
        scheduler = self.build_scheduler(self._scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def build_optimizer(
        cfg: Dict[str, Any], parameters: Iterator[Tuple[str, torch.Tensor]]
    ) -> Optimizer:
        module_params = dict(cfg.get("params", {}))
        memory_lr_scale = float(module_params.pop("memory_lr_scale", 1.0))
        base_lr = float(module_params.get("lr", 1.0))
        parameters = list(parameters)

        no_decay = ["bias", "norm", "ln", "rotary_emb", "adaln"]

        def is_no_decay(name: str) -> bool:
            lower = name.lower()
            return any(token in lower for token in no_decay)

        def is_memory(name: str) -> bool:
            return (
                name.startswith("model.memory_")
                or ".memory_cross_attn." in name
                or ".memory_layernorm." in name
                or ".memory_gate" in name
            )

        groups = []
        for memory_group in (False, True):
            for no_decay_group in (False, True):
                params = [
                    p
                    for n, p in parameters
                    if p.requires_grad
                    and is_memory(n) == memory_group
                    and is_no_decay(n) == no_decay_group
                ]
                if not params:
                    continue
                group = {
                    "params": params,
                    "weight_decay": 0.0
                    if no_decay_group
                    else module_params.get("weight_decay", 0.1),
                }
                if memory_group and memory_lr_scale != 1.0:
                    group["lr"] = base_lr * memory_lr_scale
                groups.append(group)

        module_type = cfg["type"]
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        optimizer_class = getattr(module, class_name)
        return optimizer_class(groups, **module_params)

    @staticmethod
    def build_scheduler(
        cfg: Optional[Dict[str, Any]], optimizer: Optimizer
    ) -> Optional[LRScheduler]:
        if cfg is None:
            return None

        module_type = cfg["type"]
        module_params = cfg.get("params", {})
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        scheduler_class = getattr(module, class_name)

        return scheduler_class(optimizer, **module_params)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        self.log("train/token", batch["input_ids"].shape[1])
        loss_dict = self.model(batch, return_loss=True)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train/{loss_name}", loss_value.detach())
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return loss_dict
