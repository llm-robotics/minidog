"""Checkpoint save/load utilities for Stage 1 and Stage 2 training."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


def save_stage2_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 2 training checkpoint."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_stage2_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 2 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def load_stage2_weights_only(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
) -> None:
    """Load only model + EMA weights from a Stage 2 checkpoint.

    Skips optimizer/scheduler/epoch/step, for warm-starting a fine-tune with a fresh
    optimizer/scheduler/epoch-counter — e.g. when the new run's steps-per-epoch is too
    different from the checkpoint's original run for its saved scheduler state to make sense.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])


__all__ = [
    "save_stage1_checkpoint",
    "load_stage1_checkpoint",
    "save_stage2_checkpoint",
    "load_stage2_checkpoint",
    "load_stage2_weights_only",
]
