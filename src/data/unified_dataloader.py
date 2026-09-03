"""Unified dataloader interface for MiniDog (dog-breed T2I) training."""
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


@dataclass
class DataloaderResult:
    """
    Unified result from prepare_unified_dataloader.
    Provides consistent interface for map-style and iterable datasets.
    """
    loader: Union[DataLoader, wds.WebLoader]
    sampler: Optional[DistributedSampler]
    dataset_size: int
    is_iterable: bool = False
    _wds_pipeline: Optional[object] = field(default=None, repr=False)
    _batch_size: int = 1
    _num_workers: int = 4
    _world_size: int = 1
    virtual_epoch_steps: Optional[int] = None

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling. Works for both dataset types.

        For map-style: calls sampler.set_epoch()
        For WebDataset: recreates pipeline with new seed (uses virtual_epoch_steps if set)
        """
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        elif self._wds_pipeline is not None:
            self._recreate_wds_loader(epoch)

    def _recreate_wds_loader(self, epoch: int):
        """Recreate WebDataset loader for new epoch."""
        dataset = self._wds_pipeline.create_pipeline(epoch=epoch)
        steps = self.virtual_epoch_steps or (self.dataset_size // (self._batch_size * self._world_size))
        loader = wds.WebLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self.loader = loader.with_epoch(steps)

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        if self.virtual_epoch_steps is not None:
            return self.virtual_epoch_steps
        if self.is_iterable:
            return self.dataset_size // (self._batch_size * self._world_size)
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)


def prepare_unified_dataloader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose] = None,
    condition_type: str = "text",
    shuffle: bool = True,
    virtual_epoch_steps: Optional[int] = None,
) -> DataloaderResult:
    """
    Dataloader factory for the dog-breed WebDataset shards.

    Args:
        config: Dataset configuration dict with structure:
            {
                'target': 'dogs',
                'type': 'wds' | 'latents',   # raw image shards or precomputed latents
                'data_dir': str,
                'shuffle_buffer': int,
                'seed': int,
            }
        image_size: Target image resolution (ignored for precomputed latents)
        batch_size: Per-GPU batch size
        num_workers: Number of dataloader workers
        rank: Distributed training rank (unused; WebDataset splits shards by node)
        world_size: Total number of GPUs
        transform: Optional custom transform (raw image shards only)
        condition_type: Kept for interface compatibility; the dog data is text-conditioned
        shuffle: Whether to shuffle data (False for eval)
        virtual_epoch_steps: Optional fixed number of steps per epoch

    Returns:
        DataloaderResult with unified interface
    """
    target = config.get("target", "dogs")
    if target != "dogs":
        raise ValueError(f"Unsupported dataset target {target!r}; MiniDog only supports 'dogs'.")

    if config.get("type", "wds") == "latents":
        result = _prepare_dogs_latents_loader(
            config, batch_size, num_workers, world_size, shuffle
        )
    else:
        result = _prepare_dogs_loader(
            config, image_size, batch_size, num_workers, world_size, transform, shuffle
        )
    result.virtual_epoch_steps = virtual_epoch_steps
    return result


def _prepare_dogs_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int,
    transform: Optional[transforms.Compose],
    shuffle: bool = True,
) -> DataloaderResult:
    """Prepare Dogs WebDataset loader."""
    from .dogs_wds_dataset import DogsWebDataset

    data_dir = config.get("data_dir", "./data/dogs_wds")
    shuffle_buffer = config.get("shuffle_buffer", 5000)
    seed = config.get("seed", 42)

    wds_pipeline = DogsWebDataset(
        data_dir=data_dir,
        transform=transform,
        image_size=image_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    dataset = wds_pipeline.create_pipeline(epoch=0, shuffle=shuffle)
    total_samples = wds_pipeline.estimated_size
    steps = total_samples // (batch_size * world_size)

    # For eval (shuffle=False) skip node splitting so all shards are visible on each rank
    actual_num_workers = num_workers if shuffle else 0
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=actual_num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if actual_num_workers > 0 else None,
    )
    if shuffle:
        loader = loader.with_epoch(steps)

    return DataloaderResult(
        loader=loader,
        sampler=None,
        dataset_size=total_samples,
        is_iterable=True,
        _wds_pipeline=wds_pipeline,
        _batch_size=batch_size,
        _num_workers=num_workers,
        _world_size=world_size,
    )


def _prepare_dogs_latents_loader(
    config: dict,
    batch_size: int,
    num_workers: int,
    world_size: int,
    shuffle: bool = True,
) -> DataloaderResult:
    """Prepare Dogs pre-computed latents WebDataset loader."""
    from .dogs_latents_wds_dataset import DogsLatentsWebDataset

    data_dir = config.get("data_dir", "./data/dogs_latents_wds")
    shuffle_buffer = config.get("shuffle_buffer", 5000)
    seed = config.get("seed", 42)

    wds_pipeline = DogsLatentsWebDataset(
        data_dir=data_dir,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    dataset = wds_pipeline.create_pipeline(epoch=0, shuffle=shuffle)
    total_samples = wds_pipeline.estimated_size
    steps = total_samples // (batch_size * world_size)

    actual_num_workers = num_workers if shuffle else 0
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=actual_num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if actual_num_workers > 0 else None,
    )
    if shuffle:
        loader = loader.with_epoch(steps)

    return DataloaderResult(
        loader=loader,
        sampler=None,
        dataset_size=total_samples,
        is_iterable=True,
        _wds_pipeline=wds_pipeline,
        _batch_size=batch_size,
        _num_workers=num_workers,
        _world_size=world_size,
    )

