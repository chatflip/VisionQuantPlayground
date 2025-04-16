import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.datasets import Food101Dataset
from data.transforms import get_train_transforms, get_val_transforms
from utils.reproducibility import seed_worker


def get_train_dataloader(
    cfg: Any,
    image_height: int,
    image_width: int,
    mean: list[float],
    std: list[float],
) -> DataLoader:
    transform = get_train_transforms(image_height, image_width, mean, std, cfg.seed)
    dataset = Food101Dataset(
        os.path.join(cfg.dataset_root),
        "train",
        transform=transform,
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.arch.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return data_loader


def get_val_dataloader(
    cfg: Any,
    image_height: int,
    image_width: int,
    mean: list[float],
    std: list[float],
) -> DataLoader:
    transform = get_val_transforms(image_height, image_width, mean, std)
    dataset = Food101Dataset(cfg.dataset_root, "test", transform=transform)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.arch.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return data_loader
