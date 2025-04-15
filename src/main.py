# -*- coding: utf-8 -*-
import os
import time
from logging import getLogger

import albumentations as A
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet

from datasets import Food101Dataset
from MlflowWriter import MlflowExperimentManager
from mobilenet_v2 import mobilenet_v2
from resnet import resnet50, resnet101
from train_val import train, validate
from utils import seed_everything, seed_worker

logger = getLogger(__name__)


def load_data(cfg):
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
    )
    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(
                size=(cfg.arch.crop_size, cfg.arch.crop_size),
            ),
            A.HorizontalFlip(),
            normalize,
            ToTensorV2(),
        ],
        seed=cfg.seed,
    )

    val_transform = A.Compose(
        [
            A.Resize(cfg.arch.image_size, cfg.arch.image_size),
            A.CenterCrop(cfg.arch.crop_size, cfg.arch.crop_size),
            normalize,
            ToTensorV2(),
        ]
    )

    # AnimeFaceの学習用データ設定
    train_dataset = Food101Dataset(
        os.path.join(cfg.dataset_root),
        "train",
        transform=train_transform,
    )

    # Food101の評価用データ設定
    val_dataset = Food101Dataset(cfg.dataset_root, "test", transform=val_transform)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.arch.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.arch.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader


@hydra.main(config_path="./../config", config_name="config", version_base="1.3")
def main(cfg):
    logger.info(cfg)
    seed_everything(cfg.seed)

    cfg.ckpt_root.mkdir(parents=True, exist_ok=True)
    ckp_path = cfg.ckpt_root / f"{cfg.exp_name}_checkpoint.pth"
    weight_path = cfg.ckpt_root / f"{cfg.exp_name}_{cfg.arch.name}_best.pth"

    mlflow_manager = MlflowExperimentManager(cfg.exp_name)
    mlflow_manager.log_param_from_omegaconf_dict(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = load_data(cfg)

    if cfg.arch.name == "mobilenet_v2":
        model = mobilenet_v2(pretrained=True, num_classes=cfg.num_classes)
    elif cfg.arch.name == "resnet50":
        model = resnet50(pretrained=True)
        in_channels = model.fc.in_features
        model.fc = nn.Linear(in_channels, cfg.num_classes)
    elif cfg.arch.name == "resnet101":
        model = resnet101(pretrained=True)
        in_channels = model.fc.in_features
        model.fc = nn.Linear(in_channels, cfg.num_classes)
    elif "efficientnet" in cfg.arch.name:
        model = EfficientNet.from_pretrained(cfg.arch.name)
        in_channels = model._fc.in_features
        model._fc = nn.Linear(in_channels, cfg.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.arch.max_lr,
    )

    iteration = 0  # 反復回数保存用

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader),
        T_mult=1,
        eta_min=cfg.arch.min_lr,
        last_epoch=-1,
    )  # 学習率の軽減スケジュール

    best_acc = 0.0

    # 学習と評価
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        train(
            cfg,
            model,
            device,
            train_loader,
            mlflow_manager,
            criterion,
            optimizer,
            scheduler,
            epoch,
            iteration,
            cfg.apex,
        )
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す
        acc = validate(
            cfg, model, device, val_loader, criterion, mlflow_manager, iteration
        )
        scheduler.step()  # 学習率のスケジューリング更新
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            logger.info("Acc@1 best: {:6.2f}%".format(best_acc))

            torch.save(model.cpu().state_dict(), weight_path)
            mlflow_manager.log_artifact(weight_path)

            checkpoint = {
                "model": model.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "cfg": cfg,
            }
            torch.save(checkpoint, ckp_path)
            mlflow_manager.log_artifact(ckp_path)

            model.to(device)


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    interval = end_time - start_time
    hours = int(interval // 3600)
    minutes = int((interval % 3600) // 60)
    seconds = int((interval % 3600) % 60)
    logger.info(f"elapsed time: {hours}h {minutes}m {seconds}s")
