# -*- coding: utf-8 -*-
import os
import time
from logging import getLogger

import albumentations as A
import hydra
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets import Food101Dataset
from MlflowExperimentManager import MlflowExperimentManager
from reproducibility import seed_everything, seed_worker
from train_val import train, validate

logger = getLogger(__name__)


def load_data(
    cfg: DictConfig,
    image_height: int,
    image_width: int,
    mean: list[float],
    std: list[float],
) -> tuple[DataLoader, DataLoader]:
    """データローダーを初期化する

    Args:
        cfg: DictConfig - 設定パラメータ
        image_height: int - 画像の高さ
        image_width: int - 画像の幅
        mean: list[float] - 正規化のための平均値
        std: list[float] - 正規化のための標準偏差

    Returns:
        tuple[DataLoader, DataLoader] - 訓練用と検証用のデータローダー
    """
    normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_height, image_width),
            ),
            A.HorizontalFlip(),
            normalize,
            ToTensorV2(),
        ],
        seed=cfg.seed,
    )

    val_transform = A.Compose(
        [
            A.Resize(image_height, image_width),
            A.CenterCrop(image_height, image_width),
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
def main(cfg: DictConfig) -> None:
    """メイン実行関数

    学習の実行、モデルの保存、MLflowによる実験管理を行う

    Args:
        cfg: DictConfig - hydraによって読み込まれた設定パラメータ
    """
    logger.info(cfg)
    seed_everything(cfg.seed)

    cfg.ckpt_root.mkdir(parents=True, exist_ok=True)
    ckp_path = cfg.ckpt_root / f"{cfg.exp_name}_{cfg.arch.name}_checkpoint.pth"
    weight_path = cfg.ckpt_root / f"{cfg.exp_name}_{cfg.arch.name}_best.pth"
    model_path = cfg.ckpt_root / f"{cfg.exp_name}_{cfg.arch.name}.pt"

    mlflow_manager = MlflowExperimentManager(cfg.exp_name)
    mlflow_manager.log_param_from_omegaconf_dict(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available_models = timm.list_models()
    if cfg.arch.name not in available_models:
        raise ValueError(f"Model {cfg.arch.name} not found in available models")
    model = timm.create_model(
        cfg.arch.name,
        pretrained=True,
        num_classes=cfg.num_classes,
    )
    _, image_height, image_width = model.default_cfg["input_size"]
    mean = model.default_cfg["mean"]
    std = model.default_cfg["std"]

    model.to(device)

    train_loader, val_loader = load_data(cfg, image_height, image_width, mean, std)

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
        )
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す
        acc = validate(
            cfg, model, device, val_loader, criterion, mlflow_manager, iteration
        )
        scheduler.step()  # type: ignore[no-untyped-call]
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            logger.info("Acc@1 best: {:6.2f}%".format(best_acc))

            torch.save(model.cpu().state_dict(), weight_path)
            mlflow_manager.log_artifact(weight_path)

            checkpoint = {
                "model": model.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),  # type: ignore[no-untyped-call]
                "epoch": epoch,
                "best_acc": best_acc,
                "cfg": cfg,
            }
            torch.save(checkpoint, ckp_path)
            mlflow_manager.log_artifact(ckp_path)

            torch.save(model.cpu(), model_path)
            mlflow_manager.log_artifact(model_path)

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
