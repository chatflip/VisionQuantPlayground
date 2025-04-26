import time
from logging import getLogger

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from data.dataloader import get_train_dataloader, get_val_dataloader
from model.timm_model_builder import create_model
from monitoring.MlflowExperimentManager import MlflowExperimentManager
from training.trainer import train, validate
from utils.reproducibility import seed_everything
from utils.torch_compile import get_device_compute_capability

logger = getLogger(__name__)


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

    model = create_model(cfg.arch.timm_name, cfg.num_classes)
    model.to(device)

    cc = get_device_compute_capability()
    if cc is None:
        pass
    elif cc >= 8.0:
        model = torch.compile(model)
    else:
        model = torch.compile(model, backend="aot_eager")

    _, image_height, image_width = model.default_cfg["input_size"]
    mean = model.default_cfg["mean"]
    std = model.default_cfg["std"]

    train_loader = get_train_dataloader(cfg, image_height, image_width, mean, std)
    val_loader = get_val_dataloader(cfg, image_height, image_width, mean, std)

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
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            logger.info("Acc@1 best: {:6.2f}%".format(best_acc))

            torch.save(model._orig_mod.cpu().state_dict(), weight_path)
            mlflow_manager.log_artifact(weight_path)

            checkpoint = {
                "model": model._orig_mod.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),  # type: ignore[no-untyped-call]
                "epoch": epoch,
                "best_acc": best_acc,
                "cfg": cfg,
            }
            torch.save(checkpoint, ckp_path)
            mlflow_manager.log_artifact(ckp_path)

            torch.save(model._orig_mod.cpu(), model_path)
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
