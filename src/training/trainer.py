import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monitoring.metrics_tracker import AverageMeter, ProgressMeter
from monitoring.MlflowExperimentManager import MlflowExperimentManager
from training.metrics import accuracy


def train(
    cfg: Any,
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    mlflow_manager: MlflowExperimentManager,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    iteration: int,
) -> None:
    """モデルの学習を行う。

    Args:
        cfg (Any): 設定
        model (nn.Module): モデル
        device (torch.device): デバイス
        data_loader (DataLoader): データローダー
        mlflow_manager (MlflowExperimentManager): mlflowのマネージャー
        criterion (nn.Module): 損失関数
        optimizer (Optimizer): オプティマイザ
        scheduler (LRScheduler): 学習率スケジューラ
        epoch (int): エポック数
        iteration (int): 反復回数
    """
    # ProgressMeter, AverageMeterの値初期化
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.5f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # ネットワークを学習用に設定
    # ex.)dropout,batchnormを有効
    model.train()

    end = time.time()  # 1回目の読み込み時間計測用
    for i, (images, target) in enumerate(data_loader):
        data_time.update(time.time() - end)  # 画像のロード時間記録

        images = images.to(device, non_blocking=True)  # gpu使うなら画像をgpuに転送
        target = target.to(device, non_blocking=True)  # gpu使うならラベルをgpuに転送

        output = model(images)  # sofmaxm前まで出力(forward)
        loss = criterion(
            output, target
        )  # ネットワークの出力をsoftmax + ラベルとのloss計算

        # losss, accuracyを計算して更新
        acc1, acc5 = accuracy(
            output, target, topk=(1, 5)
        )  # 予測した中で1番目と3番目までに正解がある率
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        optimizer.zero_grad()  # 勾配初期化

        loss.backward()
        optimizer.step()  # パラメータ更新
        scheduler.step()
        batch_time.update(
            time.time() - end
        )  # 画像ロードからパラメータ更新にかかった時間記録
        end = time.time()  # 基準の時間更新

        # print_freqごとに進行具合とloss表示
        if i % cfg.print_freq == 0:
            progress.display(i)
        mlflow_manager.log_metric("lr", optimizer.param_groups[0]["lr"], iteration)
        mlflow_manager.log_metric("loss.train", loss.item(), iteration)
        mlflow_manager.log_metric("acc1.train", acc1, iteration)
        mlflow_manager.log_metric("acc5.train", acc5, iteration)
        mlflow_manager.log_metric("top1.train", top1.avg, iteration)
        mlflow_manager.log_metric("top5.train", top5.avg, iteration)
        iteration += 1


def validate(
    cfg: Any,
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    criterion: nn.Module,
    mlflow_manager: MlflowExperimentManager,
    iteration: int,
) -> float:
    """モデルの評価を行う。

    Args:
        cfg (Any): 設定
        model (nn.Module): モデル
        device (torch.device): デバイス
        data_loader (DataLoader): データローダー
        criterion (nn.Module): 損失関数
        mlflow_manager (MlflowExperimentManager): mlflowのマネージャー
        iteration (int): 反復回数

    Returns:
        float: 精度
    """
    # ProgressMeter, AverageMeterの値初期化
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.5f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Validate: ",
    )

    # ネットワークを評価用に設定
    # ex.)dropout,batchnormを恒等関数に
    model.eval()

    # 勾配計算しない(計算量低減)
    with torch.no_grad():
        end = time.time()  # 基準の時間更新
        for i, (images, target) in enumerate(data_loader):
            data_time.update(time.time() - end)  # 画像のロード時間記録

            images = images.to(device, non_blocking=True)  # gpu使うなら画像をgpuに転送
            target = target.to(
                device, non_blocking=True
            )  # gpu使うならラベルをgpuに転送
            output = model(
                images
            )  # sofmaxm前まで出力(forward)#評価データセットでのloss計算
            loss = criterion(output, target)  # sum up batch loss

            # losss, accuracyを計算して更新
            losses.update(loss.item(), images.size(0))
            acc1, acc5 = accuracy(
                output, target, topk=(1, 5)
            )  # ラベルと合ってる率を算出
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

            batch_time.update(
                time.time() - end
            )  # 画像ロードからパラメータ更新にかかった時間記録
            end = time.time()  # 基準の時間更新
            if i % cfg.print_freq == 0:
                progress.display(i)
            mlflow_manager.log_metric("loss.val", loss.item(), iteration)
            mlflow_manager.log_metric("acc1.val", acc1, iteration)
            mlflow_manager.log_metric("acc5.val", acc5, iteration)
            mlflow_manager.log_metric("top1.val", top1.avg, iteration)
            mlflow_manager.log_metric("top5.val", top5.avg, iteration)

    # 精度等格納
    progress.display(i + 1)
    return top1.avg
