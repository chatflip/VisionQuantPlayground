from logging import getLogger

import torch

logger = getLogger(__name__)


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)
) -> list[float]:
    """指定されたk値に対する上位k個の予測の精度を計算する

    Args:
        output (torch.Tensor): モデルの出力テンソル
        target (torch.Tensor): 正解ラベルのテンソル
        topk (tuple[int, ...], optional): 計算する上位k個の値. Defaults to (1,).

    Returns:
        list[float]: 各k値に対する精度のリスト（パーセンテージ）
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res: list[float] = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(float(correct_k * (100.0 / batch_size)))
        return res
