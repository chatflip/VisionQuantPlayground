import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 1234) -> None:
    """乱数シードを設定し、再現性を確保する

    Args:
        seed (int, optional): 乱数シード. Defaults to 1234.

    Note:
        - Pythonの標準乱数生成器
        - NumPyの乱数生成器
        - PyTorchの乱数生成器（CPUとGPU）
        - CuDNNの動作設定
        - CUDAの動作設定
        の全てにシードを設定します。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:
    """DataLoaderのワーカープロセスのシードを設定する

    Args:
        worker_id (int): ワーカープロセスのID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
