# -*- coding: utf-8 -*-
import json
import os
from logging import getLogger
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = getLogger(__name__)


class Food101Dataset(Dataset):
    """Food101データセットを扱うためのデータセットクラス。

    このクラスは、Food101データセットの画像とそのラベルを読み込み、
    データセットとして提供します。

    Attributes:
        transform (A.Compose | None): 画像に適用する変換処理
        image_paths (list[str]): 画像ファイルへのパスのリスト
        image_labels (list[int]): 画像のラベルのリスト
    """

    def __init__(
        self, root: str, phase: str, transform: A.Compose | None = None
    ) -> None:
        """Food101Datasetの初期化を行う。

        Args:
            root (str): データセットのルートディレクトリへのパス
            phase (str): データセットのフェーズ（'train' または 'test'）
            transform (A.Compose | None): 画像に適用する変換処理
        """
        self.transform = transform
        self.image_paths: list[str] = []
        self.image_labels: list[int] = []
        class_names: np.ndarray = np.genfromtxt(
            os.path.join(root, "meta", "classes.txt"), dtype=str
        )
        with open(os.path.join(root, "meta", f"{phase}.json")) as f:
            filenames: dict[str, list[str]] = json.load(f)

        for class_index, class_name in enumerate(class_names):
            image_paths = filenames[class_name]
            num_image = len(image_paths)
            fullpaths = [
                os.path.join(f"{root}/images/{image_path}.jpg")
                for image_path in image_paths
            ]
            self.image_paths.extend(fullpaths)
            self.image_labels.extend([class_index] * num_image)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, int]:
        """指定されたインデックスの画像とラベルを取得する。

        Args:
            index (int): 取得する画像のインデックス

        Returns:
            tuple[torch.FloatTensor, int]: 変換処理が適用された画像とそのラベルのタプル
                第1要素は画像データ（shape: [C, H, W]）、
                第2要素はラベルのインデックス。
        """
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented: dict[str, Any] = {"image": image}
        if self.transform is not None:
            augmented = self.transform(image=image)
        return augmented["image"], self.image_labels[index]

    def __len__(self) -> int:
        """データセットの総サンプル数を返す。

        Returns:
            int: データセットに含まれる画像の総数
        """
        return len(self.image_paths)
