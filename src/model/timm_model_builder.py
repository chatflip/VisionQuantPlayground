from typing import Any

import timm


def create_model(name: str, num_classes: int) -> Any:
    """timmのモデルを作成する

    Args:
        name (str): モデル名
        num_classes (int): クラス数

    Raises:
        ValueError: モデルが見つからない場合

    Returns:
        Any: モデル
    """
    available_models = timm.list_models()
    if name not in available_models:
        raise ValueError(f"Model {name} not found in available models")
    model = timm.create_model(
        name,
        pretrained=True,
        num_classes=num_classes,
    )
    return model
