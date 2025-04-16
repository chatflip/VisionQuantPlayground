import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_height: int,
    image_width: int,
    mean: list[float],
    std: list[float],
    seed: int,
) -> A.Compose:
    """訓練用の画像変換を生成する。

    Args:
        image_height (int): 画像の高さ
        image_width (int): 画像の幅
        mean (list[float]): 正規化のための平均値
        std (list[float]): 正規化のための標準偏差
        seed (int): シード値

    Returns:
        A.Compose: 訓練用の画像変換
    """
    normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_height, image_width),
            ),
            A.HorizontalFlip(),
            normalize,
            ToTensorV2(),
        ],
        seed=seed,
    )


def get_val_transforms(
    image_height: int,
    image_width: int,
    mean: list[float],
    std: list[float],
) -> A.Compose:
    """検証用の画像変換を生成する。

    Args:
        image_height (int): 画像の高さ
        image_width (int): 画像の幅
        mean (list[float]): 正規化のための平均値
        std (list[float]): 正規化のための標準偏差

    Returns:
        A.Compose: 検証用の画像変換
    """
    normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
    return A.Compose(
        [
            A.Resize(image_height, image_width),
            A.CenterCrop(image_height, image_width),
            normalize,
            ToTensorV2(),
        ]
    )
