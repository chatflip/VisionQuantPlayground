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
    # 基本的な幾何学変換（高確率で適用）
    geometric_transforms = [
        # まず短辺をtarget_sizeに合わせてリサイズ
        A.SmallestMaxSize(max_size=max(image_height, image_width), p=1.0),
        # その後ランダムクロップで指定サイズに
        A.RandomCrop(
            height=image_height,
            width=image_width,
            p=1.0,
        ),
        A.Affine(
            scale=(0.5, 2.0),
            translate_percent=(-0.05, 0.05),
            rotate=(-45, 45),
            shear=(-15, 15),
            p=0.7,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]

    # 画質・ノイズ関連の変換（中確率で適用）
    quality_transforms = [
        A.OneOf(
            [
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0,
                ),
            ],
            p=0.4,
        ),
    ]

    # 高度なデータ拡張（低確率で適用）
    advanced_transforms = [
        A.OneOf(
            [
                A.CoarseDropout(
                    num_holes_range=(5, 8),
                    p=1.0,
                ),
                A.GridDistortion(
                    distort_limit=0.2,
                    p=1.0,
                ),
                A.ElasticTransform(
                    alpha=120,
                    sigma=6.0,
                    p=1.0,
                ),
            ],
            p=0.3,
        ),
    ]

    # 最終的な変換パイプライン
    transforms = (
        geometric_transforms
        + quality_transforms
        + advanced_transforms
        + [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms, p=1.0, seed=seed)


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
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=max(image_height, image_width), p=1.0),
            A.CenterCrop(height=image_height, width=image_width, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
