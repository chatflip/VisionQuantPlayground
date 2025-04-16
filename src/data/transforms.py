import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_height: int,
    image_width: int,
    mean: list[float],
    std: list[float],
    seed: int,
) -> A.Compose:
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
    normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
    return A.Compose(
        [
            A.Resize(image_height, image_width),
            A.CenterCrop(image_height, image_width),
            normalize,
            ToTensorV2(),
        ]
    )
