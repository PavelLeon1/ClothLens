"""Albumentations factories for segmentation training and validation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class MaskLongTransform:
    transform: Any

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        transformed = self.transform(**kwargs)
        mask = transformed.get("mask")
        if hasattr(mask, "long"):
            transformed["mask"] = mask.long()
        return transformed


def _resolve_transform_dependencies(
    albumentations_module: Any | None,
    to_tensor_cls: Any | None,
) -> tuple[Any, Any]:
    if albumentations_module is None:
        try:
            albumentations_module = import_module("albumentations")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Albumentations is not installed. Install the ML dependencies "
                r'with venv\Scripts\python.exe -m pip install -e ".[ml]".'
            ) from error

    if to_tensor_cls is None:
        try:
            to_tensor_cls = import_module(
                "albumentations.pytorch"
            ).ToTensorV2
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Albumentations PyTorch transforms are unavailable. Install "
                r'the ML dependencies with venv\Scripts\python.exe -m pip '
                r'install -e ".[ml]".'
            ) from error
    return albumentations_module, to_tensor_cls


def build_train_transform(
    image_size: int = 512,
    *,
    albumentations_module: Any | None = None,
    to_tensor_cls: Any | None = None,
) -> Any:
    augmentations, tensor_transform = _resolve_transform_dependencies(
        albumentations_module,
        to_tensor_cls,
    )
    return MaskLongTransform(
        augmentations.Compose(
            [
                augmentations.Resize(height=image_size, width=image_size),
                augmentations.HorizontalFlip(p=0.5),
                augmentations.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    p=0.5,
                ),
                augmentations.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                tensor_transform(),
            ]
        )
    )


def build_validation_transform(
    image_size: int = 512,
    *,
    albumentations_module: Any | None = None,
    to_tensor_cls: Any | None = None,
) -> Any:
    augmentations, tensor_transform = _resolve_transform_dependencies(
        albumentations_module,
        to_tensor_cls,
    )
    return MaskLongTransform(
        augmentations.Compose(
            [
                augmentations.Resize(height=image_size, width=image_size),
                augmentations.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                tensor_transform(),
            ]
        )
    )
