"""Extract category regions from original images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from clothing_search.segmentation.categories import ClothingCategory

BoundingBox = tuple[int, int, int, int]


class CategoryNotFoundError(LookupError):
    """Raised when a requested clothing category is absent from a mask."""


@dataclass(frozen=True, slots=True)
class CategoryCrop:
    image: Image.Image
    box: BoundingBox
    category: ClothingCategory


def bounding_box_from_mask(
    mask: NDArray[np.integer],
    category: ClothingCategory,
    *,
    padding_ratio: float = 0.1,
) -> BoundingBox:
    if mask.ndim != 2:
        raise ValueError("Segmentation mask must be two-dimensional")
    if padding_ratio < 0:
        raise ValueError("padding_ratio must not be negative")

    rows, columns = np.where(mask == int(category))
    if rows.size == 0:
        raise CategoryNotFoundError(
            f"Clothing category '{category.name.lower()}' was not found"
        )

    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    width = right - left
    height = bottom - top
    horizontal_padding = round(width * padding_ratio)
    vertical_padding = round(height * padding_ratio)

    image_height, image_width = mask.shape
    return (
        max(0, left - horizontal_padding),
        max(0, top - vertical_padding),
        min(image_width, right + horizontal_padding),
        min(image_height, bottom + vertical_padding),
    )


def crop_category(
    image: Image.Image,
    mask: NDArray[np.integer],
    category: ClothingCategory,
    *,
    padding_ratio: float = 0.1,
) -> CategoryCrop:
    if mask.shape != (image.height, image.width):
        raise ValueError("Segmentation mask dimensions must match image size")

    box = bounding_box_from_mask(
        mask,
        category,
        padding_ratio=padding_ratio,
    )
    return CategoryCrop(
        image=image.convert("RGB").crop(box),
        box=box,
        category=category,
    )
