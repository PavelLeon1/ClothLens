import numpy as np
import pytest
from PIL import Image

from clothing_search.segmentation.categories import ClothingCategory
from clothing_search.segmentation.crop import (
    CategoryNotFoundError,
    bounding_box_from_mask,
    crop_category,
)


def test_bounding_box_uses_exclusive_right_and_bottom_coordinates() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:6, 3:8] = ClothingCategory.TOP

    box = bounding_box_from_mask(
        mask,
        ClothingCategory.TOP,
        padding_ratio=0,
    )

    assert box == (3, 2, 8, 6)


def test_bounding_box_adds_bounded_padding() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:12, 1:11] = ClothingCategory.DRESS

    box = bounding_box_from_mask(
        mask,
        ClothingCategory.DRESS,
        padding_ratio=0.2,
    )

    assert box == (0, 0, 13, 14)


def test_crop_category_uses_original_image_pixels() -> None:
    pixels = np.zeros((6, 8, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(8, dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:5, 2:7] = ClothingCategory.BOTTOM

    crop = crop_category(
        image,
        mask,
        ClothingCategory.BOTTOM,
        padding_ratio=0,
    )

    assert crop.image.size == (5, 4)
    assert crop.box == (2, 1, 7, 5)
    assert np.asarray(crop.image)[0, 0, 0] == 2


def test_crop_rejects_mask_with_different_size() -> None:
    image = Image.new("RGB", (8, 6))
    mask = np.zeros((3, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="match image size"):
        crop_category(image, mask, ClothingCategory.TOP)


def test_missing_category_raises_domain_error() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(CategoryNotFoundError, match="dress"):
        bounding_box_from_mask(mask, ClothingCategory.DRESS)
