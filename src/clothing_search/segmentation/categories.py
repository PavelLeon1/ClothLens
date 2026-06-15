"""Application clothing categories and dataset mappings."""

from enum import IntEnum


class ClothingCategory(IntEnum):
    BACKGROUND = 0
    TOP = 1
    BOTTOM = 2
    DRESS = 3
    OUTERWEAR = 4
    SHOES = 5
    BAG = 6
    ACCESSORIES = 7


DEEPFASHION2_CATEGORY_MAP = {
    1: ClothingCategory.TOP,
    2: ClothingCategory.TOP,
    3: ClothingCategory.OUTERWEAR,
    4: ClothingCategory.OUTERWEAR,
    5: ClothingCategory.TOP,
    6: ClothingCategory.TOP,
    7: ClothingCategory.BOTTOM,
    8: ClothingCategory.BOTTOM,
    9: ClothingCategory.BOTTOM,
    10: ClothingCategory.DRESS,
    11: ClothingCategory.DRESS,
    12: ClothingCategory.DRESS,
    13: ClothingCategory.DRESS,
}


def category_from_name(name: str) -> ClothingCategory:
    normalized_name = name.strip().upper()
    try:
        return ClothingCategory[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unsupported clothing category: {name}") from error


def deepfashion2_target_id(source_id: int) -> ClothingCategory:
    try:
        return DEEPFASHION2_CATEGORY_MAP[source_id]
    except KeyError as error:
        raise ValueError(
            f"Unsupported DeepFashion2 category: {source_id}"
        ) from error
