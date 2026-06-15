import numpy as np
import pytest

from clothing_search.segmentation.base import SegmentationResult
from clothing_search.segmentation.categories import (
    ClothingCategory,
    category_from_name,
    deepfashion2_target_id,
)


@pytest.mark.parametrize(
    ("source_id", "expected"),
    [
        (1, ClothingCategory.TOP),
        (4, ClothingCategory.OUTERWEAR),
        (9, ClothingCategory.BOTTOM),
        (13, ClothingCategory.DRESS),
    ],
)
def test_deepfashion_categories_collapse_to_application_classes(
    source_id: int,
    expected: ClothingCategory,
) -> None:
    assert deepfashion2_target_id(source_id) is expected


def test_unknown_deepfashion_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported DeepFashion2 category"):
        deepfashion2_target_id(14)


def test_category_name_is_case_insensitive() -> None:
    assert category_from_name("OuterWear") is ClothingCategory.OUTERWEAR


def test_unknown_category_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported clothing category"):
        category_from_name("cape")


def test_segmentation_result_requires_two_dimensional_mask() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        SegmentationResult(mask=np.zeros((1, 2, 3)), scores={})


def test_segmentation_result_reports_present_category() -> None:
    result = SegmentationResult(
        mask=np.array([[0, 1], [1, 0]], dtype=np.uint8),
        scores={ClothingCategory.TOP: 0.91},
    )

    assert result.contains(ClothingCategory.TOP)
    assert not result.contains(ClothingCategory.DRESS)
