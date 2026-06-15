import numpy as np
import pytest

from clothing_search.segmentation.metrics import intersection_over_union


def test_iou_ignores_classes_absent_from_prediction_and_target() -> None:
    target = np.array([[0, 1], [1, 0]])
    prediction = np.array([[0, 1], [0, 0]])

    result = intersection_over_union(prediction, target, num_classes=8)

    assert result.per_class[0] == pytest.approx(2 / 3)
    assert result.per_class[1] == pytest.approx(1 / 2)
    assert result.per_class[2] is None
    assert result.mean == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_iou_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        intersection_over_union(
            np.zeros((2, 2)),
            np.zeros((3, 3)),
            num_classes=8,
        )
