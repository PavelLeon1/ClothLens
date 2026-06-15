"""Framework-independent segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class IoUResult:
    per_class: tuple[float | None, ...]
    mean: float


def intersection_over_union(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    num_classes: int,
) -> IoUResult:
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    if prediction_array.shape != target_array.shape:
        raise ValueError("Prediction and target must have the same shape")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")

    per_class: list[float | None] = []
    for class_id in range(num_classes):
        predicted = prediction_array == class_id
        expected = target_array == class_id
        union = np.logical_or(predicted, expected).sum()
        if union == 0:
            per_class.append(None)
            continue
        intersection = np.logical_and(predicted, expected).sum()
        per_class.append(float(intersection / union))

    present_scores = [score for score in per_class if score is not None]
    mean = float(np.mean(present_scores)) if present_scores else 0.0
    return IoUResult(per_class=tuple(per_class), mean=mean)
