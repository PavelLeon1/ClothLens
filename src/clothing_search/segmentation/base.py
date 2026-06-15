"""Backend-independent segmentation contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from clothing_search.segmentation.categories import ClothingCategory


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    mask: NDArray[np.integer]
    scores: dict[ClothingCategory, float]

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("Segmentation mask must be two-dimensional")

    def contains(self, category: ClothingCategory) -> bool:
        return bool(np.any(self.mask == int(category)))


class Segmenter(Protocol):
    def segment(self, image: Image.Image) -> SegmentationResult:
        """Return a semantic class mask for the supplied RGB image."""
