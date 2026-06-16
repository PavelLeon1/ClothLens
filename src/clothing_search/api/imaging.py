"""Image serialization helpers for the web API."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from clothing_search.segmentation.categories import ClothingCategory

MASK_COLORS = np.array(
    [
        (0, 0, 0),
        (66, 135, 245),
        (72, 199, 116),
        (219, 83, 117),
        (245, 166, 35),
        (144, 97, 249),
        (38, 198, 218),
        (255, 238, 88),
    ],
    dtype=np.uint8,
)


def image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def mask_to_data_uri(mask: NDArray[np.integer]) -> str:
    mask_array = np.asarray(mask, dtype=np.int64)
    if mask_array.ndim != 2:
        raise ValueError("Segmentation mask must be two-dimensional")
    clipped = np.clip(mask_array, 0, int(ClothingCategory.ACCESSORIES))
    colored = MASK_COLORS[clipped]
    return image_to_data_uri(Image.fromarray(colored, mode="RGB"))
