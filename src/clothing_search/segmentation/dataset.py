"""DeepFashion2 dataset adapter and polygon mask rasterization."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from clothing_search.segmentation.categories import deepfashion2_target_id

try:
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:

    class TorchDataset:  # type: ignore[no-redef]
        """Fallback base that preserves map-style dataset behavior."""


Transform = Callable[..., Mapping[str, Any]]


def _iter_polygons(segmentation: object) -> Iterator[list[float]]:
    if not isinstance(segmentation, list) or not segmentation:
        return

    if all(isinstance(value, int | float) for value in segmentation):
        yield [float(value) for value in segmentation]
        return

    for polygon in segmentation:
        if isinstance(polygon, list):
            yield [float(value) for value in polygon]


def rasterize_deepfashion2_mask(
    annotation: Mapping[str, Any],
    image_size: tuple[int, int],
) -> NDArray[np.uint8]:
    mask_image = Image.new("L", image_size, color=0)
    draw = ImageDraw.Draw(mask_image)

    for key, item in annotation.items():
        if not key.startswith("item") or not isinstance(item, Mapping):
            continue

        category = deepfashion2_target_id(int(item["category_id"]))
        for polygon in _iter_polygons(item.get("segmentation")):
            if len(polygon) < 6 or len(polygon) % 2:
                continue
            points = list(zip(polygon[::2], polygon[1::2], strict=True))
            draw.polygon(points, fill=int(category))

    return np.asarray(mask_image, dtype=np.uint8)


def _default_tensor_transform(
    *,
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required for tensor conversion. Install the ML "
            r'dependencies with venv\Scripts\python.exe -m pip install -e ".[ml]".'
        ) from error

    image_tensor = (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div(255.0)
    )
    mask_tensor = torch.from_numpy(mask.copy()).to(dtype=torch.long)
    return {"image": image_tensor, "mask": mask_tensor}


class DeepFashion2Dataset(TorchDataset):
    """Map-style dataset for native DeepFashion2 per-image annotations."""

    SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        transform: Transform | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = "validation" if split == "val" else split
        self.image_dir = self.root / self.split / "image"
        self.annotation_dir = self.root / self.split / "annos"
        self.transform = transform or _default_tensor_transform

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"DeepFashion2 image directory does not exist: {self.image_dir}"
            )
        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(
                "DeepFashion2 annotation directory does not exist: "
                f"{self.annotation_dir}"
            )

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in self.SUPPORTED_SUFFIXES
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        image_path = self.image_paths[index]
        annotation_path = self.annotation_dir / f"{image_path.stem}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Annotation does not exist for {image_path.name}: "
                f"{annotation_path}"
            )

        with Image.open(image_path) as source_image:
            rgb_image = source_image.convert("RGB")
            image = np.asarray(rgb_image, dtype=np.uint8)
            image_size = rgb_image.size

        with annotation_path.open(encoding="utf-8") as annotation_file:
            annotation = json.load(annotation_file)
        mask = rasterize_deepfashion2_mask(annotation, image_size)

        return self.transform(image=image, mask=mask)
