import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clothing_search.segmentation.categories import ClothingCategory
from clothing_search.segmentation.dataset import DeepFashion2Dataset


def identity_transform(*, image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {"image": image, "mask": mask}


def create_sample(root: Path, split: str = "train") -> None:
    image_dir = root / split / "image"
    annotation_dir = root / split / "annos"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    image = Image.new("RGB", (16, 16), color=(120, 80, 40))
    image.save(image_dir / "000001.jpg")
    annotation = {
        "source": "consumer",
        "pair_id": 1,
        "item1": {
            "category_id": 1,
            "segmentation": [[2, 2, 12, 2, 12, 12, 2, 12]],
        },
    }
    (annotation_dir / "000001.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )


def test_dataset_loads_native_deepfashion2_sample(tmp_path: Path) -> None:
    create_sample(tmp_path)
    dataset = DeepFashion2Dataset(
        tmp_path,
        split="train",
        transform=identity_transform,
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].shape == (16, 16, 3)
    assert sample["mask"].shape == (16, 16)
    assert sample["mask"].dtype == np.uint8
    assert sample["mask"][5, 5] == ClothingCategory.TOP
    assert sample["mask"][0, 0] == ClothingCategory.BACKGROUND


def test_val_alias_uses_validation_directory(tmp_path: Path) -> None:
    create_sample(tmp_path, split="validation")

    dataset = DeepFashion2Dataset(
        tmp_path,
        split="val",
        transform=identity_transform,
    )

    assert len(dataset) == 1


def test_dataset_rejects_missing_split(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DeepFashion2 image directory"):
        DeepFashion2Dataset(tmp_path, split="train")
