import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from clothing_search.catalog import index_catalog, load_catalog
from clothing_search.segmentation.categories import ClothingCategory


def write_catalog(
    root: Path,
    metadata: list[dict[str, Any]],
    image_ids: list[str],
) -> None:
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    for item_id in image_ids:
        Image.new("RGB", (4, 4), color=(10, 20, 30)).save(
            image_dir / f"{item_id}.jpg"
        )
    (root / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_load_catalog_validates_and_resolves_images(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        [
            {
                "item_id": "sku-1",
                "category": "top",
                "brand": "ClothLens",
                "image_url": "https://example.test/sku-1.jpg",
            }
        ],
        ["sku-1"],
    )

    items = load_catalog(tmp_path)

    assert len(items) == 1
    assert items[0].item_id == "sku-1"
    assert items[0].category is ClothingCategory.TOP
    assert items[0].image_path == tmp_path / "images" / "sku-1.jpg"
    assert items[0].metadata["brand"] == "ClothLens"


def test_load_catalog_rejects_missing_image(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        [{"item_id": "missing", "category": "dress"}],
        [],
    )

    with pytest.raises(FileNotFoundError, match="missing"):
        load_catalog(tmp_path)


class FakeEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode_batch(
        self,
        images: list[Image.Image],
        *,
        batch_size: int,
    ) -> np.ndarray:
        self.batch_sizes.append(len(images))
        return np.eye(len(images), 3, dtype=np.float32)


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, list[dict[str, Any]]]] = []

    def upsert(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        self.calls.append((vectors, metadata))


def test_index_catalog_encodes_and_upserts_in_batches(tmp_path: Path) -> None:
    metadata = [
        {"item_id": f"sku-{index}", "category": "top"}
        for index in range(3)
    ]
    write_catalog(
        tmp_path,
        metadata,
        [item["item_id"] for item in metadata],
    )
    encoder = FakeEncoder()
    store = FakeStore()

    count = index_catalog(
        tmp_path,
        encoder=encoder,
        store=store,
        batch_size=2,
    )

    assert count == 3
    assert encoder.batch_sizes == [2, 1]
    assert [len(metadata_batch) for _, metadata_batch in store.calls] == [2, 1]
    assert store.calls[0][1][0]["item_id"] == "sku-0"
