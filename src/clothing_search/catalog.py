"""Catalog metadata loading and batch indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from clothing_search.segmentation.categories import (
    ClothingCategory,
    category_from_name,
)


@dataclass(frozen=True, slots=True)
class CatalogItem:
    item_id: str
    category: ClothingCategory
    image_path: Path
    metadata: dict[str, Any]


class BatchEncoder(Protocol):
    def encode_batch(
        self,
        images: list[Image.Image | None],
        *,
        batch_size: int,
    ) -> NDArray[np.float32]:
        """Encode catalog images to normalized vectors."""


class VectorStore(Protocol):
    def upsert(
        self,
        vectors: NDArray[np.floating],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Insert or replace catalog vectors."""


def _resolve_image(image_dir: Path, item_id: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = image_dir / f"{item_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Catalog image is missing for item '{item_id}'")


def load_catalog(catalog_dir: str | Path) -> list[CatalogItem]:
    root = Path(catalog_dir)
    metadata_path = root / "metadata.json"
    image_dir = root / "images"
    with metadata_path.open(encoding="utf-8") as metadata_file:
        payload = json.load(metadata_file)
    if not isinstance(payload, list):
        raise ValueError("Catalog metadata root must be a list")

    items: list[CatalogItem] = []
    seen_ids: set[str] = set()
    for raw_item in payload:
        if not isinstance(raw_item, dict):
            raise ValueError("Every catalog metadata item must be an object")
        item_id = str(raw_item.get("item_id", "")).strip()
        if not item_id:
            raise ValueError("Every catalog item requires item_id")
        if item_id in seen_ids:
            raise ValueError(f"Duplicate catalog item_id: {item_id}")
        seen_ids.add(item_id)

        category = category_from_name(str(raw_item.get("category", "")))
        metadata = dict(raw_item)
        metadata["item_id"] = item_id
        metadata["category"] = category.name.lower()
        items.append(
            CatalogItem(
                item_id=item_id,
                category=category,
                image_path=_resolve_image(image_dir, item_id),
                metadata=metadata,
            )
        )
    return items


def index_catalog(
    catalog_dir: str | Path,
    *,
    encoder: BatchEncoder,
    store: VectorStore,
    batch_size: int = 64,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    items = load_catalog(catalog_dir)

    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        images = []
        for item in batch:
            with Image.open(item.image_path) as source_image:
                images.append(source_image.convert("RGB").copy())
        vectors = encoder.encode_batch(images, batch_size=batch_size)
        store.upsert(vectors, [item.metadata for item in batch])

    return len(items)
