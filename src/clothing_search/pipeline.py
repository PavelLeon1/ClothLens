"""End-to-end clothing search orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from clothing_search.config import AppConfig
from clothing_search.embeddings.encoder import FashionEncoder
from clothing_search.search.models import SearchResult
from clothing_search.search.qdrant_store import QdrantStore
from clothing_search.segmentation.base import Segmenter
from clothing_search.segmentation.categories import (
    ClothingCategory,
    category_from_name,
)
from clothing_search.segmentation.crop import CategoryCrop, crop_category
from clothing_search.segmentation.segformer import SegFormerSegmenter


class ImageEncoder(Protocol):
    def encode(self, image: Image.Image) -> NDArray[np.float32]:
        """Encode one image to a normalized vector."""


class SearchStore(Protocol):
    def search(
        self,
        query_vector: NDArray[np.floating],
        *,
        top_k: int,
        category: ClothingCategory,
    ) -> list[SearchResult]:
        """Find nearest catalog vectors."""


@dataclass(frozen=True, slots=True)
class SearchResponse:
    category: ClothingCategory
    mask: NDArray[np.integer]
    crop: CategoryCrop
    segmentation_score: float | None
    results: list[SearchResult]


class SearchPipeline:
    def __init__(
        self,
        *,
        segmenter: Segmenter,
        encoder: ImageEncoder,
        store: SearchStore,
        default_top_k: int = 10,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be positive")
        self.segmenter = segmenter
        self.encoder = encoder
        self.store = store
        self.default_top_k = default_top_k

    def search(
        self,
        image: Image.Image,
        *,
        category: str | ClothingCategory,
        top_k: int | None = None,
        padding_ratio: float = 0.1,
    ) -> SearchResponse:
        resolved_top_k = self.default_top_k if top_k is None else top_k
        if resolved_top_k < 1:
            raise ValueError("top_k must be positive")

        resolved_category = (
            category_from_name(category)
            if isinstance(category, str)
            else category
        )
        segmentation = self.segmenter.segment(image)
        crop = crop_category(
            image,
            segmentation.mask,
            resolved_category,
            padding_ratio=padding_ratio,
        )
        query_vector = self.encoder.encode(crop.image)
        results = self.store.search(
            query_vector,
            top_k=resolved_top_k,
            category=resolved_category,
        )
        return SearchResponse(
            category=resolved_category,
            mask=segmentation.mask,
            crop=crop,
            segmentation_score=segmentation.scores.get(resolved_category),
            results=results,
        )


def build_search_pipeline(
    config: AppConfig,
    *,
    segmenter_factory: Callable[..., Any] = SegFormerSegmenter,
    encoder_factory: Callable[..., Any] = FashionEncoder,
    store_factory: Callable[..., Any] = QdrantStore,
) -> SearchPipeline:
    if config.segmentation.backend != "segformer":
        raise RuntimeError(
            "The U-Net inference backend requires a trained checkpoint and "
            "will be connected after model training."
        )

    return SearchPipeline(
        segmenter=segmenter_factory(config.segmentation.model_name),
        encoder=encoder_factory(config.embedding.model_name),
        store=store_factory(
            collection_name=config.search.collection_name,
            path=config.search.path,
            vector_size=config.search.vector_size,
        ),
        default_top_k=config.search.top_k,
    )
