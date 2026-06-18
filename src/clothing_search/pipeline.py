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

SEGFORMER_MODE = "segformer"
UNET_MODE = "unet"
HYBRID_MODE = "hybrid"
SUPPORTED_SEGMENTATION_MODES = frozenset({SEGFORMER_MODE, UNET_MODE, HYBRID_MODE})
UNET_TRAINED_CATEGORIES = frozenset(
    {
        ClothingCategory.TOP,
        ClothingCategory.BOTTOM,
        ClothingCategory.DRESS,
        ClothingCategory.OUTERWEAR,
    }
)


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
    segmentation_mode: str = SEGFORMER_MODE
    segmentation_backend: str = SEGFORMER_MODE


class LazySegmenter:
    def __init__(self, factory: Callable[..., Any], **kwargs: Any) -> None:
        self.factory = factory
        self.kwargs = kwargs
        self._segmenter: Segmenter | None = None

    def segment(self, image: Image.Image) -> Any:
        if self._segmenter is None:
            self._segmenter = self.factory(**self.kwargs)
        return self._segmenter.segment(image)


class SearchPipeline:
    def __init__(
        self,
        *,
        segmenter: Segmenter,
        unet_segmenter: Segmenter | None = None,
        encoder: ImageEncoder,
        store: SearchStore,
        default_top_k: int = 10,
        default_segmentation_mode: str = SEGFORMER_MODE,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be positive")
        self.segmenter = segmenter
        self.unet_segmenter = unet_segmenter
        self.encoder = encoder
        self.store = store
        self.default_top_k = default_top_k
        self.default_segmentation_mode = default_segmentation_mode

    @property
    def supported_segmentation_modes(self) -> tuple[str, ...]:
        modes = [SEGFORMER_MODE]
        if self.unet_segmenter is not None:
            modes.append(HYBRID_MODE)
        if self.default_segmentation_mode == UNET_MODE:
            modes = [UNET_MODE]
        return tuple(modes)

    def _select_segmenter(
        self,
        *,
        category: ClothingCategory,
        segmentation_mode: str | None,
    ) -> tuple[Segmenter, str, str]:
        resolved_mode = segmentation_mode or self.default_segmentation_mode
        if resolved_mode not in SUPPORTED_SEGMENTATION_MODES:
            raise ValueError(f"Unsupported segmentation mode: {resolved_mode}")
        if resolved_mode == UNET_MODE:
            if self.unet_segmenter is None:
                raise RuntimeError("U-Net segmentation mode requires U-Net")
            return self.unet_segmenter, UNET_MODE, UNET_MODE
        if resolved_mode == HYBRID_MODE and category in UNET_TRAINED_CATEGORIES:
            if self.unet_segmenter is None:
                raise RuntimeError("Hybrid segmentation mode requires U-Net")
            return self.unet_segmenter, HYBRID_MODE, UNET_MODE
        return self.segmenter, resolved_mode, SEGFORMER_MODE

    def search(
        self,
        image: Image.Image,
        *,
        category: str | ClothingCategory,
        top_k: int | None = None,
        padding_ratio: float = 0.1,
        segmentation_mode: str | None = None,
    ) -> SearchResponse:
        resolved_top_k = self.default_top_k if top_k is None else top_k
        if resolved_top_k < 1:
            raise ValueError("top_k must be positive")

        resolved_category = (
            category_from_name(category)
            if isinstance(category, str)
            else category
        )
        segmenter, resolved_mode, backend = self._select_segmenter(
            category=resolved_category,
            segmentation_mode=segmentation_mode,
        )
        segmentation = segmenter.segment(image)
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
            segmentation_mode=resolved_mode,
            segmentation_backend=backend,
        )


def build_search_pipeline(
    config: AppConfig,
    *,
    segmenter_factory: Callable[..., Any] | None = None,
    unet_segmenter_factory: Callable[..., Any] | None = None,
    encoder_factory: Callable[..., Any] = FashionEncoder,
    store_factory: Callable[..., Any] = QdrantStore,
) -> SearchPipeline:
    unet_segmenter: Segmenter | None = None
    default_segmentation_mode = config.segmentation.backend

    if config.segmentation.backend == "segformer":
        resolved_segmenter_factory = segmenter_factory or SegFormerSegmenter
        segmenter = resolved_segmenter_factory(config.segmentation.model_name)
        if config.segmentation.checkpoint_path:
            if unet_segmenter_factory is None:
                from clothing_search.segmentation.unet_segmenter import UnetSegmenter

                unet_segmenter_factory = UnetSegmenter
            unet_segmenter = LazySegmenter(
                unet_segmenter_factory,
                checkpoint_path=config.segmentation.checkpoint_path,
                image_size=config.segmentation.image_size,
                encoder_name=config.segmentation.encoder,
                encoder_weights=config.segmentation.encoder_weights,
                num_classes=config.segmentation.num_classes,
            )
    elif config.segmentation.backend == "unet":
        if not config.segmentation.checkpoint_path:
            raise RuntimeError("U-Net backend requires segmentation.checkpoint_path")
        if unet_segmenter_factory is None:
            from clothing_search.segmentation.unet_segmenter import UnetSegmenter

            unet_segmenter_factory = UnetSegmenter
        segmenter = unet_segmenter_factory(
            checkpoint_path=config.segmentation.checkpoint_path,
            image_size=config.segmentation.image_size,
            encoder_name=config.segmentation.encoder,
            encoder_weights=config.segmentation.encoder_weights,
            num_classes=config.segmentation.num_classes,
        )
        unet_segmenter = segmenter
    else:
        raise RuntimeError(
            f"Unsupported segmentation backend: {config.segmentation.backend}"
        )

    return SearchPipeline(
        segmenter=segmenter,
        unet_segmenter=unet_segmenter,
        encoder=encoder_factory(config.embedding.model_name),
        store=store_factory(
            collection_name=config.search.collection_name,
            path=config.search.path,
            vector_size=config.search.vector_size,
        ),
        default_top_k=config.search.top_k,
        default_segmentation_mode=default_segmentation_mode,
    )
