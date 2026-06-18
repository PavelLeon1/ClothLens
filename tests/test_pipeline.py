from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from clothing_search.config import load_app_config
from clothing_search.pipeline import SearchPipeline, build_search_pipeline
from clothing_search.search.models import SearchResult
from clothing_search.segmentation.base import SegmentationResult
from clothing_search.segmentation.categories import ClothingCategory


class FakeSegmenter:
    def __init__(self) -> None:
        self.image: Image.Image | None = None

    def segment(self, image: Image.Image) -> SegmentationResult:
        self.image = image
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[2:6, 1:5] = ClothingCategory.TOP
        return SegmentationResult(
            mask=mask,
            scores={ClothingCategory.TOP: 0.88},
        )


class FakeEncoder:
    def __init__(self) -> None:
        self.image: Image.Image | None = None

    def encode(self, image: Image.Image) -> np.ndarray:
        self.image = image
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeStore:
    def __init__(self) -> None:
        self.call: tuple[np.ndarray, int, ClothingCategory] | None = None

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        category: ClothingCategory,
    ) -> list[SearchResult]:
        self.call = query_vector, top_k, category
        return [
            SearchResult(
                item_id="sku-1",
                score=0.94,
                metadata={"category": "top"},
            )
        ]


def test_pipeline_runs_segmentation_crop_embedding_and_search() -> None:
    segmenter = FakeSegmenter()
    encoder = FakeEncoder()
    store = FakeStore()
    pipeline = SearchPipeline(
        segmenter=segmenter,
        encoder=encoder,
        store=store,
        default_top_k=10,
    )
    image = Image.new("RGB", (8, 8), color=(10, 20, 30))

    response = pipeline.search(
        image,
        category="top",
        top_k=3,
        padding_ratio=0,
    )

    assert segmenter.image is image
    assert encoder.image is not None
    assert encoder.image.size == (4, 4)
    assert store.call is not None
    assert store.call[1:] == (3, ClothingCategory.TOP)
    assert response.category is ClothingCategory.TOP
    assert response.crop.box == (1, 2, 5, 6)
    assert response.segmentation_score == 0.88
    assert response.results[0].item_id == "sku-1"


def test_pipeline_uses_default_top_k() -> None:
    store = FakeStore()
    pipeline = SearchPipeline(
        segmenter=FakeSegmenter(),
        encoder=FakeEncoder(),
        store=store,
        default_top_k=7,
    )

    pipeline.search(
        Image.new("RGB", (8, 8)),
        category=ClothingCategory.TOP,
        padding_ratio=0,
    )

    assert store.call is not None
    assert store.call[1] == 7


def test_pipeline_rejects_non_positive_top_k() -> None:
    pipeline = SearchPipeline(
        segmenter=FakeSegmenter(),
        encoder=FakeEncoder(),
        store=FakeStore(),
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        pipeline.search(
            Image.new("RGB", (8, 8)),
            category=ClothingCategory.TOP,
            top_k=0,
        )


class CaptureFactory:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.args: tuple[Any, ...] | None = None
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.args = args
        self.kwargs = kwargs
        return self.result


def test_build_search_pipeline_uses_application_config(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        "segmentation:\n"
        "  backend: segformer\n"
        "  model_name: example/segformer\n"
        "embedding:\n"
        "  model_name: example/fashion-clip\n"
        "search:\n"
        "  collection_name: example_catalog\n"
        "  path: example/qdrant\n"
        "  vector_size: 3\n"
        "  top_k: 4\n",
        encoding="utf-8",
    )
    config = load_app_config(config_path)
    segmenter = FakeSegmenter()
    encoder = FakeEncoder()
    store = FakeStore()
    segmenter_factory = CaptureFactory(segmenter)
    encoder_factory = CaptureFactory(encoder)
    store_factory = CaptureFactory(store)

    pipeline = build_search_pipeline(
        config,
        segmenter_factory=segmenter_factory,
        encoder_factory=encoder_factory,
        store_factory=store_factory,
    )

    assert pipeline.segmenter is segmenter
    assert segmenter_factory.args == ("example/segformer",)
    assert encoder_factory.args == ("example/fashion-clip",)
    assert store_factory.kwargs == {
        "collection_name": "example_catalog",
        "path": "example/qdrant",
        "vector_size": 3,
    }
    assert pipeline.default_top_k == 4


def test_build_search_pipeline_uses_unet_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "app_unet.yaml"
    config_path.write_text(
        "segmentation:\n"
        "  backend: unet\n"
        "  checkpoint_path: models/unet_best.ckpt\n"
        "  image_size: 512\n"
        "embedding:\n"
        "  model_name: example/fashion-clip\n"
        "search:\n"
        "  collection_name: example_catalog\n"
        "  path: example/qdrant\n"
        "  vector_size: 3\n"
        "  top_k: 5\n",
        encoding="utf-8",
    )
    config = load_app_config(config_path)
    segmenter = FakeSegmenter()
    encoder = FakeEncoder()
    store = FakeStore()
    unet_segmenter_factory = CaptureFactory(segmenter)

    pipeline = build_search_pipeline(
        config,
        unet_segmenter_factory=unet_segmenter_factory,
        encoder_factory=CaptureFactory(encoder),
        store_factory=CaptureFactory(store),
    )

    assert pipeline.segmenter is segmenter
    assert unet_segmenter_factory.kwargs == {
        "checkpoint_path": "models/unet_best.ckpt",
        "image_size": 512,
        "encoder_name": "resnet34",
        "encoder_weights": None,
        "num_classes": 8,
    }
    assert pipeline.default_top_k == 5
