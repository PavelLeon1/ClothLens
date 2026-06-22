from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from clothing_search.api.app import create_app
from clothing_search.pipeline import SearchResponse
from clothing_search.search.models import SearchResult
from clothing_search.segmentation.categories import ClothingCategory
from clothing_search.segmentation.crop import CategoryCrop, CategoryNotFoundError


def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), color=(20, 80, 140)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEncoder:
    def __init__(self) -> None:
        self.images: list[Image.Image] = []

    def encode(self, image: Image.Image) -> np.ndarray:
        self.images.append(image)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[np.ndarray, list[dict[str, Any]]]] = []

    def upsert(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        self.upserts.append((vectors, metadata))


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[Image.Image, str, int | None, str | None]] = []
        self.encoder = FakeEncoder()
        self.store = FakeStore()
        self.default_segmentation_mode = "segformer"
        self.supported_segmentation_modes = ("segformer", "hybrid")

    def search(
        self,
        image: Image.Image,
        *,
        category: str,
        top_k: int | None = None,
        segmentation_mode: str | None = None,
    ) -> SearchResponse:
        self.calls.append((image, category, top_k, segmentation_mode))
        if category == "unknown":
            raise ValueError("Unsupported clothing category: unknown")
        if category == "dress":
            raise CategoryNotFoundError("Clothing category 'dress' was not found")
        if category == "bag":
            raise RuntimeError("U-Net checkpoint is not available")

        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[1:3, 2:5] = ClothingCategory.TOP
        crop = CategoryCrop(
            image=image.crop((2, 1, 5, 3)),
            box=(2, 1, 5, 3),
            category=ClothingCategory.TOP,
        )
        return SearchResponse(
            category=ClothingCategory.TOP,
            mask=mask,
            crop=crop,
            segmentation_score=0.91,
            segmentation_mode=segmentation_mode or "segformer",
            segmentation_backend="segformer",
            segmentation_fallback_reason=(
                "unet_selected_category_too_small"
                if segmentation_mode == "hybrid"
                else None
            ),
            results=[
                SearchResult(
                    item_id="sku-1",
                    score=0.88,
                    metadata={
                        "category": "top",
                        "brand": "ClothLens",
                        "image_url": "https://example.test/sku-1.jpg",
                    },
                )
            ],
        )


def make_client(tmp_path: Path) -> tuple[TestClient, FakePipeline]:
    pipeline = FakePipeline()
    app = create_app(pipeline=pipeline, catalog_dir=tmp_path)
    return TestClient(app), pipeline


def test_health_reports_service_status(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "clothlens"}


def test_search_returns_serialized_crop_mask_and_results(tmp_path: Path) -> None:
    client, pipeline = make_client(tmp_path)

    response = client.post(
        "/search",
        files={"file": ("query.png", image_bytes(), "image/png")},
        data={"category": "top", "top_k": "3", "segmentation_mode": "hybrid"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "top"
    assert payload["segmentation_score"] == 0.91
    assert payload["segmentation_mode"] == "hybrid"
    assert payload["segmentation_backend"] == "segformer"
    assert payload["segmentation_fallback_reason"] == (
        "unet_selected_category_too_small"
    )
    assert payload["crop_box"] == [2, 1, 5, 3]
    assert payload["crop_image"].startswith("data:image/png;base64,")
    assert payload["mask_image"].startswith("data:image/png;base64,")
    assert payload["results"] == [
        {
            "item_id": "sku-1",
            "score": 0.88,
            "metadata": {
                "category": "top",
                "brand": "ClothLens",
                "image_url": "https://example.test/sku-1.jpg",
            },
        }
    ]
    assert pipeline.calls[0][1:] == ("top", 3, "hybrid")


def test_search_rejects_invalid_image(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.post(
        "/search",
        files={"file": ("broken.txt", b"not an image", "text/plain")},
        data={"category": "top"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image"


def test_search_maps_domain_errors_to_http_errors(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    invalid = client.post(
        "/search",
        files={"file": ("query.png", image_bytes(), "image/png")},
        data={"category": "unknown"},
    )
    missing = client.post(
        "/search",
        files={"file": ("query.png", image_bytes(), "image/png")},
        data={"category": "dress"},
    )

    assert invalid.status_code == 400
    assert "Unsupported clothing category" in invalid.json()["detail"]
    assert missing.status_code == 404
    assert "was not found" in missing.json()["detail"]


def test_search_maps_model_loading_errors_to_service_unavailable(
    tmp_path: Path,
) -> None:
    client, _ = make_client(tmp_path)

    response = client.post(
        "/search",
        files={"file": ("query.png", image_bytes(), "image/png")},
        data={"category": "bag", "segmentation_mode": "hybrid"},
    )

    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"]


def test_catalog_add_saves_image_and_indexes_metadata(tmp_path: Path) -> None:
    client, pipeline = make_client(tmp_path)

    response = client.post(
        "/catalog/add",
        files={"file": ("catalog.png", image_bytes(), "image/png")},
        data={
            "item_id": "sku-1",
            "category": "top",
            "brand": "ClothLens",
            "color": "blue",
            "price": "3990",
            "image_url": "https://example.test/sku-1.jpg",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "item_id": "sku-1",
        "category": "top",
        "indexed": True,
    }
    saved_image = tmp_path / "images" / "sku-1.jpg"
    assert saved_image.is_file()
    with Image.open(saved_image) as image:
        assert image.size == (8, 6)
    metadata_payload = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata_payload == [
        {
            "item_id": "sku-1",
            "category": "top",
            "brand": "ClothLens",
            "color": "blue",
            "price": 3990.0,
            "image_url": "https://example.test/sku-1.jpg",
        }
    ]
    assert len(pipeline.encoder.images) == 1
    assert len(pipeline.store.upserts) == 1
    vectors, metadata = pipeline.store.upserts[0]
    assert vectors.tolist() == [[1.0, 0.0, 0.0]]
    assert metadata == [
        {
            "item_id": "sku-1",
            "category": "top",
            "brand": "ClothLens",
            "color": "blue",
            "price": 3990.0,
            "image_url": "https://example.test/sku-1.jpg",
        }
    ]


def test_catalog_add_uses_local_image_url_when_missing(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.post(
        "/catalog/add",
        files={"file": ("catalog.png", image_bytes(), "image/png")},
        data={"item_id": "sku-2", "category": "top"},
    )

    assert response.status_code == 200
    metadata_payload = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata_payload == [
        {
            "item_id": "sku-2",
            "category": "top",
            "image_url": "/catalog-images/sku-2.jpg",
        }
    ]

    image_response = client.get("/catalog-images/sku-2.jpg")

    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"


def test_catalog_add_rejects_unsafe_item_id(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.post(
        "/catalog/add",
        files={"file": ("catalog.png", image_bytes(), "image/png")},
        data={"item_id": "../sku-1", "category": "top"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "item_id contains unsupported characters"


def test_index_page_renders_upload_form(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert '<html lang="ru">' in response.text
    assert "ClothLens" in response.text
    assert "Визуальный поиск одежды" in response.text
    assert "Найти похожую одежду" in response.text
    assert "Добавить изображения в каталог" in response.text
    assert "Результаты" in response.text
    assert "Visual clothing search" not in response.text
    assert "Search Similar Clothes" not in response.text
    assert 'id="search-form" class="form-grid search-form-grid"' in response.text
    assert 'name="file"' in response.text
    assert 'name="category"' in response.text
    assert 'name="segmentation_mode"' in response.text
    assert "SegFormer" in response.text
    assert "U-Net" in response.text
    assert 'id="catalog-form" class="form-grid catalog-form-grid"' in response.text
    assert 'name="catalog_files"' in response.text
    assert 'multiple' in response.text
    assert 'id="catalog-status"' in response.text
    assert 'class="form-actions"' in response.text
    assert 'class="preview-card"' in response.text
    assert 'class="preview-placeholder"' in response.text
    assert 'class="preview-image"' in response.text
    assert 'id="results"' in response.text


def test_static_assets_are_served(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    script = client.get("/static/app.js")
    styles = client.get("/static/styles.css")

    assert script.status_code == 200
    assert "fetch('/search'" in script.text
    assert "fetch('/catalog/add'" in script.text
    assert "SEGMENTATION_BACKEND_LABELS" in script.text
    assert "segmentation_backend" in script.text
    assert "segmentation_fallback_reason" in script.text
    assert "catalog-form" in script.text
    assert "Идёт поиск..." in script.text
    assert "Сходство:" in script.text
    assert "Проиндексировано" in script.text
    assert "Searching..." not in script.text
    assert "Score:" not in script.text
    assert styles.status_code == 200
    assert ".form-grid" in styles.text
    assert ".search-form-grid" in styles.text
    assert ".catalog-form-grid" in styles.text
    assert ".preview-card.has-image" in styles.text
    assert ".preview-placeholder" in styles.text
    assert ".result-card" in styles.text


def test_api_module_entrypoint_runs_uvicorn_factory(monkeypatch: Any) -> None:
    from clothing_search.api import __main__ as entrypoint

    calls: dict[str, Any] = {}

    def fake_run(application: str, **kwargs: Any) -> None:
        calls["application"] = application
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entrypoint.uvicorn, "run", fake_run)

    entrypoint.main()

    assert calls == {
        "application": "clothing_search.api.app:create_app",
        "kwargs": {
            "factory": True,
            "host": "127.0.0.1",
            "port": 8000,
        },
    }
