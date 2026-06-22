"""FastAPI application factory."""
# ruff: noqa: RUF001

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from clothing_search.api.imaging import image_to_data_uri, mask_to_data_uri
from clothing_search.config import load_app_config
from clothing_search.pipeline import (
    HYBRID_MODE,
    SEGFORMER_MODE,
    UNET_MODE,
    SearchPipeline,
    SearchResponse,
    build_search_pipeline,
)
from clothing_search.search.models import SearchResult
from clothing_search.segmentation.categories import ClothingCategory, category_from_name
from clothing_search.segmentation.crop import CategoryNotFoundError

SAFE_ITEM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
API_DIR = Path(__file__).resolve().parent
CATEGORY_LABELS = {
    "top": "Верх: футболки, рубашки, блузки",
    "bottom": "Низ: брюки, джинсы, юбки",
    "dress": "Платье",
    "outerwear": "Верхняя одежда",
    "shoes": "Обувь",
    "bag": "Сумка",
    "accessories": "Аксессуары",
}
SEGMENTATION_MODE_LABELS = {
    SEGFORMER_MODE: "SegFormer для всех категорий",
    HYBRID_MODE: "Гибрид: U-Net для 4 классов, SegFormer для остальных",
    UNET_MODE: "U-Net",
}


async def _read_upload_image(file: UploadFile) -> Image.Image:
    contents = await file.read()
    try:
        with Image.open(BytesIO(contents)) as image:
            return image.convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image",
        ) from error


def _serialize_result(result: SearchResult) -> dict[str, Any]:
    return {
        "item_id": result.item_id,
        "score": result.score,
        "metadata": result.metadata,
    }


def _serialize_search_response(response: SearchResponse) -> dict[str, Any]:
    return {
        "category": response.category.name.lower(),
        "segmentation_score": response.segmentation_score,
        "segmentation_mode": response.segmentation_mode,
        "segmentation_backend": response.segmentation_backend,
        "segmentation_fallback_reason": response.segmentation_fallback_reason,
        "crop_box": list(response.crop.box),
        "crop_image": image_to_data_uri(response.crop.image),
        "mask_image": mask_to_data_uri(response.mask),
        "results": [_serialize_result(result) for result in response.results],
    }


def _validate_item_id(item_id: str) -> str:
    normalized_item_id = item_id.strip()
    if not normalized_item_id or not SAFE_ITEM_ID_PATTERN.fullmatch(
        normalized_item_id
    ):
        raise HTTPException(
            status_code=400,
            detail="item_id contains unsupported characters",
        )
    return normalized_item_id


def _upsert_catalog_metadata(catalog_dir: Path, metadata: dict[str, Any]) -> None:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = catalog_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as metadata_file:
            payload = json.load(metadata_file)
        if not isinstance(payload, list):
            raise HTTPException(
                status_code=400,
                detail="Catalog metadata root must be a list",
            )
        items = payload
    else:
        items = []

    item_id = metadata["item_id"]
    updated_items = [
        item
        for item in items
        if isinstance(item, dict) and str(item.get("item_id")) != item_id
    ]
    updated_items.append(metadata)
    metadata_path.write_text(
        json.dumps(updated_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_app(
    *,
    config_path: str | Path = "configs/app.yaml",
    pipeline: SearchPipeline | None = None,
    catalog_dir: str | Path = "data/catalog",
) -> FastAPI:
    app = FastAPI(title="ClothLens", version="0.1.0")
    app.state.pipeline = pipeline or build_search_pipeline(load_app_config(config_path))
    app.state.catalog_dir = Path(catalog_dir)
    templates = Jinja2Templates(directory=str(API_DIR / "templates"))
    app.mount(
        "/static",
        StaticFiles(directory=str(API_DIR / "static")),
        name="static",
    )
    app.mount(
        "/catalog-images",
        StaticFiles(directory=str(app.state.catalog_dir / "images"), check_dir=False),
        name="catalog-images",
    )

    @app.get("/")
    def index(request: Request) -> Any:
        categories = [
            category.name.lower()
            for category in ClothingCategory
            if category is not ClothingCategory.BACKGROUND
        ]
        category_options = [
            {
                "value": category,
                "label": CATEGORY_LABELS.get(category, category),
            }
            for category in categories
        ]
        return templates.TemplateResponse(
            name="index.html",
            request=request,
            context={
                "categories": categories,
                "category_options": category_options,
                "segmentation_mode_options": [
                    {
                        "value": mode,
                        "label": SEGMENTATION_MODE_LABELS.get(mode, mode),
                    }
                    for mode in getattr(
                        app.state.pipeline,
                        "supported_segmentation_modes",
                        (SEGFORMER_MODE,),
                    )
                ],
                "default_segmentation_mode": getattr(
                    app.state.pipeline,
                    "default_segmentation_mode",
                    SEGFORMER_MODE,
                ),
                "default_top_k": getattr(app.state.pipeline, "default_top_k", 10),
            },
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "clothlens"}

    @app.post("/search")
    async def search(
        file: Annotated[UploadFile, File()],
        category: Annotated[str, Form()],
        top_k: Annotated[int | None, Form()] = None,
        segmentation_mode: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        image = await _read_upload_image(file)
        try:
            response = app.state.pipeline.search(
                image,
                category=category,
                top_k=top_k,
                segmentation_mode=segmentation_mode,
            )
        except CategoryNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (OSError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return _serialize_search_response(response)

    @app.post("/catalog/add")
    async def add_catalog_item(
        file: Annotated[UploadFile, File()],
        item_id: Annotated[str, Form()],
        category: Annotated[str, Form()],
        brand: Annotated[str | None, Form()] = None,
        color: Annotated[str | None, Form()] = None,
        price: Annotated[float | None, Form()] = None,
        image_url: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        safe_item_id = _validate_item_id(item_id)
        try:
            resolved_category = category_from_name(category).name.lower()
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        image = await _read_upload_image(file)
        image_dir = app.state.catalog_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{safe_item_id}.jpg"
        image.save(image_path, format="JPEG")

        metadata: dict[str, Any] = {
            "item_id": safe_item_id,
            "category": resolved_category,
        }
        effective_image_url = image_url or f"/catalog-images/{safe_item_id}.jpg"
        optional_fields = {
            "brand": brand,
            "color": color,
            "price": price,
            "image_url": effective_image_url,
        }
        metadata.update(
            {
                name: value
                for name, value in optional_fields.items()
                if value not in (None, "")
            }
        )
        _upsert_catalog_metadata(app.state.catalog_dir, metadata)

        vector = app.state.pipeline.encoder.encode(image)
        app.state.pipeline.store.upsert(
            np.asarray([vector], dtype=np.float32),
            [metadata],
        )
        return {
            "item_id": safe_item_id,
            "category": resolved_category,
            "indexed": True,
        }

    return app
