"""Qdrant-backed catalog storage with persistent local mode support."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray

from clothing_search.search.models import SearchResult
from clothing_search.segmentation.categories import ClothingCategory


class QdrantStore:
    def __init__(
        self,
        *,
        collection_name: str = "clothing_catalog",
        vector_size: int = 512,
        path: str | Path = "data/qdrant",
        client: Any | None = None,
        models_module: Any | None = None,
    ) -> None:
        if vector_size < 1:
            raise ValueError("vector_size must be positive")

        if client is None or models_module is None:
            try:
                qdrant_client = import_module("qdrant_client")
                qdrant_models = import_module("qdrant_client.models")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "Qdrant dependencies are not installed. Run "
                    r'venv\Scripts\python.exe -m pip install -e ".[search]".'
                ) from error
            client = client or qdrant_client.QdrantClient(path=str(path))
            models_module = models_module or qdrant_models

        self.client = client
        self.models = models_module
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.models.VectorParams(
                size=self.vector_size,
                distance=self.models.Distance.COSINE,
            ),
        )

    def upsert(
        self,
        vectors: NDArray[np.floating],
        metadata: list[dict[str, Any]],
    ) -> None:
        vector_array = np.asarray(vectors, dtype=np.float32)
        if vector_array.ndim != 2 or vector_array.shape[1] != self.vector_size:
            raise ValueError(
                f"Expected vectors with shape (N, {self.vector_size})"
            )
        if vector_array.shape[0] != len(metadata):
            raise ValueError("Vector and metadata counts must match")

        points = []
        for vector, payload in zip(vector_array, metadata, strict=True):
            item_id = str(payload.get("item_id", "")).strip()
            if not item_id:
                raise ValueError("Every catalog payload requires item_id")
            points.append(
                self.models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, item_id)),
                    vector=vector.tolist(),
                    payload=dict(payload),
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query_vector: NDArray[np.floating],
        *,
        top_k: int = 10,
        category: ClothingCategory | None = None,
    ) -> list[SearchResult]:
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.shape != (self.vector_size,):
            raise ValueError(
                f"Expected query vector with shape ({self.vector_size},)"
            )
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_filter = None
        if category is not None:
            query_filter = self.models.Filter(
                must=[
                    self.models.FieldCondition(
                        key="category",
                        match=self.models.MatchValue(
                            value=category.name.lower()
                        ),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
        )
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            item_id = str(payload.pop("item_id", point.id))
            results.append(
                SearchResult(
                    item_id=item_id,
                    score=float(point.score),
                    metadata=payload,
                )
            )
        return results
