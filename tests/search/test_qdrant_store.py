from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from clothing_search.search.qdrant_store import QdrantStore
from clothing_search.segmentation.categories import ClothingCategory


@dataclass
class VectorParams:
    size: int
    distance: str


class Distance:
    COSINE = "cosine"


@dataclass
class PointStruct:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class MatchValue:
    value: str


@dataclass
class FieldCondition:
    key: str
    match: MatchValue


@dataclass
class Filter:
    must: list[FieldCondition]


class FakeModels:
    VectorParams = VectorParams
    Distance = Distance
    PointStruct = PointStruct
    MatchValue = MatchValue
    FieldCondition = FieldCondition
    Filter = Filter


class FakeClient:
    def __init__(self, collection_exists: bool = False) -> None:
        self.exists = collection_exists
        self.created: dict[str, Any] | None = None
        self.upserted: dict[str, Any] | None = None
        self.query: dict[str, Any] | None = None
        self.points = [
            SimpleNamespace(
                id="uuid",
                score=0.92,
                payload={
                    "item_id": "sku-1",
                    "category": "top",
                    "brand": "ClothLens",
                },
            )
        ]

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs: Any) -> None:
        self.created = kwargs
        self.exists = True

    def upsert(self, **kwargs: Any) -> None:
        self.upserted = kwargs

    def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.query = kwargs
        return SimpleNamespace(points=self.points)


def test_store_creates_cosine_collection_when_missing() -> None:
    client = FakeClient()

    QdrantStore(
        client=client,
        models_module=FakeModels,
        collection_name="catalog",
        vector_size=3,
    )

    assert client.created == {
        "collection_name": "catalog",
        "vectors_config": VectorParams(size=3, distance="cosine"),
    }


def test_store_preserves_existing_collection() -> None:
    client = FakeClient(collection_exists=True)

    QdrantStore(
        client=client,
        models_module=FakeModels,
        collection_name="catalog",
        vector_size=3,
    )

    assert client.created is None


def test_upsert_uses_deterministic_ids_and_payload() -> None:
    client = FakeClient(collection_exists=True)
    store = QdrantStore(
        client=client,
        models_module=FakeModels,
        collection_name="catalog",
        vector_size=3,
    )
    metadata = [
        {"item_id": "sku-1", "category": "top"},
        {"item_id": "sku-2", "category": "dress"},
    ]

    store.upsert(
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        metadata,
    )

    assert client.upserted is not None
    points = client.upserted["points"]
    assert client.upserted["collection_name"] == "catalog"
    assert points[0].id == "4997c622-f939-5808-903d-2bf6c88a0679"
    assert points[0].payload == metadata[0]
    assert points[0].vector == [1.0, 0.0, 0.0]


def test_search_filters_category_and_maps_results() -> None:
    client = FakeClient(collection_exists=True)
    store = QdrantStore(
        client=client,
        models_module=FakeModels,
        collection_name="catalog",
        vector_size=3,
    )

    results = store.search(
        np.array([1, 0, 0], dtype=np.float32),
        top_k=5,
        category=ClothingCategory.TOP,
    )

    assert client.query is not None
    assert client.query["limit"] == 5
    assert client.query["query"] == [1.0, 0.0, 0.0]
    condition = client.query["query_filter"].must[0]
    assert condition.key == "category"
    assert condition.match.value == "top"
    assert results[0].item_id == "sku-1"
    assert results[0].score == 0.92
    assert results[0].metadata["brand"] == "ClothLens"
