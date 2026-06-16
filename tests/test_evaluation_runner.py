import json
from pathlib import Path
from typing import Any

from PIL import Image

from clothing_search.evaluation import (
    evaluate_search_pipeline,
    load_evaluation_queries,
)
from clothing_search.search.models import SearchResult
from clothing_search.segmentation.categories import ClothingCategory


def save_query_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(40, 80, 120)).save(path)


def write_manifest(path: Path, payload: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_evaluation_queries_resolves_images_and_categories(
    tmp_path: Path,
) -> None:
    query_image = tmp_path / "queries" / "query-1.jpg"
    query_image.parent.mkdir()
    save_query_image(query_image)
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [
            {
                "query_id": "q1",
                "image_path": "queries/query-1.jpg",
                "category": "top",
                "relevant_item_ids": ["sku-1", "sku-2"],
            }
        ],
    )

    queries = load_evaluation_queries(manifest)

    assert len(queries) == 1
    assert queries[0].query_id == "q1"
    assert queries[0].image_path == query_image
    assert queries[0].category is ClothingCategory.TOP
    assert queries[0].relevant_item_ids == frozenset({"sku-1", "sku-2"})


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], ClothingCategory, int]] = []

    def search(
        self,
        image: Image.Image,
        *,
        category: ClothingCategory,
        top_k: int,
    ) -> Any:
        self.calls.append((image.size, category, top_k))
        item_ids = (
            ["sku-1", "sku-x"]
            if category is ClothingCategory.TOP
            else ["sku-3", "sku-4"]
        )
        return type(
            "FakeResponse",
            (),
            {
                "results": [
                    SearchResult(item_id=item_id, score=1.0, metadata={})
                    for item_id in item_ids
                ]
            },
        )()


def test_evaluate_search_pipeline_computes_report(tmp_path: Path) -> None:
    first_image = tmp_path / "q1.jpg"
    second_image = tmp_path / "q2.jpg"
    save_query_image(first_image)
    save_query_image(second_image)
    queries = load_evaluation_queries(
        tmp_path / "manifest.json",
        payload=[
            {
                "query_id": "q1",
                "image_path": str(first_image),
                "category": "top",
                "relevant_item_ids": ["sku-1", "sku-2"],
            },
            {
                "query_id": "q2",
                "image_path": str(second_image),
                "category": "dress",
                "relevant_item_ids": ["sku-4"],
            },
        ],
    )
    pipeline = FakePipeline()
    clock_values = iter([10.0, 10.2, 20.0, 20.5])

    report = evaluate_search_pipeline(
        pipeline,
        queries,
        top_k=2,
        clock=lambda: next(clock_values),
    )

    assert pipeline.calls == [
        ((4, 4), ClothingCategory.TOP, 2),
        ((4, 4), ClothingCategory.DRESS, 2),
    ]
    assert report.to_dict() == {
        "top_k": 2,
        "query_count": 2,
        "summary": {
            "precision_at_k": 0.5,
            "recall_at_k": 0.75,
            "mean_average_precision_at_k": 0.5,
            "latency_seconds": {
                "count": 2,
                "min": 0.2,
                "max": 0.5,
                "mean": 0.35,
                "p50": 0.2,
                "p95": 0.5,
            },
        },
        "queries": [
            {
                "query_id": "q1",
                "category": "top",
                "retrieved_item_ids": ["sku-1", "sku-x"],
                "relevant_item_ids": ["sku-1", "sku-2"],
                "precision_at_k": 0.5,
                "recall_at_k": 0.5,
                "average_precision_at_k": 0.5,
                "latency_seconds": 0.2,
            },
            {
                "query_id": "q2",
                "category": "dress",
                "retrieved_item_ids": ["sku-3", "sku-4"],
                "relevant_item_ids": ["sku-4"],
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "average_precision_at_k": 0.5,
                "latency_seconds": 0.5,
            },
        ],
    }
