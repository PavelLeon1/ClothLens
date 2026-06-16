"""Evaluation helpers for clothing retrieval experiments."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from PIL import Image

from clothing_search.segmentation.categories import (
    ClothingCategory,
    category_from_name,
)


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be positive")


def _top_k_hits(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    k: int,
) -> list[bool]:
    _validate_k(k)
    relevant = set(relevant_ids)
    return [item_id in relevant for item_id in retrieved_ids[:k]]


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    k: int,
) -> float:
    hits = _top_k_hits(retrieved_ids, relevant_ids, k=k)
    return sum(hits) / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    k: int,
) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = _top_k_hits(retrieved_ids, relevant, k=k)
    return sum(hits) / len(relevant)


def average_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    k: int,
) -> float:
    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    score = 0.0
    hits = 0
    for rank, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def mean_average_precision_at_k(
    queries: Iterable[tuple[Sequence[str], Iterable[str]]],
    *,
    k: int,
) -> float:
    _validate_k(k)
    values = [
        average_precision_at_k(retrieved_ids, relevant_ids, k=k)
        for retrieved_ids, relevant_ids in queries
    ]
    if not values:
        return 0.0
    return mean(values)


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    rank = max(1, ceil(percentile * len(values)))
    return values[rank - 1]


def summarize_latency(latencies: Sequence[float]) -> dict[str, float | int]:
    if not latencies:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }

    sorted_latencies = sorted(float(latency) for latency in latencies)
    return {
        "count": len(sorted_latencies),
        "min": sorted_latencies[0],
        "max": sorted_latencies[-1],
        "mean": mean(sorted_latencies),
        "p50": _nearest_rank_percentile(sorted_latencies, 0.5),
        "p95": _nearest_rank_percentile(sorted_latencies, 0.95),
    }


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    image_path: Path
    category: ClothingCategory
    relevant_item_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: str
    category: ClothingCategory
    retrieved_item_ids: tuple[str, ...]
    relevant_item_ids: tuple[str, ...]
    precision_at_k: float
    recall_at_k: float
    average_precision_at_k: float
    latency_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "query_id": self.query_id,
            "category": self.category.name.lower(),
            "retrieved_item_ids": list(self.retrieved_item_ids),
            "relevant_item_ids": list(self.relevant_item_ids),
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "average_precision_at_k": self.average_precision_at_k,
            "latency_seconds": self.latency_seconds,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    top_k: int
    queries: tuple[QueryEvaluation, ...]
    precision_at_k: float
    recall_at_k: float
    mean_average_precision_at_k: float
    latency_seconds: dict[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "query_count": len(self.queries),
            "summary": {
                "precision_at_k": self.precision_at_k,
                "recall_at_k": self.recall_at_k,
                "mean_average_precision_at_k": self.mean_average_precision_at_k,
                "latency_seconds": self.latency_seconds,
            },
            "queries": [query.to_dict() for query in self.queries],
        }


def load_evaluation_queries(
    manifest_path: str | Path,
    *,
    payload: object | None = None,
) -> list[EvaluationQuery]:
    path = Path(manifest_path)
    if payload is None:
        with path.open(encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
    if not isinstance(payload, list):
        raise ValueError("Evaluation manifest root must be a list")

    queries: list[EvaluationQuery] = []
    for index, raw_query in enumerate(payload, start=1):
        if not isinstance(raw_query, dict):
            raise ValueError("Every evaluation query must be an object")
        image_path = Path(str(raw_query.get("image_path", "")))
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        query_id = str(raw_query.get("query_id") or image_path.stem).strip()
        relevant_item_ids = raw_query.get("relevant_item_ids", [])
        if not isinstance(relevant_item_ids, list):
            raise ValueError("relevant_item_ids must be a list")
        queries.append(
            EvaluationQuery(
                query_id=query_id or f"query-{index}",
                image_path=image_path,
                category=category_from_name(str(raw_query.get("category", ""))),
                relevant_item_ids=frozenset(str(item) for item in relevant_item_ids),
            )
        )
    return queries


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return mean(values)


def evaluate_search_pipeline(
    pipeline: Any,
    queries: Sequence[EvaluationQuery],
    *,
    top_k: int,
    clock: Callable[[], float] = perf_counter,
) -> EvaluationReport:
    _validate_k(top_k)
    query_reports: list[QueryEvaluation] = []

    for query in queries:
        with Image.open(query.image_path) as source_image:
            image = source_image.convert("RGB").copy()
        relevant_item_ids = tuple(sorted(query.relevant_item_ids))
        started_at = clock()
        try:
            response = pipeline.search(
                image,
                category=query.category,
                top_k=top_k,
            )
        except (LookupError, ValueError) as error:
            latency = round(clock() - started_at, 10)
            query_reports.append(
                QueryEvaluation(
                    query_id=query.query_id,
                    category=query.category,
                    retrieved_item_ids=(),
                    relevant_item_ids=relevant_item_ids,
                    precision_at_k=0.0,
                    recall_at_k=0.0,
                    average_precision_at_k=0.0,
                    latency_seconds=latency,
                    error=str(error),
                )
            )
            continue

        latency = round(clock() - started_at, 10)
        retrieved_item_ids = tuple(result.item_id for result in response.results)
        query_reports.append(
            QueryEvaluation(
                query_id=query.query_id,
                category=query.category,
                retrieved_item_ids=retrieved_item_ids,
                relevant_item_ids=relevant_item_ids,
                precision_at_k=precision_at_k(
                    retrieved_item_ids,
                    relevant_item_ids,
                    k=top_k,
                ),
                recall_at_k=recall_at_k(
                    retrieved_item_ids,
                    relevant_item_ids,
                    k=top_k,
                ),
                average_precision_at_k=average_precision_at_k(
                    retrieved_item_ids,
                    relevant_item_ids,
                    k=top_k,
                ),
                latency_seconds=latency,
            )
        )

    return EvaluationReport(
        top_k=top_k,
        queries=tuple(query_reports),
        precision_at_k=_mean_or_zero(
            [query.precision_at_k for query in query_reports]
        ),
        recall_at_k=_mean_or_zero([query.recall_at_k for query in query_reports]),
        mean_average_precision_at_k=_mean_or_zero(
            [query.average_precision_at_k for query in query_reports]
        ),
        latency_seconds=summarize_latency(
            [query.latency_seconds for query in query_reports]
        ),
    )
