import pytest

from clothing_search.evaluation import (
    average_precision_at_k,
    mean_average_precision_at_k,
    precision_at_k,
    recall_at_k,
    summarize_latency,
)


def test_precision_and_recall_at_k_use_top_k_window() -> None:
    retrieved = ["sku-1", "sku-x", "sku-2", "sku-3"]
    relevant = {"sku-1", "sku-2", "sku-9"}

    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_average_precision_at_k_scores_hit_ranks() -> None:
    retrieved = ["sku-1", "sku-x", "sku-2", "sku-3"]
    relevant = {"sku-1", "sku-2", "sku-9"}

    assert average_precision_at_k(retrieved, relevant, k=3) == pytest.approx(
        (1.0 + 2 / 3) / 3
    )


def test_mean_average_precision_at_k_averages_queries() -> None:
    queries = [
        (["sku-1", "sku-2"], {"sku-1", "sku-2"}),
        (["sku-x", "sku-3"], {"sku-3"}),
    ]

    assert mean_average_precision_at_k(queries, k=2) == pytest.approx(
        (1.0 + 0.5) / 2
    )


def test_metrics_return_zero_when_no_relevant_items() -> None:
    assert precision_at_k(["sku-1"], set(), k=2) == 0.0
    assert recall_at_k(["sku-1"], set(), k=2) == 0.0
    assert average_precision_at_k(["sku-1"], set(), k=2) == 0.0
    assert mean_average_precision_at_k([], k=2) == 0.0


def test_metrics_reject_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(["sku-1"], {"sku-1"}, k=0)


def test_summarize_latency_reports_core_statistics() -> None:
    summary = summarize_latency([0.4, 0.1, 0.2])

    assert summary == {
        "count": 3,
        "min": 0.1,
        "max": 0.4,
        "mean": pytest.approx(0.2333333333),
        "p50": 0.2,
        "p95": 0.4,
    }


def test_summarize_latency_handles_empty_input() -> None:
    assert summarize_latency([]) == {
        "count": 0,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "p50": 0.0,
        "p95": 0.0,
    }
