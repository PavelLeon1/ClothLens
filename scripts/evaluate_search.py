"""Evaluate ClothLens search quality on a local query manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clothing_search.config import load_app_config
from clothing_search.evaluation import (
    evaluate_search_pipeline,
    load_evaluation_queries,
)
from clothing_search.pipeline import build_search_pipeline


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    config = load_app_config(arguments.config)
    pipeline = build_search_pipeline(config)
    queries = load_evaluation_queries(arguments.manifest)
    report = evaluate_search_pipeline(
        pipeline,
        queries,
        top_k=arguments.top_k,
    )
    payload = json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        indent=2 if arguments.pretty else None,
    )

    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{payload}\n", encoding="utf-8")
    else:
        sys.stdout.write(f"{payload}\n")


if __name__ == "__main__":
    main()
