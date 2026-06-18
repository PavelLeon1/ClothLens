"""Compare two ClothLens evaluation reports and write a Markdown summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METRIC_NAMES = (
    "precision_at_k",
    "recall_at_k",
    "mean_average_precision_at_k",
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline-name", default="Baseline")
    parser.add_argument("--current-name", default="Current")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def load_summary(path: str | Path) -> dict[str, float]:
    with Path(path).open(encoding="utf-8") as report_file:
        payload = json.load(report_file)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Report has no summary object: {path}")
    return {name: float(summary.get(name, 0.0)) for name in METRIC_NAMES}


def format_delta(value: float) -> str:
    return f"{value:+.4f}"


def render_markdown(
    *,
    baseline: dict[str, float],
    current: dict[str, float],
    baseline_name: str,
    current_name: str,
) -> str:
    lines = [
        "# Search Report Comparison",
        "",
        f"Baseline: `{baseline_name}`",
        "",
        f"Current: `{current_name}`",
        "",
        f"| Metric | {baseline_name} | {current_name} | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name in METRIC_NAMES:
        baseline_value = baseline[name]
        current_value = current[name]
        lines.append(
            "| "
            f"{name} | {baseline_value:.4f} | {current_value:.4f} | "
            f"{format_delta(current_value - baseline_value)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    markdown = render_markdown(
        baseline=load_summary(arguments.baseline),
        current=load_summary(arguments.current),
        baseline_name=arguments.baseline_name,
        current_name=arguments.current_name,
    )
    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
