import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def load_script_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "compare_search_reports.py"
    )
    spec = importlib.util.spec_from_file_location("compare_search_reports", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_report(path: Path, precision: float, recall: float, map_score: float) -> None:
    path.write_text(
        json.dumps(
            {
                "top_k": 2,
                "query_count": 6,
                "summary": {
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "mean_average_precision_at_k": map_score,
                    "latency_seconds": {"mean": 0.25},
                },
            }
        ),
        encoding="utf-8",
    )


def test_compare_search_reports_writes_markdown(tmp_path: Path) -> None:
    module = load_script_module()
    baseline = tmp_path / "segformer.json"
    current = tmp_path / "unet.json"
    output = tmp_path / "comparison.md"
    write_report(baseline, precision=0.8, recall=0.7, map_score=0.75)
    write_report(current, precision=0.9, recall=0.65, map_score=0.8)

    module.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--baseline-name",
            "SegFormer",
            "--current-name",
            "U-Net",
            "--output",
            str(output),
        ]
    )

    markdown = output.read_text(encoding="utf-8")
    assert "| Metric | SegFormer | U-Net | Delta |" in markdown
    assert "| precision_at_k | 0.8000 | 0.9000 | +0.1000 |" in markdown
    assert "| recall_at_k | 0.7000 | 0.6500 | -0.0500 |" in markdown
    assert "| mean_average_precision_at_k | 0.7500 | 0.8000 | +0.0500 |" in markdown
