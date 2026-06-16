import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from clothing_search.search.models import SearchResult
from clothing_search.segmentation.categories import ClothingCategory


def load_script_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "evaluate_search.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_search_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakePipeline:
    def search(
        self,
        image: Image.Image,
        *,
        category: ClothingCategory,
        top_k: int,
    ) -> Any:
        assert image.size == (4, 4)
        assert category is ClothingCategory.TOP
        assert top_k == 2
        return SimpleNamespace(
            results=[
                SearchResult(item_id="sku-x", score=0.7, metadata={}),
                SearchResult(item_id="sku-1", score=0.6, metadata={}),
            ]
        )


def test_evaluate_search_script_writes_json_report(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = load_script_module()
    query_image = tmp_path / "query.jpg"
    Image.new("RGB", (4, 4), color=(100, 80, 60)).save(query_image)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "image_path": str(query_image),
                    "category": "top",
                    "relevant_item_ids": ["sku-1"],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "results" / "report.json"
    monkeypatch.setattr(module, "load_app_config", lambda path: object())
    monkeypatch.setattr(module, "build_search_pipeline", lambda config: FakePipeline())

    module.main(
        [
            "--manifest",
            str(manifest),
            "--config",
            "configs/app.yaml",
            "--top-k",
            "2",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["top_k"] == 2
    assert payload["query_count"] == 1
    assert payload["summary"]["precision_at_k"] == 0.5
    assert payload["summary"]["recall_at_k"] == 1.0
    assert payload["summary"]["mean_average_precision_at_k"] == 0.5
