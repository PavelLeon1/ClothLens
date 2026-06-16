import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


def load_script_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "create_demo_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "create_demo_data_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_create_demo_data_writes_catalog_and_evaluation_manifest(
    tmp_path: Path,
) -> None:
    module = load_script_module()

    module.main(["--root", str(tmp_path)])

    catalog_dir = tmp_path / "catalog"
    metadata = json.loads((catalog_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata) == 12
    assert {item["category"] for item in metadata} == {
        "top",
        "bottom",
        "dress",
        "shoes",
        "bag",
        "accessories",
    }
    assert (catalog_dir / "images" / "demo-top-blue-shirt.jpg").is_file()
    with Image.open(catalog_dir / "images" / "demo-top-blue-shirt.jpg") as image:
        assert image.size == (384, 512)

    manifest = json.loads(
        (tmp_path / "evaluation" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest) == 6
    assert manifest[0] == {
        "query_id": "query-top-blue-shirt",
        "image_path": "queries/query-top-blue-shirt.jpg",
        "category": "top",
        "relevant_item_ids": ["demo-top-blue-shirt", "demo-top-red-shirt"],
    }
    assert (
        tmp_path / "evaluation" / "queries" / "query-top-blue-shirt.jpg"
    ).is_file()
