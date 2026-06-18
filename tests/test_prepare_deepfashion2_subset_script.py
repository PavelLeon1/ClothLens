import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


def load_script_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "prepare_deepfashion2_subset.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_deepfashion2_subset_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_deepfashion2_sample(
    root: Path,
    *,
    split: str,
    sample_id: str,
    category_id: int,
) -> None:
    image_dir = root / split / "image"
    annotation_dir = root / split / "annos"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{sample_id}.jpg").write_bytes(b"fake image")
    annotation = {
        "source": "user",
        "pair_id": int(sample_id),
        "item1": {
            "category_id": category_id,
            "segmentation": [[1, 1, 8, 1, 8, 8, 1, 8]],
        },
    }
    (annotation_dir / f"{sample_id}.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )


def create_source_dataset(root: Path) -> None:
    category_ids = [1, 7, 10, 3]
    for split, multiplier in (("train", 100), ("validation", 200)):
        for category_index, category_id in enumerate(category_ids, start=1):
            for item_index in range(3):
                write_deepfashion2_sample(
                    root,
                    split=split,
                    sample_id=str(multiplier + category_index * 10 + item_index),
                    category_id=category_id,
                )


def test_prepare_subset_copies_balanced_pairs_and_manifest(tmp_path: Path) -> None:
    module = load_script_module()
    source = tmp_path / "DeepFashion2"
    output = tmp_path / "DeepFashion2_subset"
    create_source_dataset(source)

    module.main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--train-size",
            "8",
            "--validation-size",
            "4",
            "--seed",
            "123",
        ]
    )

    train_images = sorted((output / "train" / "image").glob("*.jpg"))
    validation_images = sorted((output / "validation" / "image").glob("*.jpg"))
    assert len(train_images) == 8
    assert len(validation_images) == 4
    for image_path in train_images + validation_images:
        annotation_path = (
            image_path.parents[1] / "annos" / f"{image_path.stem}.json"
        )
        assert annotation_path.is_file()

    manifest = json.loads(
        (output / "subset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["splits"]["train"]["selected_count"] == 8
    assert manifest["splits"]["validation"]["selected_count"] == 4
    assert manifest["splits"]["train"]["assigned_category_counts"] == {
        "bottom": 2,
        "dress": 2,
        "outerwear": 2,
        "top": 2,
    }
    assert manifest["splits"]["validation"]["assigned_category_counts"] == {
        "bottom": 1,
        "dress": 1,
        "outerwear": 1,
        "top": 1,
    }


def test_prepare_subset_cleans_output_and_writes_archive(tmp_path: Path) -> None:
    module = load_script_module()
    source = tmp_path / "DeepFashion2"
    output = tmp_path / "DeepFashion2_subset"
    archive = tmp_path / "DeepFashion2_subset.zip"
    create_source_dataset(source)
    stale_file = output / "train" / "image" / "stale.jpg"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_bytes(b"old")

    module.main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--train-size",
            "4",
            "--validation-size",
            "4",
            "--archive",
            str(archive),
            "--clean",
        ]
    )

    assert not stale_file.exists()
    assert archive.is_file()
    with zipfile.ZipFile(archive) as zip_file:
        names = set(zip_file.namelist())
    assert "DeepFashion2_subset/subset_manifest.json" in names
    assert any(name.startswith("DeepFashion2_subset/train/image/") for name in names)
    assert any(
        name.startswith("DeepFashion2_subset/validation/annos/") for name in names
    )
