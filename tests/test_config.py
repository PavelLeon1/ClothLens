from pathlib import Path

import pytest

from clothing_search.config import load_app_config, load_training_config


def test_load_app_config_selects_segformer_backend(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        "segmentation:\n"
        "  backend: segformer\n"
        "  model_name: example/model\n",
        encoding="utf-8",
    )

    config = load_app_config(path)

    assert config.segmentation.backend == "segformer"
    assert config.segmentation.model_name == "example/model"


def test_load_app_config_rejects_unknown_backend(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text("segmentation:\n  backend: unknown\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported segmentation backend"):
        load_app_config(path)


def test_training_config_keeps_defaults_for_unspecified_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("model:\n  num_classes: 8\n", encoding="utf-8")

    config = load_training_config(path)

    assert config.model.num_classes == 8
    assert config.model.encoder == "resnet34"
    assert config.data.image_size == 512
    assert config.trainer.max_epochs == 20
