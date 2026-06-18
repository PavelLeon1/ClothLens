from pathlib import Path

import pytest

from clothing_search.config import load_app_config, load_training_config


def test_load_app_config_selects_segformer_backend(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        "segmentation:\n"
        "  backend: segformer\n"
        "  model_name: example/model\n"
        "embedding:\n"
        "  model_name: example/clip\n"
        "search:\n"
        "  path: example/qdrant\n"
        "  vector_size: 256\n",
        encoding="utf-8",
    )

    config = load_app_config(path)

    assert config.segmentation.backend == "segformer"
    assert config.segmentation.model_name == "example/model"
    assert config.embedding.model_name == "example/clip"
    assert config.search.path == "example/qdrant"
    assert config.search.vector_size == 256


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


def test_training_config_loads_runtime_logging_fields(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text(
        "trainer:\n"
        "  accelerator: gpu\n"
        "  devices: 1\n"
        "  log_dir: results/training_logs\n"
        "  run_name: unet_t4\n"
        "  log_every_n_steps: 5\n"
        "  progress_refresh_rate: 2\n"
        "  gradient_clip_val: 1.0\n"
        "  save_last: true\n",
        encoding="utf-8",
    )

    config = load_training_config(path)

    assert config.trainer.accelerator == "gpu"
    assert config.trainer.devices == 1
    assert config.trainer.log_dir == "results/training_logs"
    assert config.trainer.run_name == "unet_t4"
    assert config.trainer.log_every_n_steps == 5
    assert config.trainer.progress_refresh_rate == 2
    assert config.trainer.gradient_clip_val == 1.0
    assert config.trainer.save_last is True
