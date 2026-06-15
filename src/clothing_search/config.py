"""Typed configuration loaded from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

SUPPORTED_SEGMENTATION_BACKENDS = frozenset({"segformer", "unet"})


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    backend: str = "segformer"
    model_name: str = "mattmdjaga/segformer_b2_clothes"
    checkpoint_path: str | None = None


@dataclass(frozen=True, slots=True)
class SearchConfig:
    collection_name: str = "clothing_catalog"
    top_k: int = 10


@dataclass(frozen=True, slots=True)
class AppConfig:
    segmentation: SegmentationConfig
    search: SearchConfig


@dataclass(frozen=True, slots=True)
class ModelConfig:
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    num_classes: int = 8
    in_channels: int = 3


@dataclass(frozen=True, slots=True)
class DataConfig:
    root: str = "data/raw/DeepFashion2"
    image_size: int = 512
    batch_size: int = 16
    num_workers: int = 4


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    max_epochs: int = 20
    early_stopping_patience: int = 5
    precision: str = "16-mixed"
    checkpoint_path: str = "models/unet_best.ckpt"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig


ConfigType = TypeVar("ConfigType")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file) or {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")
    return payload


def _build_section(
    config_type: type[ConfigType],
    payload: object,
) -> ConfigType:
    if payload is None:
        return config_type()
    if not isinstance(payload, dict):
        raise ValueError(f"{config_type.__name__} must be a mapping")

    allowed_fields = {field.name for field in fields(config_type)}
    unknown_fields = set(payload) - allowed_fields
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown {config_type.__name__} fields: {names}")
    return config_type(**payload)


def load_app_config(path: str | Path) -> AppConfig:
    payload = _read_yaml(path)
    segmentation = _build_section(
        SegmentationConfig,
        payload.get("segmentation"),
    )
    if segmentation.backend not in SUPPORTED_SEGMENTATION_BACKENDS:
        raise ValueError(
            f"Unsupported segmentation backend: {segmentation.backend}"
        )

    return AppConfig(
        segmentation=segmentation,
        search=_build_section(SearchConfig, payload.get("search")),
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    payload = _read_yaml(path)
    return TrainingConfig(
        model=_build_section(ModelConfig, payload.get("model")),
        data=_build_section(DataConfig, payload.get("data")),
        optimizer=_build_section(OptimizerConfig, payload.get("optimizer")),
        trainer=_build_section(TrainerConfig, payload.get("trainer")),
    )
