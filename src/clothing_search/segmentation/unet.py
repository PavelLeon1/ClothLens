"""U-Net model construction without import-time ML dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True, slots=True)
class UnetSettings:
    encoder_name: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 3
    num_classes: int = 8


def build_unet(
    settings: UnetSettings | None = None,
    *,
    smp_module: Any | None = None,
) -> Any:
    resolved_settings = settings or UnetSettings()
    if smp_module is None:
        try:
            smp_module = import_module("segmentation_models_pytorch")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "U-Net dependencies are not installed. Run "
                r'venv\Scripts\python.exe -m pip install -e ".[ml]".'
            ) from error

    return smp_module.Unet(
        encoder_name=resolved_settings.encoder_name,
        encoder_weights=resolved_settings.encoder_weights,
        in_channels=resolved_settings.in_channels,
        classes=resolved_settings.num_classes,
    )
