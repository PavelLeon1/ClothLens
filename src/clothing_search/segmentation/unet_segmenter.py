"""Inference wrapper for a trained U-Net checkpoint."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from clothing_search.segmentation.base import SegmentationResult
from clothing_search.segmentation.categories import ClothingCategory
from clothing_search.segmentation.transforms import IMAGENET_MEAN, IMAGENET_STD
from clothing_search.segmentation.unet import UnetSettings, build_unet


def _load_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("U-Net checkpoint must contain a state dict")

    cleaned_state_dict: dict[str, Any] = {}
    for name, value in state_dict.items():
        cleaned_name = str(name)
        if cleaned_name.startswith("model."):
            cleaned_name = cleaned_name.removeprefix("model.")
        cleaned_state_dict[cleaned_name] = value
    return cleaned_state_dict


class UnetSegmenter:
    """Segmenter implementation backed by a trained U-Net checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        image_size: int = 512,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,
        num_classes: int = 8,
        device: str | None = None,
    ) -> None:
        try:
            self.torch = import_module("torch")
            self.functional = import_module("torch.nn.functional")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "U-Net inference dependencies are not installed. Run "
                r'venv\Scripts\python.exe -m pip install -e ".[ml]".'
            ) from error

        self.image_size = image_size
        self.num_classes = num_classes
        self.device = self.torch.device(
            device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        )
        self.model = build_unet(
            UnetSettings(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                num_classes=num_classes,
            )
        )
        try:
            checkpoint = self.torch.load(
                Path(checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = self.torch.load(
                Path(checkpoint_path),
                map_location=self.device,
            )
        self.model.load_state_dict(_load_state_dict(checkpoint))
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, image: Image.Image) -> Any:
        resized = image.convert("RGB").resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
        normalized = (array - mean) / std
        tensor = self.torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def segment(self, image: Image.Image) -> SegmentationResult:
        original_size = image.size
        with self.torch.no_grad():
            logits = self.model(self._to_tensor(image))
            probabilities = self.functional.softmax(logits, dim=1)
            mask = probabilities.argmax(dim=1).squeeze(0)
            resized_mask = self.functional.interpolate(
                mask[None, None].float(),
                size=(original_size[1], original_size[0]),
                mode="nearest",
            ).squeeze()

        mask_array: NDArray[np.integer] = (
            resized_mask.detach().cpu().numpy().astype(np.uint8)
        )
        scores = {}
        for category in ClothingCategory:
            if category is ClothingCategory.BACKGROUND:
                continue
            class_id = int(category)
            if class_id >= self.num_classes:
                continue
            class_pixels = mask == class_id
            if bool(class_pixels.any()):
                class_score = probabilities[0, class_id][class_pixels].mean()
                scores[category] = float(class_score.detach().cpu().item())
        return SegmentationResult(mask=mask_array, scores=scores)
