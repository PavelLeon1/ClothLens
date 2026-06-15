"""Pretrained clothes SegFormer backend."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from clothing_search.segmentation.base import SegmentationResult
from clothing_search.segmentation.categories import ClothingCategory

DEFAULT_MODEL_NAME = "mattmdjaga/segformer_b2_clothes"

LABEL_CATEGORY_MAP = {
    "upper clothes": ClothingCategory.TOP,
    "upper clothing": ClothingCategory.TOP,
    "shirt": ClothingCategory.TOP,
    "top": ClothingCategory.TOP,
    "skirt": ClothingCategory.BOTTOM,
    "pants": ClothingCategory.BOTTOM,
    "trousers": ClothingCategory.BOTTOM,
    "shorts": ClothingCategory.BOTTOM,
    "dress": ClothingCategory.DRESS,
    "coat": ClothingCategory.OUTERWEAR,
    "jacket": ClothingCategory.OUTERWEAR,
    "outerwear": ClothingCategory.OUTERWEAR,
    "left shoe": ClothingCategory.SHOES,
    "right shoe": ClothingCategory.SHOES,
    "shoe": ClothingCategory.SHOES,
    "shoes": ClothingCategory.SHOES,
    "bag": ClothingCategory.BAG,
    "hat": ClothingCategory.ACCESSORIES,
    "sunglasses": ClothingCategory.ACCESSORIES,
    "belt": ClothingCategory.ACCESSORIES,
    "scarf": ClothingCategory.ACCESSORIES,
}


def _normalize_label(label: str) -> str:
    return " ".join(label.lower().replace("-", " ").replace("_", " ").split())


def source_label_mapping(
    id2label: dict[int | str, str],
) -> dict[int, ClothingCategory]:
    mapping: dict[int, ClothingCategory] = {}
    for source_id, label in id2label.items():
        normalized_label = _normalize_label(label)
        mapping[int(source_id)] = LABEL_CATEGORY_MAP.get(
            normalized_label,
            ClothingCategory.BACKGROUND,
        )
    return mapping


def map_source_mask(
    source_mask: NDArray[np.integer],
    id2label: dict[int | str, str],
) -> NDArray[np.uint8]:
    target_mask = np.zeros(source_mask.shape, dtype=np.uint8)
    for source_id, category in source_label_mapping(id2label).items():
        target_mask[source_mask == source_id] = int(category)
    return target_mask


class PredictionAdapter(Protocol):
    def __call__(
        self,
        logits: Any,
        image_size: tuple[int, int],
    ) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
        """Convert logits to a source mask and confidence map."""


class TorchPredictionAdapter:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.functional = import_module("torch.nn.functional")

    def __call__(
        self,
        logits: Any,
        image_size: tuple[int, int],
    ) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
        resized_logits = self.functional.interpolate(
            logits,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        probabilities = self.torch.softmax(resized_logits, dim=1)
        confidence, source_mask = probabilities.max(dim=1)
        return (
            source_mask[0].detach().cpu().numpy(),
            confidence[0].detach().cpu().numpy(),
        )


class SegFormerSegmenter:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        processor: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
        prediction_adapter: PredictionAdapter | None = None,
    ) -> None:
        if torch_module is None:
            try:
                torch_module = import_module("torch")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "SegFormer dependencies are not installed. Run "
                    r'venv\Scripts\python.exe -m pip install -e ".[search]".'
                ) from error
        self.torch = torch_module
        self.device = device or (
            "cuda" if self.torch.cuda.is_available() else "cpu"
        )

        if processor is None or model is None:
            try:
                transformers = import_module("transformers")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "Transformers is not installed. Run "
                    r'venv\Scripts\python.exe -m pip install -e ".[search]".'
                ) from error
            processor = processor or transformers.AutoImageProcessor.from_pretrained(
                model_name
            )
            model = (
                model
                or transformers.AutoModelForSemanticSegmentation.from_pretrained(
                    model_name
                )
            )

        self.processor = processor
        self.model = model.to(self.device)
        self.model.eval()
        self.prediction_adapter = prediction_adapter or TorchPredictionAdapter(
            self.torch
        )

    def segment(self, image: Image.Image) -> SegmentationResult:
        rgb_image = image.convert("RGB")
        encoded = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in encoded.items()
        }
        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        source_mask, confidence = self.prediction_adapter(
            outputs.logits,
            (rgb_image.height, rgb_image.width),
        )
        target_mask = map_source_mask(source_mask, self.model.config.id2label)
        scores: dict[ClothingCategory, float] = {}
        for category in ClothingCategory:
            if category is ClothingCategory.BACKGROUND:
                continue
            category_pixels = target_mask == int(category)
            if np.any(category_pixels):
                mean_confidence = confidence[category_pixels].mean()
                scores[category] = round(float(mean_confidence), 6)

        return SegmentationResult(mask=target_mask, scores=scores)
