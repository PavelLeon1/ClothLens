"""FashionCLIP image encoder."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

DEFAULT_MODEL_NAME = "patrickjohncyh/fashion-clip"


class FashionEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        processor: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if torch_module is None:
            try:
                torch_module = import_module("torch")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "FashionCLIP dependencies are not installed. Run "
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
            processor = processor or transformers.CLIPProcessor.from_pretrained(
                model_name
            )
            model = model or transformers.CLIPModel.from_pretrained(model_name)

        self.processor = processor
        self.model = model.to(self.device)
        self.model.eval()
        self.embedding_dim = int(
            getattr(self.model.config, "projection_dim", 512)
        )

    @staticmethod
    def _is_valid_image(image: Image.Image | None) -> bool:
        return image is not None and image.width > 0 and image.height > 0

    def encode(self, image: Image.Image | None) -> NDArray[np.float32]:
        return self.encode_batch([image], batch_size=1)[0]

    def encode_batch(
        self,
        images: list[Image.Image | None],
        *,
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        vectors = np.zeros(
            (len(images), self.embedding_dim),
            dtype=np.float32,
        )
        valid_indices = [
            index
            for index, image in enumerate(images)
            if self._is_valid_image(image)
        ]

        for offset in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[offset : offset + batch_size]
            batch_images = [
                images[index].convert("RGB")
                for index in batch_indices
                if images[index] is not None
            ]
            encoded = self.processor(
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in encoded.items()
            }
            with self.torch.inference_mode():
                features = self.model.get_image_features(**inputs)

            batch_vectors = (
                features.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            normalized = np.divide(
                batch_vectors,
                norms,
                out=np.zeros_like(batch_vectors),
                where=norms > 0,
            )
            vectors[batch_indices] = normalized

        return vectors
