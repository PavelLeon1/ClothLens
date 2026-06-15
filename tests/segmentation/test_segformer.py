from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from clothing_search.segmentation.categories import ClothingCategory
from clothing_search.segmentation.segformer import (
    SegFormerSegmenter,
    map_source_mask,
)


def test_atr_labels_collapse_to_application_categories() -> None:
    source_mask = np.array(
        [
            [0, 4, 5, 6],
            [7, 9, 10, 16],
            [1, 3, 8, 17],
            [2, 11, 14, 15],
        ],
        dtype=np.uint8,
    )
    labels = {
        0: "Background",
        1: "Hat",
        2: "Hair",
        3: "Sunglasses",
        4: "Upper-clothes",
        5: "Skirt",
        6: "Pants",
        7: "Dress",
        8: "Belt",
        9: "Left-shoe",
        10: "Right-shoe",
        11: "Face",
        14: "Left-arm",
        15: "Right-arm",
        16: "Bag",
        17: "Scarf",
    }

    mapped = map_source_mask(source_mask, labels)

    assert mapped.tolist() == [
        [0, 1, 2, 2],
        [3, 5, 5, 6],
        [7, 7, 7, 7],
        [0, 0, 0, 0],
    ]


class FakeInput:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> "FakeInput":
        self.device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.image: Image.Image | None = None
        self.input = FakeInput()

    def __call__(
        self,
        *,
        images: Image.Image,
        return_tensors: str,
    ) -> dict[str, FakeInput]:
        self.image = images
        assert return_tensors == "pt"
        return {"pixel_values": self.input}


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            id2label={0: "Background", 4: "Upper-clothes", 7: "Dress"}
        )
        self.device: str | None = None
        self.eval_called = False
        self.inputs: dict[str, Any] | None = None

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        self.inputs = inputs
        return SimpleNamespace(logits="raw-logits")


class FakeInferenceMode:
    def __init__(self, torch_module: "FakeTorch") -> None:
        self.torch_module = torch_module

    def __enter__(self) -> None:
        self.torch_module.inside_inference = True

    def __exit__(self, *args: object) -> None:
        self.torch_module.inside_inference = False


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    cuda = FakeCuda()

    def __init__(self) -> None:
        self.inside_inference = False
        self.entered_inference = False

    def inference_mode(self) -> FakeInferenceMode:
        self.entered_inference = True
        return FakeInferenceMode(self)


class FakePredictionAdapter:
    def __init__(self) -> None:
        self.call: tuple[Any, tuple[int, int]] | None = None

    def __call__(
        self,
        logits: Any,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        self.call = logits, image_size
        return (
            np.array([[4, 4, 0], [7, 0, 0]], dtype=np.uint8),
            np.array([[0.8, 0.6, 0.9], [0.75, 0.95, 0.9]], dtype=np.float32),
        )


def test_segmenter_runs_inference_and_returns_original_size_mask() -> None:
    processor = FakeProcessor()
    model = FakeModel()
    torch_module = FakeTorch()
    adapter = FakePredictionAdapter()
    segmenter = SegFormerSegmenter(
        processor=processor,
        model=model,
        torch_module=torch_module,
        prediction_adapter=adapter,
    )
    image = Image.new("RGBA", (3, 2), color=(10, 20, 30, 128))

    result = segmenter.segment(image)

    assert processor.image is not None
    assert processor.image.mode == "RGB"
    assert processor.input.device == "cpu"
    assert model.device == "cpu"
    assert model.eval_called
    assert torch_module.entered_inference
    assert adapter.call == ("raw-logits", (2, 3))
    assert result.mask.tolist() == [[1, 1, 0], [3, 0, 0]]
    assert result.scores[ClothingCategory.TOP] == np.float32(0.7)
    assert result.scores[ClothingCategory.DRESS] == np.float32(0.75)
