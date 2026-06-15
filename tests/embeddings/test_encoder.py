from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from clothing_search.embeddings.encoder import FashionEncoder


class FakeInput:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.device: str | None = None

    def to(self, device: str) -> "FakeInput":
        self.device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.batches: list[list[Image.Image]] = []
        self.inputs: list[FakeInput] = []

    def __call__(
        self,
        *,
        images: list[Image.Image],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, FakeInput]:
        assert return_tensors == "pt"
        assert padding
        self.batches.append(images)
        current_input = FakeInput(len(images))
        self.inputs.append(current_input)
        return {"pixel_values": current_input}


class FakeFeatureTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> "FakeFeatureTensor":
        return self

    def cpu(self) -> "FakeFeatureTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(projection_dim=3)
        self.device: str | None = None
        self.eval_called = False
        self.calls = 0

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def get_image_features(self, **inputs: Any) -> FakeFeatureTensor:
        self.calls += 1
        batch_size = inputs["pixel_values"].batch_size
        values = np.array(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]][:batch_size],
            dtype=np.float32,
        )
        return FakeFeatureTensor(values)


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeInferenceMode:
    def __init__(self, torch_module: "FakeTorch") -> None:
        self.torch_module = torch_module

    def __enter__(self) -> None:
        self.torch_module.entered += 1

    def __exit__(self, *args: object) -> None:
        return None


class FakeTorch:
    cuda = FakeCuda()

    def __init__(self) -> None:
        self.entered = 0

    def inference_mode(self) -> FakeInferenceMode:
        return FakeInferenceMode(self)


def test_encode_batch_normalizes_vectors_and_uses_batches() -> None:
    processor = FakeProcessor()
    model = FakeModel()
    torch_module = FakeTorch()
    encoder = FashionEncoder(
        processor=processor,
        model=model,
        torch_module=torch_module,
    )
    images = [
        Image.new("RGBA", (4, 4)),
        Image.new("RGB", (4, 4)),
        Image.new("RGB", (4, 4)),
    ]

    vectors = encoder.encode_batch(images, batch_size=2)

    assert vectors.shape == (3, 3)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], [0.6, 0.8, 0.0])
    assert [len(batch) for batch in processor.batches] == [2, 1]
    assert all(image.mode == "RGB" for batch in processor.batches for image in batch)
    assert all(current.device == "cpu" for current in processor.inputs)
    assert model.device == "cpu"
    assert model.eval_called
    assert torch_module.entered == 2


def test_encode_returns_single_vector() -> None:
    encoder = FashionEncoder(
        processor=FakeProcessor(),
        model=FakeModel(),
        torch_module=FakeTorch(),
    )

    vector = encoder.encode(Image.new("RGB", (4, 4)))

    assert vector.shape == (3,)
    assert np.linalg.norm(vector) == np.float32(1.0)


def test_invalid_images_keep_position_as_zero_vectors() -> None:
    processor = FakeProcessor()
    model = FakeModel()
    encoder = FashionEncoder(
        processor=processor,
        model=model,
        torch_module=FakeTorch(),
    )

    vectors = encoder.encode_batch(
        [None, Image.new("RGB", (4, 4)), Image.new("RGB", (0, 0))],
        batch_size=2,
    )

    assert vectors.shape == (3, 3)
    assert vectors[0].tolist() == [0.0, 0.0, 0.0]
    assert np.linalg.norm(vectors[1]) == np.float32(1.0)
    assert vectors[2].tolist() == [0.0, 0.0, 0.0]
    assert len(processor.batches) == 1
