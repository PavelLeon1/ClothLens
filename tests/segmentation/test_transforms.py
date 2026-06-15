from typing import Any

from clothing_search.segmentation.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_train_transform,
    build_validation_transform,
)


class FakeAlbumentations:
    @staticmethod
    def _operation(name: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return name, kwargs

    @classmethod
    def Resize(cls, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return cls._operation("Resize", **kwargs)

    @classmethod
    def HorizontalFlip(cls, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return cls._operation("HorizontalFlip", **kwargs)

    @classmethod
    def ColorJitter(cls, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return cls._operation("ColorJitter", **kwargs)

    @classmethod
    def Normalize(cls, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return cls._operation("Normalize", **kwargs)

    @staticmethod
    def Compose(operations: list[Any]) -> list[Any]:
        return operations


class FakeToTensor:
    def __new__(cls) -> tuple[str, dict[str, Any]]:
        return "ToTensorV2", {}


def test_train_transform_resizes_augments_normalizes_and_tensorizes() -> None:
    operations = build_train_transform(
        image_size=256,
        albumentations_module=FakeAlbumentations,
        to_tensor_cls=FakeToTensor,
    )

    assert [operation[0] for operation in operations] == [
        "Resize",
        "HorizontalFlip",
        "ColorJitter",
        "Normalize",
        "ToTensorV2",
    ]
    assert operations[0][1] == {"height": 256, "width": 256}
    assert operations[3][1]["mean"] == IMAGENET_MEAN
    assert operations[3][1]["std"] == IMAGENET_STD


def test_validation_transform_is_deterministic() -> None:
    operations = build_validation_transform(
        image_size=512,
        albumentations_module=FakeAlbumentations,
        to_tensor_cls=FakeToTensor,
    )

    assert [operation[0] for operation in operations] == [
        "Resize",
        "Normalize",
        "ToTensorV2",
    ]
