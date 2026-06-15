from typing import Any, ClassVar

from clothing_search.segmentation.unet import UnetSettings, build_unet


class FakeSegmentationModels:
    calls: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def Unet(cls, **kwargs: Any) -> dict[str, Any]:
        cls.calls.append(kwargs)
        return kwargs


def test_build_unet_uses_required_practice_project_architecture() -> None:
    FakeSegmentationModels.calls.clear()

    model = build_unet(
        UnetSettings(),
        smp_module=FakeSegmentationModels,
    )

    assert model == {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 8,
    }
    assert FakeSegmentationModels.calls == [model]


def test_build_unet_accepts_explicit_settings() -> None:
    settings = UnetSettings(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=1,
        num_classes=4,
    )

    model = build_unet(settings, smp_module=FakeSegmentationModels)

    assert model["encoder_name"] == "efficientnet-b0"
    assert model["encoder_weights"] is None
    assert model["in_channels"] == 1
    assert model["classes"] == 4
