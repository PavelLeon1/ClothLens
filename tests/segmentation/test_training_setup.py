from pathlib import Path
from typing import Any

from clothing_search.config import load_training_config
from clothing_search.segmentation.train import (
    CombinedLoss,
    build_training_components,
    run_training,
)


class FakeLoss:
    def __init__(self, value: float, **settings: Any) -> None:
        self.value = value
        self.settings = settings

    def __call__(self, logits: object, target: object) -> float:
        return self.value


class FakeLosses:
    @staticmethod
    def DiceLoss(**settings: Any) -> FakeLoss:
        return FakeLoss(0.25, **settings)

    @staticmethod
    def SoftCrossEntropyLoss(**settings: Any) -> FakeLoss:
        return FakeLoss(0.75, **settings)


class FakeSmp:
    losses = FakeLosses()


class FakeOptimizer:
    def __init__(self, parameters: object, **settings: Any) -> None:
        self.parameters = list(parameters)
        self.settings = settings


class FakeScheduler:
    def __init__(self, optimizer: FakeOptimizer, **settings: Any) -> None:
        self.optimizer = optimizer
        self.settings = settings


class FakeOptim:
    AdamW = FakeOptimizer

    class lr_scheduler:
        CosineAnnealingLR = FakeScheduler


class FakeTorch:
    optim = FakeOptim()


class FakeModel:
    @staticmethod
    def parameters() -> list[str]:
        return ["weight"]


def test_combined_loss_adds_dice_and_cross_entropy() -> None:
    loss = CombinedLoss(FakeLoss(0.25), FakeLoss(0.75))

    assert loss(object(), object()) == 1.0


def test_training_components_follow_yaml_settings(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text(
        "optimizer:\n"
        "  learning_rate: 0.0002\n"
        "  weight_decay: 0.00003\n"
        "trainer:\n"
        "  max_epochs: 12\n",
        encoding="utf-8",
    )
    config = load_training_config(path)

    components = build_training_components(
        FakeModel(),
        config,
        smp_module=FakeSmp,
        torch_module=FakeTorch,
    )

    assert components.loss_fn.primary.settings == {"mode": "multiclass"}
    assert components.loss_fn.secondary.settings == {"smooth_factor": 0.1}
    assert components.optimizer.parameters == ["weight"]
    assert components.optimizer.settings == {
        "lr": 0.0002,
        "weight_decay": 0.00003,
    }
    assert components.scheduler.optimizer is components.optimizer
    assert components.scheduler.settings == {"T_max": 12}


def test_run_training_reports_missing_ml_dependencies(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("{}\n", encoding="utf-8")

    def missing_import(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "clothing_search.segmentation.train.import_module",
        missing_import,
    )

    try:
        run_training(path)
    except RuntimeError as error:
        assert ".[ml]" in str(error)
    else:
        raise AssertionError("Expected a missing dependency error")
