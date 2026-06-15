"""Explicit U-Net training entry point.

Importing this module does not import PyTorch. Heavy dependencies are loaded
only when ``run_training`` is called.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from clothing_search.config import TrainingConfig, load_training_config
from clothing_search.segmentation.dataset import DeepFashion2Dataset
from clothing_search.segmentation.transforms import (
    build_train_transform,
    build_validation_transform,
)
from clothing_search.segmentation.unet import UnetSettings, build_unet


@dataclass(frozen=True, slots=True)
class CombinedLoss:
    primary: Any
    secondary: Any

    def __call__(self, logits: Any, target: Any) -> Any:
        return self.primary(logits, target) + self.secondary(logits, target)


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    loss_fn: CombinedLoss
    optimizer: Any
    scheduler: Any


def build_training_components(
    model: Any,
    config: TrainingConfig,
    *,
    smp_module: Any | None = None,
    torch_module: Any | None = None,
) -> TrainingComponents:
    if smp_module is None:
        smp_module = import_module("segmentation_models_pytorch")
    if torch_module is None:
        torch_module = import_module("torch")

    loss_fn = CombinedLoss(
        primary=smp_module.losses.DiceLoss(mode="multiclass"),
        secondary=smp_module.losses.SoftCrossEntropyLoss(smooth_factor=0.1),
    )
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch_module.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.trainer.max_epochs,
    )
    return TrainingComponents(
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def run_training(config_path: str | Path = "configs/train.yaml") -> None:
    config = load_training_config(config_path)
    try:
        pl = import_module("pytorch_lightning")
        torch = import_module("torch")
        data_module = import_module("torch.utils.data")
        callbacks = import_module("pytorch_lightning.callbacks")
        torchmetrics = import_module("torchmetrics.classification")
        smp = import_module("segmentation_models_pytorch")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "U-Net training dependencies are not installed. Run "
            r'venv\Scripts\python.exe -m pip install -e ".[ml]".'
        ) from error

    model = build_unet(
        UnetSettings(
            encoder_name=config.model.encoder,
            encoder_weights=config.model.encoder_weights,
            in_channels=config.model.in_channels,
            num_classes=config.model.num_classes,
        ),
        smp_module=smp,
    )
    components = build_training_components(
        model,
        config,
        smp_module=smp,
        torch_module=torch,
    )

    class SegmentationModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.loss_fn = components.loss_fn
            self.validation_iou = torchmetrics.MulticlassJaccardIndex(
                num_classes=config.model.num_classes,
                average="macro",
            )

        def forward(self, images: Any) -> Any:
            return self.model(images)

        def training_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
            logits = self(batch["image"])
            loss = self.loss_fn(logits, batch["mask"])
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            return loss

        def validation_step(
            self,
            batch: dict[str, Any],
            batch_idx: int,
        ) -> None:
            logits = self(batch["image"])
            loss = self.loss_fn(logits, batch["mask"])
            predictions = logits.argmax(dim=1)
            self.validation_iou.update(predictions, batch["mask"])
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        def on_validation_epoch_end(self) -> None:
            mean_iou = self.validation_iou.compute()
            self.log("val_miou", mean_iou, prog_bar=True)
            self.validation_iou.reset()

        def configure_optimizers(self) -> dict[str, Any]:
            return {
                "optimizer": components.optimizer,
                "lr_scheduler": {
                    "scheduler": components.scheduler,
                    "interval": "epoch",
                },
            }

    train_dataset = DeepFashion2Dataset(
        config.data.root,
        split="train",
        transform=build_train_transform(config.data.image_size),
    )
    validation_dataset = DeepFashion2Dataset(
        config.data.root,
        split="validation",
        transform=build_validation_transform(config.data.image_size),
    )
    train_loader = data_module.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    validation_loader = data_module.DataLoader(
        validation_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    checkpoint_path = Path(config.trainer.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoint_path.parent,
        filename=checkpoint_path.stem,
        monitor="val_miou",
        mode="max",
        save_top_k=1,
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_miou",
        mode="max",
        patience=config.trainer.early_stopping_patience,
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=config.trainer.max_epochs,
        precision=config.trainer.precision,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(
        SegmentationModule(),
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the clothing U-Net")
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to the training YAML configuration",
    )
    arguments = parser.parse_args()
    run_training(arguments.config)
