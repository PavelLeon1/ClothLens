"""Explicit U-Net training entry point.

Importing this module does not import PyTorch. Heavy dependencies are loaded
only when ``run_training`` is called.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from math import ceil
from pathlib import Path
from time import perf_counter
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
        if hasattr(target, "long"):
            target = target.long()
        return self.primary(logits, target) + self.secondary(logits, target)


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    loss_fn: CombinedLoss
    optimizer: Any
    scheduler: Any


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    data_root: str
    image_size: int
    batch_size: int
    train_size: int
    validation_size: int
    steps_per_epoch: int
    validation_steps: int
    total_train_steps: int
    max_epochs: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "data_root": self.data_root,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "steps_per_epoch": self.steps_per_epoch,
            "validation_steps": self.validation_steps,
            "total_train_steps": self.total_train_steps,
            "max_epochs": self.max_epochs,
        }


def _ceil_div(value: int, divisor: int) -> int:
    if divisor < 1:
        raise ValueError("divisor must be positive")
    return ceil(value / divisor) if value else 0


def summarize_training_run(
    config: TrainingConfig,
    *,
    train_size: int,
    validation_size: int,
    max_epochs: int | None = None,
) -> TrainingRunSummary:
    resolved_max_epochs = (
        config.trainer.max_epochs if max_epochs is None else max_epochs
    )
    steps_per_epoch = _ceil_div(train_size, config.data.batch_size)
    validation_steps = _ceil_div(validation_size, config.data.batch_size)
    return TrainingRunSummary(
        data_root=config.data.root,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        train_size=train_size,
        validation_size=validation_size,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        total_train_steps=steps_per_epoch * resolved_max_epochs,
        max_epochs=resolved_max_epochs,
    )


def print_training_summary(summary: TrainingRunSummary) -> None:
    print("U-Net training summary")
    for name, value in summary.to_dict().items():
        print(f"  {name}: {value}")


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


def run_training(
    config_path: str | Path = "configs/train.yaml",
    *,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    max_epochs: int | None = None,
    limit_train_batches: float | int | None = None,
    limit_val_batches: float | int | None = None,
) -> None:
    config = load_training_config(config_path)
    resolved_max_epochs = (
        config.trainer.max_epochs if max_epochs is None else max_epochs
    )
    try:
        pl = import_module("pytorch_lightning")
        torch = import_module("torch")
        data_module = import_module("torch.utils.data")
        callbacks = import_module("pytorch_lightning.callbacks")
        loggers = import_module("pytorch_lightning.loggers")
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
    summary = summarize_training_run(
        config,
        train_size=len(train_dataset),
        validation_size=len(validation_dataset),
        max_epochs=resolved_max_epochs,
    )
    print_training_summary(summary)
    if dry_run:
        print("Dry run requested; datasets are readable, training was not started.")
        return

    train_loader = data_module.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
    )
    validation_loader = data_module.DataLoader(
        validation_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
    )

    checkpoint_path = Path(config.trainer.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoint_path.parent,
        filename=checkpoint_path.stem,
        monitor="val_miou",
        mode="max",
        save_top_k=1,
        save_last=config.trainer.save_last,
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_miou",
        mode="max",
        patience=config.trainer.early_stopping_patience,
    )

    class EpochTimingCallback(callbacks.Callback):
        def __init__(self) -> None:
            self.started_at = 0.0
            self.durations: list[float] = []

        def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
            self.started_at = perf_counter()

        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            duration = perf_counter() - self.started_at
            self.durations.append(duration)
            average_duration = sum(self.durations) / len(self.durations)
            remaining_epochs = max(trainer.max_epochs - trainer.current_epoch - 1, 0)
            eta_minutes = (average_duration * remaining_epochs) / 60
            print(
                "Epoch "
                f"{trainer.current_epoch + 1}/{trainer.max_epochs} finished in "
                f"{duration / 60:.1f} min; ETA: {eta_minutes:.1f} min"
            )

    csv_logger = loggers.CSVLogger(
        save_dir=config.trainer.log_dir,
        name=config.trainer.run_name,
    )
    callback_list = [
        checkpoint_callback,
        early_stopping,
        callbacks.TQDMProgressBar(refresh_rate=config.trainer.progress_refresh_rate),
        callbacks.LearningRateMonitor(logging_interval="epoch"),
        EpochTimingCallback(),
    ]
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        max_epochs=resolved_max_epochs,
        precision=config.trainer.precision,
        callbacks=callback_list,
        logger=csv_logger,
        log_every_n_steps=config.trainer.log_every_n_steps,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        deterministic=config.trainer.deterministic,
        limit_train_batches=1.0 if limit_train_batches is None else limit_train_batches,
        limit_val_batches=1.0 if limit_val_batches is None else limit_val_batches,
    )
    resume_path = resume_from_checkpoint or config.trainer.resume_from_checkpoint
    trainer.fit(
        SegmentationModule(),
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=resume_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the clothing U-Net")
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to the training YAML configuration",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--limit-train-batches", type=float)
    parser.add_argument("--limit-val-batches", type=float)
    arguments = parser.parse_args()
    run_training(
        arguments.config,
        dry_run=arguments.dry_run,
        resume_from_checkpoint=arguments.resume_from_checkpoint,
        max_epochs=arguments.max_epochs,
        limit_train_batches=arguments.limit_train_batches,
        limit_val_batches=arguments.limit_val_batches,
    )
