from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingStepResult:
    device: str
    steps_completed: int
    batch_size: int
    loss: float
    doppler_channels: int | None = None
    epochs_completed: int = 1
    train_loss: float | None = None
    val_loss: float | None = None
    log_path: str | None = None
    checkpoint_path: str | None = None
