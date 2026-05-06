from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import torch
from torch import nn

from echoxflow.plotting import PlotStyle, PlotViewMode
from tasks.utils.optim import AdamMuon

SampleT = TypeVar("SampleT")
_LOG_LOSS_EMA_GAMMA = 0.99
_PREVIEW_FPS = 4.0
_PREVIEW_MAX_FPS = 60.0
_PREVIEW_VIEW_MODE: PlotViewMode = "both"
_PREVIEW_STYLE = PlotStyle(width_px=1440, height_px=1080, dpi=100)
_COLOR_POWER_FLOOR = 0.05


@dataclass(frozen=True)
class FitResult:
    device: str
    epochs_completed: int
    steps_completed: int
    batch_size: int
    loss: float
    train_loss: float
    val_loss: float | None
    log_path: str | None
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class EpochMetrics:
    task: str
    epoch: int
    train_loss: float
    val_loss: float | None
    train_loss_terms: Mapping[str, float]
    val_loss_terms: Mapping[str, float] | None
    train_steps: int
    val_steps: int
    elapsed_seconds: float
    device: str
    precision: str


@dataclass(frozen=True)
class BatchMetrics:
    task: str
    split: str
    epoch: int
    step: int
    loss: float
    loss_terms: Mapping[str, float]
    batch_size: int | None
    load_seconds: float
    step_seconds: float
    device: str
    precision: str


@dataclass(frozen=True)
class AmpConfig:
    precision: str
    enabled: bool
    dtype: torch.dtype
    scaler_enabled: bool


def configure_training_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_optimizer(module: nn.Module, optimizer_config: Mapping[str, object]) -> torch.optim.Optimizer:
    name = str(optimizer_config.get("name", "adam")).lower()
    lr = float(cast(Any, optimizer_config.get("learning_rate", 1e-3)))
    weight_decay = float(cast(Any, optimizer_config.get("weight_decay", 0.0)))
    if name == "adam":
        return torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adammuon":
        betas_config = cast(Any, optimizer_config.get("betas", (0.9, 0.999)))
        betas = (float(betas_config[0]), float(betas_config[1]))
        return AdamMuon(
            module.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=float(cast(Any, optimizer_config.get("eps", 1e-8))),
            muon_momentum=float(cast(Any, optimizer_config.get("muon_momentum", 0.95))),
        )
    raise ValueError(f"Unsupported optimizer {name!r}; expected 'adam' or 'adammuon'")
