from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, cast

import torch
from torch import Tensor, nn

from tasks.utils.training.device import move_to_device
from tasks.utils.training.module import LossFn, ValMetricsFn

TargetExtractor = Callable[[object], Tensor]


class TemporalMeanModel(nn.Module):
    def __init__(self, *, task_kind: str, mean: Tensor, temporal_upsample_factor: int = 2) -> None:
        super().__init__()
        self.task_kind = str(task_kind)
        self.temporal_upsample_factor = int(temporal_upsample_factor)
        values = mean.detach().to(dtype=torch.float32)
        if values.ndim != 3:
            raise ValueError(f"temporal mean must be [channels,height,width], got {tuple(values.shape)}")
        if self.task_kind == "segmentation":
            foreground = values.clamp(1e-4, 1.0 - 1e-4)
            background = (1.0 - foreground.sum(dim=0, keepdim=True)).clamp(1e-4, 1.0 - 1e-4)
            values = torch.log(torch.cat([background, foreground], dim=0))
        self.mean: Tensor
        self.register_buffer("mean", values)

    def forward(self, frames: Tensor, *args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        target_time = self._target_time(frames)
        return self.mean[None, :, None].expand(int(frames.shape[0]), -1, target_time, -1, -1)

    def _target_time(self, frames: Tensor) -> int:
        frame_count = int(frames.shape[2])
        if self.task_kind == "color_doppler":
            return max(1, (frame_count - 1) * self.temporal_upsample_factor)
        if self.task_kind == "tissue_doppler":
            return max(1, frame_count - 1)
        return max(1, frame_count)


class TemporalMeanModule(nn.Module):
    def __init__(
        self,
        *,
        config: object,
        model: TemporalMeanModel,
        loss_fn: LossFn,
        val_metrics_fn: ValMetricsFn | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self._loss_fn = loss_fn
        self._val_metrics_fn = val_metrics_fn

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        return cast(Tensor, self.model(*args, **kwargs))

    def compute_loss_terms(self, sample: object) -> dict[str, Tensor]:
        return self._loss_fn(self.model, sample, self.config)

    def validation_metric_terms(self, sample: object) -> dict[str, Tensor]:
        if self._val_metrics_fn is None:
            return {}
        return self._val_metrics_fn(self.model, sample, self.config)


def fit_temporal_mean(
    train_loader: Iterable[object],
    *,
    task_kind: str,
    target_extractor: TargetExtractor | None = None,
    device: torch.device | str = "cpu",
    temporal_upsample_factor: int = 2,
) -> TemporalMeanModel:
    extractor = target_extractor or default_target_extractor
    total: Tensor | None = None
    count_total: Tensor | None = None
    batches = 0
    for sample in train_loader:
        sample_on_device = move_to_device(sample, torch.device(device))
        target = extractor(sample_on_device).detach().to(dtype=torch.float32)
        if target.ndim != 5:
            raise ValueError(f"baseline target must be [batch,channels,time,height,width], got {tuple(target.shape)}")
        finite = torch.isfinite(target)
        values = torch.where(finite, target, torch.zeros_like(target))
        reduced = values.sum(dim=(0, 2))
        reduced_count = finite.to(dtype=torch.float32).sum(dim=(0, 2))
        total = reduced if total is None else total + reduced
        count_total = reduced_count if count_total is None else count_total + reduced_count
        batches += 1
    if total is None or count_total is None or batches == 0:
        raise ValueError("train_loader produced no samples for temporal mean baseline")
    mean = total / torch.clamp(count_total, min=1.0)
    return TemporalMeanModel(
        task_kind=task_kind,
        mean=mean.cpu(),
        temporal_upsample_factor=temporal_upsample_factor,
    )


def default_target_extractor(sample: object) -> Tensor:
    if hasattr(sample, "doppler_target"):
        return cast(Tensor, getattr(sample, "doppler_target"))
    if hasattr(sample, "target_masks"):
        return cast(Tensor, getattr(sample, "target_masks"))
    raise TypeError(f"Unsupported temporal mean sample type: {type(sample).__name__}")
