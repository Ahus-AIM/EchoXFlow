from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from torch import Tensor

from echoxflow import RecordingRecord
from tasks.utils.types import TrainingStepResult as TrainingStepResult

__all__ = ["Sample", "TrainingStepResult"]


@dataclass
class Sample:
    task_kind: ClassVar[Literal["color_doppler"]] = "color_doppler"

    frames: Tensor
    frame_timestamps: Tensor
    target_timestamps: Tensor
    conditioning: Tensor
    doppler_target: Tensor
    color_box_target: Tensor
    valid_mask: Tensor
    velocity_loss_mask: Tensor
    nyquist_mps: Tensor
    sample_id: str = "sample"
    record: RecordingRecord | None = None
    data_root: str | Path | None = None
    clip_start: int | None = None
    clip_stop: int | None = None
    coordinate_space: str | None = None
    cartesian_metric_doppler_target: Tensor | None = None
    cartesian_metric_valid_mask: Tensor | None = None
    cartesian_metric_velocity_loss_mask: Tensor | None = None
    cartesian_metric_color_box_target: Tensor | None = None
    cartesian_metric_sample_grid: Tensor | None = None
