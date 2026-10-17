from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from torch import Tensor

from echoxflow import RecordingRecord
from tasks.utils.types import TrainingStepResult as TrainingStepResult

__all__ = ["DenseSectorSample", "Sample", "TrainingStepResult"]


@dataclass
class DenseSectorSample:
    task_kind: ClassVar[Literal["tissue_doppler"]] = "tissue_doppler"

    frames: Tensor
    frame_timestamps: Tensor
    doppler_timestamps: Tensor
    doppler_target: Tensor
    sector_velocity_limit_mps: Tensor
    velocity_scale_mps_per_px_frame: Tensor
    marker_target: Tensor | None = None
    sample_id: str = "sample"
    record: RecordingRecord | None = None
    data_root: str | Path | None = None
    clip_start: int | None = None
    clip_stop: int | None = None
    coordinate_space: str | None = None
    cartesian_metric_doppler_target: Tensor | None = None
    cartesian_metric_sample_grid: Tensor | None = None
    # [1,1,1,H,W] metric-grid pixels inside the scan sector; outside it the Cartesian target is fill, not data.
    cartesian_metric_valid_mask: Tensor | None = None
    # [1,1,1,H,W] training-target pixels inside the scan sector (Cartesian domain only); None trains on every pixel.
    loss_valid_mask: Tensor | None = None
    # [1,T] exported velocity scale (m/s) of each target frame. Clips only span frames sharing one calibrated scale,
    # so every entry equals sector_velocity_limit_mps; it is kept to make the calibration of each frame explicit.
    doppler_velocity_limit_mps_per_frame: Tensor | None = None


Sample = DenseSectorSample
