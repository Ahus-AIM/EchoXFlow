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
    task_kind: ClassVar[Literal["segmentation"]] = "segmentation"

    frames: Tensor
    target_masks: Tensor
    valid_mask: Tensor | None = None
    target_mask_valid: Tensor | None = None
    target_absent_masks: Tensor | None = None
    conditioning: Tensor | None = None
    sample_id: str = "sample"
    record: RecordingRecord | None = None
    data_root: str | Path | None = None
    clip_start: int | None = None
    clip_stop: int | None = None
    role_id: str | None = None
    coordinate_space: str | None = None
    cartesian_metric_target_masks: Tensor | None = None
    cartesian_metric_valid_mask: Tensor | None = None
    cartesian_metric_sample_grid: Tensor | None = None
