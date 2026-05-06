from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from echoxflow import RecordingRecord
from tasks.utils.types import TrainingStepResult as TrainingStepResult

__all__ = ["Sample", "TrainingStepResult"]


@dataclass
class Sample:
    task_kind: ClassVar[Literal["template_task"]] = "template_task"

    # TODO: Add task input tensors, target tensors, masks, and metadata.
    sample_id: str = "sample"
    record: RecordingRecord | None = None
    data_root: str | Path | None = None
