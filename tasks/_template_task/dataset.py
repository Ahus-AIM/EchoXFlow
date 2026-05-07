from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from echoxflow import RecordingRecord
from tasks._template_task.types import Sample


@dataclass(frozen=True)
class SampleRef:
    # TODO: Add enough metadata to locate one training sample.
    record: RecordingRecord | None = None
    sample_id: str = "sample"


class RawDataset(Dataset[Sample]):
    # TODO: Implement task-specific discovery, filtering, and sample construction.
    pass


def build_dataloaders(
    *,
    config: object,
    data_root: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    # TODO: Return train and validation dataloaders for this task.
    raise NotImplementedError("_template_task dataloaders are not implemented")
