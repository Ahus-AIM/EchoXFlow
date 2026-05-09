"""LV contour segmentation task package."""

from __future__ import annotations

from typing import Any

_DATASET_EXPORTS = {"RawDataset", "build_dataloaders", "discover_case_dirs", "discover_records"}
_TYPE_EXPORTS = {"Sample", "TrainingStepResult"}
_TASK_API_EXPORTS = {
    "evaluate",
    "load_config",
    "run_training",
    "run_training_step",
}


def __getattr__(name: str) -> Any:
    if name in _DATASET_EXPORTS:
        from tasks.segmentation import dataset

        return getattr(dataset, name)
    if name in _TYPE_EXPORTS:
        from tasks.segmentation import types

        return getattr(types, name)
    if name == "build_model":
        from tasks.utils.models import build_model

        return build_model
    if name == "TrainingModule":
        from tasks.utils.training import TrainingModule

        return TrainingModule
    if name in _TASK_API_EXPORTS:
        from tasks.registry import bind_task_api

        api = bind_task_api("segmentation")
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RawDataset",
    "Sample",
    "TrainingModule",
    "TrainingStepResult",
    "build_dataloaders",
    "build_model",
    "discover_case_dirs",
    "discover_records",
    "evaluate",
    "load_config",
    "run_training",
    "run_training_step",
]
