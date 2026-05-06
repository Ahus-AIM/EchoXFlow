"""LV contour segmentation task package."""

from tasks.registry import bind_task_api
from tasks.segmentation.dataset import RawDataset, build_dataloaders, discover_case_dirs, discover_records
from tasks.segmentation.types import Sample, TrainingStepResult
from tasks.utils.models import build_model
from tasks.utils.training import TrainingModule

bind_task_api("segmentation", globals())

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
