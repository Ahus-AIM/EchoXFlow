"""Color Doppler task package."""

from tasks.color_doppler.dataset import RawDataset, build_dataloaders, discover_case_dirs, discover_records
from tasks.color_doppler.types import Sample, TrainingStepResult
from tasks.registry import bind_task_api
from tasks.utils.models import build_model
from tasks.utils.training import TrainingModule

bind_task_api("color_doppler", globals())

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
    "run_cpu_training_step",
    "run_training",
    "run_training_step",
]
