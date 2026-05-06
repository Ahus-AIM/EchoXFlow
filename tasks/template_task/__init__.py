"""Template task package."""

from tasks.registry import bind_task_api
from tasks.template_task.dataset import RawDataset, SampleRef, build_dataloaders
from tasks.template_task.types import Sample, TrainingStepResult
from tasks.utils.models import build_model
from tasks.utils.training import TrainingModule

bind_task_api("template_task", globals())

__all__ = [
    "RawDataset",
    "Sample",
    "SampleRef",
    "TrainingModule",
    "TrainingStepResult",
    "build_dataloaders",
    "build_model",
    "evaluate",
    "load_config",
    "run_training",
    "run_training_step",
]
