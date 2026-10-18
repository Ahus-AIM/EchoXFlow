from __future__ import annotations

from tasks.utils.training.evaluation import EvaluationResult
from tasks.utils.training.module import TrainingModule
from tasks.utils.training.runners import evaluate_module
from tasks.utils.training.runtime import configure_training_logging

__all__ = ["EvaluationResult", "TrainingModule", "configure_training_logging", "evaluate_module"]
