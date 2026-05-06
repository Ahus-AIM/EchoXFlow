from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tasks._config import TrainingConfig, load_task_training_config


def make_load_config(module_file: str) -> Callable[[str | Path | None], TrainingConfig]:
    def load_config(path: str | Path | None = None) -> TrainingConfig:
        return load_task_training_config(path, module_file=module_file)

    return load_config
