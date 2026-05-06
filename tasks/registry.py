from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

from tasks._config import TrainingConfig, load_training_config
from tasks.utils.training.module import LossFn, ValMetricsFn
from tasks.utils.training.runners import (
    run_task_evaluation,
    run_task_training,
    run_task_training_step,
)
from tasks.utils.types import TrainingStepResult

RunTraining = Callable[..., TrainingStepResult]
TaskEval = Callable[..., dict[str, float]]
BuildDataloaders = Callable[..., Any]
BatchSizeFn = Callable[[Any], int]
PreviewPredictionFn = Callable[[Any, object], Any]
PreviewWriter = Callable[..., None]
TemporalUpsampleFactorFn = Callable[[TrainingConfig], int]


def _default_batch_size(sample: Any) -> int:
    return int(sample.frames.shape[0])


def _default_temporal_upsample_factor(config: TrainingConfig) -> int:
    return int(config.model.get("temporal_upsample_factor", 2))


@dataclass(frozen=True)
class PreviewSpec:
    prediction_fn: PreviewPredictionFn
    writer: PreviewWriter


@dataclass(frozen=True)
class BaselineSpec:
    target_extractor: Callable[[object], Any] | None = None
    temporal_upsample_factor_fn: TemporalUpsampleFactorFn = _default_temporal_upsample_factor


@dataclass(frozen=True)
class TaskSpec:
    name: str
    loss_fn: LossFn
    dataset_module: str | None = None
    config_path: str | Path | None = None
    batch_size_fn: BatchSizeFn = _default_batch_size
    val_metrics_fn: ValMetricsFn | None = None
    metric_keys: Mapping[str, str] = field(default_factory=dict)
    result_factory: Callable[..., TrainingStepResult] | None = None
    preview: PreviewSpec | None = None
    baseline: BaselineSpec | None = None
    model_factory: Callable[..., Any] | None = None
    _task_dir: Path | None = field(default=None, repr=False, compare=False)

    @property
    def resolved_dataset_module(self) -> str:
        return self.dataset_module or f"tasks.{self.name}.dataset"

    @property
    def train_yaml(self) -> Path:
        if self.config_path is None:
            if self._task_dir is not None:
                return self._task_dir / "train.yaml"
            return Path(__file__).resolve().parent / self.name / "train.yaml"
        path = Path(self.config_path)
        if path.is_absolute():
            return path
        if self._task_dir is not None:
            return self._task_dir / path
        return Path(__file__).resolve().parents[1] / path


class UnsupportedBaselineError(ValueError):
    pass


def available_task_names() -> tuple[str, ...]:
    return tuple(sorted(_task_specs()))


def task_package(task: str) -> Any:
    _validate_task_name(task)
    return import_module(f"tasks.{task}")


def task_train_yaml(task: str) -> Path:
    return _task_spec(task).train_yaml


def task_load_config(task: str) -> Callable[[str | Path | None], TrainingConfig]:
    spec = _task_spec(task)

    def load_config(path: str | Path | None = None) -> TrainingConfig:
        return load_training_config(path, default_path=spec.train_yaml)

    return load_config


def task_build_dataloaders(task: str) -> BuildDataloaders:
    module = import_module(_task_spec(task).resolved_dataset_module)
    return cast(BuildDataloaders, getattr(module, "build_dataloaders"))


def task_run_training(task: str) -> RunTraining:
    _validate_task_name(task)
    return cast(RunTraining, partial(_run_training_for_task, task))


def task_evaluate(task: str) -> TaskEval:
    _validate_task_name(task)
    return cast(TaskEval, partial(_evaluate_task, task))


def task_run_training_step(task: str) -> RunTraining:
    _validate_task_name(task)
    return cast(RunTraining, partial(_run_training_step_for_task, task))


def task_run_cpu_training_step(task: str) -> RunTraining:
    return task_run_training_step(task)


def task_benchmark_loss_fn(task: str) -> LossFn:
    return _task_spec(task).loss_fn


def task_benchmark_val_metrics_fn(task: str) -> ValMetricsFn | None:
    return _task_spec(task).val_metrics_fn


def task_benchmark_metric_keys(task: str) -> dict[str, str]:
    return dict(_task_spec(task).metric_keys)


def task_preview_spec(task: str) -> PreviewSpec | None:
    return _task_spec(task).preview


def task_baseline_spec(task: str, baseline: str) -> BaselineSpec:
    spec = _task_spec(task).baseline
    if spec is None:
        raise UnsupportedBaselineError(f"Task {task!r} does not support baseline {baseline!r}")
    return spec


def bind_task_api(task_name: str, namespace: dict[str, Any] | None = None) -> SimpleNamespace:
    api = {
        "load_config": task_load_config(task_name),
        "run_training": task_run_training(task_name),
        "evaluate": task_evaluate(task_name),
        "run_training_step": task_run_training_step(task_name),
        "run_cpu_training_step": task_run_cpu_training_step(task_name),
    }
    if namespace is not None:
        namespace.update(api)
    return SimpleNamespace(**api)


def _run_training_for_task(task: str, **kwargs: Any) -> TrainingStepResult:
    spec = _task_spec(task)
    options: dict[str, Any] = {
        "task_name": spec.name,
        "load_config": task_load_config(spec.name),
        "build_dataloaders": task_build_dataloaders(spec.name),
        "loss_fn": spec.loss_fn,
        "val_metrics_fn": spec.val_metrics_fn,
        "batch_size": spec.batch_size_fn,
        **kwargs,
    }
    if spec.result_factory is not None:
        options["result_factory"] = spec.result_factory
    return run_task_training(**options)


def _evaluate_task(task: str, **kwargs: Any) -> dict[str, float]:
    spec = _task_spec(task)
    return run_task_evaluation(
        task_name=spec.name,
        load_config=task_load_config(spec.name),
        build_dataloaders=task_build_dataloaders(spec.name),
        loss_fn=spec.loss_fn,
        val_metrics_fn=spec.val_metrics_fn,
        metric_keys=spec.metric_keys,
        **kwargs,
    )


def _run_training_step_for_task(task: str, **kwargs: Any) -> TrainingStepResult:
    return run_task_training_step(task_run_training(task), **kwargs)


def _task_spec(task: str) -> TaskSpec:
    specs = _task_specs()
    _validate_task_name(task, specs=specs)
    return specs[task]


def _validate_task_name(task: str, *, specs: Mapping[str, TaskSpec] | None = None) -> None:
    specs = _task_specs() if specs is None else specs
    if task not in specs:
        choices = ", ".join(sorted(specs))
        raise ValueError(f"Unsupported task {task!r}; expected one of: {choices}")


def _task_specs() -> dict[str, TaskSpec]:
    specs: dict[str, TaskSpec] = {}
    for task_dir in _task_module_dirs():
        name = task_dir.name
        if name.startswith("_"):
            continue
        module = import_module(f"tasks.{name}.task")
        raw_spec = getattr(module, "TASK_SPEC", None)
        if not isinstance(raw_spec, TaskSpec):
            raise TypeError(f"tasks.{name}.task must define TASK_SPEC = TaskSpec(...)")
        if raw_spec.name != name:
            raise ValueError(f"tasks.{name}.task TASK_SPEC.name must be {name!r}, got {raw_spec.name!r}")
        specs.setdefault(name, replace(raw_spec, _task_dir=task_dir))
    return specs


def _task_module_dirs() -> tuple[Path, ...]:
    import tasks as tasks_package

    roots: list[Path] = []
    package_path = getattr(tasks_package, "__path__", ())
    for entry in sys.path:
        root = Path(entry or ".").resolve() / "tasks"
        if root.is_dir() and root not in roots:
            roots.append(root)
    if hasattr(package_path, "__setitem__"):
        package_path[:] = [str(root) for root in roots]

    task_dirs: list[Path] = []
    for root in roots:
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "task.py").is_file() and child not in task_dirs:
                task_dirs.append(child)
    return tuple(task_dirs)
