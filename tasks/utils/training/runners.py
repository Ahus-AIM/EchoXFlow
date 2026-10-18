from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TypeVar

import torch

from tasks._config import TrainingConfig
from tasks.utils.dataset import exam_split_summary, shutdown_dataloader_workers
from tasks.utils.seed import seed_everything
from tasks.utils.training.core import fit
from tasks.utils.training.device import _autocast_context, _resolve_amp_config, _resolve_device, move_to_device
from tasks.utils.training.evaluation import ClipMetricAccumulator, EvaluationResult, per_item_terms
from tasks.utils.training.module import LossFn, TrainingModule, ValMetricsFn
from tasks.utils.training.runtime import FitResult, configure_training_logging
from tasks.utils.types import TrainingStepResult

SampleT = TypeVar("SampleT")
RunTraining = Callable[..., TrainingStepResult]


def _default_result(result: FitResult, config: TrainingConfig) -> TrainingStepResult:
    del config
    return TrainingStepResult(
        device=result.device,
        steps_completed=result.steps_completed,
        batch_size=result.batch_size,
        loss=result.loss,
        epochs_completed=result.epochs_completed,
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        log_path=result.log_path,
        checkpoint_path=result.checkpoint_path,
        best_epoch=result.best_epoch,
        best_selection_loss=result.best_selection_loss,
        stopped_early=result.stopped_early,
        training_state_path=result.training_state_path,
    )


def run_task_training(
    *,
    task_name: str,
    load_config: Callable[[str | Path | None], TrainingConfig],
    build_dataloaders: Callable[..., tuple[Iterable[SampleT], ...]],
    loss_fn: LossFn,
    val_metrics_fn: ValMetricsFn | None = None,
    batch_size: Callable[[SampleT], int],
    result_factory: Callable[[FitResult, TrainingConfig], TrainingStepResult] = _default_result,
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
    train_dataloader: Iterable[SampleT] | None = None,
    val_dataloader: Iterable[SampleT] | None = None,
    max_epochs: int | None = None,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    selection_dataloader: Iterable[SampleT] | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    selection_split: Mapping[str, object] | None = None,
    resume: bool = False,
    write_epoch_log: bool = True,
    log_every_steps: int = 10,
    config_overrides: Mapping[str, Mapping[str, object]] | None = None,
    config_hash: str | None = None,
) -> TrainingStepResult:
    config = _with_overrides(load_config(config_path), config_overrides)
    seed_everything(int(config.data.get("seed", 0)))
    configure_training_logging()
    if train_dataloader is None:
        loaders = build_dataloaders(config=config, data_root=data_root)
        train_dataloader, val_dataloader = loaders[0], loaders[1]
        if selection_dataloader is None and len(loaders) > 2:
            selection_dataloader = loaders[2]
            selection_split = exam_split_summary(
                train=train_dataloader,
                val=val_dataloader,
                selection=selection_dataloader,
            )
    if train_dataloader is None:
        raise ValueError("train_dataloader is required")
    try:
        result = fit(
            module=TrainingModule(config=config, loss_fn=loss_fn, val_metrics_fn=val_metrics_fn),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            task_name=task_name,
            batch_size=batch_size,
            max_epochs=max_epochs,
            max_train_steps=max_train_steps,
            max_val_steps=max_val_steps,
            selection_dataloader=selection_dataloader,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            selection_split=selection_split,
            resume=resume,
            write_epoch_log=write_epoch_log,
            log_every_steps=log_every_steps,
            config_hash=config_hash,
        )
    finally:
        # Persistent workers live across epochs; stop them when training ends rather than at garbage collection.
        shutdown_dataloader_workers(train_dataloader, val_dataloader, selection_dataloader)
    return result_factory(result, config)


def _with_overrides(
    config: TrainingConfig,
    overrides: Mapping[str, Mapping[str, object]] | None,
) -> TrainingConfig:
    if not overrides:
        return config
    raw = dict(config.raw)
    for section, values in overrides.items():
        current = raw.get(section) or {}
        if not isinstance(current, Mapping):
            raise TypeError(f"config section {section!r} must be a mapping")
        raw[section] = {**current, **dict(values)}
    return TrainingConfig(raw=raw)


def run_task_evaluation(
    *,
    task_name: str,
    load_config: Callable[[str | Path | None], TrainingConfig],
    build_dataloaders: Callable[..., tuple[Iterable[SampleT], ...]],
    loss_fn: LossFn,
    val_metrics_fn: ValMetricsFn | None = None,
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
    weights: str | Path | None = None,
    eval_dataloader: Iterable[SampleT] | None = None,
    max_eval_steps: int | None = None,
    metric_keys: Mapping[str, str] | None = None,
) -> dict[str, float]:
    config = load_config(config_path)
    seed_everything(int(config.data.get("seed", 0)))
    configure_training_logging()
    if eval_dataloader is None:
        eval_dataloader = build_dataloaders(config=config, data_root=data_root)[1]
    module = TrainingModule(config=config, loss_fn=loss_fn, val_metrics_fn=val_metrics_fn)
    if weights is not None:
        state = torch.load(Path(weights), map_location="cpu")
        module.model.load_state_dict(state)
    terms = evaluate_module(
        module=module,
        dataloader=eval_dataloader,
        config=config,
        max_steps=max_eval_steps,
    ).exam_means
    if metric_keys is None:
        return terms
    return {out_key: terms[src_key] for src_key, out_key in metric_keys.items() if src_key in terms}


def evaluate_module(
    *,
    module: torch.nn.Module,
    dataloader: Iterable[SampleT],
    config: TrainingConfig,
    max_steps: int | None = None,
) -> EvaluationResult:
    """Evaluate ``module`` with per-clip metrics (see :mod:`tasks.utils.training.evaluation`)."""
    trainer_config = config.section("trainer")
    device = _resolve_device(trainer_config)
    amp = _resolve_amp_config(trainer_config, device)
    module.to(device)
    module.eval()
    clips = ClipMetricAccumulator()
    steps = 0
    try:
        with torch.no_grad():
            iterator = iter(dataloader)
            while max_steps is None or steps < int(max_steps):
                try:
                    sample = next(iterator)
                except StopIteration:
                    break
                sample_on_device = move_to_device(sample, device)
                with _autocast_context(amp, device):
                    terms = per_item_terms(module, sample_on_device)
                clips.add(sample_on_device, terms)
                steps += 1
    finally:
        shutdown_dataloader_workers(dataloader)
    if steps == 0:
        raise ValueError("evaluation dataloader produced no batches")
    return EvaluationResult(
        clip_means=clips.clip_means(),
        exam_means=clips.exam_means(),
        n_clips=clips.n_clips,
        n_exams=clips.n_exams,
        accumulator=clips,
    )


def run_task_training_step(run_training: RunTraining, **kwargs: object) -> TrainingStepResult:
    return run_training(
        **kwargs,
        max_epochs=1,
        max_train_steps=1,
        max_val_steps=0,
        write_epoch_log=False,
    )


def doppler_result(result: FitResult, config: TrainingConfig) -> TrainingStepResult:
    return TrainingStepResult(
        device=result.device,
        steps_completed=result.steps_completed,
        batch_size=result.batch_size,
        doppler_channels=int(config.model.get("doppler_channels", 0)),
        loss=result.loss,
        epochs_completed=result.epochs_completed,
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        log_path=result.log_path,
        checkpoint_path=result.checkpoint_path,
        best_epoch=result.best_epoch,
        best_selection_loss=result.best_selection_loss,
        stopped_early=result.stopped_early,
        training_state_path=result.training_state_path,
    )
