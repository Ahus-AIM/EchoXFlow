from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TypeVar

import torch

from tasks._config import TrainingConfig
from tasks.utils.seed import seed_everything
from tasks.utils.training.core import fit
from tasks.utils.training.device import _autocast_context, _resolve_amp_config, _resolve_device, move_to_device
from tasks.utils.training.loss import (
    _accumulate_loss_terms,
    _compute_loss_and_terms,
    _loss_terms_to_floats,
    _mean_loss_terms,
)
from tasks.utils.training.metrics import _compute_validation_metric_terms
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
    )


def run_task_training(
    *,
    task_name: str,
    load_config: Callable[[str | Path | None], TrainingConfig],
    build_dataloaders: Callable[..., tuple[Iterable[SampleT], Iterable[SampleT]]],
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
    write_epoch_log: bool = True,
    log_every_steps: int = 10,
) -> TrainingStepResult:
    config = load_config(config_path)
    seed_everything(int(config.data.get("seed", 0)))
    configure_training_logging()
    if train_dataloader is None or val_dataloader is None:
        loaders = build_dataloaders(config=config, data_root=data_root)
        train_dataloader, val_dataloader = loaders[0], loaders[1]
    if train_dataloader is None:
        raise ValueError("train_dataloader is required")
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
        write_epoch_log=write_epoch_log,
        log_every_steps=log_every_steps,
    )
    return result_factory(result, config)


def run_task_evaluation(
    *,
    task_name: str,
    load_config: Callable[[str | Path | None], TrainingConfig],
    build_dataloaders: Callable[..., tuple[Iterable[SampleT], Iterable[SampleT]]],
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
        _train_dataloader, eval_dataloader = build_dataloaders(config=config, data_root=data_root)
    module = TrainingModule(config=config, loss_fn=loss_fn, val_metrics_fn=val_metrics_fn)
    if weights is not None:
        state = torch.load(Path(weights), map_location="cpu")
        module.model.load_state_dict(state)
    terms, _steps = evaluate_module(
        module=module,
        dataloader=eval_dataloader,
        config=config,
        max_steps=max_eval_steps,
    )
    if metric_keys is None:
        return terms
    return {out_key: terms[src_key] for src_key, out_key in metric_keys.items() if src_key in terms}


def evaluate_module(
    *,
    module: torch.nn.Module,
    dataloader: Iterable[SampleT],
    config: TrainingConfig,
    max_steps: int | None = None,
) -> tuple[dict[str, float], int]:
    trainer_config = config.section("trainer")
    device = _resolve_device(trainer_config)
    amp = _resolve_amp_config(trainer_config, device)
    module.to(device)
    module.eval()
    totals: dict[str, float] = {}
    steps = 0
    with torch.no_grad():
        iterator = iter(dataloader)
        while max_steps is None or steps < int(max_steps):
            try:
                sample = next(iterator)
            except StopIteration:
                break
            sample_on_device = move_to_device(sample, device)
            with _autocast_context(amp, device):
                _loss, loss_terms = _compute_loss_and_terms(module, sample_on_device)
                metric_terms = _compute_validation_metric_terms(module, sample_on_device)
            _accumulate_loss_terms(totals, _loss_terms_to_floats({**loss_terms, **metric_terms}))
            steps += 1
    if steps == 0:
        raise ValueError("evaluation dataloader produced no batches")
    return _mean_loss_terms(totals, steps), steps


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
    )
