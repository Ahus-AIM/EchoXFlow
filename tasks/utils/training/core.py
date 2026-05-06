from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar, cast

import torch
from torch import nn

from tasks.utils.training.debug import write_input_target_debug_pngs
from tasks.utils.training.device import _autocast_context, _resolve_amp_config, _resolve_device, move_to_device
from tasks.utils.training.logs import (
    _prepare_batch_log,
    _prepare_epoch_log,
    _write_batch_loss_plot,
    _write_batch_metrics,
    _write_epoch_metrics,
)
from tasks.utils.training.loss import (
    _accumulate_loss_terms,
    _compute_loss_and_terms,
    _loss_terms_to_floats,
    _mean_loss_terms,
    _round_terms,
    _validate_loss,
)
from tasks.utils.training.metrics import _compute_validation_metric_terms
from tasks.utils.training.preview import _stable_preview_sample, _write_epoch_previews
from tasks.utils.training.progress import _progress_bar, _should_update_progress, _update_logged_loss
from tasks.utils.training.runtime import AmpConfig, BatchMetrics, EpochMetrics, FitResult, build_optimizer

SampleT = TypeVar("SampleT")


def fit(
    *,
    module: nn.Module,
    train_dataloader: Iterable[SampleT],
    val_dataloader: Iterable[SampleT] | None,
    config: Any,
    task_name: str,
    batch_size: Callable[[Any], int],
    max_epochs: int | None = None,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    write_epoch_log: bool = True,
    log_every_steps: int = 10,
    logger: logging.Logger | None = None,
) -> FitResult:
    trainer_config = config.section("trainer")
    optimizer = build_optimizer(module, config.section("optimizer"))
    device = _resolve_device(trainer_config)
    amp = _resolve_amp_config(trainer_config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp.scaler_enabled)
    module.to(device)
    epochs = int(max_epochs if max_epochs is not None else trainer_config.get("max_epochs", 1))
    if epochs <= 0:
        raise ValueError("max_epochs must be positive")
    gradient_clip_val = float(cast(Any, trainer_config.get("gradient_clip_val", 0.0) or 0.0))
    log = logger or logging.getLogger(f"tasks.{task_name}.training")
    log_path = _prepare_epoch_log(config=config, task_name=task_name) if write_epoch_log else None
    batch_log_path = _prepare_batch_log(log_path)
    debug_batch_dir = _debug_batch_dir(config=config, log_path=log_path)
    write_epoch_previews = bool(config.section("artifacts").get("write_epoch_previews", True))
    total_train_steps = 0
    last_batch_size = 0
    last_train_loss = 0.0
    last_val_loss: float | None = None
    preview_seed = int(cast(Any, config.data.get("seed", 0)))
    train_preview_sample = (
        _stable_preview_sample(train_dataloader, device=device, seed=preview_seed) if write_epoch_previews else None
    )
    val_preview_sample = (
        None
        if val_dataloader is None or max_val_steps == 0 or not write_epoch_previews
        else _stable_preview_sample(val_dataloader, device=device, seed=preview_seed + 1)
    )

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, train_terms, train_steps, last_batch_size, _ = _run_train_epoch(
            module=module,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            max_steps=max_train_steps,
            gradient_clip_val=gradient_clip_val,
            amp=amp,
            scaler=scaler,
            task_name=task_name,
            epoch=epoch,
            epochs=epochs,
            log_every_steps=log_every_steps,
            batch_log_path=batch_log_path,
            debug_batch_dir=debug_batch_dir,
        )
        total_train_steps += train_steps
        val_loss, val_terms, val_steps, _ = _run_validation_epoch(
            module=module,
            dataloader=val_dataloader,
            device=device,
            batch_size=batch_size,
            max_steps=max_val_steps,
            amp=amp,
            task_name=task_name,
            epoch=epoch,
            epochs=epochs,
            log_every_steps=log_every_steps,
            batch_log_path=batch_log_path,
        )
        if write_epoch_previews:
            _write_epoch_previews(
                module=module,
                samples={
                    "train": train_preview_sample,
                    "val": (None if val_steps == 0 else val_preview_sample),
                },
                run_dir=None if log_path is None else log_path.parent,
                epoch=epoch,
                amp=amp,
                device=device,
                logger=log,
            )
        last_train_loss = train_loss
        last_val_loss = val_loss
        metrics = EpochMetrics(
            task=task_name,
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=None if val_loss is None else round(val_loss, 4),
            train_loss_terms=_round_terms(train_terms),
            val_loss_terms=None if val_terms is None else _round_terms(val_terms),
            train_steps=train_steps,
            val_steps=val_steps,
            elapsed_seconds=round(time.perf_counter() - started, 2),
            device=str(device),
            precision=amp.precision,
        )
        _write_epoch_metrics(log_path, metrics)
        _write_batch_loss_plot(batch_log_path)
        log.info(
            "%s epoch %d/%d train=%.4g val=%s steps=%d/%d elapsed=%.1fs device=%s precision=%s artifacts=%s",
            task_name,
            epoch,
            epochs,
            train_loss,
            "n/a" if val_loss is None else f"{val_loss:.4g}",
            train_steps,
            val_steps,
            metrics.elapsed_seconds,
            metrics.device,
            metrics.precision,
            "disabled" if log_path is None else log_path.parent,
        )

    checkpoint_path = None
    if log_path is not None:
        checkpoint_path = str(log_path.parent / "weights.pt")
        checkpoint_module = cast(nn.Module, getattr(module, "model", module))
        torch.save(checkpoint_module.state_dict(), checkpoint_path)

    return FitResult(
        device=str(device),
        epochs_completed=epochs,
        steps_completed=total_train_steps,
        batch_size=last_batch_size,
        loss=last_train_loss,
        train_loss=last_train_loss,
        val_loss=last_val_loss,
        log_path=None if log_path is None else str(log_path),
        checkpoint_path=checkpoint_path,
    )


def _run_train_epoch(
    *,
    module: nn.Module,
    dataloader: Iterable[SampleT],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: Callable[[Any], int],
    max_steps: int | None,
    gradient_clip_val: float,
    amp: AmpConfig,
    scaler: torch.amp.GradScaler,
    task_name: str,
    epoch: int,
    epochs: int,
    log_every_steps: int,
    batch_log_path: Path | None,
    debug_batch_dir: Path | None,
) -> tuple[float, dict[str, float], int, int, SampleT | None]:
    module.train()
    total_loss = 0.0
    total_terms: dict[str, float] = {}
    logged_loss: float | None = None
    steps = 0
    last_batch_size = 0
    preview_sample: SampleT | None = None
    progress = _progress_bar(
        dataloader=dataloader,
        max_steps=max_steps,
        description=f"{task_name} {epoch}/{epochs} train",
    )
    iterator = iter(dataloader)
    try:
        while max_steps is None or steps < int(max_steps):
            if progress is not None and steps == 0:
                progress.set_postfix_str("loading")
            load_started = time.perf_counter()
            try:
                sample = next(iterator)
            except StopIteration:
                break
            load_seconds = time.perf_counter() - load_started
            step_started = time.perf_counter()
            sample_on_device = cast(SampleT, move_to_device(sample, device))
            if preview_sample is None:
                preview_sample = sample_on_device
            if debug_batch_dir is not None and epoch == 1 and steps == 0:
                write_input_target_debug_pngs(
                    sample_on_device,
                    output_dir=debug_batch_dir,
                    split="train",
                    epoch=epoch,
                    step=steps + 1,
                )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(amp, device):
                loss, loss_terms = _compute_loss_and_terms(module, sample_on_device)
            _validate_loss(loss)
            if amp.scaler_enabled:
                scaler.scale(loss).backward()
                if gradient_clip_val > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_val)
                optimizer.step()
            loss_value = float(loss.detach().cpu())
            term_values = _loss_terms_to_floats(loss_terms)
            _accumulate_loss_terms(total_terms, term_values)
            logged_loss = _update_logged_loss(logged_loss, loss_value)
            total_loss += loss_value
            steps += 1
            last_batch_size = int(batch_size(sample_on_device))
            _write_batch_metrics(
                batch_log_path,
                BatchMetrics(
                    task=task_name,
                    split="train",
                    epoch=epoch,
                    step=steps,
                    loss=round(loss_value, 4),
                    loss_terms=_round_terms(term_values),
                    batch_size=last_batch_size,
                    load_seconds=round(load_seconds, 2),
                    step_seconds=round(time.perf_counter() - step_started, 2),
                    device=str(device),
                    precision=amp.precision,
                ),
            )
            if progress is not None:
                progress.update(1)
                if _should_update_progress(steps, log_every_steps):
                    progress.set_postfix(
                        {
                            "loss": f"{logged_loss:.4g}",
                            "load": f"{load_seconds:.1f}s",
                            "step": f"{time.perf_counter() - step_started:.1f}s",
                        }
                    )
    finally:
        if progress is not None:
            progress.close()
    if steps == 0:
        raise ValueError("train_dataloader produced no batches")
    return total_loss / steps, _mean_loss_terms(total_terms, steps), steps, last_batch_size, preview_sample


def _debug_batch_dir(*, config: Any, log_path: Path | None) -> Path | None:
    if log_path is None:
        return None
    artifacts = config.section("artifacts")
    if not bool(artifacts.get("debug_input_target_pngs", False)):
        return None
    return log_path.parent / "debug_batch"


def _run_validation_epoch(
    *,
    module: nn.Module,
    dataloader: Iterable[SampleT] | None,
    device: torch.device,
    batch_size: Callable[[SampleT], int],
    max_steps: int | None,
    amp: AmpConfig,
    task_name: str,
    epoch: int,
    epochs: int,
    log_every_steps: int,
    batch_log_path: Path | None,
) -> tuple[float | None, dict[str, float] | None, int, SampleT | None]:
    if dataloader is None or max_steps == 0:
        return None, None, 0, None
    module.eval()
    total_loss = 0.0
    total_terms: dict[str, float] = {}
    logged_loss: float | None = None
    steps = 0
    preview_sample: SampleT | None = None
    progress = _progress_bar(
        dataloader=dataloader,
        max_steps=max_steps,
        description=f"{task_name} {epoch}/{epochs} val",
    )
    with torch.no_grad():
        iterator = iter(dataloader)
        try:
            while max_steps is None or steps < int(max_steps):
                if progress is not None and steps == 0:
                    progress.set_postfix_str("loading")
                load_started = time.perf_counter()
                try:
                    sample = next(iterator)
                except StopIteration:
                    break
                load_seconds = time.perf_counter() - load_started
                step_started = time.perf_counter()
                sample_on_device = cast(SampleT, move_to_device(sample, device))
                if preview_sample is None:
                    preview_sample = sample_on_device
                with _autocast_context(amp, device):
                    loss, loss_terms = _compute_loss_and_terms(module, sample_on_device)
                    metric_terms = _compute_validation_metric_terms(module, sample_on_device)
                    if metric_terms:
                        loss_terms = {**loss_terms, **metric_terms}
                _validate_loss(loss)
                loss_value = float(loss.detach().cpu())
                term_values = _loss_terms_to_floats(loss_terms)
                _accumulate_loss_terms(total_terms, term_values)
                logged_loss = _update_logged_loss(logged_loss, loss_value)
                total_loss += loss_value
                steps += 1
                step_seconds = time.perf_counter() - step_started
                _write_batch_metrics(
                    batch_log_path,
                    BatchMetrics(
                        task=task_name,
                        split="val",
                        epoch=epoch,
                        step=steps,
                        loss=round(loss_value, 4),
                        loss_terms=_round_terms(term_values),
                        batch_size=int(batch_size(sample_on_device)),
                        load_seconds=round(load_seconds, 2),
                        step_seconds=round(step_seconds, 2),
                        device=str(device),
                        precision=amp.precision,
                    ),
                )
                if progress is not None:
                    progress.update(1)
                    if _should_update_progress(steps, log_every_steps):
                        progress.set_postfix(
                            {
                                "loss": f"{logged_loss:.4g}",
                                "load": f"{load_seconds:.1f}s",
                                "step": f"{step_seconds:.1f}s",
                            }
                        )
        finally:
            if progress is not None:
                progress.close()
    if steps == 0:
        return None, None, 0, None
    return total_loss / steps, _mean_loss_terms(total_terms, steps), steps, preview_sample
