from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import torch
from torch import nn

from tasks.utils.atomic import atomic_open, atomic_torch_save, atomic_write_json, remove_stale_temporaries
from tasks.utils.training.debug import write_input_target_debug_pngs
from tasks.utils.training.device import _autocast_context, _resolve_amp_config, _resolve_device, move_to_device
from tasks.utils.training.evaluation import ClipMetricAccumulator, per_item_terms
from tasks.utils.training.logs import (
    BatchLogWindow,
    _prepare_batch_log,
    _prepare_epoch_log,
    _write_epoch_metrics,
    batch_log_every_steps,
)
from tasks.utils.training.loss import _round_terms, per_clip_loss_sum
from tasks.utils.training.preview import _stable_preview_sample, _write_epoch_previews
from tasks.utils.training.progress import (
    ProgressReporter,
    _progress_bar,
    _should_update_progress,
    _update_logged_loss,
)
from tasks.utils.training.runtime import AmpConfig, EpochMetrics, FitResult, build_optimizer

TRAINING_STATE_NAME = "training_state.pt"
WEIGHTS_NAME = "weights.pt"
#: Best-epoch weights of epoch N are staged as ``weights.epochNNNN.next.pt`` and promoted to ``weights.pt`` once
#: ``training_state.pt`` records epoch N; the epoch in the name tells a finished promotion from an interrupted epoch.
PENDING_WEIGHTS_GLOB = "weights.epoch*.next.pt"
TRAINING_STATE_SCHEMA_VERSION = 2


class TrainingStateMismatchError(RuntimeError):
    """``training_state.pt`` belongs to a different resolved configuration than the current run."""


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
    selection_dataloader: Iterable[SampleT] | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    selection_split: Mapping[str, object] | None = None,
    resume: bool = False,
    write_epoch_log: bool = True,
    log_every_steps: int = 10,
    logger: logging.Logger | None = None,
    config_hash: str | None = None,
) -> FitResult:
    """Train ``module`` for a fixed epoch budget and keep the last epoch's weights.

    ``val_dataloader`` is monitoring only: its per-epoch loss is logged but never selects a checkpoint. Model
    selection happens only when a separate ``selection_dataloader`` is given (it must not be the reported evaluation
    data): its examination-equal per-clip loss over the whole selection set picks the best epoch's ``weights.pt``
    and, with ``early_stopping_patience``, stops training. ``selection_split`` (for example the selection exam ids
    and the train/selection/val exam counts) is recorded in ``early_stopping.json``.

    Crash safety: after every epoch the complete training state (model, optimizer, AMP scaler, Python/NumPy/Torch/CUDA
    RNGs, sampler epochs and data-loader generators, early-stopping state and step counters) is written atomically to
    ``training_state.pt``. With ``resume=True`` training continues after the last completed epoch as if uninterrupted;
    an interrupted epoch is rerun from its start (checkpoints are epoch-granular). Best-epoch weights are staged in
    ``weights.next.pt`` and promoted to ``weights.pt`` only after the state recording that epoch is on disk, so
    ``weights.pt`` always matches ``best_epoch``. ``config_hash`` is stored in the state; resuming a state written
    for a different hash raises :class:`TrainingStateMismatchError`.
    """
    trainer_config = config.section("trainer")
    accumulate_grad_batches = _positive_integer_config(
        trainer_config,
        "accumulate_grad_batches",
        default=1,
    )
    optimizer = build_optimizer(module, config.section("optimizer"))
    device = _resolve_device(trainer_config)
    amp = _resolve_amp_config(trainer_config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp.scaler_enabled)
    module.to(device)
    epochs = int(max_epochs if max_epochs is not None else trainer_config.get("max_epochs", 1))
    if epochs <= 0:
        raise ValueError("max_epochs must be positive")
    gradient_clip_val = float(cast(Any, trainer_config.get("gradient_clip_val", 0.0) or 0.0))
    patience = _optional_positive_integer(early_stopping_patience, "early_stopping_patience")
    min_delta = float(early_stopping_min_delta)
    if min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if patience is not None and selection_dataloader is None:
        raise ValueError("early stopping requires a selection_dataloader separate from the evaluation data")
    if selection_dataloader is not None and selection_dataloader is val_dataloader:
        raise ValueError("selection_dataloader must not be the reported validation dataloader")
    log = logger or logging.getLogger(f"tasks.{task_name}.training")
    log_path = _prepare_epoch_log(config=config, task_name=task_name, resume=True) if write_epoch_log else None
    run_dir = None if log_path is None else log_path.parent
    if run_dir is not None:
        remove_stale_temporaries(run_dir)
    selecting = selection_dataloader is not None
    state_path = None if run_dir is None or (not selecting and not resume) else run_dir / TRAINING_STATE_NAME
    resumed = bool(resume and state_path is not None and state_path.is_file())
    if log_path is not None and not resumed:
        log_path.write_text("", encoding="utf-8")
    batch_every = batch_log_every_steps(config)
    batch_log_path = _prepare_batch_log(log_path, resume=resumed, enabled=batch_every > 0)
    debug_batch_dir = _debug_batch_dir(config=config, log_path=log_path)
    write_epoch_previews = bool(config.section("artifacts").get("write_epoch_previews", True))
    total_train_steps = 0
    total_optimizer_steps = 0
    last_batch_size = 0
    last_train_loss = 0.0
    last_val_loss: float | None = None
    best_epoch: int | None = None
    best_selection_loss: float | None = None
    non_improving_epochs = 0
    start_epoch = 1
    stopped_early = False
    if resumed and state_path is not None:
        state = _load_training_state(
            state_path,
            module=module,
            optimizer=optimizer,
            scaler=scaler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            selection_dataloader=selection_dataloader,
            device=device,
            config_hash=config_hash,
        )
        completed_epoch = int(state["completed_epoch"])
        if completed_epoch > epochs:
            raise ValueError(
                f"training_state.pt already completed epoch {completed_epoch}, which exceeds max_epochs={epochs}"
            )
        start_epoch = completed_epoch + 1
        total_train_steps = int(state.get("total_train_steps", 0))
        total_optimizer_steps = int(state.get("total_optimizer_steps", 0))
        last_batch_size = int(state.get("last_batch_size", 0))
        last_train_loss = float(state.get("last_train_loss", 0.0))
        raw_last_val_loss = state.get("last_val_loss")
        last_val_loss = None if raw_last_val_loss is None else float(raw_last_val_loss)
        raw_best_epoch = state.get("best_epoch")
        best_epoch = None if raw_best_epoch is None else int(raw_best_epoch)
        raw_best_selection_loss = state.get("best_selection_loss")
        best_selection_loss = None if raw_best_selection_loss is None else float(raw_best_selection_loss)
        non_improving_epochs = int(state.get("non_improving_epochs", 0))
        stopped_early = patience is not None and non_improving_epochs >= patience
        _truncate_jsonl_after_epoch(log_path, completed_epoch)
        _truncate_jsonl_after_epoch(batch_log_path, completed_epoch)
        _reconcile_pending_weights(run_dir, best_epoch=best_epoch, completed_epoch=completed_epoch)
        log.info(
            "resuming %s after completed epoch %d/%d from %s (best epoch=%s, non-improving epochs=%d)",
            task_name,
            completed_epoch,
            epochs,
            state_path,
            best_epoch,
            non_improving_epochs,
        )
    elif run_dir is not None:
        discard_pending_weights(run_dir)
    preview_seed = int(cast(Any, config.data.get("seed", 0)))
    split_metadata = dict(selection_split or {})
    train_preview_sample = (
        _stable_preview_sample(train_dataloader, device=device, seed=preview_seed) if write_epoch_previews else None
    )
    val_preview_sample = (
        None
        if val_dataloader is None or max_val_steps == 0 or not write_epoch_previews
        else _stable_preview_sample(val_dataloader, device=device, seed=preview_seed + 1)
    )

    epochs_completed = start_epoch - 1
    epoch_range = range(start_epoch, start_epoch) if stopped_early else range(start_epoch, epochs + 1)
    fit_started = time.perf_counter()
    recent_epoch_seconds: deque[float] = deque(maxlen=5)
    for epoch in epoch_range:
        started = time.perf_counter()
        (
            train_loss,
            train_terms,
            train_steps,
            optimizer_steps,
            train_examples,
            last_batch_size,
            _,
        ) = _run_train_epoch(
            module=module,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            max_steps=max_train_steps,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            amp=amp,
            scaler=scaler,
            task_name=task_name,
            epoch=epoch,
            epochs=epochs,
            log_every_steps=log_every_steps,
            batch_log_path=batch_log_path,
            batch_log_every=batch_every,
            debug_batch_dir=debug_batch_dir,
            log=log,
        )
        total_train_steps += train_steps
        total_optimizer_steps += optimizer_steps
        val_loss, val_terms, val_steps, val_examples, _ = _run_validation_epoch(
            module=module,
            dataloader=val_dataloader,
            device=device,
            batch_size=batch_size,
            max_steps=max_val_steps,
            compute_metric_terms=True,
            amp=amp,
            task_name=task_name,
            epoch=epoch,
            epochs=epochs,
            log_every_steps=log_every_steps,
            batch_log_path=batch_log_path,
            batch_log_every=batch_every,
            log=log,
        )
        selection_loss: float | None = None
        if selecting:
            selection_loss, _terms, _steps, _examples, _ = _run_validation_epoch(
                module=module,
                dataloader=selection_dataloader,
                device=device,
                batch_size=batch_size,
                max_steps=None,
                compute_metric_terms=False,
                aggregate="exam",
                amp=amp,
                task_name=task_name,
                epoch=epoch,
                epochs=epochs,
                log_every_steps=log_every_steps,
                batch_log_path=None,
                batch_log_every=0,
                split="selection",
                log=log,
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
            train_loss=float(train_loss),
            val_loss=None if val_loss is None else float(val_loss),
            selection_loss=None if selection_loss is None else float(selection_loss),
            train_loss_terms=_round_terms(train_terms),
            val_loss_terms=None if val_terms is None else _round_terms(val_terms),
            train_steps=train_steps,
            optimizer_steps=optimizer_steps,
            val_steps=val_steps,
            train_examples=train_examples,
            val_examples=val_examples,
            accumulate_grad_batches=accumulate_grad_batches,
            elapsed_seconds=round(time.perf_counter() - started, 2),
            device=str(device),
            precision=amp.precision,
        )
        _write_epoch_metrics(log_path, metrics)
        improved = (
            selection_loss is not None
            and math.isfinite(selection_loss)
            and (best_selection_loss is None or selection_loss < best_selection_loss - min_delta)
        )
        if improved:
            best_epoch = epoch
            best_selection_loss = selection_loss
            non_improving_epochs = 0
            if run_dir is not None:
                target = pending_weights_path(run_dir, epoch) if state_path is not None else run_dir / WEIGHTS_NAME
                _atomic_torch_save(_checkpoint_module(module).state_dict(), target)
        elif selecting:
            non_improving_epochs += 1
        epochs_completed = epoch
        stopped_early = patience is not None and non_improving_epochs >= patience
        if state_path is not None:
            _save_training_state(
                state_path,
                module=module,
                optimizer=optimizer,
                scaler=scaler,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                selection_dataloader=selection_dataloader,
                completed_epoch=epoch,
                total_train_steps=total_train_steps,
                total_optimizer_steps=total_optimizer_steps,
                last_batch_size=last_batch_size,
                last_train_loss=last_train_loss,
                last_val_loss=last_val_loss,
                best_epoch=best_epoch,
                best_selection_loss=best_selection_loss,
                non_improving_epochs=non_improving_epochs,
                config_hash=config_hash,
            )
            if improved:
                _promote_pending_weights(state_path.parent, epoch)
        if state_path is not None and selecting:
            _write_early_stopping_metadata(
                state_path.parent / "early_stopping.json",
                patience=patience,
                min_delta=min_delta,
                split_metadata=split_metadata,
                completed_epochs=epochs_completed,
                max_epochs=epochs,
                best_epoch=best_epoch,
                best_selection_loss=best_selection_loss,
                non_improving_epochs=non_improving_epochs,
                stopped_early=stopped_early,
            )
        epoch_seconds = time.perf_counter() - started
        recent_epoch_seconds.append(epoch_seconds)
        log.info(
            "%s epoch %d/%d train=%.4g selection=%s val=%s best=%s patience=%s batches=%d/%d updates=%d "
            "examples=%d/%d accumulation=%d epoch_time=%s elapsed=%s eta<=%s device=%s precision=%s",
            task_name,
            epoch,
            epochs,
            train_loss,
            "n/a" if selection_loss is None else f"{selection_loss:.5g}",
            "n/a" if val_loss is None else f"{val_loss:.4g}",
            "n/a" if best_epoch is None else f"{best_epoch} ({best_selection_loss:.5g})",
            "n/a" if patience is None else f"{non_improving_epochs}/{patience}",
            train_steps,
            val_steps,
            optimizer_steps,
            train_examples,
            val_examples,
            accumulate_grad_batches,
            _duration(epoch_seconds),
            _duration(time.perf_counter() - fit_started),
            _duration(
                _remaining_epochs(
                    epoch=epoch, epochs=epochs, patience=patience, non_improving_epochs=non_improving_epochs
                )
                * (sum(recent_epoch_seconds) / len(recent_epoch_seconds))
            ),
            metrics.device,
            metrics.precision,
        )
        if stopped_early:
            log.info(
                "%s early stopping at epoch %d; best epoch=%d selection loss=%.6g patience=%d",
                task_name,
                epoch,
                best_epoch,
                best_selection_loss,
                patience,
            )
            break

    if state_path is not None and selecting and epochs_completed > 0:
        _write_early_stopping_metadata(
            state_path.parent / "early_stopping.json",
            patience=patience,
            min_delta=min_delta,
            split_metadata=split_metadata,
            completed_epochs=epochs_completed,
            max_epochs=epochs,
            best_epoch=best_epoch,
            best_selection_loss=best_selection_loss,
            non_improving_epochs=non_improving_epochs,
            stopped_early=stopped_early,
        )

    checkpoint_path = None
    if log_path is not None:
        checkpoint_path = str(log_path.parent / WEIGHTS_NAME)
        if best_epoch is None:
            _atomic_torch_save(_checkpoint_module(module).state_dict(), Path(checkpoint_path))
        elif not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"best-epoch weights {checkpoint_path} are missing (best epoch {best_epoch})")

    return FitResult(
        device=str(device),
        epochs_completed=epochs_completed,
        steps_completed=total_train_steps,
        batch_size=last_batch_size,
        loss=last_train_loss,
        train_loss=last_train_loss,
        val_loss=last_val_loss,
        log_path=None if log_path is None else str(log_path),
        checkpoint_path=checkpoint_path,
        optimizer_steps_completed=total_optimizer_steps,
        best_epoch=best_epoch,
        best_selection_loss=best_selection_loss,
        stopped_early=stopped_early,
        training_state_path=None if state_path is None else str(state_path),
    )


def _positive_integer_config(config: Any, name: str, *, default: int) -> int:
    value: object = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"trainer.{name} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"trainer.{name} must be a positive integer, got {value!r}")
    return value


def _optional_positive_integer(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checkpoint_module(module: nn.Module) -> nn.Module:
    return cast(nn.Module, getattr(module, "model", module))


def _atomic_torch_save(payload: object, path: Path) -> None:
    atomic_torch_save(payload, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_json(path, payload)


def pending_weights_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / f"weights.epoch{int(epoch):04d}.next.pt"


def discard_pending_weights(run_dir: Path) -> None:
    for path in run_dir.glob(PENDING_WEIGHTS_GLOB):
        path.unlink(missing_ok=True)


def _promote_pending_weights(run_dir: Path, epoch: int) -> None:
    """Move epoch ``epoch``'s staged weights to ``weights.pt`` (after the state recording that epoch is on disk)."""
    pending = pending_weights_path(run_dir, epoch)
    if pending.is_file():
        os.replace(pending, run_dir / WEIGHTS_NAME)


def _reconcile_pending_weights(run_dir: Path | None, *, best_epoch: int | None, completed_epoch: int) -> None:
    """Finish or discard a weights promotion interrupted by a crash, so ``weights.pt`` matches ``best_epoch``.

    Staged weights of epoch N are written before the training state of epoch N. If that state was saved (N is the
    completed best epoch) the promotion is finished; staged weights of any other epoch belong to an interrupted epoch
    and are discarded.
    """
    if run_dir is None:
        return
    for pending in run_dir.glob(PENDING_WEIGHTS_GLOB):
        epoch_text = pending.name.removeprefix("weights.epoch").removesuffix(".next.pt")
        epoch = int(epoch_text) if epoch_text.isdigit() else None
        if epoch is not None and best_epoch == epoch == completed_epoch:
            _promote_pending_weights(run_dir, epoch)
        else:
            pending.unlink(missing_ok=True)


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90.0:
        return f"{seconds:.0f}s"
    if seconds < 5400.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.2f}h"


def _remaining_epochs(*, epoch: int, epochs: int, patience: int | None, non_improving_epochs: int) -> int:
    """Upper bound on the epochs still to run: the budget, or fewer if early stopping can trigger first."""
    remaining = max(0, int(epochs) - int(epoch))
    if patience is not None:
        remaining = min(remaining, max(0, int(patience) - int(non_improving_epochs)))
    return remaining


def _dataloader_generator(dataloader: object) -> torch.Generator | None:
    generator = getattr(dataloader, "generator", None)
    return generator if isinstance(generator, torch.Generator) else None


def _generator_state(dataloader: object) -> torch.Tensor | None:
    generator = _dataloader_generator(dataloader)
    return None if generator is None else generator.get_state()


def _set_generator_state(dataloader: object, state: object) -> None:
    generator = _dataloader_generator(dataloader)
    if generator is not None and isinstance(state, torch.Tensor):
        generator.set_state(state.cpu())


def _sampler_epoch(dataloader: object) -> int | None:
    sampler = getattr(dataloader, "sampler", None)
    epoch = getattr(sampler, "epoch", None)
    return int(epoch) if isinstance(epoch, int) else None


def _set_sampler_epoch(dataloader: object, epoch: object) -> None:
    if not isinstance(epoch, int):
        return
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "epoch"):
        sampler.epoch = int(epoch)


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: object) -> None:
    if not isinstance(state, Mapping):
        return
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_state = state.get("torch")
    cuda_state = state.get("cuda")
    if python_state is not None:
        random.setstate(cast(tuple[Any, ...], python_state))
    if numpy_state is not None:
        np.random.set_state(cast(tuple[Any, ...], numpy_state))
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        # ``torch.load(..., map_location=device)`` also moves the saved RNG
        # tensors to CUDA, while set_rng_state_all requires CPU ByteTensors.
        torch.cuda.set_rng_state_all(
            [saved_state.cpu() if isinstance(saved_state, torch.Tensor) else saved_state for saved_state in cuda_state]
        )


def _save_training_state(
    path: Path,
    *,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_dataloader: object,
    val_dataloader: object,
    selection_dataloader: object,
    completed_epoch: int,
    total_train_steps: int,
    total_optimizer_steps: int,
    last_batch_size: int,
    last_train_loss: float,
    last_val_loss: float | None,
    best_epoch: int | None,
    best_selection_loss: float | None,
    non_improving_epochs: int,
    config_hash: str | None = None,
) -> None:
    _atomic_torch_save(
        {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "config_hash": config_hash,
            "completed_epoch": int(completed_epoch),
            "module": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": _rng_state(),
            "train_sampler_epoch": _sampler_epoch(train_dataloader),
            "val_sampler_epoch": _sampler_epoch(val_dataloader),
            "selection_sampler_epoch": _sampler_epoch(selection_dataloader),
            # Data-loader generators seed the worker RNGs of every epoch's iterator.
            "train_generator": _generator_state(train_dataloader),
            "val_generator": _generator_state(val_dataloader),
            "selection_generator": _generator_state(selection_dataloader),
            "total_train_steps": int(total_train_steps),
            "total_optimizer_steps": int(total_optimizer_steps),
            "last_batch_size": int(last_batch_size),
            "last_train_loss": float(last_train_loss),
            "last_val_loss": None if last_val_loss is None else float(last_val_loss),
            "best_epoch": best_epoch,
            "best_selection_loss": best_selection_loss,
            "non_improving_epochs": int(non_improving_epochs),
        },
        path,
    )


def _load_training_state(
    path: Path,
    *,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_dataloader: object,
    val_dataloader: object,
    selection_dataloader: object,
    device: torch.device,
    config_hash: str | None = None,
) -> Mapping[str, Any]:
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"cannot read {path} ({exc}); it is written atomically, so this indicates outside damage. "
            "Delete it to restart this cell from scratch."
        ) from exc
    if not isinstance(state, Mapping) or int(state.get("schema_version", 0)) not in {1, TRAINING_STATE_SCHEMA_VERSION}:
        raise ValueError(f"unsupported training state in {path}")
    saved_hash = state.get("config_hash")
    if config_hash is not None and saved_hash is not None and str(saved_hash) != str(config_hash):
        raise TrainingStateMismatchError(
            f"{path} was written for config hash {saved_hash}, but this run resolves to {config_hash}. "
            "Use a new run name, or --force-rerun to discard the old state."
        )
    module_state = state.get("module")
    optimizer_state = state.get("optimizer")
    if not isinstance(module_state, Mapping) or not isinstance(optimizer_state, Mapping):
        raise ValueError(f"incomplete training state in {path}")
    module.load_state_dict(cast(Mapping[str, Any], module_state))
    optimizer.load_state_dict(cast(dict[str, Any], optimizer_state))
    scaler_state = state.get("scaler")
    if isinstance(scaler_state, Mapping):
        scaler.load_state_dict(cast(dict[str, Any], scaler_state))
    _set_sampler_epoch(train_dataloader, state.get("train_sampler_epoch"))
    _set_sampler_epoch(val_dataloader, state.get("val_sampler_epoch"))
    _set_sampler_epoch(selection_dataloader, state.get("selection_sampler_epoch"))
    _set_generator_state(train_dataloader, state.get("train_generator"))
    _set_generator_state(val_dataloader, state.get("val_generator"))
    _set_generator_state(selection_dataloader, state.get("selection_generator"))
    _restore_rng_state(state.get("rng"))
    return state


def _truncate_jsonl_after_epoch(path: Path | None, completed_epoch: int) -> None:
    if path is None or not path.is_file():
        return
    retained: list[str] = []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_index, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if line_index == len(lines) - 1:
                break
            raise
        if not isinstance(payload, Mapping):
            raise ValueError(f"non-mapping JSON row in {path}")
        epoch = payload.get("epoch")
        if isinstance(epoch, int) and epoch <= completed_epoch:
            retained.append(json.dumps(payload, sort_keys=True))
    with atomic_open(path, "w", encoding="utf-8") as stream:
        stream.write("".join(f"{line}\n" for line in retained))


def _write_early_stopping_metadata(
    path: Path,
    *,
    patience: int | None,
    min_delta: float,
    split_metadata: Mapping[str, object],
    completed_epochs: int,
    max_epochs: int,
    best_epoch: int | None,
    best_selection_loss: float | None,
    non_improving_epochs: int,
    stopped_early: bool,
) -> None:
    _atomic_json(
        path,
        {
            **dict(split_metadata),
            "schema_version": 2,
            "monitor": "selection_loss",
            "selection_aggregation": "exam_equal_mean_of_clip_means",
            "mode": "min",
            "patience": patience,
            "min_delta": float(min_delta),
            "completed_epochs": int(completed_epochs),
            "max_epochs": int(max_epochs),
            "best_epoch": best_epoch,
            "best_selection_loss": best_selection_loss,
            "non_improving_epochs": int(non_improving_epochs),
            "stopped_early": bool(stopped_early),
            "stopping_reason": (
                "patience_exhausted"
                if stopped_early
                else ("max_epochs_reached" if completed_epochs >= max_epochs else "in_progress")
            ),
        },
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
    accumulate_grad_batches: int,
    amp: AmpConfig,
    scaler: torch.amp.GradScaler,
    task_name: str,
    epoch: int,
    epochs: int,
    log_every_steps: int,
    batch_log_path: Path | None,
    debug_batch_dir: Path | None,
    batch_log_every: int = 1,
    log: logging.Logger | None = None,
) -> tuple[float, dict[str, float], int, int, int, int, SampleT | None]:
    module.train()
    total_loss = 0.0
    total_terms: dict[str, float] = {}
    total_term_counts: dict[str, int] = {}
    total_loss_clips = 0
    total_examples = 0
    logged_loss: float | None = None
    steps = 0
    optimizer_steps = 0
    pending_microbatches = 0
    pending_examples = 0
    last_batch_size = 0
    preview_sample: SampleT | None = None
    progress = _progress_bar(
        dataloader=dataloader,
        max_steps=max_steps,
        description=f"{task_name} {epoch}/{epochs} train",
    )
    reporter = (
        None
        if progress is not None or log is None
        else ProgressReporter(
            log, dataloader=dataloader, max_steps=max_steps, description=f"{task_name} {epoch}/{epochs} train"
        )
    )
    batch_window = BatchLogWindow(
        batch_log_path,
        every=batch_log_every,
        task=task_name,
        split="train",
        epoch=epoch,
        device=str(device),
        precision=amp.precision,
    )
    iterator = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
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
            current_batch_size = int(batch_size(sample_on_device))
            if current_batch_size <= 0:
                raise ValueError(f"batch_size must return a positive integer, got {current_batch_size}")
            with _autocast_context(amp, device):
                # Losses are averaged per clip: the micro-batch contributes the sum of its clip losses, and the
                # accumulated gradients are divided by the number of clips (see _apply_optimizer_update), so every
                # batch-size/accumulation layout optimizes the same mean over clips.
                clip_loss = per_clip_loss_sum(module, sample_on_device, batch_size=current_batch_size)
            loss_sum_value = clip_loss.loss_value
            if not math.isfinite(loss_sum_value):
                raise FloatingPointError(f"non-finite loss: {loss_sum_value}")
            if amp.scaler_enabled:
                scaler.scale(clip_loss.loss_sum).backward()
            else:
                clip_loss.loss_sum.backward()
            pending_microbatches += 1
            pending_examples += clip_loss.n_clips
            if pending_microbatches == accumulate_grad_batches:
                _apply_optimizer_update(
                    module=module,
                    optimizer=optimizer,
                    accumulated_examples=pending_examples,
                    gradient_clip_val=gradient_clip_val,
                    amp=amp,
                    scaler=scaler,
                )
                optimizer_steps += 1
                pending_microbatches = 0
                pending_examples = 0
            loss_value = clip_loss.mean_loss
            for name, value in clip_loss.term_sums.items():
                total_terms[name] = total_terms.get(name, 0.0) + value
                total_term_counts[name] = total_term_counts.get(name, 0) + clip_loss.term_counts[name]
            logged_loss = _update_logged_loss(logged_loss, loss_value)
            total_loss += loss_sum_value
            total_loss_clips += clip_loss.n_clips
            total_examples += current_batch_size
            steps += 1
            last_batch_size = current_batch_size
            batch_window.add(
                step=steps,
                loss_sum=loss_sum_value,
                loss_count=clip_loss.n_clips,
                term_sums=clip_loss.term_sums,
                term_counts=clip_loss.term_counts,
                batch_size=current_batch_size,
                load_seconds=load_seconds,
                step_seconds=time.perf_counter() - step_started,
            )
            if reporter is not None:
                reporter.update(steps, logged_loss)
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
    batch_window.flush()
    if steps == 0:
        raise ValueError("train_dataloader produced no batches")
    if pending_microbatches:
        _apply_optimizer_update(
            module=module,
            optimizer=optimizer,
            accumulated_examples=pending_examples,
            gradient_clip_val=gradient_clip_val,
            amp=amp,
            scaler=scaler,
        )
        optimizer_steps += 1
    return (
        total_loss / max(total_loss_clips, 1),
        {name: value / max(total_term_counts[name], 1) for name, value in total_terms.items()},
        steps,
        optimizer_steps,
        total_examples,
        last_batch_size,
        preview_sample,
    )


def _apply_optimizer_update(
    *,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    accumulated_examples: int,
    gradient_clip_val: float,
    amp: AmpConfig,
    scaler: torch.amp.GradScaler,
) -> None:
    if accumulated_examples < 0:
        raise ValueError(f"accumulated_examples must be non-negative, got {accumulated_examples}")
    if accumulated_examples == 0:
        # No clip in this window had a defined loss: there is no gradient to apply.
        optimizer.zero_grad(set_to_none=True)
        return
    if amp.scaler_enabled:
        scaler.unscale_(optimizer)
    # Each micro-batch backpropagates the sum of its per-clip losses. Dividing by the clips actually present in
    # this window gives the mean over clips for any batch-size/accumulation layout and corrects a partial window.
    gradient_divisor = float(accumulated_examples)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad.div_(gradient_divisor)
    if gradient_clip_val > 0.0:
        torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_val)
    if amp.scaler_enabled:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


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
    compute_metric_terms: bool,
    amp: AmpConfig,
    task_name: str,
    epoch: int,
    epochs: int,
    log_every_steps: int,
    batch_log_path: Path | None,
    split: str = "val",
    aggregate: str = "clip",
    batch_log_every: int = 1,
    log: logging.Logger | None = None,
) -> tuple[float | None, dict[str, float] | None, int, int, SampleT | None]:
    """Evaluate per clip; ``aggregate="exam"`` returns examination-equal means instead of clip means."""
    if aggregate not in {"clip", "exam"}:
        raise ValueError(f"aggregate must be 'clip' or 'exam', got {aggregate!r}")
    if dataloader is None or max_steps == 0:
        return None, None, 0, 0, None
    module.eval()
    clips = ClipMetricAccumulator()
    logged_loss: float | None = None
    steps = 0
    preview_sample: SampleT | None = None
    progress = _progress_bar(
        dataloader=dataloader,
        max_steps=max_steps,
        description=f"{task_name} {epoch}/{epochs} {split}",
    )
    reporter = (
        None
        if progress is not None or log is None
        else ProgressReporter(
            log, dataloader=dataloader, max_steps=max_steps, description=f"{task_name} {epoch}/{epochs} {split}"
        )
    )
    batch_window = BatchLogWindow(
        batch_log_path,
        every=batch_log_every,
        task=task_name,
        split=split,
        epoch=epoch,
        device=str(device),
        precision=amp.precision,
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
                    item_terms = per_item_terms(module, sample_on_device, include_metrics=compute_metric_terms)
                current_batch_size = clips.add(sample_on_device, item_terms)
                finite_terms = {name: values[values.isfinite()] for name, values in item_terms.items()}
                loss_values = finite_terms.get("loss")
                if loss_values is not None and loss_values.numel() > 0:
                    logged_loss = _update_logged_loss(logged_loss, float(loss_values.mean()))
                steps += 1
                step_seconds = time.perf_counter() - step_started
                batch_window.add(
                    step=steps,
                    loss_sum=0.0 if loss_values is None else float(loss_values.sum()),
                    loss_count=0 if loss_values is None else int(loss_values.numel()),
                    term_sums={name: float(values.sum()) for name, values in finite_terms.items()},
                    term_counts={name: int(values.numel()) for name, values in finite_terms.items()},
                    batch_size=current_batch_size,
                    load_seconds=load_seconds,
                    step_seconds=step_seconds,
                )
                if reporter is not None:
                    reporter.update(steps, logged_loss)
                if progress is not None:
                    progress.update(1)
                    if _should_update_progress(steps, log_every_steps):
                        progress.set_postfix(
                            {
                                "loss": "n/a" if logged_loss is None else f"{logged_loss:.4g}",
                                "load": f"{load_seconds:.1f}s",
                                "step": f"{step_seconds:.1f}s",
                            }
                        )
        finally:
            if progress is not None:
                progress.close()
    batch_window.flush()
    if steps == 0:
        return None, None, 0, 0, None
    terms = clips.exam_means() if aggregate == "exam" else clips.clip_means()
    if "loss" not in terms:
        raise FloatingPointError(f"{split} produced no finite per-clip loss")
    return terms["loss"], terms, steps, clips.n_clips, preview_sample
