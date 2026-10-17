"""Run one benchmark cell: train a method on a task and domain, then evaluate the selected weights.

A cell can be killed at any moment and rerun with the same arguments.

- Checkpoints and result files are written atomically (``tasks.utils.atomic``), so a crash leaves the previous or the
  new complete file. After every epoch ``training_state.pt`` stores model, optimizer, scaler, RNG states, sampler and
  loader generators, early-stopping state and step counters; ``--resume-training`` continues after the last completed
  epoch and truncates the epoch and batch logs to it. Checkpoints are epoch-granular. Training is seeded but not
  bit-reproducible on GPUs.
- Stages: ``trained.json`` is written when training finishes (best epoch), ``.complete.json``
  last, after the evaluation outputs. A complete cell is skipped; a trained one is only evaluated. ``training_state.pt``
  is deleted on completion unless ``--keep-training-state``.
- ``cell.json``, ``training_state.pt``, ``trained.json``, ``.complete.json`` and ``val_metrics.json`` record a hash
  of the resolved config and step limits (workers, caches and output locations excluded). A different hash is
  refused with exit code 3; use a new ``--run-name`` or ``--force-rerun``.
- A per-cell ``.lock`` stops two launchers from running the same cell (exit code 75); a lock whose process is gone is
  taken over.
- ``train_scores.jsonl`` has one row per epoch, ``batch_scores.jsonl`` one per ``artifacts.batch_log_every_steps``
  steps. Off a terminal, progress is logged every ``ECHOXFLOW_PROGRESS_SECONDS`` (default 300) and once per epoch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import FrameType
from typing import Any, cast

import torch
import yaml

from tasks._config import TrainingConfig
from tasks.registry import (
    UnsupportedBaselineError,
    available_task_names,
    task_baseline_spec,
    task_benchmark_loss_fn,
    task_benchmark_metric_keys,
    task_benchmark_val_metrics_fn,
    task_build_dataloaders,
    task_load_config,
    task_run_training,
    task_train_yaml,
)
from tasks.utils.atomic import (
    LockHeldError,
    RunLock,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    read_json,
    remove_stale_temporaries,
)
from tasks.utils.dataset import exam_split_summary, shutdown_dataloader_workers, validated_selection_fraction
from tasks.utils.models.temporal_mean import TemporalMeanModule, fit_temporal_mean
from tasks.utils.seed import seed_everything
from tasks.utils.training import EvaluationResult, TrainingModule, configure_training_logging, evaluate_module
from tasks.utils.training.core import (
    TRAINING_STATE_NAME,
    TrainingStateMismatchError,
    discard_pending_weights,
)

_TASKS = tuple(name for name in available_task_names() if name != "video_diffusion")
_DOMAINS = ("beamspace", "cartesian")
_PROTOCOL_PROFILES = ("task_default", "echoxflow_paper_v1")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_METHODS_CONFIG = _REPO_ROOT / "configs" / "benchmark_methods.yaml"

# Cell stage files. A cell is "trained" once trained.json is written (after training finished and weights.pt is
# final) and "complete" once .complete.json is written, last, after the evaluation outputs.
CELL_FILE = "cell.json"
TRAINED_FILE = "trained.json"
COMPLETE_FILE = ".complete.json"
LOCK_FILE = ".lock"
_CELL_OUTPUTS = (
    COMPLETE_FILE,
    TRAINED_FILE,
    CELL_FILE,
    TRAINING_STATE_NAME,
    "weights.pt",
    "mean.pt",
    "val_metrics.json",
    "val_clip_metrics.csv",
    "early_stopping.json",
    "train_scores.jsonl",
    "batch_scores.jsonl",
    "loss.png",
)
# Exit codes the launcher interprets: a config mismatch or a held lock is not fixed by retrying.
EXIT_CONFIG_MISMATCH = 3
EXIT_LOCKED = 75
# Resolved-config keys that change speed or storage but not samples, model updates or metrics; left out of the hash.
_OPERATIONAL_DATA_KEYS = frozenset(
    {
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "multiprocessing_sharing_strategy",
        "sample_ref_cache",
        "sample_ref_cache_dir",
        "recording_cache",
        "recording_cache_dir",
        "recording_cache_read_only",
        "recording_cache_include",
        "recording_cache_exclude",
    }
)

LOG = logging.getLogger("tasks.bench")


class CellConfigMismatchError(RuntimeError):
    """The run directory holds results or state of a different resolved configuration."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tasks.bench", description="Run one EchoXFlow benchmark cell.")
    add = parser.add_argument
    add("--task", choices=_TASKS, required=True, help="Task to train and evaluate.")
    add("--method", required=True, help="Method from configs/benchmark_methods.yaml.")
    add("--domain", choices=_DOMAINS, required=True, help="Coordinate space of the model input.")
    add("--config", help="Task training YAML (default: the task's train.yaml).")
    add("--data-root", help="EchoXFlow data root with croissant.json (default: ECHOXFLOW_DATA_ROOT).")
    add("--seed", type=int, default=0, help="Training seed.")
    add("--output-dir", default="outputs/bench", help="Root of the run directories.")
    add("--run-name", help="Run name under the output root (default: a timestamp).")
    add("--validation-fold", type=int, help="Fold held out for the reported evaluation.")
    add("--split-csv", help="CSV with exam_id and five_fold_split columns (default: splits.csv in the data root).")
    add("--max-epochs", type=int, help="Override the epoch budget.")
    add("--max-train-steps", type=int, help="Stop each epoch after this many training steps.")
    add("--max-val-steps", type=int, help="Stop each validation pass after this many steps.")
    add("--data-fraction", type=float, help="Use this fraction of the recordings.")
    add("--resolution", type=int, help="Square input, target and Cartesian grid size in pixels.")
    add("--num-workers", type=int, help="Data-loader workers (does not change results).")
    add("--smoke-config", action="store_true", help="Apply the tiny task config from configs/bench_smoke/.")
    add(
        "--protocol-profile",
        choices=_PROTOCOL_PROFILES,
        default="task_default",
        help="Named protocol, see configs/protocols/.",
    )
    add("--resume-training", action="store_true", help="Resume from training_state.pt in the run directory if present.")
    add("--force-rerun", action="store_true", help="Discard the cell's previous state and results and run it again.")
    add("--keep-training-state", action="store_true", help="Keep training_state.pt after the cell is complete.")
    return parser


@dataclass(frozen=True)
class PlannedCell:
    run_dir: Path
    config_hash: str
    #: Data-loader workers the cell is configured with (an operational setting, outside the config hash).
    num_workers: int = 0


def plan_cell(argv: Sequence[str]) -> PlannedCell:
    """Run directory and config hash that ``tasks.bench argv`` would use, without touching the disk.

    The benchmark launcher compares the hash with the one a cell recorded, to refuse a changed configuration up front.
    """
    args = build_parser().parse_args(argv)
    method_name, method_spec = _resolve_method(args.method, _load_method_specs(_METHODS_CONFIG))
    run_dir = _cell_run_dir(args, _run_name(args.run_name))
    config = _resolved_config(args, method_name=method_name, method_spec=method_spec, run_dir=run_dir)
    return PlannedCell(
        run_dir=run_dir,
        config_hash=_config_hash_from_args(args, config),
        num_workers=int(config.raw.get("data", {}).get("num_workers", 0) or 0),
    )


def recorded_config_hash(run_dir: Path) -> str | None:
    """Config hash recorded by a started, trained or complete cell, if any."""
    for name in (COMPLETE_FILE, TRAINED_FILE, CELL_FILE):
        recorded = read_json(run_dir / name)
        if recorded is not None and recorded.get("config_hash"):
            return str(recorded["config_hash"])
    return None


def _cell_run_dir(args: argparse.Namespace, run_name: str) -> Path:
    run_dir = Path(str(args.output_dir)) / str(args.task) / run_name / str(args.method) / str(args.domain)
    if args.validation_fold is not None:
        run_dir /= f"fold={args.validation_fold}"
    return (run_dir / f"seed={args.seed}").resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    method_specs = _load_method_specs(_METHODS_CONFIG)
    method_name, method_spec = _resolve_method(args.method, method_specs)

    run_name = _run_name(args.run_name)
    run_dir = _cell_run_dir(args, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_training_logging()
    label = f"{args.task}/{run_name}/{args.method}/{args.domain}" + (
        "" if args.validation_fold is None else f"/fold={args.validation_fold}"
    )
    lock = RunLock(run_dir / LOCK_FILE, label=f"benchmark cell {label}")
    try:
        lock.acquire()
    except LockHeldError as exc:
        LOG.error("%s", exc)
        return EXIT_LOCKED
    previous_handler = signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        return _run_cell(
            args,
            parser=parser,
            method_name=method_name,
            method_spec=method_spec,
            run_name=run_name,
            run_dir=run_dir,
            label=label,
        )
    except (CellConfigMismatchError, TrainingStateMismatchError) as exc:
        LOG.error("refusing to run %s: %s", label, exc)
        return EXIT_CONFIG_MISMATCH
    except Exception:
        LOG.exception("benchmark cell %s failed", label)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        lock.release()


def _config_hash_from_args(args: argparse.Namespace, config: TrainingConfig) -> str:
    return cell_config_hash(config, max_train_steps=args.max_train_steps, max_val_steps=args.max_val_steps)


def _raise_on_sigterm(signum: int, frame: FrameType | None) -> None:
    del frame
    LOG.warning("received signal %d; stopping (a rerun with --resume-training resumes from the last epoch)", signum)
    # Unwind normally so the lock is released and temporary files are removed; every output is written atomically.
    raise SystemExit(128 + signum)


def _run_cell(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    method_name: str,
    method_spec: Mapping[str, Any],
    run_name: str,
    run_dir: Path,
    label: str,
) -> int:
    started = time.perf_counter()
    config = _resolved_config(args, method_name=method_name, method_spec=method_spec, run_dir=run_dir)
    try:
        selection = _selection_settings(config)
    except ValueError as exc:
        parser.error(str(exc))
    config_hash = _config_hash_from_args(args, config)
    if args.force_rerun:
        removed = _reset_cell(run_dir)
        if removed:
            LOG.info("--force-rerun: removed %s", ", ".join(removed))
    remove_stale_temporaries(run_dir)
    _check_cell_identity(run_dir, config_hash)
    _log_header(label=label, run_dir=run_dir, config_hash=config_hash)

    complete = read_json(run_dir / COMPLETE_FILE)
    if complete is not None:
        LOG.info("cell already complete (%s); nothing to do", run_dir / COMPLETE_FILE)
        _discard_training_state(run_dir, keep=args.keep_training_state)
        print(json.dumps(complete.get("metrics", {}), sort_keys=True))
        return 0
    _write_cell_identity(run_dir, config_hash=config_hash, label=label, args=args, run_name=run_name)
    config_path = run_dir / "config.yaml"
    _write_config(config_path, config)

    evaluation: EvaluationResult | None = None
    val_loader: Any | None = None
    trained: Mapping[str, Any] | None = None
    weights = run_dir / "weights.pt"
    if _method_kind(method_spec) == "trainable":
        trained = _finished_training(run_dir, config_hash)
        if trained is not None:
            LOG.info(
                "training already finished (%d epochs, best epoch %s); reusing %s",
                int(trained.get("epochs_completed", 0)),
                trained.get("best_epoch"),
                weights,
            )
        else:
            trained, val_loader = _train_trainable(
                args, config_path=config_path, selection=selection, config_hash=config_hash, run_dir=run_dir
            )
        if val_loader is None:
            _train_loader, val_loader, _selection_loader = _build_benchmark_dataloaders(
                args.task, config_path=config_path, data_root=args.data_root
            )
        LOG.info("evaluating %s on the validation fold", weights)
        evaluation = _evaluate_trainable(args.task, config_path=config_path, weights=weights, val_loader=val_loader)
    else:
        try:
            baseline_name = str(method_spec.get("baseline", method_name))
            task_baseline_spec(args.task, baseline_name)
            train_loader, val_loader, _selection_loader = _build_benchmark_dataloaders(
                args.task, config_path=config_path, data_root=args.data_root
            )
            model = _fit_baseline(
                args.task,
                baseline=baseline_name,
                config_path=config_path,
                data_root=args.data_root,
                run_dir=run_dir,
                train_loader=train_loader,
            )
            evaluation = _evaluate_baseline(
                args.task,
                baseline=baseline_name,
                config_path=config_path,
                data_root=args.data_root,
                model=model,
                val_loader=val_loader,
            )
        except UnsupportedBaselineError as exc:
            parser.error(str(exc))

    if evaluation is not None:
        metric_keys = task_benchmark_metric_keys(args.task)
        evaluation.accumulator.write_csv(run_dir / "val_clip_metrics.csv", rename=metric_keys)
        metrics = _select_metrics(evaluation.exam_means, metric_keys)
        wallclock_s = time.perf_counter() - started
        _write_val_metrics(
            run_dir=run_dir,
            task=args.task,
            method=args.method,
            domain=args.domain,
            seed=args.seed,
            run_name=run_name,
            validation_fold=args.validation_fold,
            fold_split_path=args.split_csv,
            metrics=metrics,
            clip_mean_metrics=_select_metrics(evaluation.clip_means, metric_keys),
            n_val_clips=evaluation.n_clips,
            n_val_exams=evaluation.n_exams,
            wallclock_s=wallclock_s,
            config_hash=config_hash,
        )
        # Written last: a cell is complete only when every output above is on disk.
        atomic_write_json(
            run_dir / COMPLETE_FILE,
            {
                "schema_version": 1,
                "config_hash": config_hash,
                "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_sha": _git_sha(),
                "metrics": metrics,
                "n_val_clips": int(evaluation.n_clips),
                "n_val_exams": int(evaluation.n_exams),
                "training": None if trained is None else dict(trained),
                "outputs": ["val_metrics.json", "val_clip_metrics.csv"],
            },
        )
        _discard_training_state(run_dir, keep=args.keep_training_state)
        LOG.info("cell complete in %.1fs: %s", wallclock_s, json.dumps(metrics, sort_keys=True))
        print(json.dumps(metrics, sort_keys=True))
    return 0


def _train_trainable(
    args: argparse.Namespace,
    *,
    config_path: Path,
    selection: _SelectionSettings,
    config_hash: str,
    run_dir: Path,
) -> tuple[dict[str, Any], Any]:
    """Train (or resume) the cell, write trained.json, and return it with the validation loader."""
    train_loader, val_loader, selection_loader = _build_benchmark_dataloaders(
        args.task,
        config_path=config_path,
        data_root=args.data_root,
    )
    # The reported validation fold never reaches training: no in-training validation and no checkpoint selection on
    # it. Without a selection split, training runs the fixed epoch budget and keeps the last epoch's weights; with
    # one, the selection exams (removed from training) pick the best epoch and drive early stopping, and the
    # validation fold is only used for the final evaluation.
    selection_split = (
        None
        if selection_loader is None
        else {
            **exam_split_summary(train=train_loader, val=val_loader, selection=selection_loader),
            "selection_fraction": selection.fraction,
            "selection_seed": selection.seed,
        }
    )
    result = task_run_training(args.task)(
        config_path=config_path,
        data_root=args.data_root,
        train_dataloader=train_loader,
        val_dataloader=None,
        selection_dataloader=selection_loader,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        early_stopping_patience=selection.patience,
        early_stopping_min_delta=selection.min_delta,
        selection_split=selection_split,
        resume=args.resume_training,
        config_hash=config_hash,
    )
    weights = run_dir / "weights.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"training finished without {weights}")
    trained: dict[str, Any] = {
        "schema_version": 1,
        "config_hash": config_hash,
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "epochs_completed": int(getattr(result, "epochs_completed", 0) or 0),
        "best_epoch": getattr(result, "best_epoch", None),
        "best_selection_loss": getattr(result, "best_selection_loss", None),
        "stopped_early": bool(getattr(result, "stopped_early", False)),
        "weights_bytes": weights.stat().st_size,
    }
    atomic_write_json(run_dir / TRAINED_FILE, trained)
    LOG.info(
        "training finished: %d epochs, best epoch %s, weights %.1f MB",
        trained["epochs_completed"],
        trained["best_epoch"],
        trained["weights_bytes"] / 1e6,
    )
    return trained, val_loader


def _finished_training(run_dir: Path, config_hash: str) -> dict[str, Any] | None:
    """The trained.json of a finished training run of this config whose weights.pt exists, else ``None``."""
    trained = read_json(run_dir / TRAINED_FILE)
    weights = run_dir / "weights.pt"
    if trained is None or trained.get("config_hash") != config_hash or not weights.is_file():
        return None
    return trained


def cell_config_hash(
    config: TrainingConfig,
    *,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
) -> str:
    """Hash of everything that determines a cell's results: the resolved config and the step limits.

    Data-loading performance and cache settings and the artifact locations are left out, so moving the output root
    or changing worker counts still resumes; any change to the model, data, protocol, optimizer, loss, trainer or
    metric settings gives a different hash.
    """
    raw = copy.deepcopy(dict(config.raw))
    raw.pop("artifacts", None)
    data = raw.get("data")
    if isinstance(data, dict):
        raw["data"] = {key: value for key, value in data.items() if key not in _OPERATIONAL_DATA_KEYS}
    payload = {
        "config": raw,
        "limits": {
            "max_train_steps": max_train_steps,
            "max_val_steps": max_val_steps,
            "max_eval_steps": None,
        },
        "weights": None,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _check_cell_identity(run_dir: Path, config_hash: str) -> None:
    for name in (COMPLETE_FILE, TRAINED_FILE, CELL_FILE):
        recorded = read_json(run_dir / name)
        if recorded is None or recorded.get("config_hash") in {None, config_hash}:
            continue
        raise CellConfigMismatchError(
            f"{run_dir} holds a run with config hash {recorded.get('config_hash')} ({name}), but the current "
            f"settings resolve to {config_hash}. Use a new --run-name, or --force-rerun to discard the old cell."
        )


def _write_cell_identity(
    run_dir: Path, *, config_hash: str, label: str, args: argparse.Namespace, run_name: str
) -> None:
    previous = read_json(run_dir / CELL_FILE) or {}
    atomic_write_json(
        run_dir / CELL_FILE,
        {
            "schema_version": 1,
            "cell": label,
            "task": args.task,
            "method": args.method,
            "domain": args.domain,
            "validation_fold": args.validation_fold,
            "seed": int(args.seed),
            "run_name": run_name,
            "config_hash": config_hash,
            "created_utc": previous.get("created_utc") or datetime.now(UTC).isoformat(timespec="seconds"),
            "attempts": int(previous.get("attempts", 0)) + 1,
            "last_attempt_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )


def _reset_cell(run_dir: Path) -> list[str]:
    discard_pending_weights(run_dir)
    removed = []
    for name in _CELL_OUTPUTS:
        path = run_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def _discard_training_state(run_dir: Path, *, keep: bool) -> None:
    discard_pending_weights(run_dir)
    state = run_dir / TRAINING_STATE_NAME
    if keep or not state.is_file():
        return
    size = state.stat().st_size
    state.unlink()
    LOG.info("removed %s (%.1f MB); pass --keep-training-state to keep it", state.name, size / 1e6)


def _log_header(*, label: str, run_dir: Path, config_hash: str) -> None:
    LOG.info("cell %s", label)
    LOG.info("run_dir=%s config_hash=%s git=%s", run_dir, config_hash, _git_description())
    LOG.info(
        "host=%s pid=%d CUDA_VISIBLE_DEVICES=%s device=%s",
        socket.gethostname(),
        os.getpid(),
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )


def _write_config(path: Path, config: TrainingConfig) -> None:
    atomic_write_text(path, yaml.safe_dump(config.raw, sort_keys=False))


def _resolved_config(
    args: argparse.Namespace, *, method_name: str, method_spec: Mapping[str, Any], run_dir: Path
) -> TrainingConfig:
    task, domain, method = args.task, args.domain, method_name
    raw = _load_yaml_mapping(Path(args.config) if args.config else task_train_yaml(task))
    if args.smoke_config:
        raw = _merge_config(raw, _load_yaml_mapping(_smoke_config_path(task)))
    raw = _apply_method_config(raw, method_spec, method=method)
    raw = _apply_protocol_profile(raw, task=task, profile=args.protocol_profile)
    raw = _apply_method_overrides(raw, method_spec)
    raw = _apply_spatial_resolution(raw, task=task, spatial_resolution=args.resolution)
    raw = _apply_resolution_method_overrides(raw, method_spec, spatial_resolution=args.resolution)
    data = _section(raw, "data")
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        data["num_workers"] = int(args.num_workers)
    data["seed"] = int(args.seed)
    if args.validation_fold is not None:
        data["validation_fold"] = int(args.validation_fold)
    if args.split_csv is not None:
        data["fold_split_path"] = str(args.split_csv)
    if args.data_fraction is not None:
        data["data_fraction"] = float(args.data_fraction)
    data["coordinate_space"] = str(domain)
    # Learned methods finish each sample's geometric preprocessing on their GPU. The temporal-mean baseline runs on
    # CPU, where the data-loader workers keep doing it in parallel.
    data["device_preprocessing"] = str(method) != "temporal_mean"
    if domain == "cartesian":
        data.setdefault("cartesian_height", data.get("input_spatial_shape", (256, 256))[0])
    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ValueError("--max-epochs must be positive")
        _section(raw, "trainer")["max_epochs"] = int(args.max_epochs)
    artifacts = _section(raw, "artifacts")
    artifacts["root_dir"] = str(run_dir.parent)
    artifacts["run_name"] = run_dir.name
    artifacts["append_timestamp"] = False
    artifacts.setdefault("epoch_scores_name", "train_scores.jsonl")
    artifacts["write_epoch_previews"] = False
    artifacts["debug_input_target_pngs"] = False
    benchmark = _section(raw, "benchmark")
    benchmark["method"] = str(method)
    benchmark["protocol_profile"] = str(args.protocol_profile)
    return TrainingConfig(raw=raw)


def _apply_spatial_resolution(
    raw: Mapping[str, Any],
    *,
    task: str,
    spatial_resolution: int | None,
) -> dict[str, Any]:
    if spatial_resolution is None:
        return dict(raw)
    resolution = int(spatial_resolution)
    if resolution <= 0:
        raise ValueError("--resolution must be positive")
    shape = [resolution, resolution]
    data_override: dict[str, Any] = {
        "input_spatial_shape": shape,
        "cartesian_height": resolution,
    }
    if task in {"tissue_doppler", "color_doppler"}:
        data_override["target_spatial_shape"] = shape
    elif task == "segmentation":
        data_override["spatial_shape"] = shape
    else:
        raise ValueError(f"Spatial-resolution override does not support task {task!r}")
    return _merge_config(
        raw,
        {
            "model": {"target_spatial_shape": shape},
            "data": data_override,
        },
    )


def _apply_protocol_profile(raw: Mapping[str, Any], *, task: str, profile: str) -> dict[str, Any]:
    if profile == "task_default":
        return dict(raw)
    spec = _load_yaml_mapping(_REPO_ROOT / "configs" / "protocols" / f"{profile}.yaml")
    if task not in spec["tasks"]:
        raise ValueError(f"Protocol profile {profile!r} does not support task {task!r}")
    return _merge_config(_merge_config(raw, spec["common"]), spec["tasks"][task])


def _load_method_specs(path: Path) -> dict[str, Mapping[str, Any]]:
    raw = _load_yaml_mapping(path)
    methods = raw.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise ValueError(f"benchmark methods config must define a non-empty `methods` mapping: {path}")
    specs: dict[str, Mapping[str, Any]] = {}
    for name, spec in methods.items():
        if not isinstance(spec, Mapping):
            raise TypeError(f"benchmark method {name!r} must be a mapping")
        specs[str(name)] = cast(Mapping[str, Any], spec)
    aliases = raw.get("aliases", {})
    if aliases is not None and not isinstance(aliases, Mapping):
        raise TypeError("benchmark methods `aliases` must be a mapping")
    for alias, target in cast(Mapping[str, object], aliases).items():
        target_name = str(target)
        if target_name not in specs:
            raise ValueError(f"benchmark method alias {alias!r} points to unknown method {target_name!r}")
        specs[str(alias)] = {"architecture": target_name, **specs[target_name]}
    return specs


def _resolve_method(method: str, specs: Mapping[str, Mapping[str, Any]]) -> tuple[str, Mapping[str, Any]]:
    if method not in specs:
        choices = ", ".join(sorted(specs))
        raise SystemExit(f"Unsupported benchmark method {method!r}; expected one of: {choices}")
    spec = specs[method]
    return str(spec.get("name", method)), spec


def _method_kind(spec: Mapping[str, Any]) -> str:
    kind = str(spec.get("kind", "trainable")).strip()
    if kind not in {"trainable", "baseline"}:
        raise ValueError(f"benchmark method kind must be 'trainable' or 'baseline', got {kind!r}")
    return kind


def _apply_method_config(raw: Mapping[str, Any], spec: Mapping[str, Any], *, method: str) -> dict[str, Any]:
    updated = _apply_method_overrides(raw, spec)
    transform = spec.get("config_transform")
    if transform is not None:
        transformed = _load_dotted_callable(str(transform))(updated)
        if not isinstance(transformed, dict):
            raise TypeError(f"benchmark method config_transform {transform!r} must return a dict")
        return transformed
    if _method_kind(spec) != "trainable":
        return updated
    from tasks.utils.models.registry import METHODS, method_config

    architecture = str(spec.get("architecture", method))
    if architecture not in METHODS:
        raise ValueError(f"benchmark method {method!r} uses unknown architecture {architecture!r}")
    return method_config(updated, architecture)


def _apply_method_overrides(raw: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(raw)
    overrides = spec.get("config_overrides")
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise TypeError("benchmark method config_overrides must be a mapping")
        updated = _merge_config(updated, cast(Mapping[str, Any], overrides))
    return updated


def _apply_resolution_method_overrides(
    raw: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    spatial_resolution: int | None,
) -> dict[str, Any]:
    if spatial_resolution is None:
        return dict(raw)
    overrides = spec.get("resolution_overrides")
    if overrides is None:
        return dict(raw)
    if not isinstance(overrides, Mapping):
        raise TypeError("benchmark method resolution_overrides must be a mapping")
    resolution_override = overrides.get(str(int(spatial_resolution)), overrides.get(int(spatial_resolution)))
    if resolution_override is None:
        return dict(raw)
    if not isinstance(resolution_override, Mapping):
        raise TypeError(f"benchmark resolution override {spatial_resolution} must be a mapping")
    return _merge_config(raw, cast(Mapping[str, Any], resolution_override))


def _load_dotted_callable(path: str) -> Any:
    module_name, _, attr = path.replace(":", ".").rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid dotted callable path: {path!r}")
    candidate = getattr(import_module(module_name), attr)
    if not callable(candidate):
        raise TypeError(f"{path!r} must resolve to a callable")
    return candidate


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"benchmark config must be a YAML mapping: {path}")
    return cast(dict[str, Any], raw)


def _smoke_config_path(task: str) -> Path:
    return _REPO_ROOT / "configs" / "bench_smoke" / f"{task}.yaml"


def _merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[str(key)] = _merge_config(current, value)
        else:
            merged[str(key)] = value
    return merged


def _fit_baseline(
    task: str,
    *,
    baseline: str,
    config_path: Path,
    data_root: str | None,
    run_dir: Path,
    train_loader: Any | None = None,
) -> torch.nn.Module:
    config = task_load_config(task)(config_path)
    baseline_spec = task_baseline_spec(task, baseline)
    if train_loader is None:
        train_loader = _build_benchmark_dataloaders(task, config_path=config_path, data_root=data_root)[0]
    try:
        model = fit_temporal_mean(
            train_loader,
            task_kind=task,
            target_extractor=baseline_spec.target_extractor,
            temporal_upsample_factor=baseline_spec.temporal_upsample_factor_fn(config),
        )
    finally:
        shutdown_dataloader_workers(train_loader)
    atomic_torch_save(model.state_dict(), run_dir / "mean.pt")
    return model


def _evaluate_baseline(
    task: str,
    *,
    baseline: str,
    config_path: Path,
    data_root: str | None,
    model: torch.nn.Module,
    val_loader: Any | None = None,
) -> EvaluationResult:
    task_baseline_spec(task, baseline)
    config = task_load_config(task)(config_path)
    if val_loader is None:
        val_loader = _build_benchmark_dataloaders(task, config_path=config_path, data_root=data_root)[1]
    module = TemporalMeanModule(
        config=config,
        model=cast(Any, model),
        loss_fn=task_benchmark_loss_fn(task),
        val_metrics_fn=task_benchmark_val_metrics_fn(task),
    )
    return evaluate_module(module=module, dataloader=val_loader, config=config, max_steps=None)


def _evaluate_trainable(
    task: str,
    *,
    config_path: Path,
    weights: Path,
    val_loader: Any,
) -> EvaluationResult:
    config = task_load_config(task)(config_path)
    seed_everything(int(config.data.get("seed", 0)))
    module = TrainingModule(
        config=config,
        loss_fn=task_benchmark_loss_fn(task),
        val_metrics_fn=task_benchmark_val_metrics_fn(task),
    )
    module.model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    return evaluate_module(module=module, dataloader=val_loader, config=config, max_steps=None)


def _select_metrics(terms: Mapping[str, float], metric_keys: Mapping[str, str]) -> dict[str, float]:
    return {out_key: float(terms[src_key]) for src_key, out_key in metric_keys.items() if src_key in terms}


def _build_benchmark_dataloaders(task: str, *, config_path: Path, data_root: str | None) -> tuple[Any, Any, Any]:
    """Return ``(train, val, selection)``; ``selection`` is ``None`` without a selection split."""
    config = task_load_config(task)(config_path)
    loaders = task_build_dataloaders(task)(config=config, data_root=data_root)
    selection = loaders[2] if len(loaders) > 2 else None
    exam_split_summary(train=loaders[0], val=loaders[1], selection=selection)
    return loaders[0], loaders[1], selection


@dataclass(frozen=True)
class _SelectionSettings:
    fraction: float
    seed: int
    patience: int | None
    min_delta: float


def _selection_settings(config: TrainingConfig) -> _SelectionSettings:
    """Validate the model-selection settings of a resolved config: early stopping needs a selection split."""
    data = config.section("data")
    trainer = config.section("trainer")
    fraction = validated_selection_fraction(data.get("selection_fraction"))
    raw_patience = trainer.get("early_stopping_patience")
    patience = None if raw_patience is None else int(cast(Any, raw_patience))
    min_delta = float(cast(Any, trainer.get("early_stopping_min_delta", 0.0) or 0.0))
    if patience is not None and patience <= 0:
        raise ValueError("trainer.early_stopping_patience must be positive")
    if min_delta < 0:
        raise ValueError("trainer.early_stopping_min_delta must be non-negative")
    if fraction <= 0.0 and (patience is not None or "early_stopping_min_delta" in trainer):
        raise ValueError(
            "early stopping needs a model-selection split separate from the reported validation fold; "
            "set data.selection_fraction > 0 (the echoxflow_paper_v1 profile uses 0.125)"
        )
    return _SelectionSettings(
        fraction=fraction,
        seed=int(cast(Any, data.get("selection_seed", 0))),
        patience=patience,
        min_delta=min_delta,
    )


def _write_val_metrics(
    *,
    run_dir: Path,
    task: str,
    method: str,
    domain: str,
    seed: int,
    run_name: str,
    validation_fold: int | None,
    fold_split_path: str | None,
    metrics: Mapping[str, float],
    clip_mean_metrics: Mapping[str, float],
    n_val_clips: int,
    n_val_exams: int,
    wallclock_s: float,
    config_hash: str | None = None,
) -> None:
    payload = {
        "task": task,
        "method": method,
        "domain": domain,
        "seed": int(seed),
        "run_name": str(run_name),
        # Headline metrics are examination-equal: per-clip values averaged within each exam, then over exams.
        "metrics": {key: float(value) for key, value in metrics.items()},
        "metric_aggregation": "exam_equal_mean_of_clip_means",
        "clip_mean_metrics": {key: float(value) for key, value in clip_mean_metrics.items()},
        "n_val_clips": int(n_val_clips),
        "n_val_exams": int(n_val_exams),
        "git_sha": _git_sha(),
        "config_hash": config_hash,
        "wallclock_s": round(float(wallclock_s), 2),
    }
    if validation_fold is not None:
        payload["validation_fold"] = int(validation_fold)
    if fold_split_path is not None:
        payload["fold_split_path"] = str(fold_split_path)
    atomic_write_json(run_dir / "val_metrics.json", payload)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.setdefault(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"config section {name!r} must be a mapping")
    return cast(dict[str, Any], value)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "--short", "HEAD"), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_description() -> str:
    sha = _git_sha()
    try:
        status = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=no"), text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return sha
    return f"{sha}{'-dirty' if status.strip() else ''}"


def _run_name(value: str | None) -> str:
    text = datetime.now().strftime("%Y%m%d_%H%M%S") if value is None else str(value).strip()
    if not text:
        raise ValueError("--run-name must not be empty")
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in text)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
