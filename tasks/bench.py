from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import import_module
from pathlib import Path
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
    task_evaluate,
    task_load_config,
    task_run_training,
    task_train_yaml,
)
from tasks.utils.models.temporal_mean import TemporalMeanModule, fit_temporal_mean
from tasks.utils.training import evaluate_module

_DOMAINS = ("beamspace", "cartesian")
_STAGES = ("train", "evaluate", "train_and_evaluate")
_DEFAULT_METHODS_CONFIG = Path("configs/benchmark_methods.yaml")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EchoXFlow benchmark cells.")
    parser.add_argument("--task", choices=available_task_names(), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--domain", choices=_DOMAINS, required=True)
    parser.add_argument("--stage", choices=_STAGES, default="train_and_evaluate")
    parser.add_argument("--methods-config", default=str(_DEFAULT_METHODS_CONFIG))
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="outputs/bench")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--validation-fold", type=int, default=None)
    parser.add_argument("--fold-split-path", default=None)
    parser.add_argument("--weights")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--max-eval-steps", type=int)
    parser.add_argument("--data-fraction", type=float)
    parser.add_argument(
        "--sample-cache-dir",
        help="Enable the generic preprocessed sample cache and store cached samples under this directory.",
    )
    parser.add_argument(
        "--sample-cache-read-only",
        action="store_true",
        help="Read samples from --sample-cache-dir without writing cache misses.",
    )
    parser.add_argument(
        "--recording-cache-dir",
        help="Cache each full source recording array once and slice task samples from those cached arrays.",
    )
    parser.add_argument(
        "--recording-cache-read-only",
        action="store_true",
        help="Read arrays from --recording-cache-dir without writing cache misses.",
    )
    parser.add_argument(
        "--recording-cache-include",
        action="append",
        help="Only cache matching recording array paths; may be repeated and supports shell-style globs.",
    )
    parser.add_argument(
        "--recording-cache-exclude",
        action="append",
        help="Do not cache matching recording array paths; may be repeated and supports shell-style globs.",
    )
    parser.add_argument(
        "--smoke-config",
        action="store_true",
        help="Use the task smoke benchmark config from configs/bench_smoke/.",
    )
    args = parser.parse_args(argv)
    method_specs = _load_method_specs(Path(args.methods_config))
    method_name, method_spec = _resolve_method(args.method, method_specs)

    started = time.perf_counter()
    run_name = _run_name(args.run_name)
    run_dir = Path(args.output_root) / args.task / run_name / args.method / args.domain
    if args.validation_fold is not None:
        run_dir /= f"fold={args.validation_fold}"
    run_dir = (run_dir / f"seed={args.seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _resolved_config(
        task=args.task,
        method=method_name,
        method_spec=method_spec,
        domain=args.domain,
        seed=args.seed,
        config_path=args.config,
        run_dir=run_dir,
        data_fraction=args.data_fraction,
        validation_fold=args.validation_fold,
        fold_split_path=args.fold_split_path,
        sample_cache_dir=args.sample_cache_dir,
        sample_cache_read_only=args.sample_cache_read_only,
        recording_cache_dir=args.recording_cache_dir,
        recording_cache_read_only=args.recording_cache_read_only,
        recording_cache_include=args.recording_cache_include,
        recording_cache_exclude=args.recording_cache_exclude,
        smoke_config=args.smoke_config,
    )
    config_path = run_dir / "config.yaml"
    _write_config(config_path, config)

    metrics: dict[str, float] | None = None
    val_loader: Any | None = None
    weights = Path(args.weights) if args.weights else None
    if _method_kind(method_spec) == "trainable":
        if args.stage in {"train", "train_and_evaluate"}:
            train_loader = None
            if args.stage == "train_and_evaluate":
                train_loader, val_loader = _build_benchmark_dataloaders(
                    args.task,
                    config_path=config_path,
                    data_root=args.data_root,
                )
            result = task_run_training(args.task)(
                config_path=config_path,
                data_root=args.data_root,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                max_epochs=args.max_epochs,
                max_train_steps=args.max_train_steps,
                max_val_steps=args.max_val_steps,
            )
            checkpoint = getattr(result, "checkpoint_path", None)
            weights = Path(checkpoint) if checkpoint else weights
        if args.stage in {"evaluate", "train_and_evaluate"}:
            _write_config(config_path, config)
            if weights is None:
                weights = run_dir / "weights.pt"
            if val_loader is None:
                _train_loader, val_loader = _build_benchmark_dataloaders(
                    args.task,
                    config_path=config_path,
                    data_root=args.data_root,
                )
            metrics = task_evaluate(args.task)(
                config_path=config_path,
                data_root=args.data_root,
                weights=weights,
                eval_dataloader=val_loader,
                max_eval_steps=args.max_eval_steps,
            )
    else:
        try:
            baseline_name = str(method_spec.get("baseline", method_name))
            task_baseline_spec(args.task, baseline_name)
            train_loader, val_loader = _build_benchmark_dataloaders(
                args.task,
                config_path=config_path,
                data_root=args.data_root,
            )
            if args.stage == "train":
                _fit_baseline(
                    args.task,
                    baseline=baseline_name,
                    config_path=config_path,
                    data_root=args.data_root,
                    run_dir=run_dir,
                    train_loader=train_loader,
                )
            else:
                model = _fit_baseline(
                    args.task,
                    baseline=baseline_name,
                    config_path=config_path,
                    data_root=args.data_root,
                    run_dir=run_dir,
                    train_loader=train_loader,
                )
                _write_config(config_path, config)
                metrics = _evaluate_baseline(
                    args.task,
                    baseline=baseline_name,
                    config_path=config_path,
                    data_root=args.data_root,
                    model=model,
                    max_eval_steps=args.max_eval_steps,
                    val_loader=val_loader,
                )
        except UnsupportedBaselineError as exc:
            parser.error(str(exc))

    if metrics is not None:
        _write_val_metrics(
            run_dir=run_dir,
            task=args.task,
            method=args.method,
            domain=args.domain,
            seed=args.seed,
            run_name=run_name,
            validation_fold=args.validation_fold,
            fold_split_path=args.fold_split_path,
            metrics=metrics,
            n_val_clips=(
                _val_clip_count_from_loader(val_loader)
                if val_loader is not None
                else _val_clip_count(args.task, config_path=config_path, data_root=args.data_root)
            ),
            wallclock_s=time.perf_counter() - started,
        )
        print(json.dumps(metrics, sort_keys=True))
    return 0


def _write_config(path: Path, config: TrainingConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8")


def _resolved_config(
    *,
    task: str,
    method: str,
    method_spec: Mapping[str, Any],
    domain: str,
    seed: int,
    config_path: str | None,
    run_dir: Path,
    data_fraction: float | None = None,
    validation_fold: int | None = None,
    fold_split_path: str | None = None,
    sample_cache_dir: str | None = None,
    sample_cache_read_only: bool = False,
    recording_cache_dir: str | None = None,
    recording_cache_read_only: bool = False,
    recording_cache_include: Sequence[str] | None = None,
    recording_cache_exclude: Sequence[str] | None = None,
    smoke_config: bool = False,
) -> TrainingConfig:
    raw = _load_yaml_mapping(Path(config_path) if config_path else task_train_yaml(task))
    if smoke_config:
        raw = _merge_config(raw, _load_yaml_mapping(_smoke_config_path(task)))
    raw = _apply_method_config(raw, method_spec)
    data = _section(raw, "data")
    data["seed"] = int(seed)
    if validation_fold is not None:
        data["validation_fold"] = int(validation_fold)
    if fold_split_path is not None:
        data["fold_split_path"] = str(fold_split_path)
    if data_fraction is not None:
        data["data_fraction"] = float(data_fraction)
    if sample_cache_dir is not None:
        data["sample_cache_dir"] = str(sample_cache_dir)
        if sample_cache_read_only:
            data["sample_cache_read_only"] = True
    if recording_cache_dir is not None:
        data["recording_cache_dir"] = str(recording_cache_dir)
        if recording_cache_read_only:
            data["recording_cache_read_only"] = True
        if recording_cache_include:
            data["recording_cache_include"] = [str(path) for path in recording_cache_include]
        if recording_cache_exclude:
            data["recording_cache_exclude"] = [str(path) for path in recording_cache_exclude]
    data["coordinate_space"] = str(domain)
    if domain == "cartesian":
        data.setdefault("cartesian_height", data.get("input_spatial_shape", (256, 256))[0])
    artifacts = _section(raw, "artifacts")
    artifacts["root_dir"] = str(run_dir.parent)
    artifacts["run_name"] = run_dir.name
    artifacts["append_timestamp"] = False
    artifacts.setdefault("epoch_scores_name", "train_scores.jsonl")
    artifacts["write_epoch_previews"] = False
    artifacts["debug_input_target_pngs"] = False
    _section(raw, "benchmark")["method"] = str(method)
    return TrainingConfig(raw=raw)


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
        specs[str(alias)] = specs[target_name]
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


def _apply_method_config(raw: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(raw)
    overrides = spec.get("config_overrides")
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise TypeError("benchmark method config_overrides must be a mapping")
        updated = _merge_config(updated, cast(Mapping[str, Any], overrides))
    transform = spec.get("config_transform")
    if transform is None:
        return updated
    transformed = _load_dotted_callable(str(transform))(updated)
    if not isinstance(transformed, dict):
        raise TypeError(f"benchmark method config_transform {transform!r} must return a dict")
    return transformed


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
    return Path(__file__).resolve().parents[1] / "configs" / "bench_smoke" / f"{task}.yaml"


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
    model = fit_temporal_mean(
        train_loader,
        task_kind=task,
        target_extractor=baseline_spec.target_extractor,
        temporal_upsample_factor=baseline_spec.temporal_upsample_factor_fn(config),
    )
    torch.save(model.state_dict(), run_dir / "mean.pt")
    return model


def _evaluate_baseline(
    task: str,
    *,
    baseline: str,
    config_path: Path,
    data_root: str | None,
    model: torch.nn.Module,
    max_eval_steps: int | None,
    val_loader: Any | None = None,
) -> dict[str, float]:
    task_baseline_spec(task, baseline)
    config = task_load_config(task)(config_path)
    if val_loader is None:
        _train_loader, val_loader = _build_benchmark_dataloaders(task, config_path=config_path, data_root=data_root)
    module = TemporalMeanModule(
        config=config,
        model=cast(Any, model),
        loss_fn=task_benchmark_loss_fn(task),
        val_metrics_fn=task_benchmark_val_metrics_fn(task),
    )
    terms, _steps = evaluate_module(module=module, dataloader=val_loader, config=config, max_steps=max_eval_steps)
    return {
        out_key: terms[src_key] for src_key, out_key in task_benchmark_metric_keys(task).items() if src_key in terms
    }


def _build_benchmark_dataloaders(task: str, *, config_path: Path, data_root: str | None) -> tuple[Any, Any]:
    config = task_load_config(task)(config_path)
    loaders = task_build_dataloaders(task)(config=config, data_root=data_root)
    return loaders[0], loaders[1]


def _val_clip_count(task: str, *, config_path: Path, data_root: str | None) -> int:
    _train_loader, loader = _build_benchmark_dataloaders(task, config_path=config_path, data_root=data_root)
    return _val_clip_count_from_loader(loader)


def _val_clip_count_from_loader(loader: object) -> int:
    dataset = getattr(loader, "dataset", None)
    return 0 if dataset is None else int(len(dataset))


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
    n_val_clips: int,
    wallclock_s: float,
) -> None:
    payload = {
        "task": task,
        "method": method,
        "domain": domain,
        "seed": int(seed),
        "run_name": str(run_name),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "n_val_clips": int(n_val_clips),
        "git_sha": _git_sha(),
        "wallclock_s": round(float(wallclock_s), 2),
    }
    if validation_fold is not None:
        payload["validation_fold"] = int(validation_fold)
    if fold_split_path is not None:
        payload["fold_split_path"] = str(fold_split_path)
    (run_dir / "val_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.setdefault(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"config section {name!r} must be a mapping")
    return cast(dict[str, Any], value)


def _git_sha() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "--short", "HEAD"), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run_name(value: str | None) -> str:
    text = datetime.now().strftime("%Y%m%d_%H%M%S") if value is None else str(value).strip()
    if not text:
        raise ValueError("--run-name must not be empty")
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in text)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
