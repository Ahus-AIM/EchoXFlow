from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echoxflow import PlotStyle, RecordingRecord, render_recording_frame, write_recording  # noqa: E402
from tasks.registry import (  # noqa: E402
    task_benchmark_loss_fn,
    task_benchmark_val_metrics_fn,
    task_build_dataloaders,
    task_load_config,
)
from tasks.utils.training.device import move_to_device  # noqa: E402
from tasks.utils.training.module import TrainingModule  # noqa: E402
from tasks.utils.training.preview import (  # noqa: E402
    _prediction_recording_preview_specs,
    _preview_prediction,
    _stable_preview_sample,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_DOMAINS = ("any", "beamspace", "cartesian")
_DEFAULT_BENCHMARK_SPEC = Path("configs/benchmark_table.yaml")


@dataclass(frozen=True)
class CheckpointResolution:
    task: str
    checkpoint: Path | None
    config: Path | None
    message: str | None = None

    @property
    def usable(self) -> bool:
        return self.checkpoint is not None and self.config is not None and self.message is None


@dataclass(frozen=True)
class PredictionPanel:
    task: str
    title: str
    real: np.ndarray | None = None
    predicted: np.ndarray | None = None
    message: str | None = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot one real/predicted validation sample for benchmark tasks.")
    parser.add_argument("--output-root", default="outputs/bench")
    parser.add_argument("--output", default="outputs/bench/task_prediction_grid.png")
    parser.add_argument("--benchmark-spec", default=str(_DEFAULT_BENCHMARK_SPEC))
    parser.add_argument("--data-root")
    parser.add_argument("--domain", choices=_DOMAINS, default="any")
    parser.add_argument("--method", help="Restrict checkpoint discovery to one benchmark method.")
    parser.add_argument("--tasks", help="Comma-separated task names to plot, e.g. tissue_doppler.")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--validation-fold",
        type=int,
        help="Restrict checkpoint discovery and preview sampling to one CV validation fold.",
    )
    parser.add_argument("--checkpoint", action="append", default=[], help="Override a task checkpoint as task=path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--sample-fraction", type=float, default=0.5)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument(
        "--max-records",
        type=int,
        default=3,
        help="Maximum records to scan per task while building the preview sample. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help=(
            "Maximum train and validation samples to build per task while selecting the preview sample. "
            "Use 0 for no cap."
        ),
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-width-in", type=float, default=10.5)
    parser.add_argument("--figure-height-in", type=float, default=7.0)
    parser.add_argument("--no-ecg-strip", action="store_true", help="Omit the ECG strip from rendered panels.")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser()
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else output.parent / ".plot_task_predictions_cache"
    task_columns = _filter_task_columns(_task_columns(Path(args.benchmark_spec)), tasks=args.tasks)
    overrides = _parse_checkpoint_overrides(args.checkpoint, task_columns=task_columns)
    panels = []
    for task, title in task_columns:
        resolution = _resolve_checkpoint(
            task=task,
            output_root=Path(args.output_root).expanduser(),
            domain=args.domain,
            method=args.method,
            run_name=args.run_name,
            seed=args.seed,
            validation_fold=args.validation_fold,
            overrides=overrides,
        )
        panels.append(_prediction_panel(task=task, title=title, resolution=resolution, args=args, cache_dir=cache_dir))

    try:
        _write_prediction_grid(
            panels,
            output=output,
            dpi=int(args.dpi),
            figsize=(float(args.figure_width_in), float(args.figure_height_in)),
        )
    finally:
        if not args.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
    print(output)
    return 0


def _parse_checkpoint_overrides(values: Sequence[str], *, task_columns: Sequence[tuple[str, str]]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    task_names = {task for task, _title in task_columns}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--checkpoint must use task=path, got {value!r}")
        task, raw_path = value.split("=", maxsplit=1)
        task = task.strip()
        if task not in task_names:
            choices = ", ".join(sorted(task_names))
            raise SystemExit(f"unsupported checkpoint task {task!r}; expected one of: {choices}")
        path = Path(raw_path.strip()).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        overrides[task] = path
    return overrides


def _filter_task_columns(
    task_columns: Sequence[tuple[str, str]],
    *,
    tasks: str | None,
) -> tuple[tuple[str, str], ...]:
    if tasks is None or not tasks.strip():
        return tuple(task_columns)
    requested = tuple(task.strip() for task in tasks.split(",") if task.strip())
    available = {task for task, _title in task_columns}
    unknown = [task for task in requested if task not in available]
    if unknown:
        choices = ", ".join(sorted(available))
        raise SystemExit(f"unsupported --tasks value(s): {', '.join(unknown)}; expected one of: {choices}")
    requested_set = set(requested)
    return tuple((task, title) for task, title in task_columns if task in requested_set)


def _resolve_checkpoint(
    *,
    task: str,
    output_root: Path,
    domain: str = "any",
    method: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    validation_fold: int | None = None,
    overrides: Mapping[str, Path] | None = None,
) -> CheckpointResolution:
    explicit = dict(overrides or {}).get(task)
    if explicit is not None:
        return _checkpoint_resolution_for_path(task, explicit)

    candidates = _checkpoint_candidates(
        output_root=output_root,
        task=task,
        domain=domain,
        method=method,
        run_name=run_name,
        seed=seed,
        validation_fold=validation_fold,
    )
    usable = [path for path in candidates if (path.parent / "config.yaml").exists()]
    if usable:
        checkpoint = max(usable, key=_checkpoint_sort_key)
        return CheckpointResolution(task=task, checkpoint=checkpoint, config=checkpoint.parent / "config.yaml")
    if candidates:
        checkpoint = max(candidates, key=_checkpoint_sort_key)
        return CheckpointResolution(
            task=task,
            checkpoint=checkpoint,
            config=None,
            message=f"missing config: {checkpoint.parent / 'config.yaml'}",
        )
    return CheckpointResolution(task=task, checkpoint=None, config=None, message="missing model")


def _checkpoint_resolution_for_path(task: str, checkpoint: Path) -> CheckpointResolution:
    if not checkpoint.exists():
        return CheckpointResolution(
            task=task, checkpoint=checkpoint, config=None, message=f"missing model: {checkpoint}"
        )
    config = checkpoint.parent / "config.yaml"
    if not config.exists():
        return CheckpointResolution(task=task, checkpoint=checkpoint, config=None, message=f"missing config: {config}")
    return CheckpointResolution(task=task, checkpoint=checkpoint, config=config)


def _checkpoint_candidates(
    *,
    output_root: Path,
    task: str,
    domain: str,
    method: str | None,
    run_name: str | None,
    seed: int | None,
    validation_fold: int | None = None,
) -> list[Path]:
    run_pattern = run_name or "*"
    method_pattern = method or "*"
    domain_pattern = "*" if domain == "any" else domain
    seed_pattern = f"seed={int(seed)}" if seed is not None else "seed=*"
    fold_pattern = f"fold={int(validation_fold)}" if validation_fold is not None else "fold=*"
    base = output_root / task
    patterns = (
        (f"{run_pattern}/{method_pattern}/{domain_pattern}/{seed_pattern}/weights.pt",)
        if validation_fold is None
        else ()
    ) + (f"{run_pattern}/{method_pattern}/{domain_pattern}/{fold_pattern}/{seed_pattern}/weights.pt",)
    return sorted({path for pattern in patterns for path in base.glob(pattern)})


def _task_columns(spec_path: Path) -> tuple[tuple[str, str], ...]:
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"benchmark table spec must be a YAML mapping: {spec_path}")
    columns = raw.get("columns")
    if not isinstance(columns, list):
        raise ValueError("benchmark table spec requires a `columns` list")
    labels: dict[str, str] = {}
    for column in columns:
        if not isinstance(column, Mapping):
            raise TypeError("benchmark table columns must be mappings")
        task = str(column["task"])
        labels.setdefault(task, str(column.get("title", task)))
    return tuple(labels.items())


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    try:
        mtime = int(path.stat().st_mtime_ns)
    except OSError:
        mtime = -1
    return (mtime, str(path))


def _prediction_panel(
    *,
    task: str,
    title: str,
    resolution: CheckpointResolution,
    args: argparse.Namespace,
    cache_dir: Path,
) -> PredictionPanel:
    if not resolution.usable:
        return PredictionPanel(task=task, title=title, message=resolution.message or "missing model")
    try:
        real, predicted = _render_task_prediction_frames(
            task=task,
            checkpoint=cast(Path, resolution.checkpoint),
            config_path=cast(Path, resolution.config),
            data_root=args.data_root,
            device=_resolve_cli_device(str(args.device)),
            sample_fraction=float(args.sample_fraction),
            sample_index=args.sample_index,
            frame_index=args.frame_index,
            max_records=None if int(args.max_records) <= 0 else int(args.max_records),
            max_samples=None if int(args.max_samples) <= 0 else int(args.max_samples),
            cache_dir=cache_dir / task,
            show_ecg=not bool(args.no_ecg_strip),
            validation_fold=args.validation_fold,
        )
    except Exception as exc:
        return PredictionPanel(task=task, title=title, message=f"{type(exc).__name__}: {exc}")
    return PredictionPanel(task=task, title=title, real=real, predicted=predicted)


def _render_task_prediction_frames(
    *,
    task: str,
    checkpoint: Path,
    config_path: Path,
    data_root: str | None,
    device: torch.device,
    sample_fraction: float,
    sample_index: int | None,
    frame_index: int | None,
    max_records: int | None,
    max_samples: int | None,
    cache_dir: Path,
    show_ecg: bool = True,
    validation_fold: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    config = task_load_config(task)(config_path)
    _apply_validation_fold(config, validation_fold=validation_fold)
    _limit_preview_config(config, max_records=max_records, max_samples=max_samples)
    _train_loader, val_loader = task_build_dataloaders(task)(config=config, data_root=data_root)
    sample = _select_validation_sample(
        val_loader,
        device=device,
        seed=int(config.data.get("seed", 0)),
        sample_fraction=float(sample_fraction),
        sample_index=sample_index,
    )
    if sample is None:
        raise ValueError("validation dataset is empty")
    record = getattr(sample, "record", None)
    if not isinstance(record, RecordingRecord):
        raise TypeError("validation sample is missing a RecordingRecord")

    module = TrainingModule(
        config=config,
        loss_fn=task_benchmark_loss_fn(task),
        val_metrics_fn=task_benchmark_val_metrics_fn(task),
    )
    state = _load_model_state(checkpoint)
    module.model.load_state_dict(state)
    module.to(device)
    module.eval()
    with torch.no_grad():
        prediction = _preview_prediction(module, sample, task)
    specs = _prediction_recording_preview_specs(task_kind=task, sample=sample, record=record, prediction=prediction)
    real_frame = _render_preview_frame(
        cache_dir=cache_dir,
        record=record,
        suffix="real",
        arrays=(*specs.common, *specs.real),
        attrs=specs.attrs,
        modalities=specs.modalities,
        frame_index=frame_index,
        show_ecg=show_ecg,
    )
    predicted_frame = _render_preview_frame(
        cache_dir=cache_dir,
        record=record,
        suffix="predicted",
        arrays=(*specs.common, *specs.predicted),
        attrs=specs.attrs,
        modalities=specs.modalities,
        frame_index=frame_index,
        show_ecg=show_ecg,
    )
    return real_frame, predicted_frame


def _apply_validation_fold(config: object, *, validation_fold: int | None) -> None:
    if validation_fold is None:
        return
    raw = getattr(config, "raw", None)
    if not isinstance(raw, dict):
        return
    data = raw.setdefault("data", {})
    if isinstance(data, dict):
        data["validation_fold"] = int(validation_fold)


def _limit_preview_config(config: object, *, max_records: int | None, max_samples: int | None) -> None:
    raw = getattr(config, "raw", None)
    if not isinstance(raw, dict):
        return
    data = raw.setdefault("data", {})
    if not isinstance(data, dict):
        return
    if max_records is not None:
        if _uses_fold_split(data):
            # Keeping the CV split is more important than capping discovered files here: applying max_files before
            # splitting can consume only training-fold records and leave the validation dataset empty.
            data["max_files"] = None
        else:
            data["max_files"] = max(1, int(max_records))
            data["fold_split_path"] = False
        data["data_fraction"] = None
        data["frame_ratio_range"] = (0.0, 1.0e9)
        data["fps_range"] = (0.0, 1.0e9)
    if max_samples is not None:
        limit = max(1, int(max_samples))
        data["max_train_samples"] = limit
        data["max_val_samples"] = limit


def _uses_fold_split(data: Mapping[str, Any]) -> bool:
    fold_split_path = data.get("fold_split_path")
    return data.get("validation_fold") is not None or fold_split_path not in (None, False, "")


def _load_model_state(checkpoint: Path) -> Mapping[str, Tensor]:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
        state = state["state_dict"]
    if isinstance(state, Mapping) and isinstance(state.get("model"), Mapping):
        state = state["model"]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint does not contain a state dict: {checkpoint}")
    return cast(Mapping[str, Tensor], state)


def _select_validation_sample(
    dataloader: object,
    *,
    device: torch.device,
    seed: int,
    sample_fraction: float,
    sample_index: int | None,
) -> object | None:
    dataset = getattr(dataloader, "dataset", None)
    if sample_index is None:
        return _stable_preview_sample(
            cast(Any, dataloader),
            device=device,
            seed=int(seed),
            index_fraction=float(sample_fraction),
        )
    if dataset is None:
        return None
    sample_count = int(len(dataset))
    if sample_count <= 0:
        return None
    index = min(sample_count - 1, max(0, int(sample_index)))
    return move_to_device(dataset[index], device)


def _resolve_cli_device(value: str) -> torch.device:
    text = value.strip().lower()
    if text in {"auto", ""}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is not available")
        return torch.device("cuda:0")
    if text.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{value} requested but CUDA is not available")
        index = int(text.removeprefix("cuda:"))
        if index >= int(torch.cuda.device_count()):
            raise RuntimeError(f"CUDA device {index} requested, but only {torch.cuda.device_count()} device(s) exist")
        return torch.device(text)
    if text == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device {value!r}; expected auto, cpu, cuda, or cuda:N")


def _render_preview_frame(
    *,
    cache_dir: Path,
    record: RecordingRecord,
    suffix: str,
    arrays: Sequence[Any],
    attrs: Mapping[str, Any],
    modalities: tuple[str, ...],
    frame_index: int | None,
    show_ecg: bool = True,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    recording_id = _safe_recording_id(f"{record.recording_id}_{suffix}")
    recording_path = cache_dir / f"{recording_id}.zarr"
    write_recording(
        recording_path,
        exam_id=record.exam_id,
        recording_id=recording_id,
        source_recording_id=record.recording_id,
        arrays=arrays,
        attrs=attrs,
        overwrite=True,
    )
    index = _resolved_frame_index(_modality_arrays(arrays, modalities), frame_index)
    return np.asarray(
        render_recording_frame(
            recording_path,
            modalities=modalities,
            frame_index=index,
            view_mode="pre_converted",
            show_annotations=False,
            style=PlotStyle(show_ecg=show_ecg),
        ).image,
        dtype=np.uint8,
    )


def _resolved_frame_index(arrays: Sequence[Any], frame_index: int | None) -> int:
    count = max((_array_frame_count(array) for array in arrays), default=1)
    if frame_index is None:
        return max(0, count // 2)
    return min(max(0, int(frame_index)), max(0, count - 1))


def _array_frame_count(array: Any) -> int:
    values = np.asarray(getattr(array, "values", np.empty((1,), dtype=np.float32)))
    if values.ndim <= 0:
        return 1
    return int(values.shape[0])


def _modality_arrays(arrays: Sequence[Any], modalities: tuple[str, ...]) -> tuple[Any, ...]:
    modality_set = set(modalities)
    selected = []
    for array in arrays:
        content_type = getattr(array, "content_type", None)
        data_path = str(getattr(array, "data_path", "")).removeprefix("data/")
        if content_type in modality_set or data_path in modality_set:
            selected.append(array)
    return tuple(selected) or tuple(arrays)


def _write_prediction_grid(
    panels: Sequence[PredictionPanel],
    *,
    output: Path,
    dpi: int = 300,
    figsize: tuple[float, float] = (10.5, 7.0),
) -> Path:
    figure = _prediction_grid_figure(panels, figsize=figsize)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)
    return output


def _prediction_grid_figure(panels: Sequence[PredictionPanel], *, figsize: tuple[float, float] = (10.5, 7.0)) -> Figure:
    if not panels:
        raise ValueError("prediction grid requires at least one panel")
    figure, axes = plt.subplots(2, len(panels), figsize=figsize, squeeze=False)
    figure.patch.set_facecolor("white")
    for column, panel in enumerate(panels):
        axes[0, column].set_title(panel.title, fontsize=11, pad=4)
        _draw_panel(axes[0, column], panel.real, panel.message, task=panel.task)
        _draw_panel(axes[1, column], panel.predicted, panel.message, task=panel.task)
    axes[0, 0].set_ylabel("Real", fontsize=12, labelpad=8)
    axes[1, 0].set_ylabel("Predicted", fontsize=12, labelpad=8)
    figure.subplots_adjust(left=0.055, right=0.995, bottom=0.045, top=0.94, wspace=0.03, hspace=0.035)
    return figure


def _draw_panel(axis: Any, image: np.ndarray | None, message: str | None, *, task: str) -> None:
    axis.set_box_aspect(1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    if image is not None:
        if task == "segmentation":
            image = _trim_dark_edge_padding(image)
        axis.imshow(image, aspect="auto")
        return
    axis.set_facecolor("#f4f4f4")
    axis.text(
        0.5,
        0.5,
        _short_panel_message(message or "missing model"),
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        wrap=True,
        transform=axis.transAxes,
    )


def _trim_dark_edge_padding(image: np.ndarray, *, luminance_threshold: float = 42.0) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] < 2 or image.shape[1] < 2 or image.shape[2] < 3:
        return image
    rgb = np.asarray(image[..., :3], dtype=np.float32)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if image.shape[2] >= 4:
        luminance = np.where(np.asarray(image[..., 3], dtype=np.float32) > 0.0, luminance, 0.0)

    rows = np.flatnonzero(luminance.mean(axis=1) > luminance_threshold)
    cols = np.flatnonzero(luminance.mean(axis=0) > luminance_threshold)
    if rows.size == 0 or cols.size == 0:
        return image

    y0 = max(0, int(rows[0]) - 1)
    y1 = min(int(image.shape[0]), int(rows[-1]) + 2)
    x0 = max(0, int(cols[0]) - 1)
    x1 = min(int(image.shape[1]), int(cols[-1]) + 2)
    if y1 - y0 < 2 or x1 - x0 < 2:
        return image
    return image[y0:y1, x0:x1]


def _short_panel_message(message: str) -> str:
    text = str(message).strip()
    if len(text) <= 90:
        return text
    return text[:87].rstrip() + "..."


def _safe_recording_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
