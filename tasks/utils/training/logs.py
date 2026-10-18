from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from echoxflow.plotting import PlotStyle
from tasks.utils.atomic import atomic_open
from tasks.utils.training.runtime import _PREVIEW_STYLE, BatchMetrics, EpochMetrics

CARTESIAN_METRIC_PREFIX = "cartesian_"
#: Default number of steps aggregated into one ``batch_scores.jsonl`` row (``artifacts.batch_log_every_steps``).
#: A row is ~0.4 KB, so 25 epochs of 400 steps stay near 160 KB per run. 0 disables the batch log.
DEFAULT_BATCH_LOG_EVERY_STEPS = 25
_LOSS_PLOT_DPI = 80


def _prepare_epoch_log(*, config: Any, task_name: str, resume: bool = False) -> Path | None:
    artifacts = config.section("artifacts")
    root = Path(cast(Any, artifacts.get("root_dir", "outputs/runs")))
    run_name = _artifact_run_name(
        task_name=task_name,
        run_name=artifacts.get("run_name"),
        append_timestamp=bool(artifacts.get("append_timestamp", True)),
    )
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / str(artifacts.get("epoch_scores_name", "train_scores.jsonl"))
    if not resume:
        path.write_text("", encoding="utf-8")
    return path


def _prepare_batch_log(epoch_log_path: Path | None, *, resume: bool = False, enabled: bool = True) -> Path | None:
    if epoch_log_path is None:
        return None
    path = epoch_log_path.with_name("batch_scores.jsonl")
    if not enabled:
        path.unlink(missing_ok=True)
        return None
    if not resume:
        path.write_text("", encoding="utf-8")
    return path


def batch_log_every_steps(config: Any) -> int:
    """Steps per ``batch_scores.jsonl`` row from ``artifacts.batch_log_every_steps`` (0 disables the batch log)."""
    value = config.section("artifacts").get("batch_log_every_steps", DEFAULT_BATCH_LOG_EVERY_STEPS)
    every = int(cast(Any, DEFAULT_BATCH_LOG_EVERY_STEPS if value is None else value))
    if every < 0:
        raise ValueError(f"artifacts.batch_log_every_steps must be non-negative, got {every}")
    return every


class BatchLogWindow:
    """Aggregate consecutive steps into one ``batch_scores.jsonl`` row every ``every`` steps.

    A row reports the clip-weighted mean loss and loss terms of its window, the summed load and step time, and the
    last step of the window, so the batch log stays small while keeping the loss curve within each epoch.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        every: int,
        task: str,
        split: str,
        epoch: int,
        device: str,
        precision: str,
    ) -> None:
        self.path = path if every > 0 else None
        self.every = max(1, int(every))
        self.task = task
        self.split = split
        self.epoch = int(epoch)
        self.device = device
        self.precision = precision
        self._reset()

    def _reset(self) -> None:
        self.steps = 0
        self.last_step = 0
        self.loss_sum = 0.0
        self.loss_count = 0
        self.term_sums: dict[str, float] = {}
        self.term_counts: dict[str, int] = {}
        self.examples = 0
        self.load_seconds = 0.0
        self.step_seconds = 0.0

    def add(
        self,
        *,
        step: int,
        loss_sum: float,
        loss_count: int,
        term_sums: Mapping[str, float],
        term_counts: Mapping[str, int],
        batch_size: int,
        load_seconds: float,
        step_seconds: float,
    ) -> None:
        if self.path is None:
            return
        self.steps += 1
        self.last_step = int(step)
        self.loss_sum += float(loss_sum)
        self.loss_count += int(loss_count)
        for name, value in term_sums.items():
            self.term_sums[name] = self.term_sums.get(name, 0.0) + float(value)
            self.term_counts[name] = self.term_counts.get(name, 0) + int(term_counts.get(name, 0))
        self.examples += int(batch_size)
        self.load_seconds += float(load_seconds)
        self.step_seconds += float(step_seconds)
        if self.steps >= self.every:
            self.flush()

    def flush(self) -> None:
        if self.path is None or self.steps == 0:
            return
        loss = self.loss_sum / self.loss_count if self.loss_count else float("nan")
        terms = {
            name: round(value / self.term_counts[name], 4)
            for name, value in self.term_sums.items()
            if self.term_counts.get(name, 0) > 0
        }
        _write_batch_metrics(
            self.path,
            BatchMetrics(
                task=self.task,
                split=self.split,
                epoch=self.epoch,
                step=self.last_step,
                loss=round(loss, 4),
                loss_terms=terms,
                batch_size=self.examples,
                load_seconds=round(self.load_seconds, 2),
                step_seconds=round(self.step_seconds, 2),
                device=self.device,
                precision=self.precision,
                window_steps=self.steps,
            ),
        )
        self._reset()


def _default_run_name(task_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task_name}/{timestamp}"


def _artifact_run_name(*, task_name: str, run_name: object | None, append_timestamp: bool) -> str:
    if run_name is None or str(run_name).strip() == "":
        return _default_run_name(task_name)
    if not append_timestamp:
        return str(run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(str(run_name)) / timestamp)


def _write_epoch_metrics(path: Path | None, metrics: EpochMetrics) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")
    _write_loss_plot(path)


def _write_batch_metrics(path: Path | None, metrics: BatchMetrics) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")


def _write_loss_plot(scores_path: Path) -> Path:
    rows = _read_score_rows(scores_path)
    batch_path = scores_path.with_name("batch_scores.jsonl")
    batch_rows = _read_score_rows(batch_path) if batch_path.is_file() else []
    figure = _loss_plot_figure(rows, batch_rows=batch_rows)
    output = scores_path.with_name("loss.png")
    _save_figure(figure, output)
    _write_cartesian_domain_loss_plot(scores_path, rows)
    return output


def _save_figure(figure: Any, output: Path) -> None:
    try:
        with atomic_open(output, "wb") as handle:
            figure.savefig(
                handle, format="png", dpi=_LOSS_PLOT_DPI, facecolor=figure.get_facecolor(), bbox_inches="tight"
            )
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def _write_cartesian_domain_loss_plot(
    scores_path: Path, rows: Sequence[Mapping[str, Any]] | None = None
) -> Path | None:
    score_rows = _read_score_rows(scores_path) if rows is None else rows
    if not _has_cartesian_domain_terms(score_rows):
        return None
    figure = _cartesian_domain_loss_plot_figure(score_rows)
    output = scores_path.with_name("cartesian_domain_loss.png")
    _save_figure(figure, output)
    return output


def _read_score_rows(scores_path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by a crash; resume truncates it
        if isinstance(row, Mapping):
            rows.append(row)
    return rows


def _loss_plot_figure(rows: Sequence[Mapping[str, Any]], *, batch_rows: Sequence[Mapping[str, Any]] = ()) -> Any:
    """Epoch train/val loss terms, the selection loss, and the (window-averaged) batch train loss in one plot."""
    import matplotlib.pyplot as plt

    style = PlotStyle.from_config()
    figure, ax = plt.subplots(figsize=(8.0, 4.5), dpi=int(_PREVIEW_STYLE.dpi), facecolor=style.figure_facecolor)
    x_values = _epoch_training_batch_x_values(rows)
    batch_points = _batch_train_points(rows, batch_rows)
    if batch_points:
        batch_x, batch_y = zip(*batch_points)
        ax.plot(batch_x, batch_y, linewidth=0.8, alpha=0.35, color=style.text_dim_color, label="train (batches)")
    selection_loss = [_score_float(row.get("selection_loss")) for row in rows]
    _plot_loss_series(
        ax, x_values, selection_loss, label="selection", color=style.annotation_color, linestyle="-.", marker="o"
    )
    term_names = _epoch_loss_term_names(rows)
    if not term_names:
        train_loss = [_score_float(row.get("train_loss")) for row in rows]
        val_loss = [_score_float(row.get("val_loss")) for row in rows]
        _plot_loss_series(ax, x_values, train_loss, label="train", color=style.line_color)
        _plot_loss_series(ax, x_values, val_loss, label="val", color=style.annotation_secondary_color)
    for term_index, term_name in enumerate(term_names):
        color = _loss_term_color(style, term_index)
        train_loss = [_score_float(_plottable_loss_terms(row.get("train_loss_terms")).get(term_name)) for row in rows]
        val_loss = [_score_float(_plottable_loss_terms(row.get("val_loss_terms")).get(term_name)) for row in rows]
        _plot_loss_series(ax, x_values, train_loss, label=f"train {term_name}", color=color)
        _plot_loss_series(ax, x_values, val_loss, label=f"val {term_name}", color=color, linestyle="--")
    _style_loss_axes(ax, style=style, xlabel="training batches", ylabel="loss")
    figure.tight_layout()
    return figure


def _batch_train_points(
    epoch_rows: Sequence[Mapping[str, Any]], batch_rows: Sequence[Mapping[str, Any]]
) -> list[tuple[float, float]]:
    """``(cumulative training batch, loss)`` of the train rows of ``batch_scores.jsonl``."""
    offsets: dict[int, int] = {}
    cumulative = 0
    for row in sorted(epoch_rows, key=_score_epoch):
        offsets[_score_epoch(row)] = cumulative
        train_steps = row.get("train_steps")
        cumulative += int(train_steps) if isinstance(train_steps, int | float) else 0
    points: list[tuple[float, float]] = []
    for row in batch_rows:
        epoch = _score_epoch(row)
        value = _score_float(row.get("loss"))
        if row.get("split") != "train" or epoch not in offsets or value is None:
            continue
        points.append((float(offsets[epoch] + int(row.get("step") or 0)), value))
    return points


def _cartesian_domain_loss_plot_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    import matplotlib.pyplot as plt

    style = PlotStyle.from_config()
    figure, ax = plt.subplots(figsize=(8.0, 4.5), dpi=int(_PREVIEW_STYLE.dpi), facecolor=style.figure_facecolor)
    x_values = _epoch_training_batch_x_values(rows)
    values = [
        _metric_loss(_plottable_cartesian_domain_terms(row.get("val_loss_terms")).get("cartesian_dice_mean"))
        for row in rows
    ]
    _plot_loss_series(ax, x_values, values, label="val cartesian Dice loss", color=_loss_term_color(style, 0))
    _style_cartesian_domain_axes(ax, style=style)
    figure.tight_layout()
    return figure


def _plot_loss_series(
    ax: Any,
    x_values: Sequence[float],
    values: Sequence[float | None],
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: str | None = None,
) -> None:
    points = [(x_value, value) for x_value, value in zip(x_values, values) if value is not None]
    if not points:
        return
    plotted_x_values, y_values = zip(*points)
    ax.plot(
        plotted_x_values,
        y_values,
        linewidth=1.8,
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=3,
    )


def _epoch_training_batch_x_values(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    cumulative = 0
    for row in rows:
        train_steps = row.get("train_steps")
        if isinstance(train_steps, int | float) and np.isfinite(float(train_steps)):
            cumulative += max(0, int(train_steps))
            values.append(float(cumulative if cumulative > 0 else _score_epoch(row)))
        else:
            values.append(float(_score_epoch(row)))
    return values


def _epoch_loss_term_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for row in rows:
        names.update(_plottable_loss_terms(row.get("train_loss_terms")))
        names.update(_plottable_loss_terms(row.get("val_loss_terms")))
    return _ordered_loss_term_names(names)


def _ordered_loss_term_names(names: Iterable[str]) -> tuple[str, ...]:
    preferred = {"velocity": 0, "power": 1, "std": 2, "dice": 3, "loss": 99}
    return tuple(sorted((str(name) for name in names), key=lambda name: (preferred.get(name, 50), name)))


def _plottable_loss_terms(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    components = {
        str(name): score
        for name, score in value.items()
        if str(name) != "loss" and not str(name).startswith(CARTESIAN_METRIC_PREFIX)
    }
    if components:
        return components
    if "loss" in value:
        return {"loss": value["loss"]}
    return {}


def _plottable_cartesian_domain_terms(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(name): score for name, score in value.items() if str(name) == "cartesian_dice_mean"}


def _has_cartesian_domain_terms(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(bool(_plottable_cartesian_domain_terms(row.get("val_loss_terms"))) for row in rows)


def _metric_loss(value: object) -> float | None:
    score = _score_float(value)
    if score is None:
        return None
    return max(0.0, 1.0 - score)


def _style_cartesian_domain_axes(ax: Any, *, style: PlotStyle) -> None:
    ax.set_facecolor(style.figure_facecolor)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("training batches", color=style.text_color)
    ax.set_ylabel("cartesian-domain loss", color=style.text_color)
    ax.tick_params(colors=style.text_dim_color)
    ax.grid(True, color=style.grid_color, linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(style.text_dim_color)
    if ax.lines:
        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color(style.text_color)


def _loss_term_color(style: PlotStyle, index: int) -> str:
    palette = (
        style.line_color,
        style.annotation_secondary_color,
        style.annotation_color,
        style.ecg_marker_color,
        style.sampling_gate_color,
        style.text_color,
    )
    return palette[int(index) % len(palette)]


def _style_loss_axes(ax: Any, *, style: PlotStyle, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(style.figure_facecolor)
    ax.set_yscale("log", nonpositive="clip")
    ax.set_xlabel(xlabel, color=style.text_color)
    ax.set_ylabel(ylabel, color=style.text_color)
    ax.tick_params(colors=style.text_dim_color)
    ax.grid(True, color=style.grid_color, linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(style.text_dim_color)
    if ax.lines:
        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color(style.text_color)


def _score_epoch(row: Mapping[str, Any]) -> int:
    return int(row.get("epoch") or 0)


def _score_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
