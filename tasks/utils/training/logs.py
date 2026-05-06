from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from echoxflow.plotting import PlotStyle
from tasks.utils.training.runtime import _PREVIEW_STYLE, BatchMetrics, EpochMetrics

CARTESIAN_METRIC_PREFIX = "cartesian_"


def _prepare_epoch_log(*, config: Any, task_name: str) -> Path | None:
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
    path.write_text("", encoding="utf-8")
    return path


def _prepare_batch_log(epoch_log_path: Path | None) -> Path | None:
    if epoch_log_path is None:
        return None
    path = epoch_log_path.with_name("batch_scores.jsonl")
    path.write_text("", encoding="utf-8")
    return path


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
    figure = _loss_plot_figure(rows)
    output = scores_path.with_name("loss.png")
    try:
        figure.savefig(output, dpi=int(_PREVIEW_STYLE.dpi), facecolor=figure.get_facecolor(), bbox_inches="tight")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    _write_cartesian_domain_loss_plot(scores_path, rows)
    return output


def _write_cartesian_domain_loss_plot(
    scores_path: Path, rows: Sequence[Mapping[str, Any]] | None = None
) -> Path | None:
    score_rows = _read_score_rows(scores_path) if rows is None else rows
    if not _has_cartesian_domain_terms(score_rows):
        return None
    figure = _cartesian_domain_loss_plot_figure(score_rows)
    output = scores_path.with_name("cartesian_domain_loss.png")
    try:
        figure.savefig(output, dpi=int(_PREVIEW_STYLE.dpi), facecolor=figure.get_facecolor(), bbox_inches="tight")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return output


def _write_batch_loss_plot(scores_path: Path | None) -> Path | None:
    if scores_path is None:
        return None
    figure = _batch_loss_plot_figure(_read_score_rows(scores_path))
    output = scores_path.with_name("batch_loss.png")
    try:
        figure.savefig(output, dpi=int(_PREVIEW_STYLE.dpi), facecolor=figure.get_facecolor(), bbox_inches="tight")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return output


def _read_score_rows(scores_path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, Mapping):
            rows.append(row)
    return rows


def _loss_plot_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    import matplotlib.pyplot as plt

    style = PlotStyle.from_config()
    figure, ax = plt.subplots(figsize=(8.0, 4.5), dpi=int(_PREVIEW_STYLE.dpi), facecolor=style.figure_facecolor)
    x_values = _epoch_training_batch_x_values(rows)
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


def _batch_loss_plot_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    import matplotlib.pyplot as plt

    style = PlotStyle.from_config()
    figure, ax = plt.subplots(figsize=(8.0, 4.5), dpi=int(_PREVIEW_STYLE.dpi), facecolor=style.figure_facecolor)
    term_names = _batch_loss_term_names(rows)
    if not term_names:
        term_names = ("loss",)
    for term_index, term_name in enumerate(term_names):
        color = _loss_term_color(style, term_index)
        for split, linestyle in (("train", "-"), ("val", "--")):
            split_points = _batch_training_batch_points(rows, split=split)
            split_rows = [row for _x_value, row in split_points]
            x_values = [x_value for x_value, _row in split_points]
            y_values = [
                _score_float(_plottable_loss_terms(row.get("loss_terms")).get(term_name, row.get("loss")))
                for row in split_rows
            ]
            _plot_loss_series(ax, x_values, y_values, label=f"{split} {term_name}", color=color, linestyle=linestyle)
    _style_loss_axes(ax, style=style, xlabel="training batches", ylabel="loss")
    figure.tight_layout()
    return figure


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


def _batch_training_batch_points(
    rows: Sequence[Mapping[str, Any]], *, split: str
) -> list[tuple[float, Mapping[str, Any]]]:
    epoch_order: list[int] = []
    rows_by_epoch: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        epoch = _score_epoch(row)
        if epoch not in rows_by_epoch:
            epoch_order.append(epoch)
            rows_by_epoch[epoch] = []
        rows_by_epoch[epoch].append(row)

    points: list[tuple[float, Mapping[str, Any]]] = []
    train_offset = 0
    for epoch in epoch_order:
        epoch_rows = rows_by_epoch[epoch]
        train_rows = [row for row in epoch_rows if row.get("split") == "train"]
        val_rows = [row for row in epoch_rows if row.get("split") == "val"]
        train_count = len(train_rows)
        if split == "train":
            points.extend((float(train_offset + index), row) for index, row in enumerate(train_rows, start=1))
        elif split == "val" and val_rows:
            if train_count > 0:
                scale = float(train_count) / float(len(val_rows))
                points.extend(
                    (float(train_offset) + scale * float(index), row) for index, row in enumerate(val_rows, start=1)
                )
            else:
                points.extend((float(train_offset + index), row) for index, row in enumerate(val_rows, start=1))
        train_offset += train_count
    return points


def _epoch_loss_term_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for row in rows:
        names.update(_plottable_loss_terms(row.get("train_loss_terms")))
        names.update(_plottable_loss_terms(row.get("val_loss_terms")))
    return _ordered_loss_term_names(names)


def _batch_loss_term_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for row in rows:
        names.update(_plottable_loss_terms(row.get("loss_terms")))
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
