"""Public plotting convenience functions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from echoxflow.croissant import RecordingRecord
from echoxflow.loading import LoadedArray
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.specs import PanelSpec, PlotViewMode, RenderedFrame, TraceSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import select_timeline
from echoxflow.plotting.writers import figure_to_rgb, write_video


def plot_recording(
    record: RecordingRecord | str | Path,
    *,
    root: str | Path | None = None,
    modalities: tuple[str, ...] | list[str] | None = None,
    time_s: float | None = None,
    frame_index: int | None = None,
    view_mode: PlotViewMode | str = "pre_converted",
    show_annotations: bool = True,
    style: PlotStyle | None = None,
) -> Figure:
    renderer = RecordingPlotRenderer(style=style)
    return renderer.plot_recording(
        record,
        root=root,
        modalities=None if modalities is None else tuple(modalities),
        time_s=time_s,
        frame_index=frame_index,
        view_mode=view_mode,
        show_annotations=show_annotations,
    )


def save_recording_plot(
    record: RecordingRecord | str | Path,
    output: str | Path,
    *,
    root: str | Path | None = None,
    modalities: tuple[str, ...] | list[str] | None = None,
    time_s: float | None = None,
    frame_index: int | None = None,
    view_mode: PlotViewMode | str = "pre_converted",
    show_annotations: bool = True,
    dpi: int = 200,
    style: PlotStyle | None = None,
) -> Path:
    renderer = RecordingPlotRenderer(style=style)
    return renderer.save_plot(
        record,
        output,
        root=root,
        modalities=None if modalities is None else tuple(modalities),
        time_s=time_s,
        frame_index=frame_index,
        view_mode=view_mode,
        show_annotations=show_annotations,
        dpi=dpi,
    )


def render_recording_frame(
    record: RecordingRecord | str | Path,
    *,
    root: str | Path | None = None,
    modalities: tuple[str, ...] | list[str] | None = None,
    time_s: float | None = None,
    frame_index: int | None = None,
    view_mode: PlotViewMode | str = "pre_converted",
    show_annotations: bool = True,
    style: PlotStyle | None = None,
) -> RenderedFrame:
    renderer = RecordingPlotRenderer(style=style)
    return renderer.render_frame(
        record,
        root=root,
        modalities=None if modalities is None else tuple(modalities),
        time_s=time_s,
        frame_index=frame_index,
        view_mode=view_mode,
        show_annotations=show_annotations,
    )


def render_recording_video(
    record: RecordingRecord | str | Path,
    output: str | Path,
    *,
    root: str | Path | None = None,
    modalities: tuple[str, ...] | list[str] | None = None,
    view_mode: PlotViewMode | str = "pre_converted",
    show_annotations: bool = True,
    max_fps: float = 60.0,
    dpi: int = 150,
    style: PlotStyle | None = None,
) -> Path:
    renderer = RecordingPlotRenderer(style=style)
    return renderer.render_video(
        record,
        output,
        root=root,
        modalities=None if modalities is None else tuple(modalities),
        view_mode=view_mode,
        show_annotations=show_annotations,
        max_fps=max_fps,
        dpi=dpi,
    )


def render_panel_video(
    panels: Sequence[PanelSpec],
    output: str | Path,
    *,
    timestamps: Sequence[float] | np.ndarray | None = None,
    fps: float = 4.0,
    dpi: int = 150,
    style: PlotStyle | None = None,
) -> Path:
    panel_tuple = tuple(panels)
    if not panel_tuple:
        raise ValueError("At least one panel is required")
    timeline = _panel_video_timestamps(panel_tuple, timestamps=timestamps, fps=fps)
    renderer = RecordingPlotRenderer(style=style)
    ecg = TraceSpec(signal=np.zeros_like(timeline, dtype=np.float32), timestamps=timeline)
    frames: list[np.ndarray] = []
    for frame_index, time_s in enumerate(timeline):
        figure = renderer.render_figure_from_specs(
            panels=panel_tuple,
            ecg=ecg,
            time_s=float(time_s),
            frame_index=frame_index,
            dpi=dpi,
        )
        try:
            frames.append(figure_to_rgb(figure))
        finally:
            plt.close(figure)
    return write_video(output, np.stack(frames, axis=0), fps=float(fps))


def render_loaded_arrays_video(
    loaded_arrays: Sequence[LoadedArray],
    output: str | Path,
    *,
    view_mode: PlotViewMode | str = "pre_converted",
    max_fps: float = 60.0,
    dpi: int = 150,
    style: PlotStyle | None = None,
) -> Path:
    renderer = RecordingPlotRenderer(style=style)
    panels = renderer.build_panel_specs(tuple(loaded_arrays), view_mode=view_mode)
    timeline = select_timeline(panels, max_fps=float(max_fps))
    ecg = TraceSpec(signal=np.zeros_like(timeline.timestamps, dtype=np.float32), timestamps=timeline.timestamps)
    frames: list[np.ndarray] = []
    for frame_index, time_s in enumerate(np.asarray(timeline.timestamps, dtype=np.float64).reshape(-1)):
        figure = renderer.render_figure_from_specs(
            panels=panels,
            ecg=ecg,
            time_s=float(time_s),
            frame_index=frame_index,
            dpi=dpi,
        )
        try:
            frames.append(figure_to_rgb(figure))
        finally:
            plt.close(figure)
    return write_video(output, np.stack(frames, axis=0), fps=timeline.fps)


def _panel_video_timestamps(
    panels: tuple[PanelSpec, ...],
    *,
    timestamps: Sequence[float] | np.ndarray | None,
    fps: float,
) -> np.ndarray:
    if timestamps is not None:
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if values.size:
            return values
    for panel in panels:
        if panel.loaded.timestamps is not None:
            values = np.asarray(panel.loaded.timestamps, dtype=np.float64).reshape(-1)
            if values.size:
                return values
    count = max((int(np.asarray(panel.loaded.data).shape[0]) for panel in panels), default=1)
    return np.arange(max(1, count), dtype=np.float64) / max(float(fps), 1e-6)
