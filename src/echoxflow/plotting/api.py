"""Public plotting convenience functions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from echoxflow.croissant import RecordingRecord
from echoxflow.loading import LoadedArray
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.specs import PanelSpec, PlotViewMode, RenderedFrame, TraceSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import select_timeline
from echoxflow.plotting.writers import write_video


def plot_recording(
    record: RecordingRecord | str | Path,
    *,
    root: str | Path | None = None,
    modalities: tuple[str, ...] | list[str] | None = None,
    time_s: float | None = None,
    frame_index: int | None = None,
    view_mode: PlotViewMode | str = "beamspace",
    show_annotations: bool = True,
    style: PlotStyle | None = None,
) -> Figure:
    """Plot one frame of a recording and return the Matplotlib figure, for example in a notebook.

    The figure is `style.width_px` x `style.height_px` pixels at `style.dpi`, and the caller closes it.

    Args:
        record: A catalog record, or the path of a recording store.
        root: Directory the record's relative store path is resolved against.
        modalities: Streams to draw, as names such as `"2d_brightness_mode"`; `"2d_color_doppler"` expands to the
            velocity and power streams. `None` draws every imaging stream in the recording and adds anatomical
            curves/bullseye for recordings with regional strain (GE automated function imaging, AFI).
            `"aha_strain"` selects curves plus bullseye without cines; `"aha_curves"` and `"bullseye"` select either
            summary alone.
        view_mode: `"beamspace"` draws the native arrays, `"cartesian"` the scan-converted sector, and `"both"` puts
            the two side by side.
        show_annotations: Draw the recording's annotations (contours, gates, measurements) over the images.
        style: Figure size, colours and fonts; defaults to `PlotStyle.from_config()`.
        time_s: Time of the frame in seconds; the nearest frame of each stream is drawn.
        frame_index: Frame to draw when `time_s` is not given; defaults to the first frame, or the middle of the strip
            when no 2D stream has more than one frame.

    Returns:
        The rendered `Figure`.
    """
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
    view_mode: PlotViewMode | str = "beamspace",
    show_annotations: bool = True,
    dpi: int | None = None,
    style: PlotStyle | None = None,
) -> Path:
    """Save one frame of a recording as an image and return its path.

    The image is always `style.width_px` x `style.height_px` pixels.

    Args:
        record: A catalog record, or the path of a recording store.
        root: Directory the record's relative store path is resolved against.
        modalities: Streams to draw, as names such as `"2d_brightness_mode"`; `"2d_color_doppler"` expands to the
            velocity and power streams. `None` draws every imaging stream in the recording and adds anatomical
            curves/bullseye for recordings with regional strain (GE automated function imaging, AFI).
            `"aha_strain"` selects curves plus bullseye without cines; `"aha_curves"` and `"bullseye"` select either
            summary alone.
        view_mode: `"beamspace"` draws the native arrays, `"cartesian"` the scan-converted sector, and `"both"` puts
            the two side by side.
        show_annotations: Draw the recording's annotations (contours, gates, measurements) over the images.
        style: Figure size, colours and fonts; defaults to `PlotStyle.from_config()`.
        time_s: Time of the frame in seconds; the nearest frame of each stream is drawn.
        frame_index: Frame to draw when `time_s` is not given; defaults to the first frame, or the middle of the strip
            when no 2D stream has more than one frame.
        dpi: Scales fonts and line widths only; the pixel size comes from `style`. `None` uses `style.dpi`; a larger
            value draws larger text and thicker lines.

    Returns:
        The path of the written image.
    """
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
    view_mode: PlotViewMode | str = "beamspace",
    show_annotations: bool = True,
    style: PlotStyle | None = None,
) -> RenderedFrame:
    """Render one frame of a recording to an RGB array.

    The array is `style.height_px` x `style.width_px` pixels.

    Args:
        record: A catalog record, or the path of a recording store.
        root: Directory the record's relative store path is resolved against.
        modalities: Streams to draw, as names such as `"2d_brightness_mode"`; `"2d_color_doppler"` expands to the
            velocity and power streams. `None` draws every imaging stream in the recording and adds anatomical
            curves/bullseye for recordings with regional strain (GE automated function imaging, AFI).
            `"aha_strain"` selects curves plus bullseye without cines; `"aha_curves"` and `"bullseye"` select either
            summary alone.
        view_mode: `"beamspace"` draws the native arrays, `"cartesian"` the scan-converted sector, and `"both"` puts
            the two side by side.
        show_annotations: Draw the recording's annotations (contours, gates, measurements) over the images.
        style: Figure size, colours and fonts; defaults to `PlotStyle.from_config()`.
        time_s: Time of the frame in seconds; the nearest frame of each stream is drawn.
        frame_index: Frame to draw when `time_s` is not given; defaults to the first frame, or the middle of the strip
            when no 2D stream has more than one frame.

    Returns:
        A `RenderedFrame` with `image`, a uint8 array of shape `(height_px, width_px, 3)`, and the `time_s` drawn.
    """
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
    view_mode: PlotViewMode | str = "beamspace",
    show_annotations: bool = True,
    max_fps: float = 60.0,
    dpi: int | None = None,
    style: PlotStyle | None = None,
    time_range: tuple[float, float] | None = None,
) -> Path:
    """Render a recording to a video (`.mp4` H.264, `.webm` VP9 or animated `.webp`) and return its path.

    Frames follow the fastest 2D stream, or the longest spectral strip when no 2D stream has more than one frame.
    Every frame is `style.width_px` x `style.height_px` pixels.

    Args:
        record: A catalog record, or the path of a recording store.
        root: Directory the record's relative store path is resolved against.
        modalities: Streams to draw, as names such as `"2d_brightness_mode"`; `"2d_color_doppler"` expands to the
            velocity and power streams. `None` draws every imaging stream in the recording and adds anatomical
            curves/bullseye for recordings with regional strain (GE automated function imaging, AFI).
            `"aha_strain"` selects curves plus bullseye without cines; `"aha_curves"` and `"bullseye"` select either
            summary alone.
        view_mode: `"beamspace"` draws the native arrays, `"cartesian"` the scan-converted sector, and `"both"` puts
            the two side by side.
        show_annotations: Draw the recording's annotations (contours, gates, measurements) over the images.
        style: Figure size, colours and fonts; defaults to `PlotStyle.from_config()`.
        max_fps: Upper bound on the frame rate; faster streams are subsampled in time.
        dpi: Scales fonts and line widths only; the pixel size comes from `style`. `None` uses `style.dpi`; a larger
            value draws larger text and thicker lines.
        time_range: `(start_s, end_s)` keeps only the frames with `start_s <= time < end_s`; `None` renders all.

    Returns:
        The path of the written video.
    """
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
        time_range=time_range,
    )


def render_panel_video(
    panels: Sequence[PanelSpec],
    output: str | Path,
    *,
    timestamps: Sequence[float] | np.ndarray | None = None,
    fps: float = 4.0,
    dpi: int | None = None,
    style: PlotStyle | None = None,
) -> Path:
    """Render ready-made panels to an MP4 video and return its path.

    Args:
        panels: Panel specs, for example from `RecordingPlotRenderer.build_panel_specs`.
        output: Path of the video to write.
        timestamps: Time of each frame in seconds; defaults to the first panel's timestamps, else one frame per
            `1 / fps` seconds.
        fps: Playback frame rate of the video.
        dpi: Scales fonts and line widths only; the pixel size comes from `style`. `None` uses `style.dpi`; a larger
            value draws larger text and thicker lines.
        style: Figure size, colours and fonts; defaults to `PlotStyle.from_config()`.

    Returns:
        The path of the written video.
    """
    panel_tuple = tuple(panels)
    if not panel_tuple:
        raise ValueError("At least one panel is required")
    timeline = _panel_video_timestamps(panel_tuple, timestamps=timestamps, fps=fps)
    renderer = RecordingPlotRenderer(style=style)
    ecg = TraceSpec(signal=np.zeros_like(timeline, dtype=np.float32), timestamps=timeline)
    frames = renderer.iter_frames(panel_tuple, ecg, np.asarray(timeline, dtype=np.float64).reshape(-1), dpi=dpi)
    return write_video(output, frames, fps=float(fps))


def render_loaded_arrays_video(
    loaded_arrays: Sequence[LoadedArray],
    output: str | Path,
    *,
    view_mode: PlotViewMode | str = "beamspace",
    max_fps: float = 60.0,
    dpi: int | None = None,
    style: PlotStyle | None = None,
) -> Path:
    """Render loaded arrays to an MP4 video and return its path.

    Args:
        loaded_arrays: Arrays from `RecordingStore.load_modality`, one panel per imaging stream.
        output: Path of the video to write.
        view_mode: `"beamspace"` draws the native arrays, `"cartesian"` the scan-converted sector, and `"both"` puts
            the two side by side.
        max_fps: Upper bound on the frame rate; faster streams are subsampled in time.
        dpi: Scales fonts and line widths only; the pixel size comes from `style`. `None` uses `style.dpi`; a larger
            value draws larger text and thicker lines.
        style: Figure size, colours and fonts; defaults to `PlotStyle.from_config()`.

    Returns:
        The path of the written video.
    """
    renderer = RecordingPlotRenderer(style=style)
    panels = renderer.build_panel_specs(tuple(loaded_arrays), view_mode=view_mode)
    timeline = select_timeline(panels, max_fps=float(max_fps))
    ecg = TraceSpec(signal=np.zeros_like(timeline.timestamps, dtype=np.float32), timestamps=timeline.timestamps)
    times = np.asarray(timeline.timestamps, dtype=np.float64).reshape(-1)
    return write_video(output, renderer.iter_frames(panels, ecg, times, dpi=dpi), fps=timeline.fps)


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
