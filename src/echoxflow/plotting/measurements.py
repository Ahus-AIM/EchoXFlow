"""Draw measurements on the B-mode frame or Doppler spectrum they were made on."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure

from echoxflow.annotations import Measurement
from echoxflow.loading import RecordingStore
from echoxflow.plotting.panels import fixed_normalize, panel_colormap
from echoxflow.plotting.renderer import _with_recording_colormap
from echoxflow.plotting.style import PlotStyle
from echoxflow.scan.conversion import sector_to_cartesian
from echoxflow.spectral import SpectralMetadata


def plot_measurements(
    store: RecordingStore,
    measurements: Measurement | Sequence[Measurement],
    *,
    ax: Axes | None = None,
    height_px: int = 720,
    style: PlotStyle | None = None,
) -> Figure:
    """Draw measurements on the image they were made on.

    Spatial measurements, including left-atrial outlines, are drawn on their scan-converted B-mode frame, in metres
    (x lateral, y depth). Spectral measurements are drawn on their Doppler spectrum, in matrix columns and rows.

    Args:
        store: The opened recording the measurements were loaded from.
        measurements: One measurement or several from `store.load_measurements()`. They must be all spatial and on
            the same frame, or all spectral and on the same spectrum. A spatial measurement whose `frame_status`
            is `between_frames` is drawn on the nearest stored frame, not the one it was made on.
        ax: Axes to draw into. By default a new figure with a transparent background is created.
        height_px: Height of the image in pixels.
        style: Colours, line widths and fonts; defaults to `PlotStyle.from_config()`.

    Returns:
        The `Figure` drawn into.
    """
    items = (measurements,) if isinstance(measurements, Measurement) else tuple(measurements)
    if not items:
        raise ValueError("No measurements to plot")
    first = items[0]
    for item in items:
        if item.coordinate_type not in {"spatial", "spectral"}:
            raise ValueError(f"{item.kind} is a {item.coordinate_type} measurement, which is not drawn")
        if item.coordinate_type != first.coordinate_type or item.data_path != first.data_path:
            raise ValueError("Measurements must be made on the same image; plot each image separately")
    style = style or PlotStyle.from_config()
    if first.coordinate_type == "spatial":
        ax, curves = _draw_frame(store, items, ax=ax, height_px=height_px, style=style)
    else:
        ax, curves = _draw_spectrum(store, items, ax=ax, height_px=height_px, style=style)
    _draw_curves(ax, items, curves, style=style)
    owner = ax.figure
    return owner if isinstance(owner, Figure) else owner.figure


def _draw_frame(
    store: RecordingStore, items: tuple[Measurement, ...], *, ax: Axes | None, height_px: int, style: PlotStyle
) -> tuple[Axes, list[np.ndarray]]:
    first = items[0]
    for item in items:
        if item.frame_index is None or item.data_path is None or item.geometry is None:
            raise ValueError(f"{item.kind} has no stored B-mode frame to draw on (frame_status={item.frame_status})")
        if (item.frame_index, item.recording) != (first.frame_index, first.recording):
            raise ValueError("Measurements must share one frame; plot each frame separately")
    assert first.data_path is not None and first.frame_index is not None and first.geometry is not None
    # A left-atrial outline is drawn on a still; its frame is in the loop the still was taken from.
    source = store if first.recording is None else store.open_reference(first.recording)
    try:
        frame = source.load_array_slice(first.data_path, first.frame_index, first.frame_index + 1)[0]
    finally:
        if source is not store:
            source.close()
    image = sector_to_cartesian(frame, first.geometry, output_height=height_px)
    (x0, x1), (y0, y1) = image.grid.x_range_m, image.grid.y_range_m
    ax = ax or _new_axes(image.grid.shape[1], image.grid.shape[0], style)
    # Pixels outside the sector are masked, so they stay transparent instead of being drawn black.
    pixels = np.ma.masked_where(~image.mask, image.data)
    ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, extent=(x0, x1, y1, y0), interpolation="bilinear")
    curves = [np.asarray(item.points, dtype=np.float64) for item in items]
    return ax, [
        np.vstack([curve, curve[:1]]) if item.geometry_kind == "polygon" else curve
        for item, curve in zip(items, curves)
    ]


def _draw_spectrum(
    store: RecordingStore, items: tuple[Measurement, ...], *, ax: Axes | None, height_px: int, style: PlotStyle
) -> tuple[Axes, list[np.ndarray]]:
    data_path = items[0].data_path
    if data_path is None:
        raise ValueError(f"{items[0].kind} does not name the spectrum it was made on")
    # Loaded like the recording renderer does, so the spectrum gets the recording's own colour table when the
    # recording identifies one, and the configured Doppler colours otherwise.
    loaded = _with_recording_colormap(store, store.load_modality(data_path))
    spectrum = np.asarray(loaded.data)
    clock = loaded.timestamps
    if spectrum.ndim != 2 or clock is None or len(clock) != spectrum.shape[0]:
        raise ValueError(f"{data_path} must be a [time, velocity] matrix with one timestamp per column")
    metadata = store.spectral_metadata(data_path)
    ax = ax or _new_axes(round(1.5 * height_px), height_px, style)
    ax.imshow(
        spectrum.T,
        cmap=panel_colormap(loaded) or "gray",
        norm=fixed_normalize(loaded, spectrum),
        aspect="auto",
        interpolation="bilinear",
    )
    times = np.asarray(clock, dtype=np.float64).reshape(-1)
    curves = []
    for item in items:
        columns, runs = _columns_at_times(times, item.points[:, 0])
        rows = _rows_at_velocities(metadata, columns, item.points[:, 1], row_count=spectrum.shape[1])
        curve = np.column_stack([columns, rows])
        if item.geometry_kind == "polyline":
            # A traced curve is not joined across a break in the spectrum's clock.
            curve = np.insert(curve, np.flatnonzero(runs[1:] != runs[:-1]) + 1, np.nan, axis=0)
        curves.append(curve)
    return ax, curves


def _draw_curves(ax: Axes, items: tuple[Measurement, ...], curves: list[np.ndarray], *, style: PlotStyle) -> None:
    ax.set_axis_off()
    # A darker shade of the depth ruler's green.
    red, green, blue = to_rgb(style.ecg_trace_color)
    color = to_hex((0.7 * red, 0.7 * green, 0.7 * blue))
    linewidth = 1.95 * style.annotation_linewidth
    labelled: set[str] = set()
    for item, curve in zip(items, curves, strict=True):
        pairs = item.geometry_kind == "segments"
        for part in [curve[begin : begin + 2] for begin in range(0, len(curve) - 1, 2)] if pairs else [curve]:
            ax.plot(
                part[:, 0],
                part[:, 1],
                marker=None if item.geometry_kind in {"polyline", "polygon"} else "+",
                linestyle="None" if item.geometry_kind in {"point", "points"} or len(part) < 2 else "-",
                color=color,
                linewidth=linewidth,
                markersize=3.0 * style.annotation_markersize,
                markeredgewidth=linewidth,
            )
        label = _label(item)
        drawn = curve[np.isfinite(curve).all(axis=1)]
        if label in labelled or drawn.size == 0:
            continue
        labelled.add(label)
        fontsize = 2.7 * style.annotation_label_fontsize
        anchor = drawn[int(np.argmax(drawn[:, 0]))]
        # The label goes to the right of the measurement, or to its left when it would run off the image.
        width_px = 0.6 * fontsize * ax.figure.dpi / 72.0 * max(len(line) for line in label.split("\n"))
        fits = ax.transData.transform(anchor)[0] + width_px < ax.bbox.x1
        if not fits:
            anchor = drawn[int(np.argmin(drawn[:, 0]))]
        ax.annotate(
            label,
            xy=(float(anchor[0]), float(anchor[1])),
            xytext=(8.0 if fits else -8.0, 0.0),
            textcoords="offset points",
            ha="left" if fits else "right",
            va="center",
            color=color,
            fontsize=fontsize,
        )


def _new_axes(width_px: int, height_px: int, style: PlotStyle) -> Axes:
    figure = Figure(figsize=(width_px / style.dpi, height_px / style.dpi), dpi=style.dpi)
    figure.patch.set_alpha(0.0)
    return figure.add_axes((0.0, 0.0, 1.0, 1.0))


def _columns_at_times(clock: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractional matrix column of each time, and the run of the clock it falls in.

    A spectrum can be stitched from sweeps whose clocks overlap, so the clock is split into increasing runs and a
    time is placed only when exactly one run contains it; the others are NaN.
    """
    steps = np.diff(clock)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    gap = 3.0 * float(np.median(positive)) if positive.size else np.inf
    breaks = np.flatnonzero(~np.isfinite(steps) | (steps <= 0) | (steps > gap)) + 1
    bounds = [0, *(int(index) for index in breaks), len(clock)]
    columns = np.full(times.shape, np.nan, dtype=np.float64)
    runs = np.full(times.shape, -1, dtype=int)
    matches = np.zeros(times.shape, dtype=int)
    for run, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        inside = np.isfinite(times) & (clock[begin] <= times) & (times <= clock[end - 1])
        columns[inside] = begin + np.interp(times[inside], clock[begin:end], np.arange(end - begin))
        runs[inside] = run
        matches += inside
    columns[matches != 1] = np.nan
    runs[matches != 1] = -1
    return columns, runs


def _rows_at_velocities(
    metadata: SpectralMetadata, columns: np.ndarray, velocities: np.ndarray, *, row_count: int
) -> np.ndarray:
    """Fractional matrix row of each velocity, NaN where the column is uncalibrated or the velocity is off scale."""
    rows = np.full(velocities.shape, np.nan, dtype=np.float64)
    for index in np.flatnonzero(np.isfinite(columns)):
        if metadata.row_velocity_mps is not None:
            axis = np.asarray(metadata.row_velocity_mps, dtype=np.float64).reshape(-1)
        elif metadata.column_velocity_mps is not None:
            axis = np.asarray(metadata.column_velocity_mps, dtype=np.float64)[int(round(columns[index]))]
        else:
            continue
        if axis.size != row_count or not np.all(np.isfinite(axis)):
            continue
        order = np.argsort(axis)
        rows[index] = np.interp(velocities[index], axis[order], order.astype(np.float64), left=np.nan, right=np.nan)
    return rows


def _label(measurement: Measurement) -> str:
    parts = (measurement.measure_id or measurement.kind).split("/")
    # `Cardiac/<mode>/<group>/<name>`; the name can itself contain a slash (`LVOT Trace/Manual`).
    name = "/".join(parts[3:]) if parts[0] == "Cardiac" and len(parts) > 3 else parts[-1]
    name = "\n".join(textwrap.wrap(name, width=12, break_long_words=False)) or name
    length, area = measurement.length_m, measurement.area_m2
    if length is not None:
        return f"{name}\n{1000.0 * length:.1f} mm"
    return name if area is None else f"{name}\n{1e4 * area:.1f} cm²"
