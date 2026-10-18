"""Matplotlib panel renderers for EchoXFlow modalities."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import Polygon, Rectangle

from echoxflow.colors import named_listed_colormap
from echoxflow.loading import LoadedArray
from echoxflow.plotting.colorbar import colorbar_spec_for_modality, draw_top_right_colorbar
from echoxflow.plotting.specs import PanelKind, PanelSpec, TraceSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import nearest_index, temporal_length
from echoxflow.scan.contours import ContourGroupLayout
from echoxflow.scan.geometry import CartesianGrid, SectorGeometry, sector_geometry_from_mapping
from echoxflow.scan.matplotlib import SectorDepthRuler, draw_sector_depth_ruler, sector_depth_ticks
from echoxflow.scan.rasterization import points_to_sector_indices
from echoxflow.scan.spherical import MosaicCell, MosaicSlicePlane, SphericalGeometry, SphericalMosaicLayout
from echoxflow.schema import semantic_name
from echoxflow.spectral import SpectralMetadata
from echoxflow.streams import default_value_range_for_path


@dataclass(frozen=True)
class PanelArtists:
    """Handle returned by a panel renderer to move an already drawn panel to another frame.

    ``update(time_s, frame_index)`` touches only per-frame artists; the axes, ticks, labels,
    colour bars and layout stay as drawn by the first frame.
    """

    update: Callable[[float, int], None]


class PanelRenderer(Protocol):
    def render(
        self, ax: Axes, panel: PanelSpec, *, time_s: float, frame_index: int, style: PlotStyle
    ) -> PanelArtists | None:
        pass


@contextmanager
def _collect_new_artists(ax: Axes, sink: list[Artist]) -> Iterator[None]:
    """Record every artist added to ``ax`` inside the block into ``sink``."""
    before = set(ax.get_children())
    try:
        yield
    finally:
        sink.extend(child for child in ax.get_children() if child not in before)


def _remove_artists(artists: list[Artist]) -> None:
    for artist in artists:
        artist.remove()
    artists.clear()


class ImagePanelRenderer:
    def render(
        self, ax: Axes, panel: PanelSpec, *, time_s: float, frame_index: int, style: PlotStyle
    ) -> PanelArtists | None:
        count = temporal_length(panel.loaded.data)
        flow_rgb = panel.loaded.attrs.get("flow_color_rgb") if panel.view != "cartesian" else None
        data = np.asarray(flow_rgb if isinstance(flow_rgb, np.ndarray) else panel.loaded.data)

        def frame_at(time: float, frame: int) -> tuple[int, np.ndarray]:
            index = nearest_index(panel.loaded.timestamps, time, count=count, fallback_index=frame)
            return index, _display_frame(data[index] if _has_temporal_axis(panel, count) else data)

        index, frame = frame_at(time_s, frame_index)
        cmap = None if frame.ndim == 3 else _transparent_bad_colormap(panel_colormap(panel.loaded))
        norm = None if frame.ndim == 3 else fixed_normalize(panel.loaded, frame)
        grid = _cartesian_grid(panel)
        image = ax.imshow(
            frame,
            cmap=cmap or _transparent_bad_colormap("gray"),
            norm=norm,
            interpolation="nearest",
            aspect="equal" if _is_physical_image(panel, grid) else "auto",
            extent=_cartesian_extent(grid),
        )
        ax.set_anchor("C")
        overlay_artists: list[Artist] = []

        def draw_overlays(time: float, frame_idx: int, shown: np.ndarray) -> None:
            with _collect_new_artists(ax, overlay_artists):
                _draw_image_annotations(
                    ax, panel, time_s=time, frame_index=frame_idx, frame_shape=shown.shape[:2], style=style
                )

        draw_overlays(time_s, index, frame)
        _draw_color_doppler_extent(ax, panel, frame_shape=frame.shape[:2], style=style)
        colorbar_cmap = _colorbar_colormap(panel, cmap)
        if colorbar_cmap is not None and (colorbar_spec := _colorbar_spec(panel)) is not None:
            draw_top_right_colorbar(ax, colorbar_cmap, colorbar_spec, style=style)
        _draw_cartesian_depth_ruler(ax, panel, style=style)
        _draw_mosaic_depth_guides(ax, panel, style=style)
        _finish_panel(ax, panel.label, style)

        def update(time: float, frame_idx: int) -> None:
            # The normalisation is fixed per stream, so only the pixels and the overlays change.
            new_index, new_frame = frame_at(time, frame_idx)
            image.set_data(new_frame)
            _remove_artists(overlay_artists)
            draw_overlays(time, new_index, new_frame)

        return PanelArtists(update)


class MatrixPanelRenderer:
    def render(
        self, ax: Axes, panel: PanelSpec, *, time_s: float, frame_index: int, style: PlotStyle
    ) -> PanelArtists | None:
        del frame_index
        matrix = _display_matrix(np.asarray(panel.loaded.data))
        timestamps = panel.loaded.timestamps
        cmap = panel_colormap(panel.loaded) or "magma"
        norm = fixed_normalize(panel.loaded, matrix)
        if timestamps is not None:
            ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
            if ts.size == matrix.shape[0] and ts.size > 1:
                y_size = matrix.shape[1]
                extent = (float(ts[0]), float(ts[-1]), *_matrix_y_extent(panel, y_size))
                ax.imshow(
                    matrix.T,
                    cmap=cmap,
                    norm=norm,
                    aspect="auto",
                    origin=_matrix_origin(panel),
                    extent=extent,
                )
                cursor = ax.axvline(float(time_s), color=style.cursor_color, linewidth=1.4)
                _draw_matrix_annotations(ax, panel, time_s=time_s, y_size=y_size, style=style)
                if not isinstance(cmap, str) and (colorbar_spec := _colorbar_spec(panel)) is not None:
                    draw_top_right_colorbar(ax, cmap, colorbar_spec, style=style)
                _finish_panel(ax, panel.label, style)
                _configure_matrix_y_axis(ax, panel, y_size=y_size, style=style)
                return PanelArtists(lambda time, _frame: cursor.set_xdata([float(time), float(time)]))
        y_size = matrix.shape[1]
        ax.imshow(
            matrix.T,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            origin=_matrix_origin(panel),
            extent=(0.0, float(matrix.shape[0]), *_matrix_y_extent(panel, y_size)),
        )
        _draw_matrix_annotations(ax, panel, time_s=time_s, y_size=y_size, style=style)
        if not isinstance(cmap, str) and (colorbar_spec := _colorbar_spec(panel)) is not None:
            draw_top_right_colorbar(ax, cmap, colorbar_spec, style=style)
        _finish_panel(ax, panel.label, style)
        _configure_matrix_y_axis(ax, panel, y_size=y_size, style=style)
        return PanelArtists(lambda _time, _frame: None)  # no cursor: the panel is identical on every frame


class LinePanelRenderer:
    def render(
        self, ax: Axes, panel: PanelSpec, *, time_s: float, frame_index: int, style: PlotStyle
    ) -> PanelArtists | None:
        del frame_index
        values = np.asarray(panel.loaded.data, dtype=np.float32)
        timestamps = panel.loaded.timestamps
        x, traces = _line_trace_values(values, timestamps)
        colors = panel.loaded.attrs.get("trace_colors", ())
        for index, trace in enumerate(traces):
            ax.plot(
                x,
                trace,
                color=_line_trace_color(index, len(traces), colors, style=style),
                linewidth=1.0,
            )
        cursor = ax.axvline(float(time_s), color=style.cursor_color, linewidth=1.4) if x.size else None
        _finish_panel(ax, panel.label, style)
        _draw_line_markers(ax, panel, style)
        if cursor is None:
            return PanelArtists(lambda _time, _frame: None)
        return PanelArtists(lambda time, _frame: cursor.set_xdata([float(time), float(time)]))


def _draw_line_markers(ax: Axes, panel: PanelSpec, style: PlotStyle) -> None:
    """Mark labelled points on a line panel (such as end-diastole and end-systole) with small captions."""
    markers = panel.loaded.attrs.get("line_markers")
    if not isinstance(markers, tuple) or not markers:
        return
    ax.margins(y=0.35)
    for x, y, text in markers:
        ax.plot([x], [y], marker="o", markersize=3.0, color=style.line_color, linestyle="None")
        ax.annotate(
            text,
            (x, y),
            xytext=(4, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=0.75 * style.axis_tick_label_fontsize,
            color=style.text_color,
            clip_on=True,
        )


def panel_label_lines(panel: PanelSpec) -> tuple[str, ...]:
    """Return the stream name and, for image and matrix panels, the array dtype and shape or `cartesian`."""
    name = semantic_name(panel.loaded.data_path)
    if panel.kind == "image" and (view := panel.loaded.attrs.get("strain_source_view")):
        return (f"{view} B-mode", panel.view)
    if panel.kind == "line":
        return (str(panel.loaded.attrs.get("line_label", name)),)
    if panel.view == "cartesian" and panel.kind == "image":
        return (name.removeprefix("cartesian_"), "cartesian")
    data = np.asarray(panel.loaded.data)
    return (name, f"{data.dtype} {'×'.join(str(size) for size in data.shape)}")


def draw_panel_label(ax: Axes, panel: PanelSpec, style: PlotStyle) -> None:
    """Write the panel's stream name and array description in its top-left corner.

    Call after the layout is final: the font shrinks until the longest line fits the axes width.
    """
    if panel.kind in {"strain_curves", "bullseye"}:
        if style.show_panel_labels:
            ax.set_title(panel.label, color=style.text_color, fontsize=style.panel_label_fontsize, pad=16)
        return
    if not style.show_panel_labels:
        return
    text = ax.text(
        0.012,
        0.988,
        "\n".join(panel_label_lines(panel)),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.panel_label_fontsize,
        color=style.text_color,
        linespacing=1.2,
        zorder=20,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": style.panel_facecolor, "edgecolor": "none", "alpha": 0.6},
    )
    renderer = ax.figure.canvas.get_renderer()  # type: ignore[attr-defined]
    available = 0.9 * ax.get_window_extent(renderer).width
    width = text.get_window_extent(renderer).width
    if width > available > 0.0:
        text.set_fontsize(max(6.0, style.panel_label_fontsize * available / width))


def renderer_for(kind: PanelKind) -> PanelRenderer:
    if kind in {"strain_curves", "bullseye"}:
        from echoxflow.plotting.strain import BullseyePanelRenderer, StrainCurvePanelRenderer

        return BullseyePanelRenderer() if kind == "bullseye" else StrainCurvePanelRenderer()
    if kind == "image":
        return ImagePanelRenderer()
    if kind == "matrix":
        return MatrixPanelRenderer()
    return LinePanelRenderer()


def _draw_image_annotations(
    ax: Axes,
    panel: PanelSpec,
    *,
    time_s: float,
    frame_index: int,
    frame_shape: tuple[int, ...],
    style: PlotStyle,
) -> None:
    mosaic_polygons = panel.loaded.attrs.get("mosaic_annotation_polygons")
    if isinstance(mosaic_polygons, tuple) and mosaic_polygons:
        resolved_index = min(max(0, int(frame_index)), len(mosaic_polygons) - 1)
        for polygon in mosaic_polygons[resolved_index]:
            pts = np.asarray(polygon, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] == 2:
                ax.add_patch(
                    Polygon(
                        pts,
                        closed=True,
                        facecolor=style.annotation_color,
                        edgecolor="none",
                        alpha=0.45,
                        linewidth=0.0,
                        zorder=7.9,
                    )
                )
    mosaic_lines = panel.loaded.attrs.get("mosaic_annotation_lines")
    if isinstance(mosaic_lines, tuple) and mosaic_lines:
        resolved_index = min(max(0, int(frame_index)), len(mosaic_lines) - 1)
        for line in mosaic_lines[resolved_index]:
            pts = np.asarray(line, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 2:
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=style.annotation_edge_color,
                    linewidth=style.annotation_linewidth + 1.0,
                    alpha=0.8,
                    zorder=8.0,
                )
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=style.annotation_color,
                    linewidth=style.annotation_linewidth,
                    alpha=0.95,
                    zorder=8.1,
                )
    for overlay in _annotation_overlays(panel, kind="sampling_gate"):
        _draw_sampling_gate_annotation(
            ax, panel, overlay, frame_index=frame_index, frame_shape=frame_shape, style=style
        )
    for overlay in _annotation_overlays(panel, kind="sampling_line"):
        _draw_sampling_line_annotation(ax, panel, overlay, frame_shape=frame_shape, style=style)
    for overlay in _annotation_overlays(panel, kind="physical_points"):
        points = _annotation_points_at_time(
            np.asarray(overlay.get("points"), dtype=np.float32),
            frame_index=frame_index,
            timestamps=_overlay_timestamps(overlay),
            time_s=time_s,
        )
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            continue
        xy = _annotation_image_xy(panel, points[:, :2], frame_shape=frame_shape)
        if xy is None or xy.size == 0:
            continue
        layout = overlay.get("group_layout")
        if isinstance(layout, ContourGroupLayout):
            grid = _contour_point_grid(xy, layout)
            if grid is not None:
                _draw_contour_boundaries(ax, grid, layout, style=style)
                _draw_annotation_label(ax, overlay, grid[:, int(layout.endo_index), :], style=style)
                continue
        if xy.shape[0] >= 2:
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=style.annotation_edge_color,
                linewidth=style.annotation_linewidth + 1.0,
                alpha=0.8,
                zorder=8.0,
            )
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=style.annotation_color,
                linewidth=style.annotation_linewidth,
                alpha=0.95,
                zorder=8.1,
            )
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            marker="+",
            linestyle="None",
            color=style.annotation_edge_color,
            markersize=style.annotation_markersize + 1.5,
            markeredgewidth=2.0,
            zorder=8.2,
        )
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            marker="+",
            linestyle="None",
            color=style.annotation_color,
            markersize=style.annotation_markersize,
            markeredgewidth=1.2,
            zorder=8.3,
        )
        _draw_annotation_label(ax, overlay, xy, style=style)


def _contour_point_grid(xy: np.ndarray, layout: ContourGroupLayout) -> np.ndarray | None:
    size = int(layout.group_size)
    if size < 2 or xy.shape[0] < size or xy.shape[0] % size != 0:
        return None
    grid = xy.reshape(-1, size, 2)
    grid = grid[np.all(np.isfinite(grid), axis=(1, 2))]
    return grid if grid.shape[0] > 0 else None


def _draw_contour_boundaries(ax: Axes, grid: np.ndarray, layout: ContourGroupLayout, *, style: PlotStyle) -> None:
    """Draw a strain tracking grid with the mid-wall row emphasised.

    The mid row is a bright polyline with round markers; rows toward the
    borders fade to small dim dots, and the outermost rows keep a faint thin
    outline so the ROI stays legible.
    """
    del layout
    size = grid.shape[1]
    mid = size // 2
    marker_size = style.annotation_markersize
    for column in range(size):
        pts = grid[:, column, :]
        distance = abs(column - mid)
        if distance == 0:
            if pts.shape[0] >= 2:
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=style.annotation_edge_color,
                    linewidth=style.annotation_linewidth * 1.4 + 1.0,
                    alpha=0.8,
                    zorder=8.0,
                )
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=style.annotation_color,
                linewidth=style.annotation_linewidth * 1.4,
                alpha=0.95,
                marker="o",
                markersize=marker_size * 1.3,
                markerfacecolor=style.annotation_color,
                markeredgecolor=style.annotation_edge_color,
                markeredgewidth=0.8,
                zorder=8.3,
            )
            continue
        if distance == 1:
            scale, alpha = 0.8, 0.7
        else:
            scale, alpha = 0.55, 0.45
            if pts.shape[0] >= 2:
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=style.annotation_color,
                    linewidth=style.annotation_linewidth * 0.6,
                    alpha=0.35,
                    zorder=8.1,
                )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            linestyle="None",
            marker="o",
            markersize=marker_size * scale,
            markerfacecolor=style.annotation_color,
            markeredgecolor=style.annotation_edge_color,
            markeredgewidth=0.4,
            alpha=alpha,
            zorder=8.2,
        )


def _draw_sampling_gate_annotation(
    ax: Axes,
    panel: PanelSpec,
    overlay: Mapping[object, object],
    *,
    frame_index: int,
    frame_shape: tuple[int, ...],
    style: PlotStyle,
) -> None:
    if panel.view != "cartesian" and _draw_beamspace_sampling_gate_annotation(
        ax, panel, overlay, frame_shape=frame_shape, style=style
    ):
        return
    points = _annotation_points(np.asarray(overlay.get("points"), dtype=np.float32), frame_index=frame_index)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return
    beam_xy = _annotation_image_xy(panel, points[:, :2], frame_shape=frame_shape)
    if beam_xy is None or beam_xy.shape[0] < 2:
        return
    tick_points = _annotation_points(np.asarray(overlay.get("tick_points"), dtype=np.float32), frame_index=frame_index)
    tick_xy = None
    if tick_points.ndim == 2 and tick_points.shape[0] >= 2 and tick_points.shape[1] >= 2:
        tick_xy = _annotation_image_xy(panel, tick_points[:, :2], frame_shape=frame_shape)
    if _draw_sector_sampling_gate_annotation(ax, panel, overlay, style=style):
        return
    _plot_sampling_gate_line(ax, beam_xy, linestyle="--", style=style)
    if tick_xy is not None and tick_xy.shape[0] >= 2:
        _plot_sampling_gate_line(ax, tick_xy, linestyle="-", style=style)


def _draw_sampling_line_annotation(
    ax: Axes,
    panel: PanelSpec,
    overlay: Mapping[object, object],
    *,
    frame_shape: tuple[int, ...],
    style: PlotStyle,
) -> None:
    if panel.view != "cartesian" and _draw_beamspace_sampling_line_annotation(
        ax, panel, overlay, frame_shape=frame_shape, style=style
    ):
        return
    if _draw_sector_sampling_line_annotation(ax, panel, overlay, style=style):
        return
    points = np.asarray(overlay.get("points"), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return
    xy = _annotation_image_xy(panel, points[:, :2], frame_shape=frame_shape)
    if xy is None or xy.shape[0] < 2:
        return
    _plot_sampling_gate_line(ax, xy, linestyle="-", style=style)


def _draw_sector_sampling_gate_annotation(
    ax: Axes,
    panel: PanelSpec,
    overlay: Mapping[object, object],
    *,
    style: PlotStyle,
) -> bool:
    geometry = _cartesian_geometry(panel)
    metadata = overlay.get("metadata")
    if geometry is None or not isinstance(metadata, Mapping):
        return False
    layout = _sector_sampling_gate_layout(metadata, geometry)
    if layout is None:
        return False
    segments, markers = layout
    for segment in segments:
        _plot_sampling_gate_line(ax, segment, linestyle="--", style=style)
    for marker in markers:
        _plot_sampling_gate_line(ax, marker, linestyle="-", style=style)
    return True


def _draw_beamspace_sampling_gate_annotation(
    ax: Axes,
    panel: PanelSpec,
    overlay: Mapping[object, object],
    *,
    frame_shape: tuple[int, ...],
    style: PlotStyle,
) -> bool:
    geometry = None if panel.loaded.stream is None else panel.loaded.stream.metadata.geometry
    metadata = overlay.get("metadata")
    if not isinstance(geometry, SectorGeometry) or not isinstance(metadata, Mapping):
        return False
    coords = _beamspace_sampling_gate_coordinates(metadata, geometry, frame_shape=frame_shape)
    if coords is None:
        return False
    x, row0, row1, row_end = coords
    row_top, row_bottom = min(row0, row1), max(row0, row1)
    for segment in _split_line_segments((x, 0.0), (x, row_top), (x, row_bottom), (x, row_end)):
        _plot_sampling_gate_line(ax, segment, linestyle="--", style=style)
    _plot_sampling_gate_markers(ax, np.asarray([[x, row_top], [x, row_bottom]], dtype=np.float32), style=style)
    return True


def _draw_beamspace_sampling_line_annotation(
    ax: Axes,
    panel: PanelSpec,
    overlay: Mapping[object, object],
    *,
    frame_shape: tuple[int, ...],
    style: PlotStyle,
) -> bool:
    geometry = None if panel.loaded.stream is None else panel.loaded.stream.metadata.geometry
    if not isinstance(geometry, SectorGeometry):
        return False
    tilt_rad = _sampling_line_tilt_rad(overlay)
    if tilt_rad is None:
        return False
    coords = _beamspace_sampling_line_coordinates(tilt_rad, geometry, frame_shape=frame_shape)
    if coords is None:
        return False
    x, row0, row1 = coords
    _plot_sampling_gate_line(ax, np.asarray([[x, row0], [x, row1]], dtype=np.float32), linestyle="-", style=style)
    return True


def _beamspace_sampling_gate_coordinates(
    metadata: Mapping[object, object],
    geometry: SectorGeometry,
    *,
    frame_shape: tuple[int, ...],
) -> tuple[float, float, float, float] | None:
    if len(frame_shape) < 2:
        return None
    center_depth = _metadata_float(metadata.get("gate_center_depth_m"))
    tilt_rad = _metadata_float(metadata.get("gate_tilt_rad"))
    if center_depth is None or tilt_rad is None:
        return None
    sample_volume = _metadata_float(metadata.get("gate_sample_volume_m")) or 0.006
    height, width = int(frame_shape[0]), int(frame_shape[1])
    row0 = _scale_interval_value(
        center_depth - 0.5 * sample_volume, geometry.depth_start_m, geometry.depth_end_m, height
    )
    row1 = _scale_interval_value(
        center_depth + 0.5 * sample_volume, geometry.depth_start_m, geometry.depth_end_m, height
    )
    col = _scale_interval_value(tilt_rad, geometry.angle_start_rad, geometry.angle_end_rad, width)
    return col, row0, row1, float(max(0, height - 1))


def _beamspace_sampling_line_coordinates(
    tilt_rad: float,
    geometry: SectorGeometry,
    *,
    frame_shape: tuple[int, ...],
) -> tuple[float, float, float] | None:
    if len(frame_shape) < 2:
        return None
    height, width = int(frame_shape[0]), int(frame_shape[1])
    row0 = _scale_interval_value(geometry.depth_start_m, geometry.depth_start_m, geometry.depth_end_m, height)
    row1 = _scale_interval_value(geometry.depth_end_m, geometry.depth_start_m, geometry.depth_end_m, height)
    col = _scale_interval_value(tilt_rad, geometry.angle_start_rad, geometry.angle_end_rad, width)
    return col, row0, row1


def _sector_sampling_gate_layout(
    metadata: Mapping[object, object],
    geometry: SectorGeometry,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None:
    center_depth = _metadata_float(metadata.get("gate_center_depth_m"))
    tilt_rad = _metadata_float(metadata.get("gate_tilt_rad"))
    if center_depth is None or tilt_rad is None:
        return None
    sample_volume = _metadata_float(metadata.get("gate_sample_volume_m")) or 0.006
    gate_start = float(np.clip(center_depth - 0.5 * sample_volume, geometry.depth_start_m, geometry.depth_end_m))
    gate_end = float(np.clip(center_depth + 0.5 * sample_volume, geometry.depth_start_m, geometry.depth_end_m))
    depth_start = float(geometry.depth_start_m)
    depth_end = float(geometry.depth_end_m)
    segments = _split_line_segments(
        _sector_point(depth_start, tilt_rad),
        _sector_point(gate_start, tilt_rad),
        _sector_point(gate_end, tilt_rad),
        _sector_point(depth_end, tilt_rad),
    )
    markers = (
        _sector_gate_marker(gate_start, tilt_rad, sample_volume),
        _sector_gate_marker(gate_end, tilt_rad, sample_volume),
    )
    return segments, markers


def _draw_sector_sampling_line_annotation(
    ax: Axes,
    panel: PanelSpec,
    overlay: Mapping[object, object],
    *,
    style: PlotStyle,
) -> bool:
    geometry = _cartesian_geometry(panel)
    metadata = overlay.get("metadata")
    if geometry is None or not isinstance(metadata, Mapping):
        return False
    line = _sector_sampling_line(metadata, geometry)
    if line is None:
        return False
    _plot_sampling_gate_line(ax, line, linestyle="-", style=style)
    return True


def _sector_sampling_line(metadata: Mapping[object, object], geometry: SectorGeometry) -> np.ndarray | None:
    tilt_rad = _metadata_float(_metadata_first(metadata, "gate_tilt_rad", "tilt_rad", "tilt"))
    if tilt_rad is None:
        return None
    return np.asarray(
        [
            _sector_point(float(geometry.depth_start_m), tilt_rad),
            _sector_point(float(geometry.depth_end_m), tilt_rad),
        ],
        dtype=np.float32,
    )


def _sampling_line_tilt_rad(overlay: Mapping[object, object]) -> float | None:
    metadata = overlay.get("metadata")
    if isinstance(metadata, Mapping):
        tilt_rad = _metadata_float(_metadata_first(metadata, "gate_tilt_rad", "tilt_rad", "tilt"))
        if tilt_rad is not None:
            return tilt_rad
    points = np.asarray(overlay.get("points"), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    distances = np.linalg.norm(points[:, :2], axis=1)
    if not np.any(np.isfinite(distances) & (distances > 1e-6)):
        return None
    point = points[int(np.nanargmax(distances)), :2]
    tilt_rad = float(np.arctan2(float(point[0]), float(point[1])))
    return tilt_rad if np.isfinite(tilt_rad) else None


def _sector_point(depth_m: float, angle_rad: float) -> tuple[float, float]:
    return float(depth_m * np.sin(angle_rad)), float(depth_m * np.cos(angle_rad))


def _sector_gate_marker(depth_m: float, angle_rad: float, width_m: float) -> np.ndarray:
    center = np.asarray(_sector_point(depth_m, angle_rad), dtype=np.float32)
    normal = np.asarray([np.cos(angle_rad), -np.sin(angle_rad)], dtype=np.float32)
    half_width = 0.5 * max(float(width_m), 1e-6)
    return np.asarray([center - half_width * normal, center + half_width * normal], dtype=np.float32)


def _split_line_segments(
    start: tuple[float, float],
    gate_start: tuple[float, float],
    gate_end: tuple[float, float],
    end: tuple[float, float],
) -> tuple[np.ndarray, ...]:
    segments = []
    for point0, point1 in ((start, gate_start), (gate_end, end)):
        segment = np.asarray([point0, point1], dtype=np.float32)
        if not np.allclose(segment[0], segment[1]):
            segments.append(segment)
    return tuple(segments)


def _plot_sampling_gate_line(
    ax: Axes,
    xy: np.ndarray,
    *,
    linestyle: str,
    style: PlotStyle,
) -> None:
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=style.sampling_gate_color,
        linestyle=linestyle,
        linewidth=style.sampling_gate_linewidth,
        alpha=0.98,
        zorder=8.5,
    )


def _plot_sampling_gate_markers(ax: Axes, xy: np.ndarray, *, style: PlotStyle) -> None:
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=style.sampling_gate_color,
        linestyle="None",
        marker="_",
        markersize=style.sampling_gate_markersize,
        markeredgewidth=style.sampling_gate_linewidth,
        alpha=1.0,
        zorder=8.7,
    )


def _draw_matrix_annotations(
    ax: Axes,
    panel: PanelSpec,
    *,
    time_s: float,
    y_size: int,
    style: PlotStyle,
) -> None:
    del time_s
    for overlay in _annotation_overlays(panel, kind=("spectral_points", "spectral_trace")):
        is_trace = overlay.get("kind") == "spectral_trace"
        points = np.asarray(overlay.get("points"), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError(f"Spectral annotation {overlay.get('path')!r} needs time/value pairs, got {points.shape}")
        if points.shape[0] == 0:
            continue
        x = points[:, 0]
        if _is_mmode_panel(panel):
            y = _mmode_depth_to_display_y(panel, points[:, 1], y_size=y_size)
            if y is None:
                continue
        else:
            y = _spectral_annotation_y_to_display_y(panel, overlay, points[:, 1], y_size=y_size, times=x)
        geometry = overlay.get("geometry_kind")
        if geometry in {"segment", "segments"}:
            for begin in range(0, len(x) - 1, 2):
                trace_x, trace_y = _spectral_trace_with_breaks(
                    panel, x[begin : begin + 2], y[begin : begin + 2], overlay
                )
                _draw_spectral_trace(ax, trace_x, trace_y, style=style)
            _draw_annotation_label(ax, overlay, np.column_stack([x, y]), style=style)
            continue
        if geometry == "polyline" or (geometry is None and is_trace):
            trace_x, trace_y = _spectral_trace_with_breaks(panel, x, y, overlay)
            _draw_spectral_trace(ax, trace_x, trace_y, style=style)
            _draw_annotation_label(ax, overlay, np.column_stack([x, y]), style=style)
            continue
        ax.plot(
            x,
            y,
            marker="+",
            linestyle="None",
            color=style.annotation_edge_color,
            markersize=style.annotation_markersize + 1.5,
            markeredgewidth=2.0,
            zorder=8.0,
        )
        ax.plot(
            x,
            y,
            marker="+",
            linestyle="None",
            color=style.annotation_color,
            markersize=style.annotation_markersize,
            markeredgewidth=1.2,
            zorder=8.1,
        )
        _draw_annotation_label(ax, overlay, np.column_stack([x, y]), style=style)


def _spectral_trace_with_breaks(
    panel: PanelSpec, x: np.ndarray, y: np.ndarray, overlay: Mapping[object, object] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Do not join physical curves across native gaps, segment boundaries or calibration changes."""
    clock = panel.loaded.timestamps
    if clock is None or len(x) < 2:
        return x, y
    columns, runs = _spectral_annotation_columns(panel, x, overlay)
    metadata = panel.loaded.attrs.get("spectral_metadata")
    calibration = metadata.calibration if isinstance(metadata, SpectralMetadata) else None
    output_x, output_y = [float(x[0])], [float(y[0])]
    for index in range(1, len(x)):
        low, high = sorted((columns[index - 1], columns[index]))
        broken = low < 0 or runs[index - 1] != runs[index]
        if not broken and calibration is not None:
            broken = any(
                values is not None and not np.all(values[low : high + 1] == values[low])
                for values in (calibration.baseline_fraction, calibration.nyquist_mps)
            )
        if broken:
            output_x.append(np.nan)
            output_y.append(np.nan)
        output_x.append(float(x[index]))
        output_y.append(float(y[index]))
    return np.asarray(output_x), np.asarray(output_y)


def _draw_spectral_trace(ax: Axes, x: np.ndarray, y: np.ndarray, *, style: PlotStyle) -> None:
    ax.plot(x, y, color=style.annotation_edge_color, linewidth=3.2, zorder=8.0, solid_capstyle="round")
    ax.plot(
        x,
        y,
        color=style.annotation_color,
        linewidth=1.6,
        zorder=8.1,
        solid_capstyle="round",
        marker="+" if len(x) == 1 or np.isnan(x).any() else None,
    )


def _annotation_overlays(panel: PanelSpec, *, kind: str | tuple[str, ...]) -> tuple[Mapping[object, object], ...]:
    raw = panel.loaded.attrs.get("annotation_overlays")
    if not isinstance(raw, tuple):
        return ()
    kinds = (kind,) if isinstance(kind, str) else kind
    return tuple(overlay for overlay in raw if isinstance(overlay, Mapping) and overlay.get("kind") in kinds)


def _draw_annotation_label(
    ax: Axes,
    overlay: Mapping[object, object],
    xy: np.ndarray,
    *,
    style: PlotStyle,
) -> None:
    label = _overlay_label(overlay)
    if label is None:
        return
    point = _representative_label_point(xy)
    if point is None:
        return
    text = ax.annotate(
        label,
        xy=(float(point[0]), float(point[1])),
        xytext=(4.0, -4.0),
        textcoords="offset points",
        ha="left",
        va="top",
        color=style.annotation_color,
        fontsize=style.annotation_label_fontsize,
        zorder=9.0,
        annotation_clip=True,
    )
    text.set_path_effects(
        [
            patheffects.withStroke(linewidth=2.0, foreground=style.annotation_edge_color),
        ]
    )
    text.set_in_layout(False)


def _overlay_label(overlay: Mapping[object, object]) -> str | None:
    raw = overlay.get("label")
    if not isinstance(raw, str):
        return None
    label = raw.strip()
    label = label.removeprefix("Cardiac/SD/").removesuffix("/Manual").strip()
    return label or None


def _representative_label_point(xy: np.ndarray) -> np.ndarray | None:
    points = np.asarray(xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.size == 0:
        return None
    return np.asarray(finite[int(finite.shape[0] // 2)], dtype=np.float32)


def _annotation_points(points: np.ndarray, *, frame_index: int) -> np.ndarray:
    return _annotation_points_at_time(points, frame_index=frame_index, timestamps=None, time_s=None)


def _annotation_points_at_time(
    points: np.ndarray,
    *,
    frame_index: int,
    timestamps: np.ndarray | None,
    time_s: float | None,
) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 3:
        index = min(max(0, int(frame_index)), arr.shape[0] - 1)
        if timestamps is not None and time_s is not None:
            ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
            if ts.size == arr.shape[0] and ts.size > 0:
                distances = np.abs(ts - float(time_s))
                distances[~np.isfinite(distances)] = np.inf
                index = int(np.clip(int(np.argmin(distances)), 0, arr.shape[0] - 1))
        return np.asarray(arr[index], dtype=np.float32)
    return arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else np.zeros((0, 2), dtype=np.float32)


def _annotation_image_xy(
    panel: PanelSpec,
    points_xy_m: np.ndarray,
    *,
    frame_shape: tuple[int, ...],
) -> np.ndarray | None:
    if panel.view == "cartesian":
        return np.asarray(points_xy_m, dtype=np.float32)
    geometry = panel.loaded.attrs.get("annotation_geometry")
    if not isinstance(geometry, SectorGeometry):
        geometry = None if panel.loaded.stream is None else panel.loaded.stream.metadata.geometry
    if not isinstance(geometry, SectorGeometry):
        return None
    output_shape = (int(frame_shape[0]), int(frame_shape[1]))
    row_col = points_to_sector_indices(points_xy_m, geometry, output_shape=output_shape)
    return np.column_stack([row_col[:, 1], row_col[:, 0]]).astype(np.float32)


def _overlay_timestamps(overlay: Mapping[object, object]) -> np.ndarray | None:
    raw = overlay.get("timestamps")
    if raw is None:
        return None
    timestamps = np.asarray(raw, dtype=np.float64).reshape(-1)
    return timestamps if timestamps.size else None


def _line_trace_values(values: np.ndarray, timestamps: np.ndarray | None) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim <= 1:
        y = arr.reshape(-1)
        x = _line_x_axis(y.size, timestamps)
        return x, (y,)
    ts = None if timestamps is None else np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts is not None:
        matching_axes = [axis for axis, size in enumerate(arr.shape) if size == ts.size]
        if matching_axes:
            matrix = np.moveaxis(arr, matching_axes[0], 0).reshape(ts.size, -1)
            return ts, tuple(np.asarray(matrix[:, index], dtype=np.float32) for index in range(matrix.shape[1]))
    matrix = arr.reshape(arr.shape[0], -1)
    if ts is not None and ts.size == matrix.shape[0]:
        return ts, tuple(np.asarray(matrix[:, index], dtype=np.float32) for index in range(matrix.shape[1]))
    if ts is not None and ts.size == matrix.shape[1]:
        return ts, tuple(np.asarray(matrix[index], dtype=np.float32) for index in range(matrix.shape[0]))
    x = np.arange(matrix.shape[0], dtype=np.float64)
    return x, tuple(np.asarray(matrix[:, index], dtype=np.float32) for index in range(matrix.shape[1]))


def _line_trace_color(index: int, count: int, colors: object, *, style: PlotStyle) -> str:
    if isinstance(colors, tuple) and index < len(colors) and isinstance(colors[index], str):
        return str(colors[index])
    if count < 2:
        return style.line_color
    palette = ("#440154", "#414487", "#2A788E", "#22A884", "#7AD151", "#FDE725")
    return palette[round(index * (len(palette) - 1) / (count - 1))]


def _line_x_axis(size: int, timestamps: np.ndarray | None) -> np.ndarray:
    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.size == size:
            return ts
    return np.arange(size, dtype=np.float64)


def _spectral_annotation_y_to_display_y(
    panel: PanelSpec,
    overlay: Mapping[object, object],
    values: np.ndarray,
    *,
    y_size: int,
    times: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(values, dtype=np.float32)
    unit_hint = _spectral_annotation_y_unit(overlay)
    coordinate_hint = _spectral_annotation_y_coordinate_system(overlay)
    if _is_spectral_centimeters_per_second_hint(unit_hint):
        return _spectral_velocity_to_display_y(panel, y / 100.0, y_size=y_size, times=times, overlay=overlay)
    if _is_spectral_meters_per_second_hint(unit_hint):
        return _spectral_velocity_to_display_y(panel, y, y_size=y_size, times=times, overlay=overlay)
    if _is_spectral_row_coordinate_hint(coordinate_hint):
        return _spectral_row_to_display_y(
            panel,
            y,
            y_size=y_size,
            use_cursor_box=_is_spectral_image_coordinate_hint(coordinate_hint),
        )
    if unit_hint is not None:
        return np.full(y.shape, np.nan, dtype=np.float32)

    axis = _spectral_velocity_axis(panel)
    # Unlabelled legacy overlays without calibration may already contain panel rows.
    if coordinate_hint is None and panel.loaded.attrs.get("spectral_metadata") is None:
        finite = y[np.isfinite(y)]
        upper = float(max(0, y_size - 1))
        if finite.size and np.all((finite >= -0.5) & (finite <= upper + 0.5)):
            if np.ptp(finite) >= max(2.0, 0.25 * upper):
                return _spectral_row_to_display_y(panel, y, y_size=y_size, use_cursor_box=False)
    # Compatibility for unlabelled legacy velocity traces with a known physical axis.
    if axis is not None and coordinate_hint is None and _values_look_like_centimeters_per_second(y, axis):
        if _values_fit_range(y / 100.0, axis):
            y = y / 100.0
    return _spectral_velocity_to_display_y(panel, y, y_size=y_size, times=times, overlay=overlay)


def _spectral_annotation_columns(
    panel: PanelSpec, times: np.ndarray, overlay: Mapping[object, object] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve points to native columns and clock runs; ambiguous ownership stays -1."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    columns = np.full(times.size, -1, dtype=int)
    runs = np.full(times.size, -1, dtype=int)
    matches = np.zeros(times.size, dtype=int)
    clock = panel.loaded.timestamps
    offsets = panel.loaded.attrs.get("spectral_segment_offsets")
    links = {} if overlay is None else overlay.get("links", {})
    segment_index = links.get("segment_index") if isinstance(links, Mapping) else None
    if clock is None or (segment_index is not None and offsets is None):
        return columns, runs
    clock = np.asarray(clock, dtype=np.float64).reshape(-1)
    delta = np.diff(clock)
    positive = delta[np.isfinite(delta) & (delta > 0)]
    gap = 3 * np.median(positive) if positive.size else np.inf
    boundaries = {0, len(clock), *(np.flatnonzero(~np.isfinite(delta) | (delta <= 0) | (delta > gap)) + 1)}
    if offsets is not None:
        boundaries.update(int(i) for i in np.asarray(offsets).reshape(-1) if 0 < i < len(clock))
    ordered = sorted(boundaries)
    for run, (begin, end) in enumerate(zip(ordered[:-1], ordered[1:], strict=True)):
        if segment_index is not None and np.searchsorted(np.asarray(offsets), begin, side="right") - 1 != segment_index:
            continue
        selected = np.flatnonzero(np.isfinite(times) & (clock[begin] <= times) & (times <= clock[end - 1]))
        right = np.clip(np.searchsorted(clock[begin:end], times[selected]) + begin, begin, end - 1)
        left = np.maximum(begin, right - 1)
        columns[selected] = np.where(
            abs(times[selected] - clock[left]) <= abs(clock[right] - times[selected]), left, right
        )
        runs[selected] = run
        matches[selected] += 1
    columns[matches != 1] = runs[matches != 1] = -1
    return columns, runs


def _spectral_velocity_to_display_y(
    panel: PanelSpec,
    velocity_mps: np.ndarray,
    *,
    y_size: int,
    times: np.ndarray | None = None,
    overlay: Mapping[object, object] | None = None,
) -> np.ndarray:
    metadata = panel.loaded.attrs.get("spectral_metadata")
    velocity = np.asarray(velocity_mps, dtype=np.float32)
    result = np.full(velocity.shape, np.nan, dtype=np.float32)
    if isinstance(metadata, SpectralMetadata) and metadata.calibration is not None:
        calibration = metadata.calibration
        valid = calibration.calibrated_columns()
        if valid is None or times is None:
            return result
        if panel.loaded.timestamps is None or len(panel.loaded.timestamps) != len(valid):
            return result
        columns, _ = _spectral_annotation_columns(panel, times, overlay)
        points = np.flatnonzero(columns >= 0)
        points = points[valid[columns[points]]]
        baseline = np.asarray(calibration.baseline_fraction)[columns[points]]
        nyquist = np.asarray(calibration.nyquist_mps)[columns[points]]
        native_rows = y_size * (1 - baseline + velocity.flat[points] / (2 * nyquist))
        result.flat[points] = np.where((0 <= native_rows) & (native_rows <= y_size), native_rows, np.nan)
        return result
    axis = _spectral_velocity_axis(panel)
    if axis is not None:
        rows = np.linspace(float(y_size) - 0.5, 0.5, axis.size, dtype=np.float32)
        order = np.argsort(axis)
        return np.interp(velocity, axis[order], rows[order], left=np.nan, right=np.nan).astype(np.float32)
    return result


def _spectral_velocity_axis(panel: PanelSpec) -> np.ndarray | None:
    metadata = panel.loaded.attrs.get("spectral_metadata")
    if isinstance(metadata, SpectralMetadata) and metadata.row_velocity_mps is not None:
        axis = np.asarray(metadata.row_velocity_mps, dtype=np.float32).reshape(-1)
        return axis if axis.size and np.all(np.isfinite(axis)) else None
    return None


def _spectral_row_to_display_y(
    panel: PanelSpec,
    rows: np.ndarray,
    *,
    y_size: int,
    use_cursor_box: bool,
) -> np.ndarray:
    panel_rows = np.asarray(rows, dtype=np.float32)
    metadata = panel.loaded.attrs.get("spectral_metadata")
    if use_cursor_box and isinstance(metadata, SpectralMetadata) and metadata.cursor_box is not None:
        box = metadata.cursor_box
        denominator = max(float(box.height) - 1.0, 1.0)
        panel_rows = (panel_rows - float(box.y)) * (float(max(0, int(y_size) - 1)) / denominator)
    panel_rows = np.clip(panel_rows, 0.0, float(max(0, int(y_size) - 1)))
    return (float(y_size) - 0.5 - panel_rows).astype(np.float32)


def _spectral_annotation_y_unit(overlay: Mapping[object, object]) -> str | None:
    return _normalized_overlay_string(
        overlay,
        "y_unit",
        "y_units",
        "velocity_unit",
        "velocity_units",
        "value_unit",
        "value_units",
        "unit",
        "units",
    )


def _spectral_annotation_y_coordinate_system(overlay: Mapping[object, object]) -> str | None:
    return _normalized_overlay_string(
        overlay,
        "y_coordinate_system",
        "coordinate_system",
        "coordinate_kind",
        "value_coordinate_system",
        "geometry_kind",
        "field",
    )


def _normalized_overlay_string(overlay: Mapping[object, object], *keys: str) -> str | None:
    for key in keys:
        value = overlay.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace("-", "_").replace(" ", "_")
    return None


def _is_spectral_centimeters_per_second_hint(value: str | None) -> bool:
    return value in {
        "cm/s",
        "cm_s",
        "cms",
        "cm_per_s",
        "s_cmps",
        "cm_per_second",
        "centimeter_per_second",
        "centimeters_per_second",
    }


def _is_spectral_meters_per_second_hint(value: str | None) -> bool:
    return value in {"m/s", "m_s", "mps", "s_mps", "m_per_s", "m_per_second", "meter_per_second", "meters_per_second"}


def _is_spectral_row_coordinate_hint(value: str | None) -> bool:
    if value is None:
        return False
    return any(token in value for token in ("row", "pixel", "image"))


def _is_spectral_image_coordinate_hint(value: str | None) -> bool:
    if value is None:
        return False
    return any(token in value for token in ("pixel", "image")) and "panel" not in value


def _values_fit_range(values: np.ndarray, reference: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    ref_min = float(np.min(reference))
    ref_max = float(np.max(reference))
    tolerance = max(1e-3, 0.05 * max(abs(ref_max - ref_min), 1e-3))
    return float(np.min(finite)) >= ref_min - tolerance and float(np.max(finite)) <= ref_max + tolerance


def _values_look_like_centimeters_per_second(values: np.ndarray, velocity_axis_mps: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    return float(np.max(np.abs(finite))) >= max(10.0, 4.0 * float(np.max(np.abs(velocity_axis_mps))))


def _is_mmode_panel(panel: PanelSpec) -> bool:
    return panel.loaded.data_path.strip("/") == "data/1d_motion_mode" or panel.loaded.name == "1d_motion_mode"


def _is_1d_matrix_panel(panel: PanelSpec) -> bool:
    return panel.loaded.name.startswith("1d_") or panel.loaded.data_path.strip("/").startswith("data/1d_")


def _matrix_y_extent(panel: PanelSpec, y_size: int) -> tuple[float, float]:
    if _is_mmode_panel(panel) and (depth_range := _mmode_depth_range_cm(panel)) is not None:
        return (depth_range[1] - depth_range[0], 0.0)
    return (0.0, float(y_size))


def _matrix_origin(panel: PanelSpec) -> Literal["upper", "lower"]:
    return "upper"


def _mmode_depth_range_cm(panel: PanelSpec) -> tuple[float, float] | None:
    geometry = None if panel.loaded.stream is None else panel.loaded.stream.metadata.geometry
    if not isinstance(geometry, SectorGeometry):
        metadata = panel.loaded.attrs.get("spectral_metadata")
        if isinstance(metadata, SpectralMetadata):
            return _mmode_depth_range_from_metadata_cm(metadata.raw)
        return None
    return (100.0 * float(geometry.depth_start_m), 100.0 * float(geometry.depth_end_m))


def _mmode_depth_to_display_y(panel: PanelSpec, depth_m: np.ndarray, *, y_size: int) -> np.ndarray | None:
    depth_range = _mmode_depth_range_cm(panel)
    if depth_range is None:
        return None
    start_cm, end_cm = depth_range
    span_cm = max(0.0, end_cm - start_cm)
    raw = np.asarray(depth_m, dtype=np.float32)
    y_cm = _mmode_annotation_depth_cm(raw, depth_range_cm=depth_range, y_size=y_size)
    y = y_cm - start_cm
    return cast(np.ndarray, np.clip(y, 0.0, span_cm).astype(np.float32))


def _mmode_depth_range_from_metadata_cm(raw: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(raw, Mapping):
        return None
    for key, scale in (
        ("y_range_cm", 1.0),
        ("depth_range_cm", 1.0),
        ("distance_range_cm", 1.0),
        ("y_range_m", 100.0),
        ("depth_range_m", 100.0),
        ("distance_range_m", 100.0),
        ("y_range", 1.0),
    ):
        value = raw.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        try:
            start, end = float(value[0]) * scale, float(value[1]) * scale
        except (TypeError, ValueError):
            continue
        if np.isfinite(start) and np.isfinite(end) and end > start:
            return start, end
    return None


def _mmode_annotation_depth_cm(
    values: np.ndarray,
    *,
    depth_range_cm: tuple[float, float],
    y_size: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    start_cm, end_cm = depth_range_cm
    span_cm = max(0.0, end_cm - start_cm)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, start_cm, dtype=np.float32)
    margin_cm = max(1.0, 0.05 * span_cm)
    if float(np.nanmin(finite)) >= start_cm - margin_cm and float(np.nanmax(finite)) <= end_cm + margin_cm:
        return arr
    meters_as_cm = 100.0 * arr
    finite_cm = meters_as_cm[np.isfinite(meters_as_cm)]
    if (
        finite_cm.size
        and float(np.nanmin(finite_cm)) >= start_cm - margin_cm
        and float(np.nanmax(finite_cm)) <= end_cm + margin_cm
    ):
        return meters_as_cm
    if y_size > 1 and float(np.nanmin(finite)) >= 0.0 and float(np.nanmax(finite)) <= float(y_size - 1):
        return start_cm + (arr / float(y_size - 1)) * span_cm
    return arr


def _configure_matrix_y_axis(ax: Axes, panel: PanelSpec, *, y_size: int, style: PlotStyle) -> None:
    if not _is_1d_matrix_panel(panel):
        return
    ticks, tick_labels, unit, baseline = _matrix_y_ticks(panel, y_size=y_size)
    if not ticks:
        return
    original_ylim = ax.get_ylim()
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel("")
    ax.yaxis.set_ticks_position("left")
    tag = ax.text(
        1.008,
        baseline,
        unit,
        transform=ax.get_yaxis_transform(),
        color=style.text_dim_color,
        fontsize=0.8 * style.axis_tick_label_fontsize,
        ha="left",
        va="center",
        clip_on=False,
    )
    tag.set_in_layout(False)  # sits in the margin without shrinking the strip against the ECG
    ax.tick_params(
        axis="y",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        colors=style.text_dim_color,
        labelsize=style.axis_tick_label_fontsize,
        length=2,
    )
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(style.text_dim_color)
    ax.spines["left"].set_linewidth(0.6)
    if not _is_mmode_panel(panel):
        ax.set_ylim(original_ylim)


def _matrix_y_ticks(panel: PanelSpec, *, y_size: int) -> tuple[tuple[float, ...], tuple[str, ...], str, float]:
    """Return tick positions, tick labels, the unit tag and the display y of the baseline the tag sits at."""
    if _is_mmode_panel(panel):
        depth_range = _mmode_depth_range_cm(panel)
        if depth_range is not None:
            span_cm = max(0.0, depth_range[1] - depth_range[0])
            ticks = _axis_limit_ticks(0.0, span_cm)
            return ticks, tuple(_format_axis_tick(tick) for tick in ticks), "[cm]", ticks[0]
        return (), (), "", 0.0
    metadata = panel.loaded.attrs.get("spectral_metadata")
    if isinstance(metadata, SpectralMetadata) and metadata.row_velocity_mps is not None:
        axis = np.asarray(metadata.row_velocity_mps, dtype=np.float32).reshape(-1)
        finite = axis[np.isfinite(axis)]
        if finite.size:
            velocity_ticks = _nice_axis_ticks(float(np.min(finite)), float(np.max(finite)), count=5)
            display_ticks = tuple(
                float(value)
                for value in _spectral_velocity_to_display_y(
                    panel, np.asarray(velocity_ticks, dtype=np.float32), y_size=y_size
                )
            )
            baseline = float(
                _spectral_velocity_to_display_y(panel, np.asarray([0.0], dtype=np.float32), y_size=y_size)[0]
            )
            lo, hi = sorted(display_ticks[:1] + display_ticks[-1:])
            baseline = min(max(baseline, lo), hi)
            return display_ticks, tuple(_format_axis_tick(tick) for tick in velocity_ticks), "[m/s]", baseline
    return (), (), "", 0.0


def _nice_axis_ticks(lo: float, hi: float, *, count: int) -> tuple[float, ...]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return ()
    if hi < lo:
        lo, hi = hi, lo
    if np.isclose(lo, hi):
        return (float(lo),)
    return tuple(float(value) for value in np.linspace(float(lo), float(hi), max(2, int(count))))


def _axis_limit_ticks(lo: float, hi: float) -> tuple[float, ...]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return ()
    if np.isclose(lo, hi):
        return (float(lo),)
    return (float(lo), float(hi))


def _format_axis_tick(value: float) -> str:
    if np.isclose(value, round(value), atol=1e-3):
        return str(int(round(value)))
    return f"{value:.1f}"


def _has_temporal_axis(panel: PanelSpec, count: int) -> bool:
    timestamps = panel.loaded.timestamps
    return (
        timestamps is not None
        and np.asarray(timestamps).reshape(-1).size == count
        and np.asarray(panel.loaded.data).ndim >= 3
    )


def render_ecg(
    ax: Axes,
    ecg: TraceSpec,
    *,
    time_s: float,
    style: PlotStyle,
    marker_times: np.ndarray | None = None,
) -> Callable[[float, np.ndarray | None], None]:
    """Draw the ECG trace and return ``update(time_s, marker_times)`` that redraws only the markers."""
    signal = np.asarray(ecg.signal, dtype=np.float32).reshape(-1)
    timestamps = np.asarray(ecg.timestamps, dtype=np.float64).reshape(-1)
    count = min(signal.size, timestamps.size)
    if count:
        ax.plot(timestamps[:count], signal[:count], color=style.ecg_trace_color, linewidth=0.9)

    marker_artists: list[Artist] = []

    def draw_markers(time: float, times: np.ndarray | None) -> None:
        if not count:
            return
        markers = _ecg_marker_times(times, fallback_time=float(time))
        linewidth = 1.4 if markers.size == 1 else 1.1
        alpha = 1.0 if markers.size == 1 else 0.75
        with _collect_new_artists(ax, marker_artists):
            for marker_time in markers:
                ax.axvline(float(marker_time), color=style.ecg_marker_color, linewidth=linewidth, alpha=alpha)

    draw_markers(time_s, marker_times)
    _finish_panel(ax, "", style)

    def update(time: float, new_marker_times: np.ndarray | None) -> None:
        _remove_artists(marker_artists)
        draw_markers(time, new_marker_times)

    return update


def _ecg_marker_times(marker_times: np.ndarray | None, *, fallback_time: float) -> np.ndarray:
    if marker_times is None:
        return np.asarray([fallback_time], dtype=np.float64)
    values = np.asarray(marker_times, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return values if values.size else np.asarray([fallback_time], dtype=np.float64)


def _finish_panel(ax: Axes, title: str, style: PlotStyle) -> None:
    del title
    ax.set_title("")
    ax.set_facecolor(style.panel_facecolor)
    ax.tick_params(colors=style.text_dim_color, labelsize=style.axis_tick_label_fontsize, length=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0.0, y=0.0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def fixed_normalize(loaded: LoadedArray, values: np.ndarray) -> Normalize:
    """Return a non-adaptive Matplotlib normalization for modality values."""
    value_range = None if loaded.stream is None else loaded.stream.metadata.value_range
    if value_range is None:
        value_range = default_value_range_for_path(loaded.data_path, np.asarray(values))
    if value_range is None:
        value_range = _dtype_value_range(np.asarray(values))
    return Normalize(vmin=value_range[0], vmax=value_range[1], clip=True)


def _transparent_bad_colormap(cmap: Colormap | str | None) -> Colormap | None:
    if cmap is None:
        return None
    base = colormaps[cmap] if isinstance(cmap, str) else cmap
    out = base.copy()
    out.set_bad(alpha=0.0)
    return out


def _stream_value_range(loaded: LoadedArray) -> tuple[float, float] | None:
    if loaded.stream is None:
        return None
    return loaded.stream.metadata.value_range


def _colorbar_data_path(panel: PanelSpec) -> str:
    raw = panel.loaded.attrs.get("cartesian_colorbar_data_path") if panel.view == "cartesian" else None
    return str(raw) if isinstance(raw, str) and raw else panel.loaded.data_path


def _colorbar_value_range(panel: PanelSpec) -> tuple[float, float] | None:
    raw = panel.loaded.attrs.get("cartesian_colorbar_value_range") if panel.view == "cartesian" else None
    if isinstance(raw, tuple) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    return _stream_value_range(panel.loaded)


def panel_colormap(loaded: LoadedArray, data_path: str | None = None) -> Colormap | None:
    """Return the recording's own colormap when loaded with the array, else the configured one for its path."""
    own = loaded.attrs.get("colormap")
    if isinstance(own, Colormap):
        return own
    return named_listed_colormap(loaded.data_path if data_path is None else data_path)


def _colorbar_colormap(panel: PanelSpec, image_cmap: Colormap | None) -> Colormap | None:
    if panel.view != "cartesian":
        return None
    return image_cmap or panel_colormap(panel.loaded, _colorbar_data_path(panel))


def _colorbar_spec(panel: PanelSpec):
    if panel.view != "cartesian":
        return None
    table = panel.loaded.attrs.get("cartesian_colorbar_table")
    return colorbar_spec_for_modality(
        _colorbar_data_path(panel),
        value_range=_colorbar_value_range(panel),
        table=table if isinstance(table, np.ndarray) else None,
    )


def _cartesian_grid(panel: PanelSpec) -> CartesianGrid | None:
    if panel.view != "cartesian":
        return None
    grid = panel.loaded.attrs.get("cartesian_grid")
    return grid if isinstance(grid, CartesianGrid) else None


def _is_physical_image(panel: PanelSpec, grid: CartesianGrid | None) -> bool:
    """True when pixels are isotropic physical samples, so the axes box must keep the image's true aspect."""
    return grid is not None or (panel.view == "cartesian" and panel.loaded.attrs.get("mosaic_layout") is not None)


def _cartesian_extent(grid: CartesianGrid | None) -> tuple[float, float, float, float] | None:
    if grid is None:
        return None
    return grid.x_range_m[0], grid.x_range_m[1], grid.y_range_m[1], grid.y_range_m[0]


def _cartesian_geometry(panel: PanelSpec) -> SectorGeometry | None:
    if panel.view != "cartesian" or panel.loaded.stream is None:
        return None
    geometry = panel.loaded.stream.metadata.geometry
    return geometry if isinstance(geometry, SectorGeometry) else None


def _draw_cartesian_depth_ruler(ax: Axes, panel: PanelSpec, *, style: PlotStyle) -> None:
    geometry = _cartesian_geometry(panel)
    if geometry is None or not style.show_cartesian_depth_ruler:
        return
    side: Literal["left", "right"] = "right" if style.cartesian_depth_ruler_side == "right" else "left"
    ruler = SectorDepthRuler(
        side=side,
        tick_interval_cm=style.cartesian_depth_ruler_tick_interval_cm,
        label_interval_cm=style.cartesian_depth_ruler_label_interval_cm,
        minor_tick_length_cm=style.cartesian_depth_ruler_minor_tick_length_cm,
        major_tick_length_cm=style.cartesian_depth_ruler_major_tick_length_cm,
        label_pad_cm=style.cartesian_depth_ruler_label_pad_cm,
        color=style.ecg_trace_color,
        linewidth=style.cartesian_depth_ruler_linewidth,
        border_linewidth=style.cartesian_depth_ruler_border_linewidth,
        label_fontsize=style.cartesian_depth_ruler_label_fontsize,
        show_border=style.cartesian_depth_ruler_show_border,
    )
    draw_sector_depth_ruler(ax, geometry, ruler)


def _slice_edge_slope(geometry: SphericalGeometry, angle_deg: float) -> float:
    """Half-width per metre of depth z of a radial-axis slice, i.e. the slope of its left and right sector edges."""
    angle = math.radians(float(angle_deg))
    cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
    tan_az = math.tan(0.5 * geometry.azimuth_width_rad)
    tan_el = math.tan(0.5 * geometry.elevation_width_rad)
    slopes = [math.inf]
    if sin_a > 1e-9:
        slopes.append(tan_el / sin_a)
    denom = cos_a * cos_a - tan_az * tan_az * sin_a * sin_a
    if denom > 1e-12:
        slopes.append(tan_az / math.sqrt(denom))
    return min(slopes)


def _mosaic_level_cells(layout: SphericalMosaicLayout) -> tuple[MosaicCell, ...]:
    """The shallowest, middle and deepest depth cuts, marked on the long-axis slices."""
    by_position = {(cell.row, cell.col): cell for cell in layout.cells}
    return tuple(by_position[position] for position in ((0, 1), (1, 2), (2, 3)) if position in by_position)


def _draw_mosaic_depth_guides(ax: Axes, panel: PanelSpec, *, style: PlotStyle) -> None:
    layout = panel.loaded.attrs.get("mosaic_layout")
    geometry = panel.loaded.attrs.get("mosaic_geometry")
    if not isinstance(layout, SphericalMosaicLayout) or not isinstance(geometry, SphericalGeometry):
        return
    levels = _mosaic_level_cells(layout) if style.show_mosaic_levels else ()
    if levels:
        _draw_mosaic_level_brackets(ax, levels, style=style)
    for cell in layout.cells:
        plane = cell.plane
        if cell.kind != "radial_axis" or plane is None or cell.angle_deg is None:
            continue
        slope = _slice_edge_slope(geometry, cell.angle_deg)
        if not math.isfinite(slope):
            continue
        sx = (cell.image_width - 1) / max(plane.x_max - plane.x_min, 1e-12)
        sy = (cell.image_height - 1) / max(plane.y_max - plane.y_min, 1e-12)

        to_px = _slice_pixel_mapper(cell, plane, sx, sy)
        if style.show_cartesian_depth_ruler:
            _draw_slice_depth_ruler(ax, cell, geometry, slope, to_px, sx, style=style)
        for color, level in zip(style.mosaic_level_colors, levels, strict=False):
            if level.plane is None:
                continue
            _draw_slice_level(ax, cell, level, slope, to_px, color=color, style=style)


def _slice_pixel_mapper(
    cell: MosaicCell, plane: MosaicSlicePlane, sx: float, sy: float
) -> Callable[[float, float], tuple[float, float]]:
    def to_px(u: float, z: float) -> tuple[float, float]:
        return cell.image_x0 + (u - plane.x_min) * sx, cell.image_y0 + (z - plane.y_min) * sy

    return to_px


def _draw_slice_depth_ruler(
    ax: Axes,
    cell: MosaicCell,
    geometry: SphericalGeometry,
    slope: float,
    to_px: Callable[[float, float], tuple[float, float]],
    sx: float,
    *,
    style: PlotStyle,
) -> None:
    plane = cast("MosaicSlicePlane", cell.plane)
    half_angle = math.atan(slope)
    sector = SectorGeometry(
        depth_start_m=float(geometry.depth_start_m),
        depth_end_m=float(geometry.depth_end_m),
        angle_start_rad=-half_angle,
        angle_end_rad=half_angle,
    )
    ruler = SectorDepthRuler(
        side="left",
        tick_interval_cm=style.cartesian_depth_ruler_tick_interval_cm,
        label_interval_cm=style.cartesian_depth_ruler_label_interval_cm,
        minor_tick_length_cm=style.cartesian_depth_ruler_minor_tick_length_cm,
        major_tick_length_cm=style.cartesian_depth_ruler_major_tick_length_cm,
        label_pad_cm=style.cartesian_depth_ruler_label_pad_cm,
        color=style.ecg_trace_color,
        linewidth=style.cartesian_depth_ruler_linewidth,
        border_linewidth=style.cartesian_depth_ruler_border_linewidth,
        label_fontsize=style.cartesian_depth_ruler_label_fontsize,
        show_border=False,
        zorder=7.5,  # below the mesh lines (8.0), so reused and freshly built frames stack identically
    )
    ticks = [t for t in sector_depth_ticks(sector, ruler) if t.base_xy_m[1] >= plane.y_min]
    if not ticks:
        return
    # The apex sits above the slice; the edge is in the tick base frame (u = x, z = y).
    deepest = max(ticks, key=lambda t: t.depth_cm)
    outward_m = abs(deepest.label_xy_m[0] - deepest.base_xy_m[0])
    label_px = 0.6 * style.cartesian_depth_ruler_label_fontsize * ax.figure.dpi / 72.0 * 1.6
    free_px = to_px(deepest.base_xy_m[0], deepest.base_xy_m[1])[0] - cell.x0
    inside = free_px < outward_m * sx + label_px
    sign = -1.0 if inside else 1.0
    for tick in ticks:
        bx, by = to_px(*tick.base_xy_m)
        ex, ey = to_px(*tick.end_xy_m)
        lx, ly = to_px(*tick.label_xy_m)
        ex, ey = bx + sign * (ex - bx), by + sign * (ey - by)
        lx, ly = bx + sign * (lx - bx), by + sign * (ly - by)
        ax.plot(
            [bx, ex],
            [by, ey],
            color=ruler.color,
            linewidth=ruler.linewidth,
            antialiased=False,
            clip_on=False,
            zorder=ruler.zorder,
        )
        if tick.label:
            ax.text(
                lx,
                ly,
                tick.label,
                color=ruler.color,
                fontsize=ruler.label_fontsize,
                ha="left" if inside else "right",
                va="center",
                clip_on=False,
                zorder=ruler.zorder,
            )


def _draw_slice_level(
    ax: Axes,
    cell: MosaicCell,
    level: MosaicCell,
    slope: float,
    to_px: Callable[[float, float], tuple[float, float]],
    *,
    color: str,
    style: PlotStyle,
) -> None:
    plane = cast("MosaicSlicePlane", level.plane)
    depth_m = float(plane.plane_point[2])
    half_m = slope * depth_m
    x0, y = to_px(-half_m, depth_m)
    x1, _ = to_px(half_m, depth_m)
    ax.plot([x0, x1], [y, y], color=color, linewidth=1.6, zorder=8.3, solid_capstyle="butt")
    fontsize = max(6.0, 0.7 * style.cartesian_depth_ruler_label_fontsize)
    label_px = 4.4 * fontsize * ax.figure.dpi / 72.0 * 0.6
    fits = x1 + 3.0 + label_px <= cell.x0 + cell.width
    ax.text(
        x1 + 3.0 if fits else x1,
        y if fits else y - 2.0,
        f"{100.0 * depth_m:.1f} cm",
        color=color,
        fontsize=fontsize,
        ha="left" if fits else "right",
        va="center" if fits else "bottom",
        clip_on=False,
        zorder=8.3,
    )


def _draw_mosaic_level_brackets(ax: Axes, levels: tuple[MosaicCell, ...], *, style: PlotStyle) -> None:
    for color, cell in zip(style.mosaic_level_colors, levels, strict=False):
        size = 0.08 * min(cell.image_height, cell.image_width)
        x0, y0 = cell.image_x0 + 1.5, cell.image_y0 + 1.5
        ax.plot(
            [x0, x0, x0 + size],
            [y0 + size, y0, y0],
            color=color,
            linewidth=2.0,
            solid_joinstyle="miter",
            solid_capstyle="butt",
            clip_on=False,
            zorder=8.3,
        )


def _draw_color_doppler_extent(ax: Axes, panel: PanelSpec, *, frame_shape: tuple[int, ...], style: PlotStyle) -> None:
    if not style.show_color_doppler_extent:
        return
    if panel.view == "cartesian":
        sector = panel.loaded.attrs.get("cartesian_color_doppler_sector")
        points = _native_physical_polygon(sector)
        if points is None:
            geometry = panel.loaded.attrs.get("cartesian_color_doppler_geometry")
            points = _sector_extent_polygon(geometry) if isinstance(geometry, SectorGeometry) else None
        if points is not None:
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=style.color_doppler_extent_color,
                linewidth=style.color_doppler_extent_linewidth,
                alpha=0.95,
                zorder=7.0,
            )
        return
    rect = _beamspace_extent_rect(panel, frame_shape=frame_shape)
    if rect is None:
        return
    row0, row1, col0, col1 = rect
    ax.add_patch(
        Rectangle(
            (col0 - 0.5, row0 - 0.5),
            max(0.0, col1 - col0 + 1.0),
            max(0.0, row1 - row0 + 1.0),
            fill=False,
            edgecolor=style.color_doppler_extent_color,
            linewidth=style.color_doppler_extent_linewidth,
            alpha=0.95,
            zorder=7.0,
        )
    )


def _beamspace_extent_rect(
    panel: PanelSpec, *, frame_shape: tuple[int, ...]
) -> tuple[float, float, float, float] | None:
    if len(frame_shape) < 2:
        return None
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        return None
    color_sector = panel.loaded.attrs.get("beamspace_color_doppler_sector")
    reference_sector = panel.loaded.attrs.get("beamspace_reference_sector")
    if isinstance(reference_sector, Mapping) and isinstance(color_sector, Mapping):
        return _beamspace_rect_from_native(
            reference_sector=reference_sector, color_sector=color_sector, shape=frame_shape
        )
    return None


def _beamspace_rect_from_native(
    *, reference_sector: Mapping[object, object], color_sector: Mapping[object, object], shape: tuple[int, ...]
) -> tuple[float, float, float, float] | None:
    reference = _native_geometry(reference_sector)
    color = _native_geometry(color_sector)
    if reference is None or color is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    row0 = _scale_interval_value(color.depth_start_m, reference.depth_start_m, reference.depth_end_m, height)
    row1 = _scale_interval_value(color.depth_end_m, reference.depth_start_m, reference.depth_end_m, height)
    col0 = _scale_interval_value(color.angle_start_rad, reference.angle_start_rad, reference.angle_end_rad, width)
    col1 = _scale_interval_value(color.angle_end_rad, reference.angle_start_rad, reference.angle_end_rad, width)
    return min(row0, row1), max(row0, row1), min(col0, col1), max(col0, col1)


def _native_geometry(sector: Mapping[object, object]) -> SectorGeometry | None:
    raw = sector.get("geometry")
    if not isinstance(raw, Mapping):
        return None
    try:
        return sector_geometry_from_mapping(raw)
    except (KeyError, TypeError, ValueError):
        return None


def _native_physical_polygon(sector: object) -> np.ndarray | None:
    if not isinstance(sector, Mapping):
        return None
    overlays = sector.get("overlays")
    if not isinstance(overlays, Mapping):
        return None
    polygons = overlays.get("physical_polygons")
    if not isinstance(polygons, list):
        return None
    selected = next(
        (
            polygon
            for polygon in polygons
            if isinstance(polygon, Mapping) and str(polygon.get("label", "")) == "color_doppler_extent"
        ),
        next((polygon for polygon in polygons if isinstance(polygon, Mapping)), None),
    )
    if not isinstance(selected, Mapping):
        return None
    points = np.asarray(selected.get("points"), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return None
    if not np.allclose(points[0], points[-1]):
        points = np.concatenate([points, points[:1]], axis=0)
    return points


def _scale_interval_value(value: float, start: float, end: float, size: int) -> float:
    span = max(1e-12, float(end) - float(start))
    return float(np.clip((float(value) - float(start)) / span, 0.0, 1.0) * max(0, int(size) - 1))


def _metadata_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _metadata_first(metadata: Mapping[object, object], *keys: str) -> object | None:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _sector_extent_polygon(geometry: SectorGeometry, *, n_points: int = 96) -> np.ndarray:
    theta = np.linspace(
        float(geometry.angle_start_rad),
        float(geometry.angle_end_rad),
        int(max(8, n_points)),
        dtype=np.float32,
    )
    d0 = float(geometry.depth_start_m)
    d1 = float(geometry.depth_end_m)
    near = np.column_stack([d0 * np.sin(theta), d0 * np.cos(theta)]).astype(np.float32)
    far = np.column_stack([d1 * np.sin(theta[::-1]), d1 * np.cos(theta[::-1])]).astype(np.float32)
    points = np.concatenate([near, far, near[:1]], axis=0)
    return points


def _dtype_value_range(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def _display_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim <= 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        return arr
    return np.asarray(np.nanmax(arr, axis=0))


def _display_matrix(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)
