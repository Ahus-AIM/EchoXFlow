"""Matplotlib helpers for scan-converted displays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.axes import Axes

from echoxflow.scan.geometry import CartesianGrid, SectorGeometry


@dataclass(frozen=True)
class ScaleMarker:
    length_cm: float = 1.0
    side: str = "right"
    pad_fraction: float = 0.04
    color: str = "#E6E6E6"
    linewidth: float = 2.0
    label: str | None = None


@dataclass(frozen=True)
class SectorDepthRuler:
    """Centimeter depth ticks drawn beside one sector border."""

    side: Literal["left", "right"] = "left"
    tick_interval_cm: float = 1.0
    label_interval_cm: float = 5.0
    minor_tick_length_cm: float = 0.22
    major_tick_length_cm: float = 0.34
    label_pad_cm: float = 0.12
    color: str = "#E6E6E6"
    linewidth: float = 0.75
    border_linewidth: float = 0.65
    label_fontsize: float = 7.0
    show_border: bool = False
    include_boundary_ticks: bool = False
    minimum_tick_depth_cm: float | None = None
    omitted_tick_depths_cm: tuple[float, ...] = ()
    zorder: float = 8.0


@dataclass(frozen=True)
class SectorDepthTick:
    """One physical centimeter tick on a sector depth ruler."""

    depth_cm: float
    base_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    label_xy_m: tuple[float, float]
    label: str
    is_major: bool


def draw_cm_marker(ax: Axes, grid: CartesianGrid, marker: ScaleMarker | None = None) -> None:
    spec = marker or ScaleMarker()
    length_m = float(spec.length_cm) / 100.0
    x0, x1 = grid.x_range_m
    y0, y1 = grid.y_range_m
    x = x1 - spec.pad_fraction * (x1 - x0) if spec.side == "right" else x0 + spec.pad_fraction * (x1 - x0)
    y_bottom = y1 - spec.pad_fraction * (y1 - y0)
    y_top = y_bottom - length_m
    ax.plot([x, x], [y_bottom, y_top], color=spec.color, linewidth=spec.linewidth)
    label = spec.label if spec.label is not None else f"{spec.length_cm:g} cm"
    ax.text(x, y_top, label, color=spec.color, fontsize=8, ha="right" if spec.side == "right" else "left", va="bottom")


def sector_depth_ticks(geometry: SectorGeometry, ruler: SectorDepthRuler | None = None) -> tuple[SectorDepthTick, ...]:
    """Return centimeter tick coordinates along the selected sector side."""
    spec = ruler or SectorDepthRuler()
    depth_start_cm = 100.0 * float(geometry.depth_start_m)
    depth_end_cm = 100.0 * float(geometry.depth_end_m)
    interval = float(spec.tick_interval_cm)
    if not (
        np.isfinite(depth_start_cm)
        and np.isfinite(depth_end_cm)
        and np.isfinite(interval)
        and depth_end_cm > depth_start_cm
        and interval > 0.0
    ):
        return ()

    first_cm = math.ceil(depth_start_cm / interval) * interval
    last_cm = math.floor(depth_end_cm / interval) * interval
    if not spec.include_boundary_ticks:
        if math.isclose(first_cm, depth_start_cm, abs_tol=1e-6):
            first_cm += interval
        if math.isclose(last_cm, depth_end_cm, abs_tol=1e-6):
            last_cm -= interval
    if spec.minimum_tick_depth_cm is not None:
        first_cm = max(first_cm, math.ceil(float(spec.minimum_tick_depth_cm) / interval) * interval)
    if last_cm < first_cm:
        return ()

    angle = _sector_side_angle(geometry, side=spec.side)
    normal = _sector_side_outward_normal(geometry, angle)
    depths_cm = np.arange(first_cm, last_cm + 0.5 * interval, interval, dtype=np.float64)
    omitted_depths = tuple(float(depth) for depth in spec.omitted_tick_depths_cm)
    ticks: list[SectorDepthTick] = []
    for depth_cm in depths_cm:
        if any(math.isclose(float(depth_cm), omitted, abs_tol=1e-6) for omitted in omitted_depths):
            continue
        is_major = _is_major_tick(float(depth_cm), spec.label_interval_cm)
        tick_length_m = (spec.major_tick_length_cm if is_major else spec.minor_tick_length_cm) / 100.0
        label_pad_m = float(spec.label_pad_cm) / 100.0
        base = _sector_point(float(depth_cm) / 100.0, angle)
        end = base + normal * tick_length_m
        label_point = end + normal * label_pad_m
        label = f"{int(round(float(depth_cm)))}" if is_major else ""
        ticks.append(
            SectorDepthTick(
                depth_cm=float(depth_cm),
                base_xy_m=(float(base[0]), float(base[1])),
                end_xy_m=(float(end[0]), float(end[1])),
                label_xy_m=(float(label_point[0]), float(label_point[1])),
                label=label,
                is_major=is_major,
            )
        )
    return tuple(ticks)


def draw_sector_depth_ruler(
    ax: Axes,
    geometry: SectorGeometry,
    ruler: SectorDepthRuler | None = None,
) -> None:
    """Draw sector-side centimeter ticks in physical Cartesian coordinates."""
    spec = ruler or SectorDepthRuler()
    ticks = sector_depth_ticks(geometry, spec)
    if not ticks:
        return

    angle = _sector_side_angle(geometry, side=spec.side)
    if spec.show_border:
        start = _sector_point(float(geometry.depth_start_m), angle)
        end = _sector_point(float(geometry.depth_end_m), angle)
        ax.plot(
            [float(start[0]), float(end[0])],
            [float(start[1]), float(end[1])],
            color=spec.color,
            linewidth=spec.border_linewidth,
            antialiased=False,
            clip_on=False,
            zorder=spec.zorder,
        )

    for tick in ticks:
        ax.plot(
            [tick.base_xy_m[0], tick.end_xy_m[0]],
            [tick.base_xy_m[1], tick.end_xy_m[1]],
            color=spec.color,
            linewidth=spec.linewidth,
            antialiased=False,
            clip_on=False,
            zorder=spec.zorder,
        )
        if tick.label:
            ax.text(
                tick.label_xy_m[0],
                tick.label_xy_m[1],
                tick.label,
                color=spec.color,
                fontsize=spec.label_fontsize,
                ha="right" if spec.side == "left" else "left",
                va="center",
                clip_on=False,
                zorder=spec.zorder,
            )
    _expand_xlim_for_ruler(ax, ticks)


def set_cartesian_extent(ax: Axes, grid: CartesianGrid) -> None:
    ax.set_xlim(grid.x_range_m)
    ax.set_ylim(grid.y_range_m[1], grid.y_range_m[0])
    ax.set_aspect("equal")


def _sector_side_angle(geometry: SectorGeometry, *, side: Literal["left", "right"]) -> float:
    angles = (float(geometry.angle_start_rad), float(geometry.angle_end_rad))
    depth = 0.5 * (float(geometry.depth_start_m) + float(geometry.depth_end_m))
    points = [_sector_point(depth, angle) for angle in angles]
    x_values = [float(point[0]) for point in points]
    if side == "left":
        return angles[int(np.argmin(x_values))]
    return angles[int(np.argmax(x_values))]


def _sector_side_outward_normal(geometry: SectorGeometry, angle: float) -> np.ndarray:
    tangent = np.asarray([math.sin(float(angle)), math.cos(float(angle))], dtype=np.float64)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    depth = 0.5 * (float(geometry.depth_start_m) + float(geometry.depth_end_m))
    center_angle = 0.5 * (float(geometry.angle_start_rad) + float(geometry.angle_end_rad))
    outward = _sector_point(depth, angle) - _sector_point(depth, center_angle)
    if float(np.dot(normal, outward)) < 0.0:
        normal *= -1.0
    norm = float(np.linalg.norm(normal))
    return normal / max(norm, 1e-12)


def _sector_point(depth_m: float, angle_rad: float) -> np.ndarray:
    return np.asarray(
        [float(depth_m) * math.sin(float(angle_rad)), float(depth_m) * math.cos(float(angle_rad))],
        dtype=np.float64,
    )


def _is_major_tick(depth_cm: float, label_interval_cm: float) -> bool:
    interval = float(label_interval_cm)
    if not np.isfinite(interval) or interval <= 0.0:
        return False
    return math.isclose(depth_cm / interval, round(depth_cm / interval), abs_tol=1e-6)


def _expand_xlim_for_ruler(ax: Axes, ticks: tuple[SectorDepthTick, ...]) -> None:
    xs: list[float] = []
    for tick in ticks:
        xs.extend((tick.base_xy_m[0], tick.end_xy_m[0], tick.label_xy_m[0]))
    if not xs:
        return
    current = ax.get_xlim()
    ax.set_xlim(min(float(current[0]), min(xs)), max(float(current[1]), max(xs)))
