"""Contour rasterization helpers for echo annotations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class ContourGroupLayout:
    group_size: int
    endo_index: int
    outer_index: int


@dataclass(frozen=True)
class ContourMaskResult:
    group_layout: ContourGroupLayout
    point_grid: np.ndarray
    endo_mask: np.ndarray
    myo_mask: np.ndarray


LV_GROUP_LAYOUT = ContourGroupLayout(group_size=5, endo_index=0, outer_index=4)
AFIRV_GROUP_LAYOUT = ContourGroupLayout(group_size=3, endo_index=0, outer_index=2)


def contour_group_layout_for_metadata(
    *,
    field_type: object = "",
    content_type: object = "",
    point_count: int | None = None,
) -> ContourGroupLayout:
    field = str(field_type or "").strip().upper()
    content = str(content_type or "").strip().lower()
    if field == "AFIRV" or content == "2d_right_ventricular_strain":
        return AFIRV_GROUP_LAYOUT
    if field in {"AUTOEF", "AFI", "AFILA"} or content in {
        "2d_left_ventricular_strain",
        "2d_left_atrial_strain",
    }:
        return LV_GROUP_LAYOUT
    if point_count is not None:
        count = int(point_count)
        if count > 0 and count % AFIRV_GROUP_LAYOUT.group_size == 0 and count % LV_GROUP_LAYOUT.group_size != 0:
            return AFIRV_GROUP_LAYOUT
    return LV_GROUP_LAYOUT


def build_contour_masks(
    points_px: np.ndarray,
    *,
    image_shape: tuple[int, int],
    group_layout: ContourGroupLayout,
) -> ContourMaskResult:
    height, width = int(image_shape[0]), int(image_shape[1])
    empty = np.zeros((height, width), dtype=bool)
    point_grid = _reshape_valid_point_grid(points_px, group_layout=group_layout)
    if point_grid.shape[0] < 2:
        return ContourMaskResult(
            group_layout=group_layout,
            point_grid=point_grid,
            endo_mask=empty.copy(),
            myo_mask=empty.copy(),
        )

    endo_mask = np.zeros((height, width), dtype=bool)
    _rasterize_polygon_into(endo_mask, point_grid[:, group_layout.endo_index, :])
    myo_mask = np.zeros((height, width), dtype=bool)
    start_col = min(group_layout.endo_index, group_layout.outer_index)
    stop_col = max(group_layout.endo_index, group_layout.outer_index)
    # The myocardium is the union of the quads between neighbouring contour rows and columns, each filled and
    # outlined like any polygon; quads of (near) zero area are skipped.
    quads = np.stack(
        (
            point_grid[:-1, start_col:stop_col],
            point_grid[1:, start_col:stop_col],
            point_grid[1:, start_col + 1 : stop_col + 1],
            point_grid[:-1, start_col + 1 : stop_col + 1],
        ),
        axis=2,
    ).reshape(-1, 4, 2)
    quads = quads[np.abs(_quad_areas(quads)) > 1e-6]
    if quads.shape[0]:
        _fill_polygons_scanlines(myo_mask, quads)
        _rasterize_polygons_boundary(myo_mask, quads)
    myo_mask &= ~endo_mask

    return ContourMaskResult(
        group_layout=group_layout,
        point_grid=point_grid,
        endo_mask=endo_mask,
        myo_mask=myo_mask,
    )


def rasterize_polygon_pixels(vertices_px: np.ndarray, *, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=bool)
    _rasterize_polygon_into(mask, vertices_px)
    return mask


def _rasterize_polygon_into(mask: np.ndarray, vertices: np.ndarray) -> None:
    polygon = _finite_vertices(vertices)
    if polygon.shape[0] == 0:
        return
    if polygon.shape[0] >= 3 and abs(_polygon_area(polygon)) > 1e-6:
        _fill_polygons_scanlines(mask, polygon[None])
    _rasterize_polygons_boundary(mask, polygon[None])


def _reshape_valid_point_grid(points: np.ndarray, *, group_layout: ContourGroupLayout) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        return np.zeros((0, group_layout.group_size, 2), dtype=np.float32)
    row_count = values.shape[0] // group_layout.group_size
    if row_count <= 0 or row_count * group_layout.group_size != values.shape[0]:
        return np.zeros((0, group_layout.group_size, 2), dtype=np.float32)
    grid = values.reshape(row_count, group_layout.group_size, 2)
    finite_rows = np.isfinite(grid).all(axis=(1, 2))
    return np.asarray(grid[finite_rows], dtype=np.float32)


def _fill_polygons_scanlines(mask: np.ndarray, polygons: np.ndarray) -> None:
    """Fill ``polygons`` (``[P,V,2]`` float32 ``x,y`` pixel vertices) into ``mask`` with the even-odd scanline rule.

    Pixel ``(row, col)`` has its centre at ``x = col, y = row``, the convention of the contour pixel coordinates
    (:class:`~echoxflow.scan.coordinates.BeamspacePixelGrid`, :class:`~echoxflow.scan.coordinates.CartesianPixelGrid`)
    and of the outline in :func:`_mark_points`. Each pixel row is sampled at ``y = row``; an edge crosses it when ``y``
    lies in ``[min(y0, y1), max(y0, y1))`` (horizontal edges never do). The sorted crossings pair up into spans, and a
    span ``[left, right]`` covers the columns ``ceil(left) .. floor(right)``. The arithmetic is done in float64
    in the same order as a per-edge evaluation, so the result is independent of how polygons are batched.
    """
    height, width = mask.shape
    if polygons.shape[0] == 0 or height == 0 or width == 0:
        return
    vertices = np.asarray(polygons, dtype=np.float32)
    x0 = vertices[:, :, 0].astype(np.float64)
    y0 = vertices[:, :, 1].astype(np.float64)
    x1 = np.roll(x0, -1, axis=1)
    y1 = np.roll(y0, -1, axis=1)
    # Only the rows within a polygon's vertical extent can have crossings: scan each polygon over its own rows.
    first_row = np.clip(np.floor(np.min(y0, axis=1)), 0.0, float(height - 1)).astype(np.int64)
    last_row = np.clip(np.ceil(np.max(y0, axis=1)), -1.0, float(height - 1)).astype(np.int64)
    row_count = int(np.max(last_row - first_row)) + 1
    if row_count <= 0:
        return
    rows = first_row[:, None] + np.arange(row_count)[None, :]
    in_extent = rows <= last_row[:, None]
    y = rows.astype(np.float64)[:, :, None]
    y0b, y1b, x0b, x1b = (values[:, None, :] for values in (y0, y1, x0, x1))
    crosses = in_extent[:, :, None] & (y0b != y1b) & (((y0b <= y) & (y < y1b)) | ((y1b <= y) & (y < y0b)))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = x0b + (y - y0b) * (x1b - x0b) / (y1b - y0b)
    intersections = np.sort(np.where(crosses, x, np.inf), axis=-1)
    counts = crosses.sum(axis=-1)
    pair_count = intersections.shape[-1] // 2
    if pair_count == 0:
        return
    left = intersections[..., 0 : 2 * pair_count : 2]
    right = intersections[..., 1 : 2 * pair_count : 2]
    paired = (2 * np.arange(pair_count) + 1)[None, None, :] < counts[..., None]
    with np.errstate(invalid="ignore"):
        col_start = np.maximum(0.0, np.ceil(np.minimum(left, right)))
        col_stop = np.minimum(float(width - 1), np.floor(np.maximum(left, right)))
    spans = paired & (col_stop >= col_start)
    if not bool(spans.any()):
        return
    span_rows = np.broadcast_to(rows[:, :, None], spans.shape)[spans]
    starts = col_start[spans].astype(np.int64)
    stops = col_stop[spans].astype(np.int64) + 1
    stride = width + 1
    size = height * stride
    edges = np.bincount(span_rows * stride + starts, minlength=size + 1)[:size]
    edges -= np.bincount(span_rows * stride + stops, minlength=size + 1)[:size]
    mask |= np.cumsum(edges.reshape(height, stride), axis=1)[:, :width] > 0


def _rasterize_polygons_boundary(mask: np.ndarray, polygons: np.ndarray) -> None:
    """Mark the closed outline of each polygon (``[P,V,2]``): every edge, including the last-to-first one."""
    vertices = np.asarray(polygons, dtype=np.float32)
    if vertices.shape[0] == 0 or vertices.shape[1] == 0:
        return
    if vertices.shape[1] == 1:
        _mark_points(mask, vertices[:, 0, :])
        return
    starts = vertices.reshape(-1, 2)
    ends = np.roll(vertices, -1, axis=1).reshape(-1, 2)
    _rasterize_segments(mask, starts, ends)


def _rasterize_segments(mask: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> None:
    """Mark the pixels nearest to ``2 * max(|dx|, |dy|)`` evenly spaced float32 samples along each segment."""
    starts = np.asarray(starts, dtype=np.float32)
    deltas = np.asarray(ends, dtype=np.float32) - starts
    steps = np.maximum(1, np.ceil(np.max(np.abs(deltas), axis=1).astype(np.float64) * 2.0)).astype(np.int64)
    points: list[np.ndarray] = []
    for step_count in np.unique(steps):
        selected = steps == step_count
        samples = _segment_samples(int(step_count))
        sampled = starts[selected][:, None, :] + samples[None, :, None] * deltas[selected][:, None, :]
        points.append(sampled.reshape(-1, 2))
    if points:
        _mark_points(mask, np.concatenate(points))


@lru_cache(maxsize=1024)
def _segment_samples(step_count: int) -> np.ndarray:
    samples = np.linspace(0.0, 1.0, step_count + 1, dtype=np.float32)
    samples.setflags(write=False)
    return samples


def _mark_points(mask: np.ndarray, points: np.ndarray) -> None:
    """Set the pixels nearest (round half to even) to ``x,y`` points that fall inside ``mask``."""
    rounded = np.rint(np.asarray(points, dtype=np.float32))
    cols = rounded[:, 0]
    rows = rounded[:, 1]
    inside = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    mask[rows[inside].astype(np.int64), cols[inside].astype(np.int64)] = True


def _quad_areas(quads: np.ndarray) -> np.ndarray:
    """Signed shoelace areas of ``[Q,4,2]`` quads, equal to :func:`_polygon_area` of each quad."""
    x = quads[:, :, 0].astype(np.float64)
    y = quads[:, :, 1].astype(np.float64)
    terms = x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y
    areas = 0.5 * (((terms[:, 0] + terms[:, 1]) + terms[:, 2]) + terms[:, 3])
    # The summation order of numpy's reduction is an implementation detail; near the 1e-6 threshold use the exact
    # per-quad computation so the zero-area test matches it bit for bit.
    near = np.abs(np.abs(areas) - 1e-6) <= 1e-6 + 1e-9 * np.max(np.abs(terms), axis=1, initial=0.0)
    for index in np.flatnonzero(near):
        areas[index] = _polygon_area(quads[index])
    return areas


def _finite_vertices(vertices: np.ndarray) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(values[np.isfinite(values).all(axis=1)], dtype=np.float32)


def _polygon_area(vertices: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return 0.0
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
