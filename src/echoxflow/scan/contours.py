"""Contour rasterization helpers for echo annotations."""

from __future__ import annotations

from dataclasses import dataclass

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
    quad = np.empty((4, 2), dtype=np.float32)
    for row in range(point_grid.shape[0] - 1):
        for col in range(start_col, stop_col):
            quad[0] = point_grid[row, col]
            quad[1] = point_grid[row + 1, col]
            quad[2] = point_grid[row + 1, col + 1]
            quad[3] = point_grid[row, col + 1]
            if abs(_polygon_area(quad)) <= 1e-6:
                continue
            _rasterize_polygon_into(myo_mask, quad)
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
        _fill_polygon_scanlines(mask, polygon)
    _rasterize_polygon_boundary(mask, polygon)


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


def _fill_polygon_scanlines(mask: np.ndarray, polygon: np.ndarray) -> None:
    height, width = mask.shape
    y_min = max(0, int(np.floor(float(np.min(polygon[:, 1])))))
    y_max = min(height - 1, int(np.ceil(float(np.max(polygon[:, 1])))))
    if y_max < y_min:
        return
    vertex_count = int(polygon.shape[0])
    for row in range(y_min, y_max + 1):
        y = float(row) + 0.5
        intersections: list[float] = []
        for index in range(vertex_count):
            start = polygon[index]
            end = polygon[(index + 1) % vertex_count]
            y0 = float(start[1])
            y1 = float(end[1])
            if y0 == y1:
                continue
            if (y0 <= y < y1) or (y1 <= y < y0):
                x0 = float(start[0])
                x1 = float(end[0])
                intersections.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
        intersections.sort()
        for left, right in zip(intersections[0::2], intersections[1::2]):
            col_start = max(0, int(np.ceil(min(left, right) - 0.5)))
            col_stop = min(width - 1, int(np.floor(max(left, right) - 0.5)))
            if col_stop >= col_start:
                mask[row, col_start : col_stop + 1] = True


def _rasterize_polygon_boundary(mask: np.ndarray, polygon: np.ndarray) -> None:
    if polygon.shape[0] == 1:
        _mark_point(mask, polygon[0])
        return
    vertex_count = int(polygon.shape[0])
    for index in range(vertex_count):
        _rasterize_segment(mask, polygon[index], polygon[(index + 1) % vertex_count])


def _rasterize_segment(mask: np.ndarray, start: np.ndarray, end: np.ndarray) -> None:
    delta = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    steps = max(1, int(np.ceil(float(np.max(np.abs(delta))) * 2.0)))
    for t in np.linspace(0.0, 1.0, steps + 1, dtype=np.float32):
        _mark_point(mask, np.asarray(start, dtype=np.float32) + t * delta)


def _mark_point(mask: np.ndarray, point: np.ndarray) -> None:
    col = int(np.rint(float(point[0])))
    row = int(np.rint(float(point[1])))
    if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
        mask[row, col] = True


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
