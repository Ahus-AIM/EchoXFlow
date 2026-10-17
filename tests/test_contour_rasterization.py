"""Vectorized contour rasterization is bit-exact against the original per-pixel Python implementation.

The reference below is a copy of the scanline fill, boundary tracing and myocardium quad loop that
``echoxflow.scan.contours`` used before it was vectorized (renamed, and with the fill sampling pixel centres at integer
coordinates like the outline), so the masks, including the shared endocardium/myocardium boundary and the closing edge
across the mitral valve, must be identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from echoxflow.scan.contours import (
    AFIRV_GROUP_LAYOUT,
    LV_GROUP_LAYOUT,
    ContourGroupLayout,
    build_contour_masks,
    rasterize_polygon_pixels,
)


@dataclass(frozen=True)
class _ReferenceResult:
    group_layout: ContourGroupLayout
    point_grid: np.ndarray
    endo_mask: np.ndarray
    myo_mask: np.ndarray


# ---- reference implementation (original code, renamed) -------------------------------------------------------------
def _reference_build_contour_masks(
    points_px: np.ndarray,
    *,
    image_shape: tuple[int, int],
    group_layout: ContourGroupLayout,
) -> _ReferenceResult:
    height, width = int(image_shape[0]), int(image_shape[1])
    empty = np.zeros((height, width), dtype=bool)
    point_grid = _reference_reshape_valid_point_grid(points_px, group_layout=group_layout)
    if point_grid.shape[0] < 2:
        return _ReferenceResult(
            group_layout=group_layout,
            point_grid=point_grid,
            endo_mask=empty.copy(),
            myo_mask=empty.copy(),
        )

    endo_mask = np.zeros((height, width), dtype=bool)
    _reference_rasterize_polygon_into(endo_mask, point_grid[:, group_layout.endo_index, :])
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
            if abs(_reference_polygon_area(quad)) <= 1e-6:
                continue
            _reference_rasterize_polygon_into(myo_mask, quad)
    myo_mask &= ~endo_mask

    return _ReferenceResult(
        group_layout=group_layout,
        point_grid=point_grid,
        endo_mask=endo_mask,
        myo_mask=myo_mask,
    )


def _reference_rasterize_polygon_pixels(vertices_px: np.ndarray, *, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=bool)
    _reference_rasterize_polygon_into(mask, vertices_px)
    return mask


def _reference_rasterize_polygon_into(mask: np.ndarray, vertices: np.ndarray) -> None:
    polygon = _reference_finite_vertices(vertices)
    if polygon.shape[0] == 0:
        return
    if polygon.shape[0] >= 3 and abs(_reference_polygon_area(polygon)) > 1e-6:
        _reference_fill_polygon_scanlines(mask, polygon)
    _reference_rasterize_polygon_boundary(mask, polygon)


def _reference_reshape_valid_point_grid(points: np.ndarray, *, group_layout: ContourGroupLayout) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        return np.zeros((0, group_layout.group_size, 2), dtype=np.float32)
    row_count = values.shape[0] // group_layout.group_size
    if row_count <= 0 or row_count * group_layout.group_size != values.shape[0]:
        return np.zeros((0, group_layout.group_size, 2), dtype=np.float32)
    grid = values.reshape(row_count, group_layout.group_size, 2)
    finite_rows = np.isfinite(grid).all(axis=(1, 2))
    return np.asarray(grid[finite_rows], dtype=np.float32)


def _reference_fill_polygon_scanlines(mask: np.ndarray, polygon: np.ndarray) -> None:
    height, width = mask.shape
    y_min = max(0, int(np.floor(float(np.min(polygon[:, 1])))))
    y_max = min(height - 1, int(np.ceil(float(np.max(polygon[:, 1])))))
    if y_max < y_min:
        return
    vertex_count = int(polygon.shape[0])
    for row in range(y_min, y_max + 1):
        y = float(row)
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
            col_start = max(0, int(np.ceil(min(left, right))))
            col_stop = min(width - 1, int(np.floor(max(left, right))))
            if col_stop >= col_start:
                mask[row, col_start : col_stop + 1] = True


def _reference_rasterize_polygon_boundary(mask: np.ndarray, polygon: np.ndarray) -> None:
    if polygon.shape[0] == 1:
        _reference_mark_point(mask, polygon[0])
        return
    vertex_count = int(polygon.shape[0])
    for index in range(vertex_count):
        _reference_rasterize_segment(mask, polygon[index], polygon[(index + 1) % vertex_count])


def _reference_rasterize_segment(mask: np.ndarray, start: np.ndarray, end: np.ndarray) -> None:
    delta = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    steps = max(1, int(np.ceil(float(np.max(np.abs(delta))) * 2.0)))
    for t in np.linspace(0.0, 1.0, steps + 1, dtype=np.float32):
        _reference_mark_point(mask, np.asarray(start, dtype=np.float32) + t * delta)


def _reference_mark_point(mask: np.ndarray, point: np.ndarray) -> None:
    col = int(np.rint(float(point[0])))
    row = int(np.rint(float(point[1])))
    if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
        mask[row, col] = True


def _reference_finite_vertices(vertices: np.ndarray) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(values[np.isfinite(values).all(axis=1)], dtype=np.float32)


def _reference_polygon_area(vertices: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return 0.0
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ---- tests -----------------------------------------------------------------------------------------------------------


def _lv_like_contour(
    rng: np.random.Generator, *, rows: int, layout: ContourGroupLayout, shape: tuple[int, int]
) -> np.ndarray:
    """A horseshoe of ``rows`` contour rows (base to apex to base) with ``group_size`` points from endo to epi."""
    height, width = shape
    centre = np.array([width * rng.uniform(0.3, 0.7), height * rng.uniform(0.3, 0.7)])
    radius = rng.uniform(0.1, 0.35) * min(height, width)
    thickness = rng.uniform(0.02, 0.15) * min(height, width)
    angles = np.linspace(rng.uniform(-0.4, 0.2), np.pi + rng.uniform(-0.2, 0.4), rows)
    stretch = rng.uniform(0.8, 1.8)
    grid = np.empty((rows, layout.group_size, 2), dtype=np.float64)
    for column in range(layout.group_size):
        r = radius + thickness * column / max(1, layout.group_size - 1)
        grid[:, column, 0] = centre[0] + r * np.cos(angles)
        grid[:, column, 1] = centre[1] + stretch * r * np.sin(angles)
    grid += rng.normal(scale=rng.uniform(0.0, 1.5), size=grid.shape)
    return grid.reshape(-1, 2).astype(np.float32)


def _assert_same(points: np.ndarray, shape: tuple[int, int], layout: ContourGroupLayout) -> None:
    expected = _reference_build_contour_masks(points, image_shape=shape, group_layout=layout)
    actual = build_contour_masks(points, image_shape=shape, group_layout=layout)
    np.testing.assert_array_equal(actual.point_grid, expected.point_grid)
    np.testing.assert_array_equal(actual.endo_mask, expected.endo_mask)
    np.testing.assert_array_equal(actual.myo_mask, expected.myo_mask)
    assert actual.endo_mask.dtype == expected.endo_mask.dtype == bool


@pytest.mark.parametrize("layout", [LV_GROUP_LAYOUT, AFIRV_GROUP_LAYOUT])
@pytest.mark.parametrize("shape", [(64, 64), (128, 96), (256, 256), (300, 211)])
def test_contour_masks_match_reference_on_lv_like_contours(layout: ContourGroupLayout, shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(1234 + shape[0] + layout.group_size)
    for _ in range(25):
        points = _lv_like_contour(rng, rows=int(rng.integers(2, 60)), layout=layout, shape=shape)
        _assert_same(points, shape, layout)


def test_contour_masks_match_reference_on_degenerate_and_boundary_cases() -> None:
    rng = np.random.default_rng(7)
    shape = (40, 50)
    cases = []
    # Integer and half-integer vertices put crossings and samples exactly on pixel centres and rounding ties.
    for _ in range(40):
        rows = int(rng.integers(2, 12))
        grid = rng.integers(-5, 55, size=(rows * 5, 2)).astype(np.float32)
        cases.append(grid)
        cases.append(grid + np.float32(0.5))
    # Coincident contour rows (zero-area quads), collinear points and horizontal edges.
    flat = np.repeat(np.array([[10.0, 10.0], [12.0, 10.0], [14.0, 10.0], [16.0, 10.0], [18.0, 10.0]]), 1, axis=0)
    cases.append(np.concatenate([flat, flat, flat + [0.0, 5.0]]).astype(np.float32))
    cases.append(np.concatenate([flat, flat]).astype(np.float32))
    # Partly outside the image, tiny quads near the zero-area threshold, NaN rows, a wrong point count.
    big = _lv_like_contour(rng, rows=20, layout=LV_GROUP_LAYOUT, shape=(40, 50)) * np.float32(1.8) - np.float32(20)
    cases.append(big)
    tiny = np.zeros((10, 2), dtype=np.float32)
    tiny[:, 0] = np.float32(5.0) + np.arange(10, dtype=np.float32) * np.float32(1e-3)
    tiny[:, 1] = np.float32(5.0) + np.tile(np.arange(5, dtype=np.float32), 2) * np.float32(1e-3)
    cases.append(tiny)
    with_nan = _lv_like_contour(rng, rows=10, layout=LV_GROUP_LAYOUT, shape=shape)
    with_nan[7] = np.nan
    cases.append(with_nan)
    cases.append(np.zeros((7, 2), dtype=np.float32))
    cases.append(np.zeros((0, 2), dtype=np.float32))
    for points in cases:
        for layout in (LV_GROUP_LAYOUT, AFIRV_GROUP_LAYOUT):
            if points.shape[0] % layout.group_size == 0:
                _assert_same(points, shape, layout)


def test_quad_area_threshold_matches_reference_near_zero() -> None:
    rng = np.random.default_rng(99)
    shape = (32, 32)
    for _ in range(300):
        rows = int(rng.integers(2, 6))
        base = rng.uniform(5.0, 25.0, size=(rows, 1, 2))
        offsets = rng.normal(scale=10.0 ** rng.uniform(-5.0, -2.5), size=(rows, LV_GROUP_LAYOUT.group_size, 2))
        points = (base + offsets).reshape(-1, 2).astype(np.float32)
        _assert_same(points, shape, LV_GROUP_LAYOUT)


def test_polygon_pixels_match_reference() -> None:
    rng = np.random.default_rng(5)
    for shape in ((16, 16), (64, 80), (257, 129)):
        for vertex_count in (1, 2, 3, 4, 7, 25, 120):
            for _ in range(15):
                vertices = rng.uniform(-10.0, max(shape) + 10.0, size=(vertex_count, 2)).astype(np.float32)
                if rng.random() < 0.3:
                    vertices = np.round(vertices * 2.0).astype(np.float32) / np.float32(2.0)
                expected = _reference_rasterize_polygon_pixels(vertices, image_shape=shape)
                actual = rasterize_polygon_pixels(vertices, image_shape=shape)
                np.testing.assert_array_equal(actual, expected)
    nan_polygon = np.array([[1.0, 1.0], [np.nan, 3.0], [8.0, 2.0], [5.0, 9.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        rasterize_polygon_pixels(nan_polygon, image_shape=(12, 12)),
        _reference_rasterize_polygon_pixels(nan_polygon, image_shape=(12, 12)),
    )


def _enclosed_unlabelled(labelled: np.ndarray) -> int:
    """Pixels outside ``labelled`` that cannot reach the image border (4-connected): holes and seams."""
    outside = ~labelled
    reached = np.zeros_like(outside)
    reached[0, :], reached[-1, :], reached[:, 0], reached[:, -1] = (
        outside[0, :],
        outside[-1, :],
        outside[:, 0],
        outside[:, -1],
    )
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= outside
        if np.array_equal(grown, reached):
            return int((outside & ~reached).sum())
        reached = grown


def test_fill_samples_pixel_centres_at_integer_coordinates() -> None:
    square = np.array([[2.0, 2.0], [5.0, 2.0], [5.0, 5.0], [2.0, 5.0]], dtype=np.float32)
    mask = rasterize_polygon_pixels(square, image_shape=(8, 8))
    expected = np.zeros((8, 8), dtype=bool)
    expected[2:6, 2:6] = True  # centres on the edges belong to the polygon, as the outline marks them
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize("layout", [LV_GROUP_LAYOUT, AFIRV_GROUP_LAYOUT])
def test_cavity_and_myocardium_leave_no_unlabelled_pockets(layout: ContourGroupLayout) -> None:
    rng = np.random.default_rng(11 + layout.group_size)
    for shape in ((128, 128), (256, 256)):
        for _ in range(20):
            points = _lv_like_contour(rng, rows=int(rng.integers(20, 50)), layout=layout, shape=shape)
            result = build_contour_masks(points, image_shape=shape, group_layout=layout)
            assert not (result.endo_mask & result.myo_mask).any()
            assert _enclosed_unlabelled(result.endo_mask | result.myo_mask) == 0
