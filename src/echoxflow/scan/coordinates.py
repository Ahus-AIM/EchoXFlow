"""Coordinate conversion helpers for 2D echo scan grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoxflow.scan.geometry import CartesianGrid, SectorGeometry


@dataclass(frozen=True)
class BeamspacePixelGrid:
    """Pixel grid for a 2D polar sector image.

    Pixel coordinates use image convention: x is column and y is row.
    Physical points use EchoXFlow's 2D Cartesian convention: x is lateral
    position in meters and y is depth in meters.
    """

    geometry: SectorGeometry
    shape: tuple[int, int]

    def physical_to_pixel_xy(self, points_m: np.ndarray) -> np.ndarray:
        row_col = self.physical_to_row_col(points_m)
        return np.column_stack([row_col[:, 1], row_col[:, 0]]).astype(np.float32)

    def pixel_xy_to_physical(self, points_px: np.ndarray) -> np.ndarray:
        points = _points2(points_px, label="points_px")
        row_col = np.column_stack([points[:, 1], points[:, 0]])
        return self.row_col_to_physical(row_col)

    def physical_to_row_col(self, points_m: np.ndarray) -> np.ndarray:
        points = _points2(points_m, label="points_m")
        xy = points[:, :2]
        radii = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
        angles = np.arctan2(xy[:, 0], xy[:, 1])
        rows = (radii - self.geometry.depth_start_m) / self.geometry.depth_span_m * max(0, self.shape[0] - 1)
        cols = (angles - self.geometry.angle_start_rad) / self.geometry.angle_span_rad * max(0, self.shape[1] - 1)
        return np.column_stack([rows, cols]).astype(np.float32)

    def row_col_to_physical(self, row_col: np.ndarray) -> np.ndarray:
        points = _points2(row_col, label="row_col")
        rows = points[:, 0]
        cols = points[:, 1]
        depth = self.geometry.depth_start_m + rows / max(1, self.shape[0] - 1) * self.geometry.depth_span_m
        angle = self.geometry.angle_start_rad + cols / max(1, self.shape[1] - 1) * self.geometry.angle_span_rad
        return np.column_stack([depth * np.sin(angle), depth * np.cos(angle)]).astype(np.float32)


@dataclass(frozen=True)
class CartesianPixelGrid:
    """Pixel grid for a 2D physical Cartesian image."""

    grid: CartesianGrid
    shape: tuple[int, int] | None = None

    @property
    def resolved_shape(self) -> tuple[int, int]:
        return self.grid.shape if self.shape is None else (int(self.shape[0]), int(self.shape[1]))

    def physical_to_pixel_xy(self, points_m: np.ndarray) -> np.ndarray:
        points = _points2(points_m, label="points_m")
        height, width = self.resolved_shape
        x0, x1 = self.grid.x_range_m
        y0, y1 = self.grid.y_range_m
        px = (points[:, 0] - x0) / max(1e-12, x1 - x0) * max(0, width - 1)
        py = (points[:, 1] - y0) / max(1e-12, y1 - y0) * max(0, height - 1)
        return np.column_stack([px, py]).astype(np.float32)

    def pixel_xy_to_physical(self, points_px: np.ndarray) -> np.ndarray:
        points = _points2(points_px, label="points_px")
        height, width = self.resolved_shape
        x0, x1 = self.grid.x_range_m
        y0, y1 = self.grid.y_range_m
        x = x0 + points[:, 0] / max(1, width - 1) * (x1 - x0)
        y = y0 + points[:, 1] / max(1, height - 1) * (y1 - y0)
        return np.column_stack([x, y]).astype(np.float32)

    def physical_to_row_col(self, points_m: np.ndarray) -> np.ndarray:
        xy = self.physical_to_pixel_xy(points_m)
        return np.column_stack([xy[:, 1], xy[:, 0]]).astype(np.float32)

    def row_col_to_physical(self, row_col: np.ndarray) -> np.ndarray:
        points = _points2(row_col, label="row_col")
        return self.pixel_xy_to_physical(np.column_stack([points[:, 1], points[:, 0]]))


def resize_pixel_xy(
    points_px: np.ndarray,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Scale image xy pixel coordinates from one image shape to another."""
    points = _points2(points_px, label="points_px")
    source_h, source_w = int(source_shape[0]), int(source_shape[1])
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    out = points.astype(np.float32, copy=True)
    out[:, 0] *= max(1, target_w - 1) / max(1, source_w - 1)
    out[:, 1] *= max(1, target_h - 1) / max(1, source_h - 1)
    return out


def _points2(value: np.ndarray, *, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected {label} shaped [N,2+], got {points.shape}")
    return points[:, :2]
