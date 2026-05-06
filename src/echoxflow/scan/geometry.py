"""Geometry data structures for scan conversion and slicing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SectorGeometry:
    """Polar sector geometry in meters and radians."""

    depth_start_m: float
    depth_end_m: float
    angle_start_rad: float
    angle_end_rad: float
    grid_shape: tuple[int, int] | None = None

    @classmethod
    def from_center_width(
        cls,
        *,
        depth_start_m: float,
        depth_end_m: float,
        tilt_rad: float,
        width_rad: float,
        grid_shape: tuple[int, int] | None = None,
    ) -> "SectorGeometry":
        half_width = 0.5 * float(width_rad)
        return cls(
            depth_start_m=float(depth_start_m),
            depth_end_m=float(depth_end_m),
            angle_start_rad=float(tilt_rad) - half_width,
            angle_end_rad=float(tilt_rad) + half_width,
            grid_shape=grid_shape,
        )

    @property
    def depth_span_m(self) -> float:
        return max(1e-9, float(self.depth_end_m) - float(self.depth_start_m))

    @property
    def angle_span_rad(self) -> float:
        return max(1e-9, float(self.angle_end_rad) - float(self.angle_start_rad))


@dataclass(frozen=True)
class CartesianGrid:
    """Cartesian output grid in physical coordinates."""

    shape: tuple[int, int]
    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]

    @classmethod
    def from_sector_height(cls, geometry: SectorGeometry, height: int) -> "CartesianGrid":
        x0, x1, y0, y1 = _sector_cartesian_bounds(geometry)
        x_span = max(1e-6, x1 - x0)
        y_span = max(1e-6, y1 - y0)
        out_h = max(1, int(height))
        out_w = max(1, int(round(out_h * x_span / y_span)))
        return cls(
            shape=(out_h, out_w),
            x_range_m=(x0, x1),
            y_range_m=(y0, y1),
        )

    @property
    def spacing_m(self) -> tuple[float, float]:
        height, width = self.shape
        dx = (float(self.x_range_m[1]) - float(self.x_range_m[0])) / max(1, width - 1)
        dy = (float(self.y_range_m[1]) - float(self.y_range_m[0])) / max(1, height - 1)
        return dx, dy


@dataclass(frozen=True)
class VolumeGrid:
    """Axis-aligned 3D voxel grid in physical coordinates."""

    origin_m: tuple[float, float, float]
    spacing_m: tuple[float, float, float]

    def world_to_index(self, points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64)
        origin = np.asarray(self.origin_m, dtype=np.float64)
        spacing = np.asarray(self.spacing_m, dtype=np.float64)
        spacing = np.where(np.abs(spacing) < 1e-12, 1.0, spacing)
        return (points - origin) / spacing


@dataclass(frozen=True)
class SlicePlane:
    """2D plane embedded in 3D physical space."""

    origin_m: tuple[float, float, float]
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    u_range_m: tuple[float, float]
    v_range_m: tuple[float, float]
    shape: tuple[int, int]

    @property
    def normal(self) -> np.ndarray:
        u = _unit(self.u_axis)
        v = _unit(self.v_axis)
        normal = np.cross(u, v)
        return _unit(normal)

    def points(self) -> np.ndarray:
        height, width = self.shape
        us = np.linspace(self.u_range_m[0], self.u_range_m[1], width, dtype=np.float64)
        vs = np.linspace(self.v_range_m[0], self.v_range_m[1], height, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)
        origin = np.asarray(self.origin_m, dtype=np.float64)
        u_axis = _unit(self.u_axis)
        v_axis = _unit(self.v_axis)
        points = origin[None, None, :] + uu[..., None] * u_axis + vv[..., None] * v_axis
        return np.asarray(points)


def radial_slice_plane(
    *,
    angle_deg: float,
    radius_m: float,
    depth_m: float,
    shape: tuple[int, int],
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SlicePlane:
    angle_rad = math.radians(float(angle_deg))
    u_axis = (math.cos(angle_rad), math.sin(angle_rad), 0.0)
    v_axis = (0.0, 0.0, 1.0)
    return SlicePlane(
        origin_m=origin_m,
        u_axis=u_axis,
        v_axis=v_axis,
        u_range_m=(-float(radius_m), float(radius_m)),
        v_range_m=(0.0, float(depth_m)),
        shape=shape,
    )


def axial_slice_plane(
    *,
    x_range_m: tuple[float, float],
    y_range_m: tuple[float, float],
    z_m: float,
    shape: tuple[int, int],
) -> SlicePlane:
    return SlicePlane(
        origin_m=(0.0, 0.0, float(z_m)),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        u_range_m=x_range_m,
        v_range_m=y_range_m,
        shape=shape,
    )


def physical_points_to_pixels(points_m: np.ndarray, grid: CartesianGrid) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    height, width = grid.shape
    x0, x1 = grid.x_range_m
    y0, y1 = grid.y_range_m
    px = (points[:, 0] - x0) / max(1e-12, x1 - x0) * max(0, width - 1)
    py = (points[:, 1] - y0) / max(1e-12, y1 - y0) * max(0, height - 1)
    return np.column_stack([px, py]).astype(np.float32)


def sector_geometry_from_mapping(
    value: Mapping[str, Any],
    *,
    grid_shape: tuple[int, int] | None = None,
) -> SectorGeometry:
    """Build a 2D sector geometry from EchoXFlow metadata mappings."""
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected geometry mapping, got {type(value).__name__}")
    resolved_grid_shape = grid_shape or _grid_shape_from_mapping(value)
    depth_start = _required_float(value, "depth_start_m", "DepthStart")
    depth_end = _required_float(value, "depth_end_m", "DepthEnd")
    if _has_any(value, "angle_start_rad", "angle_end_rad", "AngleStart", "AngleEnd"):
        angle_start = _required_float(value, "angle_start_rad", "AngleStart")
        angle_end = _required_float(value, "angle_end_rad", "AngleEnd")
        return SectorGeometry(
            depth_start_m=depth_start,
            depth_end_m=depth_end,
            angle_start_rad=angle_start,
            angle_end_rad=angle_end,
            grid_shape=resolved_grid_shape,
        )
    tilt = _optional_float(value, "tilt_rad", "Tilt", "Angle") or 0.0
    width = _required_float(value, "width_rad", "Width")
    return SectorGeometry.from_center_width(
        depth_start_m=depth_start,
        depth_end_m=depth_end,
        tilt_rad=tilt,
        width_rad=width,
        grid_shape=resolved_grid_shape,
    )


def _grid_shape_from_mapping(value: Mapping[str, Any]) -> tuple[int, int] | None:
    raw = _first_value(value, "grid_shape", "grid_size", "GridShape", "GridSize")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    return None


def _has_any(value: Mapping[str, Any], *keys: str) -> bool:
    return any(key in value for key in keys)


def _required_float(value: Mapping[str, Any], *keys: str) -> float:
    result = _optional_float(value, *keys)
    if result is None:
        raise KeyError(keys[0])
    return result


def _optional_float(value: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = value.get(key)
        if raw is not None and raw != "":
            return float(raw)
    return None


def _first_value(value: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _sector_cartesian_bounds(geometry: SectorGeometry) -> tuple[float, float, float, float]:
    angles = _sector_bound_angles(float(geometry.angle_start_rad), float(geometry.angle_end_rad))
    radii = (float(geometry.depth_start_m), float(geometry.depth_end_m))
    points = np.asarray(
        [(radius * math.sin(angle), radius * math.cos(angle)) for radius in radii for angle in angles],
        dtype=np.float64,
    )
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def _sector_bound_angles(angle_start_rad: float, angle_end_rad: float) -> tuple[float, ...]:
    start = float(angle_start_rad)
    end = float(angle_end_rad)
    if end < start:
        start, end = end, start

    angles = [start, end]
    period = 2.0 * math.pi
    for base in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
        first_k = math.floor((start - base) / period) - 1
        last_k = math.ceil((end - base) / period) + 1
        for k in range(first_k, last_k + 1):
            angle = base + k * period
            if start <= angle <= end:
                angles.append(angle)
    return tuple(angles)


def _unit(vector: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError(f"Cannot normalize zero-length vector: {vector!r}")
    return arr / norm
