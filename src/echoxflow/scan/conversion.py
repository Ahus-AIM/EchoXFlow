"""Reusable scan conversion and slicing operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoxflow.scan.geometry import CartesianGrid, SectorGeometry, SlicePlane, VolumeGrid
from echoxflow.scan.interpolation import Interpolation, sample_image, sample_volume


@dataclass(frozen=True)
class CartesianImage:
    data: np.ndarray
    mask: np.ndarray
    grid: CartesianGrid


@dataclass(frozen=True)
class VolumeSlice:
    data: np.ndarray
    mask: np.ndarray
    plane: SlicePlane


def sector_to_cartesian(
    image: np.ndarray,
    geometry: SectorGeometry,
    *,
    grid: CartesianGrid | None = None,
    output_height: int | None = None,
    interpolation: Interpolation = "linear",
) -> CartesianImage:
    """Convert one beamspace sector image to a Cartesian image."""
    img = np.asarray(image)
    if img.ndim not in {2, 3}:
        raise ValueError(f"Expected one image [R,A] or [R,A,C], got {img.shape}")
    target_grid = grid or CartesianGrid.from_sector_height(geometry, output_height or int(img.shape[0]))
    rows, cols, mask = _sector_sample_coordinates(geometry, target_grid, source_shape=img.shape[:2])
    sampled = sample_image(img, rows, cols, interpolation=interpolation)
    return CartesianImage(data=_zero_outside(sampled, mask, dtype=img.dtype), mask=mask, grid=target_grid)


def sector_stack_to_cartesian(
    frames: np.ndarray,
    geometry: SectorGeometry,
    *,
    grid: CartesianGrid | None = None,
    output_height: int | None = None,
    interpolation: Interpolation = "linear",
) -> tuple[CartesianImage, ...]:
    arr = np.asarray(frames)
    if arr.ndim < 3:
        raise ValueError(f"Expected frame stack [T,R,A...], got {arr.shape}")
    return tuple(
        sector_to_cartesian(frame, geometry, grid=grid, output_height=output_height, interpolation=interpolation)
        for frame in arr
    )


def slice_volume(
    volume: np.ndarray,
    volume_grid: VolumeGrid,
    plane: SlicePlane,
    *,
    interpolation: Interpolation = "linear",
) -> VolumeSlice:
    """Sample a 3D volume on an arbitrary physical plane."""
    vol = np.asarray(volume)
    points_xyz = plane.points()
    xyz_index = volume_grid.world_to_index(points_xyz)
    zyx_index = xyz_index[..., [2, 1, 0]]
    mask = _volume_mask(zyx_index, vol.shape[:3])
    sampled = sample_volume(vol, zyx_index, interpolation=interpolation)
    return VolumeSlice(data=_zero_outside(sampled, mask, dtype=vol.dtype), mask=mask, plane=plane)


def _sector_sample_coordinates(
    geometry: SectorGeometry, grid: CartesianGrid, *, source_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_h, out_w = grid.shape
    xs = np.linspace(grid.x_range_m[0], grid.x_range_m[1], out_w, dtype=np.float64)
    ys = np.linspace(grid.y_range_m[0], grid.y_range_m[1], out_h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    radii = np.sqrt(xx**2 + yy**2)
    angles = np.arctan2(xx, yy)
    mask = (
        (radii >= geometry.depth_start_m)
        & (radii <= geometry.depth_end_m)
        & (angles >= geometry.angle_start_rad)
        & (angles <= geometry.angle_end_rad)
    )
    rows = (radii - geometry.depth_start_m) / geometry.depth_span_m * max(0, source_shape[0] - 1)
    cols = (angles - geometry.angle_start_rad) / geometry.angle_span_rad * max(0, source_shape[1] - 1)
    return rows, cols, mask


def _volume_mask(zyx_index: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return (
        (zyx_index[..., 0] >= 0.0)
        & (zyx_index[..., 0] <= shape[0] - 1)
        & (zyx_index[..., 1] >= 0.0)
        & (zyx_index[..., 1] <= shape[1] - 1)
        & (zyx_index[..., 2] >= 0.0)
        & (zyx_index[..., 2] <= shape[2] - 1)
    )


def _zero_outside(values: np.ndarray, mask: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    out = np.asarray(values)
    if not np.all(mask):
        out = out.copy()
        out[~mask] = 0
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(dtype)
    else:
        out = out.astype(dtype, copy=False)
    return out
