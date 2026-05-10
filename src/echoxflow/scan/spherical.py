"""Spherical 3D brightness-mode slicing utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from echoxflow.scan.interpolation import Interpolation, sample_volume


@dataclass(frozen=True)
class SphericalGeometry:
    """3D sector geometry for volumes shaped as elevation, azimuth, radius."""

    depth_start_m: float
    depth_end_m: float
    azimuth_width_rad: float
    elevation_width_rad: float


@dataclass(frozen=True)
class SphericalMosaic:
    """Twelve 3D slices arranged as a 3-row by 4-column frame stack."""

    frames: np.ndarray
    rows: int = 3
    cols: int = 4


def spherical_geometry_from_metadata(metadata: Mapping[str, object] | None) -> SphericalGeometry:
    """Parse native 3D geometry from the public Zarr manifest."""
    geometry = _spherical_sector_geometry(metadata)
    if not isinstance(geometry, Mapping):
        raise ValueError("3D brightness mode is missing sectors[].geometry")
    depth_start = _required_float(geometry, "DepthStart")
    depth_end = _required_float(geometry, "DepthEnd")
    width = _required_float(geometry, "Width")
    elevation_width = _optional_float(geometry, "ElevationWidth", default=width)
    if depth_end <= depth_start:
        raise ValueError(f"Invalid 3D depth range: {depth_start} to {depth_end}")
    return SphericalGeometry(
        depth_start_m=depth_start,
        depth_end_m=depth_end,
        azimuth_width_rad=width,
        elevation_width_rad=elevation_width,
    )


def _spherical_sector_geometry(metadata: Mapping[str, object] | None) -> Mapping[str, object] | None:
    raw = metadata or {}
    sectors = raw.get("sectors")
    if isinstance(sectors, list):
        for sector in sectors:
            if not isinstance(sector, Mapping):
                continue
            geometry = sector.get("geometry")
            if not isinstance(geometry, Mapping):
                continue
            if geometry.get("coordinate_system") == "spherical_sector_3d":
                return cast(Mapping[str, object], geometry)
            frames = sector.get("frames")
            if _array_ref_path(frames) == "data/3d_brightness_mode":
                return cast(Mapping[str, object], geometry)
    render_metadata = raw.get("render_metadata")
    if isinstance(render_metadata, Mapping):
        return cast(Mapping[str, object], render_metadata)
    return None


def _array_ref_path(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    path = value.get("array_path") or value.get("zarr_path") or value.get("path")
    return str(path).strip("/") if path else None


def beamspace_spherical_mosaic(
    volumes: np.ndarray,
    *,
    output_size: tuple[int, int] = (120, 120),
) -> SphericalMosaic:
    """Build a 12-panel mosaic from native beamspace slices."""
    arr = _volume_stack(volumes)
    panels: list[np.ndarray] = []
    for row, az_fraction in enumerate((0.75, 0.5, 0.25)):
        panels.append(_resize_stack(_native_radial_axis_slice(arr, az_fraction=az_fraction), output_size))
        for depth_fraction in _depth_fractions_for_row(row):
            panels.append(_resize_stack(_native_radial_normal_slice(arr, depth_fraction=depth_fraction), output_size))
    return SphericalMosaic(frames=_mosaic_frames(tuple(panels), rows=3, cols=4, fill_value=0.0))


def cartesian_spherical_mosaic(
    volumes: np.ndarray,
    geometry: SphericalGeometry,
    *,
    output_size: tuple[int, int] = (120, 120),
    interpolation: Interpolation = "linear",
    cover_depth_fraction: float = 1.0,
    depth_slice_cover_depth_fraction: float | None = None,
    depth_slice_lateral_scale: float = 1.0,
    radial_axis_output_size: tuple[int, int] | None = None,
    radial_axis_depth_start_m: float | None = None,
) -> SphericalMosaic:
    """Build a 12-panel mosaic from Cartesian slices through a spherical volume."""
    arr = _volume_stack(volumes)
    panels: list[np.ndarray] = []
    radial_axis_angles = (0.0, -60.0, -120.0)
    radial_depth_start = _radial_axis_depth_start(geometry, radial_axis_depth_start_m)
    resolved_depth_slice_cover = (
        cover_depth_fraction if depth_slice_cover_depth_fraction is None else float(depth_slice_cover_depth_fraction)
    )
    radial_axis_half_width = max(
        _radial_axis_half_width_m(geometry, cover_depth_fraction, angle_deg) for angle_deg in radial_axis_angles
    )
    radial_axis_size = _radial_axis_output_size(
        radial_axis_output_size or output_size,
        geometry,
        cover_depth_fraction,
        radial_axis_angles,
        depth_start_m=radial_depth_start,
    )
    for row, angle_deg in enumerate(radial_axis_angles):
        panels.append(
            spherical_radial_axis_stack(
                arr,
                geometry,
                angle_deg=angle_deg,
                output_size=radial_axis_size,
                interpolation=interpolation,
                cover_depth_fraction=cover_depth_fraction,
                half_width_m=radial_axis_half_width,
                depth_start_m=radial_depth_start,
            )
        )
        for depth_fraction in _depth_fractions_for_row(row):
            panels.append(
                spherical_depth_slice_stack(
                    arr,
                    geometry,
                    depth_fraction=depth_fraction,
                    output_size=output_size,
                    interpolation=interpolation,
                    cover_depth_fraction=resolved_depth_slice_cover,
                    lateral_scale=depth_slice_lateral_scale,
                )
            )
    return SphericalMosaic(frames=_mosaic_frames(tuple(panels), rows=3, cols=4, fill_value=np.nan))


def spherical_depth_slice_stack(
    volumes: np.ndarray,
    geometry: SphericalGeometry,
    *,
    depth_fraction: float,
    output_size: tuple[int, int],
    interpolation: Interpolation = "linear",
    cover_depth_fraction: float = 1.0,
    lateral_scale: float = 1.0,
) -> np.ndarray:
    arr = _volume_stack(volumes)
    height, width = _shape2(output_size)
    depth_m = geometry.depth_start_m + float(np.clip(depth_fraction, 0.0, 1.0)) * _depth_span(geometry)
    cover_depth_m = geometry.depth_start_m + float(np.clip(cover_depth_fraction, 0.0, 1.0)) * _depth_span(geometry)
    half_x = cover_depth_m * math.tan(0.5 * geometry.azimuth_width_rad) * max(float(lateral_scale), 1e-6)
    half_y = cover_depth_m * math.tan(0.5 * geometry.elevation_width_rad)
    x = np.linspace(-half_x, half_x, width, dtype=np.float32)
    y = np.linspace(-half_y, half_y, height, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    zz = np.full_like(xx, depth_m)
    coords = _cartesian_to_index(xx, yy, zz, geometry, arr.shape[1:4])
    return _sample_stack(arr, coords, interpolation=interpolation)


def spherical_radial_axis_stack(
    volumes: np.ndarray,
    geometry: SphericalGeometry,
    *,
    angle_deg: float,
    output_size: tuple[int, int],
    interpolation: Interpolation = "linear",
    cover_depth_fraction: float = 1.0,
    half_width_m: float | None = None,
    depth_start_m: float | None = None,
) -> np.ndarray:
    arr = _volume_stack(volumes)
    height, width = _shape2(output_size)
    depth_start = _radial_axis_depth_start(geometry, depth_start_m)
    half_width = (
        float(half_width_m)
        if half_width_m is not None
        else _radial_axis_half_width_m(geometry, cover_depth_fraction, angle_deg)
    )
    depth = np.linspace(depth_start, geometry.depth_end_m, height, dtype=np.float32)
    lateral = np.linspace(-half_width, half_width, width, dtype=np.float32)
    zz, lateral_grid = np.meshgrid(depth, lateral, indexing="ij")
    angle_rad = math.radians(float(angle_deg))
    xx = lateral_grid * math.cos(angle_rad)
    yy = lateral_grid * math.sin(angle_rad)
    coords = _cartesian_to_index(xx, yy, zz, geometry, arr.shape[1:4])
    return _sample_stack(arr, coords, interpolation=interpolation)


def _volume_stack(volumes: np.ndarray) -> np.ndarray:
    arr = np.asarray(volumes)
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D brightness mode with shape [T,E,A,R], got {arr.shape}")
    return arr


def _native_radial_axis_slice(volumes: np.ndarray, *, az_fraction: float) -> np.ndarray:
    az_index = int(round(float(np.clip(az_fraction, 0.0, 1.0)) * max(0, volumes.shape[2] - 1)))
    return np.asarray(np.swapaxes(volumes[:, :, az_index, :], 1, 2))


def _native_radial_normal_slice(volumes: np.ndarray, *, depth_fraction: float) -> np.ndarray:
    radial_index = int(round(float(np.clip(depth_fraction, 0.0, 1.0)) * max(0, volumes.shape[3] - 1)))
    return np.asarray(volumes[:, :, :, radial_index])


def _sample_stack(volumes: np.ndarray, coords: np.ndarray, *, interpolation: Interpolation) -> np.ndarray:
    mask = _coordinate_mask(coords, volumes.shape[1:4])
    frames = []
    for frame in volumes:
        sampled = sample_volume(frame, coords, interpolation=interpolation).astype(np.float32, copy=False)
        if not np.all(mask):
            sampled = sampled.copy()
            sampled[~mask] = np.nan
        frames.append(sampled)
    return np.asarray(frames)


def _cartesian_to_index(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    geometry: SphericalGeometry,
    shape_ear: tuple[int, int, int],
) -> np.ndarray:
    elevation_count, azimuth_count, radial_count = shape_ear
    depth = np.sqrt(x * x + y * y + z * z)
    azimuth = np.arctan2(x, np.sqrt(y * y + z * z))
    elevation = np.arctan2(y, z)
    radial_index = (depth - geometry.depth_start_m) / _depth_span(geometry) * max(0, radial_count - 1)
    azimuth_index = (
        (azimuth + 0.5 * geometry.azimuth_width_rad)
        / max(geometry.azimuth_width_rad, 1e-12)
        * max(0, azimuth_count - 1)
    )
    elevation_index = (
        (elevation + 0.5 * geometry.elevation_width_rad)
        / max(geometry.elevation_width_rad, 1e-12)
        * max(0, elevation_count - 1)
    )
    return np.stack([elevation_index, azimuth_index, radial_index], axis=-1)


def _coordinate_mask(coords: np.ndarray, shape_ear: tuple[int, int, int]) -> np.ndarray:
    return (
        (coords[..., 0] >= 0.0)
        & (coords[..., 0] <= shape_ear[0] - 1)
        & (coords[..., 1] >= 0.0)
        & (coords[..., 1] <= shape_ear[1] - 1)
        & (coords[..., 2] >= 0.0)
        & (coords[..., 2] <= shape_ear[2] - 1)
    )


def _mosaic_frames(panels: tuple[np.ndarray, ...], *, rows: int, cols: int, fill_value: float) -> np.ndarray:
    if len(panels) != rows * cols:
        raise ValueError(f"Expected {rows * cols} panels, got {len(panels)}")
    frame_count = int(panels[0].shape[0])
    height = max(int(panel.shape[1]) for panel in panels)
    width = max(int(panel.shape[2]) for panel in panels)
    out = np.full((frame_count, rows * height, cols * width), fill_value, dtype=np.float32)
    for index, panel in enumerate(panels):
        row = index // cols
        col = index % cols
        panel_arr = np.asarray(panel, dtype=np.float32)
        out[:, row * height : row * height + panel_arr.shape[1], col * width : col * width + panel_arr.shape[2]] = (
            panel_arr
        )
    return out


def _resize_stack(frames: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(frames)
    height, width = _shape2(output_size)
    if arr.shape[1:3] == (height, width):
        return arr.astype(np.float32, copy=False)
    row_idx = _resample_indices(arr.shape[1], height)
    col_idx = _resample_indices(arr.shape[2], width)
    return arr[:, row_idx][:, :, col_idx].astype(np.float32, copy=False)


def _resample_indices(source_size: int, target_size: int) -> np.ndarray:
    if target_size <= 1:
        return np.zeros((1,), dtype=np.int64)
    return np.clip(np.rint(np.linspace(0, max(0, source_size - 1), target_size)), 0, max(0, source_size - 1)).astype(
        np.int64
    )


def _depth_fractions_for_row(row: int) -> tuple[float, float, float]:
    fractions = np.linspace(4.0 / 15.0, 10.0 / 15.0, 9, dtype=np.float32)
    start = int(row) * 3
    selected = fractions[start : start + 3]
    return float(selected[0]), float(selected[1]), float(selected[2])


def _radial_axis_half_width_m(geometry: SphericalGeometry, cover_depth_fraction: float, angle_deg: float) -> float:
    cover_depth_m = geometry.depth_start_m + float(np.clip(cover_depth_fraction, 0.0, 1.0)) * _depth_span(geometry)
    az_half = cover_depth_m * math.tan(0.5 * geometry.azimuth_width_rad)
    el_half = cover_depth_m * math.tan(0.5 * geometry.elevation_width_rad)
    angle_rad = math.radians(float(angle_deg))
    return az_half * abs(math.cos(angle_rad)) + el_half * abs(math.sin(angle_rad))


def _radial_axis_output_size(
    depth_slice_size: tuple[int, int],
    geometry: SphericalGeometry,
    cover_depth_fraction: float,
    angles_deg: tuple[float, ...],
    *,
    depth_start_m: float | None = None,
) -> tuple[int, int]:
    height, _ = _shape2(depth_slice_size)
    lateral_half_width = max(
        _radial_axis_half_width_m(geometry, cover_depth_fraction, angle_deg) for angle_deg in angles_deg
    )
    physical_width = 2.0 * lateral_half_width
    physical_height = max(1e-9, float(geometry.depth_end_m) - _radial_axis_depth_start(geometry, depth_start_m))
    width = int(round(height * physical_width / physical_height))
    return height, max(2, width)


def _radial_axis_depth_start(geometry: SphericalGeometry, depth_start_m: float | None) -> float:
    if depth_start_m is None:
        return float(geometry.depth_start_m)
    return float(np.clip(float(depth_start_m), float(geometry.depth_start_m), float(geometry.depth_end_m)))


def _shape2(shape: tuple[int, int]) -> tuple[int, int]:
    return max(2, int(shape[0])), max(2, int(shape[1]))


def _depth_span(geometry: SphericalGeometry) -> float:
    return max(1e-9, float(geometry.depth_end_m) - float(geometry.depth_start_m))


def _required_float(metadata: Mapping[str, object], key: str) -> float:
    value = _optional_float(metadata, key, default=float("nan"))
    if not np.isfinite(value):
        raise ValueError(f"Missing or invalid 3D metadata value `{key}`")
    return value


def _optional_float(metadata: Mapping[str, object], key: str, *, default: float) -> float:
    value = metadata.get(key, default)
    if isinstance(value, list | tuple):
        value = value[0] if value else default
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default
