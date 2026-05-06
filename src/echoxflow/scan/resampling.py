"""Sector lookup and sector-to-sector resampling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoxflow.scan.geometry import SectorGeometry
from echoxflow.scan.interpolation import Interpolation, sample_image


@dataclass(frozen=True)
class SectorLookup:
    """Source beamspace coordinates for sampling one sector on another sector grid."""

    rows: np.ndarray
    cols: np.ndarray
    mask: np.ndarray
    output_shape: tuple[int, int]


def sector_lookup(
    source_geometry: SectorGeometry,
    target_geometry: SectorGeometry,
    *,
    source_shape: tuple[int, int],
    output_shape: tuple[int, int] | None = None,
) -> SectorLookup:
    """Return source row/column coordinates for each target-sector pixel."""
    target_shape = output_shape or target_geometry.grid_shape
    if target_shape is None:
        raise ValueError("output_shape is required when target_geometry.grid_shape is not set")
    radii, angles = _beamspace_grid(target_geometry, target_shape)
    rows = (radii - source_geometry.depth_start_m) / source_geometry.depth_span_m * max(0, source_shape[0] - 1)
    cols = (angles - source_geometry.angle_start_rad) / source_geometry.angle_span_rad * max(0, source_shape[1] - 1)
    mask = (
        (radii >= source_geometry.depth_start_m)
        & (radii <= source_geometry.depth_end_m)
        & (angles >= source_geometry.angle_start_rad)
        & (angles <= source_geometry.angle_end_rad)
    )
    return SectorLookup(rows=rows, cols=cols, mask=mask, output_shape=target_shape)


def resample_sector(
    image: np.ndarray,
    source_geometry: SectorGeometry,
    target_geometry: SectorGeometry,
    *,
    output_shape: tuple[int, int] | None = None,
    interpolation: Interpolation = "linear",
) -> np.ndarray:
    """Resample one sector image onto another sector's native beamspace grid."""
    img = np.asarray(image)
    if img.ndim not in {2, 3}:
        raise ValueError(f"Expected one image [R,A] or [R,A,C], got {img.shape}")
    lookup = sector_lookup(
        source_geometry,
        target_geometry,
        source_shape=img.shape[:2],
        output_shape=output_shape,
    )
    sampled = sample_image(img, lookup.rows, lookup.cols, interpolation=interpolation)
    return _zero_outside(sampled, lookup.mask, dtype=img.dtype)


def resample_sector_stack(
    frames: np.ndarray,
    source_geometry: SectorGeometry,
    target_geometry: SectorGeometry,
    *,
    output_shape: tuple[int, int] | None = None,
    interpolation: Interpolation = "linear",
) -> np.ndarray:
    """Resample a temporal sector stack while preserving the frame axis."""
    arr = np.asarray(frames)
    if arr.ndim < 3:
        raise ValueError(f"Expected frame stack [T,R,A...], got {arr.shape}")
    return np.asarray(
        [
            resample_sector(
                frame,
                source_geometry,
                target_geometry,
                output_shape=output_shape,
                interpolation=interpolation,
            )
            for frame in arr
        ]
    )


def _beamspace_grid(geometry: SectorGeometry, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.linspace(geometry.depth_start_m, geometry.depth_end_m, int(shape[0]), dtype=np.float64)
    cols = np.linspace(geometry.angle_start_rad, geometry.angle_end_rad, int(shape[1]), dtype=np.float64)
    return np.meshgrid(rows, cols, indexing="ij")


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
