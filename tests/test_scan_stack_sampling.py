"""Vectorized frame-stack resampling equals resampling each frame on its own, bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from echoxflow.scan import (
    CartesianGrid,
    SectorGeometry,
    resample_sector,
    resample_sector_stack,
    sector_stack_to_cartesian,
    sector_stack_to_cartesian_array,
    sector_to_cartesian,
)
from echoxflow.scan.interpolation import sample_image, sample_image_stack


def _geometry(depth: tuple[float, float], angles: tuple[float, float], shape: tuple[int, int]) -> SectorGeometry:
    return SectorGeometry(
        depth_start_m=depth[0],
        depth_end_m=depth[1],
        angle_start_rad=angles[0],
        angle_end_rad=angles[1],
        grid_shape=shape,
    )


def _frames(rng: np.random.Generator, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 256, size=shape).astype(dtype)
    values = rng.normal(size=shape).astype(dtype)
    values[rng.random(shape) < 0.05] = np.nan  # Color Doppler velocity has NaN where there is no flow estimate
    return values


def _same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8])
@pytest.mark.parametrize("channels", [None, 3])
def test_resample_sector_stack_matches_per_frame_resampling(
    interpolation: str, dtype: type, channels: int | None
) -> None:
    rng = np.random.default_rng(0)
    source = _geometry((0.002, 0.12), (-0.5, 0.45), (40, 33))
    target = _geometry((0.0, 0.15), (-0.7, 0.7), (48, 50))
    shape = (7, 40, 33) if channels is None else (7, 40, 33, channels)
    frames = _frames(rng, shape, np.dtype(dtype))

    expected = np.asarray(
        [
            resample_sector(frame, source, target, output_shape=(48, 50), interpolation=interpolation)  # type: ignore[arg-type]
            for frame in frames
        ]
    )
    actual = resample_sector_stack(frames, source, target, output_shape=(48, 50), interpolation=interpolation)  # type: ignore[arg-type]

    _same_bits(actual, expected)
    assert actual.flags.c_contiguous


def test_resample_sector_stack_keeps_empty_stack_behaviour() -> None:
    source = _geometry((0.0, 0.1), (-0.5, 0.5), (8, 8))
    empty = np.zeros((0, 8, 8), dtype=np.float32)
    _same_bits(
        resample_sector_stack(empty, source, source, output_shape=(4, 4)),
        np.asarray([resample_sector(frame, source, source, output_shape=(4, 4)) for frame in empty]),
    )


@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
def test_sector_stack_to_cartesian_array_matches_per_frame_conversion(interpolation: str, dtype: type) -> None:
    rng = np.random.default_rng(1)
    geometry = _geometry((0.001, 0.14), (-0.6, 0.6), (64, 48))
    grid = CartesianGrid.from_sector_height(geometry, 72)
    frames = _frames(rng, (5, 64, 48), np.dtype(dtype))

    converted = sector_stack_to_cartesian_array(frames, geometry, grid=grid, interpolation=interpolation)  # type: ignore[arg-type]
    per_frame = [sector_to_cartesian(frame, geometry, grid=grid, interpolation=interpolation) for frame in frames]  # type: ignore[arg-type]

    _same_bits(converted.data, np.stack([image.data for image in per_frame], axis=0))
    np.testing.assert_array_equal(converted.mask, per_frame[0].mask)
    assert converted.grid == grid
    default_height = sector_stack_to_cartesian_array(frames, geometry, interpolation=interpolation)  # type: ignore[arg-type]
    listed = sector_stack_to_cartesian(frames, geometry, interpolation=interpolation)  # type: ignore[arg-type]
    _same_bits(default_height.data, np.stack([image.data for image in listed], axis=0))


def test_sample_image_stack_matches_sample_image() -> None:
    rng = np.random.default_rng(2)
    frames = rng.normal(size=(4, 10, 12, 2)).astype(np.float32)
    rows = rng.uniform(-1.0, 11.0, size=(6, 7))
    cols = rng.uniform(-1.0, 13.0, size=(6, 7))
    for interpolation in ("linear", "nearest"):
        expected = np.stack([sample_image(frame, rows, cols, interpolation=interpolation) for frame in frames])  # type: ignore[arg-type]
        _same_bits(
            np.ascontiguousarray(sample_image_stack(frames, rows, cols, interpolation=interpolation)),  # type: ignore[arg-type]
            expected,
        )
    with pytest.raises(ValueError):
        sample_image_stack(frames[0, :, :, 0], rows, cols, interpolation="linear")
