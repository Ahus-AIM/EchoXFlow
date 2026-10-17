"""Torch versions of the per-sample geometric preprocessing, so it can run on the training device.

Data-loader workers used to resample every clip in numpy; that CPU work, not the GPU, bounded training speed. These
functions do the same operations with torch on any device. Sampling coordinates and sector masks depend only on the
geometries and shapes, so they are computed once in float64 numpy (exactly as :mod:`echoxflow.scan` does) and cached
per device. Results match the numpy path up to float rounding.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from echoxflow.scan import CartesianGrid, SectorGeometry
from echoxflow.scan.resampling import sector_lookup

Interpolation = Literal["linear", "nearest"]

_CACHE_SIZE = 512
_lookup_cache: OrderedDict[tuple[object, ...], tuple[Tensor, Tensor]] = OrderedDict()


def _cached(key: tuple[object, ...], build: Callable[[], tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    hit = _lookup_cache.get(key)
    if hit is not None:
        _lookup_cache.move_to_end(key)
        return hit
    value = build()
    _lookup_cache[key] = value
    if len(_lookup_cache) > _CACHE_SIZE:
        _lookup_cache.popitem(last=False)
    return value


def _normalized_grid(
    rows: np.ndarray, cols: np.ndarray, mask: np.ndarray, source_shape: tuple[int, int], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Pixel-unit source coordinates -> ``grid_sample`` grid (align_corners=True) and a boolean mask."""
    height, width = int(source_shape[0]), int(source_shape[1])
    x = cols / max(1, width - 1) * 2.0 - 1.0 if width > 1 else np.zeros_like(cols)
    y = rows / max(1, height - 1) * 2.0 - 1.0 if height > 1 else np.zeros_like(rows)
    grid = torch.from_numpy(np.stack([x, y], axis=-1).astype(np.float32))[None]
    return grid.to(device), torch.from_numpy(np.asarray(mask, dtype=bool)).to(device)


def sector_sampling_grid(
    source_geometry: SectorGeometry,
    target_geometry: SectorGeometry,
    *,
    source_shape: tuple[int, int],
    output_shape: tuple[int, int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Grid and mask that resample a source sector onto ``target_geometry`` at ``output_shape``."""
    source_shape = (int(source_shape[0]), int(source_shape[1]))
    output_shape = (int(output_shape[0]), int(output_shape[1]))
    key = ("sector", source_geometry, target_geometry, source_shape, output_shape, str(device))

    def build() -> tuple[Tensor, Tensor]:
        lookup = sector_lookup(source_geometry, target_geometry, source_shape=source_shape, output_shape=output_shape)
        return _normalized_grid(lookup.rows, lookup.cols, lookup.mask, source_shape, device)

    return _cached(key, build)


def cartesian_sampling_grid(
    geometry: SectorGeometry,
    grid: CartesianGrid,
    *,
    source_shape: tuple[int, int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Grid and sector mask that scan-convert a ``geometry`` sector onto the Cartesian ``grid``."""
    source_shape = (int(source_shape[0]), int(source_shape[1]))
    key = ("cartesian", geometry, grid, source_shape, str(device))

    def build() -> tuple[Tensor, Tensor]:
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
        return _normalized_grid(rows, cols, mask, source_shape, device)

    return _cached(key, build)


def sample_stack(frames: Tensor, grid: Tensor, mask: Tensor, *, interpolation: Interpolation = "linear") -> Tensor:
    """Sample a ``[T,H,W]`` stack at ``grid``; pixels outside ``mask`` are 0, like the numpy sector resamplers.

    Border padding reproduces the numpy samplers' index clamping, and NaN source samples propagate through the
    interpolation taps the same way.
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected a frame stack [T,H,W], got {tuple(frames.shape)}")
    values = frames.to(dtype=torch.float32)
    mode = "bilinear" if interpolation == "linear" else "nearest"
    sampled = F.grid_sample(values[None], grid, mode=mode, padding_mode="border", align_corners=True)[0]
    return torch.where(mask[None], sampled, torch.zeros((), dtype=sampled.dtype, device=sampled.device))


def resize_stack(frames: Tensor, shape: tuple[int, int], *, interpolation: Interpolation = "linear") -> Tensor:
    """Device version of :func:`tasks.utils.dataset.resize_stack` (``[T,H,W]`` float32)."""
    tensor = frames.to(dtype=torch.float32)[:, None]
    size = (int(shape[0]), int(shape[1]))
    if interpolation == "linear":
        resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    elif interpolation == "nearest":
        resized = F.interpolate(tensor, size=size, mode="nearest")
    else:
        raise ValueError(f"Unsupported stack interpolation: {interpolation}")
    return resized[:, 0]


def minmax_normalize(frames: Tensor) -> Tensor:
    """Device version of :func:`tasks.utils.dataset.minmax_normalize`: one min/max over the whole clip."""
    values = torch.nan_to_num(frames.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.numel() == 0:
        return torch.zeros_like(values)
    low, high = torch.aminmax(values)
    value_range = high - low
    normalized = torch.clamp((values - low) / value_range, 0.0, 1.0)
    degenerate = ~torch.isfinite(value_range) | (value_range < 1e-6)
    return torch.where(degenerate, torch.zeros_like(normalized), normalized)


def velocity_std_3x3(velocity: Tensor, region: Tensor, *, nyquist: float) -> Tensor:
    """Local velocity spread: SD over each pixel's 3x3 neighbours inside ``region``, aliasing-aware (``[T,H,W]``).

    Only finite neighbours inside the color box count, so the box edge does not read the zeros outside it as flow.
    Each neighbour's difference from the centre is wrapped into ``[-nyquist, nyquist)`` before the SD, so an aliased
    neighbour counts by its true phase distance rather than ~2 x Nyquist. Where no neighbour aliases this is the plain
    3x3 SD. Pixels outside ``region`` are 0 and non-finite centres stay NaN.
    """
    values = velocity.to(dtype=torch.float32)
    inside = region.expand_as(values) & torch.isfinite(values)
    height, width = int(values.shape[-2]), int(values.shape[-1])
    padded = F.pad(torch.where(inside, values, torch.zeros_like(values))[:, None], (1, 1, 1, 1))[:, 0]
    padded_inside = F.pad(inside[:, None].to(dtype=torch.float32), (1, 1, 1, 1))[:, 0]
    period = 2.0 * float(nyquist)
    centre = torch.where(inside, values, torch.zeros_like(values))
    count = torch.zeros_like(values)
    total = torch.zeros_like(values)
    total_sq = torch.zeros_like(values)
    for dy in range(3):
        for dx in range(3):
            weight = padded_inside[:, dy : dy + height, dx : dx + width]
            difference = padded[:, dy : dy + height, dx : dx + width] - centre
            if period > 0.0:
                difference = difference - period * torch.round(difference / period)
            count = count + weight
            total = total + weight * difference
            total_sq = total_sq + weight * difference * difference
    count = torch.clamp(count, min=1.0)
    mean = total / count
    spread = torch.sqrt(torch.clamp(total_sq / count - mean * mean, min=0.0))
    spread = torch.where(inside, spread, torch.zeros_like(spread))
    return torch.where(torch.isfinite(values), spread, values)


def as_device_tensor(values: np.ndarray | Tensor, device: torch.device) -> Tensor:
    """Move a worker-loaded array to ``device`` without changing its (usually compact) dtype."""
    tensor = values if isinstance(values, Tensor) else torch.from_numpy(np.ascontiguousarray(values))
    return tensor.to(device=device, non_blocking=True)
