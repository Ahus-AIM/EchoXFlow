from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from echoxflow.scan import CartesianGrid, SectorGeometry, resample_sector_stack, sector_stack_to_cartesian_array
from tasks.utils import dataset as numpy_ops
from tasks.utils import device_sampling as device_ops
from tasks.utils.dataset import collate_samples
from tasks.utils.deferred import DeferredBatch, DeferredSample
from tasks.utils.training.device import move_to_device

CPU = torch.device("cpu")
SOURCE = SectorGeometry(depth_start_m=0.004, depth_end_m=0.12, angle_start_rad=-0.55, angle_end_rad=0.5)
REFERENCE = SectorGeometry(depth_start_m=0.0, depth_end_m=0.15, angle_start_rad=-0.7, angle_end_rad=0.7)


def _frames(shape: tuple[int, int, int], *, with_nan: bool = False) -> np.ndarray:
    values = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    if with_nan:
        values[:, 10:13, 20:22] = np.nan
    return values


@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_sector_resampling_matches_numpy(interpolation: str) -> None:
    frames = _frames((5, 60, 40), with_nan=True)
    target = replace(REFERENCE, grid_shape=(48, 52))
    expected = resample_sector_stack(frames, SOURCE, target, output_shape=(48, 52), interpolation=interpolation)
    grid, mask = device_ops.sector_sampling_grid(
        SOURCE, target, source_shape=(60, 40), output_shape=(48, 52), device=CPU
    )
    actual = device_ops.sample_stack(torch.from_numpy(frames), grid, mask, interpolation=interpolation).numpy()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_cartesian_conversion_matches_numpy(interpolation: str) -> None:
    frames = _frames((4, 50, 30), with_nan=True)
    grid_spec = CartesianGrid.from_sector_height(SOURCE, 40)
    expected = sector_stack_to_cartesian_array(frames, SOURCE, grid=grid_spec, interpolation=interpolation)
    grid, mask = device_ops.cartesian_sampling_grid(SOURCE, grid_spec, source_shape=(50, 30), device=CPU)
    actual = device_ops.sample_stack(torch.from_numpy(frames), grid, mask, interpolation=interpolation).numpy()
    np.testing.assert_array_equal(mask.numpy(), expected.mask)
    np.testing.assert_allclose(actual, expected.data, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_value_transforms_match_numpy() -> None:
    frames = _frames((3, 20, 24))
    frames[0, 0, 0] = np.inf
    np.testing.assert_allclose(
        device_ops.minmax_normalize(torch.from_numpy(frames)).numpy(), numpy_ops.minmax_normalize(frames), atol=1e-6
    )
    flat = np.zeros((2, 4, 4), dtype=np.float32)
    assert not device_ops.minmax_normalize(torch.from_numpy(flat)).any()


@dataclass
class _ToySample:
    values: torch.Tensor
    sample_id: str


@dataclass(frozen=True)
class _ToyFinisher:
    scale: float

    def finish(self, raw: np.ndarray, device: torch.device) -> _ToySample:
        values = torch.from_numpy(raw).to(device) * self.scale
        return _ToySample(values=values[None], sample_id=f"item{int(raw[0])}")


def test_deferred_batches_finish_on_device_like_eager_samples() -> None:
    finisher = _ToyFinisher(scale=2.0)
    raws = [np.asarray([index, index + 1.0], dtype=np.float32) for index in range(3)]
    deferred = collate_samples([DeferredSample(raw=raw, finisher=finisher) for raw in raws])
    assert isinstance(deferred, DeferredBatch)
    finished = move_to_device(deferred, CPU)
    eager = collate_samples([finisher.finish(raw, CPU) for raw in raws])
    assert isinstance(finished, _ToySample)
    torch.testing.assert_close(finished.values, eager.values)
    assert finished.sample_id == "item0|item1|item2"
    single = move_to_device(DeferredSample(raw=raws[1], finisher=finisher), CPU)
    assert isinstance(single, _ToySample)
    torch.testing.assert_close(single.values, torch.tensor([[2.0, 4.0]]))


def _reference_std(values: np.ndarray, region: np.ndarray, nyquist: float) -> np.ndarray:
    """Per-pixel loop: SD of wrapped differences to the centre over finite in-region 3x3 neighbours."""
    out = np.zeros_like(values)
    frames, height, width = values.shape
    for t in range(frames):
        for row in range(height):
            for col in range(width):
                centre = values[t, row, col]
                if not np.isfinite(centre):
                    out[t, row, col] = np.nan
                    continue
                if not region[row, col]:
                    continue
                window = values[t, max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
                inside = region[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2] & np.isfinite(window)
                difference = window[inside] - centre
                difference = difference - 2 * nyquist * np.round(difference / (2 * nyquist))
                out[t, row, col] = difference.std()
    return out


def test_velocity_std_uses_in_box_neighbours_and_wraps_aliasing() -> None:
    rng = np.random.default_rng(1)
    values = rng.uniform(-0.5, 0.5, size=(2, 9, 11)).astype(np.float32)
    values[0, 4, 5] = np.nan
    region = np.zeros((9, 11), dtype=bool)
    region[2:8, 1:9] = True
    actual = device_ops.velocity_std_3x3(torch.from_numpy(values), torch.from_numpy(region), nyquist=0.5).numpy()
    np.testing.assert_allclose(actual, _reference_std(values, region, 0.5), atol=1e-5, equal_nan=True)

    # A uniform flow ending at the box edge has no spread, even though the pixels outside the box are 0.
    flat = np.where(region, 0.4, 0.0).astype(np.float32)[None]
    edge = device_ops.velocity_std_3x3(torch.from_numpy(flat), torch.from_numpy(region), nyquist=0.5).numpy()
    np.testing.assert_allclose(edge, 0.0, atol=1e-6)

    # Neighbours either side of the Nyquist limit (0.49 vs -0.49 m/s) are 0.02 m/s apart, not 0.98 m/s.
    aliased = np.full((1, 3, 3), 0.49, dtype=np.float32)
    aliased[0, :, 2] = -0.49
    full = np.ones((3, 3), dtype=bool)
    spread = device_ops.velocity_std_3x3(torch.from_numpy(aliased), torch.from_numpy(full), nyquist=0.5).numpy()
    assert spread[0, 1, 1] < 0.011
