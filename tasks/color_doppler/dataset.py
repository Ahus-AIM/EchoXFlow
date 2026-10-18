from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from echoxflow import (
    BrightnessModeStream,
    ColorDopplerPowerStream,
    ColorDopplerVelocityStream,
    CroissantCatalog,
    RecordingRecord,
    find_recordings,
    load_croissant,
    open_recording,
)
from echoxflow.scan import CartesianGrid, SectorGeometry
from echoxflow.schema import current_array_candidates
from echoxflow.timing import run_ids, within_one_run
from tasks.color_doppler.types import Sample
from tasks.utils.dataset import (
    SharedTrainingTransform,
    SkippedCase,
    _config_bool,
    build_task_dataloaders,
    discover_records_from_candidates,
    progress_iter,
    shared_training_transform_from_config,
    two_dimensional_frame_count,
)
from tasks.utils.dataset import as_spatial_shape as _as_spatial_shape
from tasks.utils.dataset import cartesian_grid_sample_grid as _cartesian_grid_sample_grid
from tasks.utils.dataset import cartesian_image_sample_grid as _cartesian_image_sample_grid
from tasks.utils.dataset import fps_from_timestamps as _fps_from_timestamps
from tasks.utils.dataset import optional_int as _optional_int
from tasks.utils.dataset import optional_ordered_float_pair as _optional_ordered_float_pair
from tasks.utils.dataset import ordered_float_pair as _ordered_float_pair
from tasks.utils.dataset import sample_indices_by_case as _sample_indices_by_case
from tasks.utils.dataset import slice_timestamps as _slice_timestamps
from tasks.utils.dataset import timestamps_or_index as _timestamps_or_index
from tasks.utils.dataset import training_transform_from_args as _training_transform_from_args
from tasks.utils.dataset import velocity_limit_from_stream as _velocity_limit_from_stream
from tasks.utils.deferred import DeferredSample
from tasks.utils.device_sampling import (
    as_device_tensor,
    cartesian_sampling_grid,
    sample_stack,
    sector_sampling_grid,
    velocity_std_3x3,
)
from tasks.utils.device_sampling import minmax_normalize as _minmax_normalize
from tasks.utils.device_sampling import resize_stack as _resize_stack
from tasks.utils.ref_cache import DEFAULT_REF_CACHE_DIR, RecordingScanCache, scan_cache_dir

__all__ = [
    "RawDataset",
    "SampleRef",
    "SkippedCase",
    "build_dataloaders",
    "discover_records",
]

_BMODE_PATH = "data/2d_brightness_mode"
_VELOCITY_PATH = "data/2d_color_doppler_velocity"
_POWER_PATH = "data/2d_color_doppler_power"
_VELOCITY_LOSS_BMODE_THRESHOLD = 0.4
_VELOCITY_LOSS_POWER_THRESHOLD = 0.3
_VELOCITY_LOSS_MASK_FLOOR = 0.01
_TEMPORAL_UPSAMPLE_FACTOR = 2
_FRAME_RATIO_RANGE = (0.45, 1.1)
_FPS_RANGE = (10.0, 13.0)
_NYQUIST_MPS_RANGE = (0.60, 0.61)
_RAW_COLOR_PATH = "data/2d_color_doppler_raw"
# 2: clip starts stay on one strictly increasing clock run and one calibrated Nyquist scale inside nyquist_mps_range.
_CLIP_STARTS_CACHE_VERSION = 2


@dataclass(frozen=True)
class SampleRef:
    record: RecordingRecord
    bmode_start: int
    bmode_stop: int
    sample_id: str


class RawDataset(Dataset[Sample]):
    def __init__(
        self,
        *,
        records: list[RecordingRecord],
        clip_length: int,
        clip_stride: int,
        min_frames_per_case: int = 32,
        input_spatial_shape: tuple[int, int] = (128, 128),
        target_spatial_shape: tuple[int, int] = (64, 64),
        data_root: str | Path | None = None,
        max_samples: int | None = None,
        velocity_loss_bmode_threshold: float = _VELOCITY_LOSS_BMODE_THRESHOLD,
        velocity_loss_power_threshold: float = _VELOCITY_LOSS_POWER_THRESHOLD,
        velocity_loss_mask_floor: float = _VELOCITY_LOSS_MASK_FLOOR,
        frame_ratio_range: tuple[float, float] = _FRAME_RATIO_RANGE,
        fps_range: tuple[float, float] = _FPS_RANGE,
        nyquist_mps_range: tuple[float, float] | None = _NYQUIST_MPS_RANGE,
        coordinate_space: str = "beamspace",
        cartesian_height: int | None = None,
        include_cartesian_metrics: bool = False,
        training_transform: SharedTrainingTransform | None = None,
        recording_cache_dir: str | Path | None = None,
        recording_cache_read_only: bool = False,
        recording_cache_include: tuple[str, ...] = (),
        recording_cache_exclude: tuple[str, ...] = (),
        sample_ref_cache_dir: str | Path | None = DEFAULT_REF_CACHE_DIR,
        sample_ref_cache_read_only: bool = False,
        device_preprocessing: bool = False,
        **_: object,
    ) -> None:
        self.records = sorted(records, key=lambda record: (record.exam_id, record.recording_id))
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride)
        self.min_frames_per_case = int(min_frames_per_case)
        self.input_spatial_shape = _as_spatial_shape(input_spatial_shape)
        self.target_spatial_shape = _as_spatial_shape(target_spatial_shape)
        self.data_root = data_root
        self.training_transform = _training_transform_from_args(training_transform=training_transform)
        self.velocity_loss_bmode_threshold = float(velocity_loss_bmode_threshold)
        self.velocity_loss_power_threshold = float(velocity_loss_power_threshold)
        self.velocity_loss_mask_floor = float(velocity_loss_mask_floor)
        self.frame_ratio_range = _ordered_float_pair(frame_ratio_range, name="frame_ratio_range")
        self.fps_range = _ordered_float_pair(fps_range, name="fps_range")
        self.nyquist_mps_range = _optional_ordered_float_pair(nyquist_mps_range, name="nyquist_mps_range")
        self.coordinate_space = str(coordinate_space).strip().lower()
        if self.coordinate_space not in {"beamspace", "cartesian"}:
            raise ValueError(
                f"Color Doppler coordinate_space must be 'beamspace' or 'cartesian', got {coordinate_space!r}"
            )
        self.cartesian_height = (
            int(cartesian_height) if cartesian_height is not None else int(self.input_spatial_shape[0])
        )
        self.include_cartesian_metrics = bool(include_cartesian_metrics)
        self.device_preprocessing = bool(device_preprocessing)
        self.recording_cache_dir = None if recording_cache_dir is None else Path(recording_cache_dir)
        self.recording_cache_read_only = bool(recording_cache_read_only)
        self.recording_cache_include = tuple(str(path) for path in recording_cache_include) or (
            tuple(
                dict.fromkeys(
                    candidate
                    for path in (_BMODE_PATH, _VELOCITY_PATH, _POWER_PATH)
                    for candidate in current_array_candidates(path)
                )
            )
            if self.recording_cache_dir is not None
            else ()
        )
        self.recording_cache_exclude = tuple(str(path) for path in recording_cache_exclude)
        self.skipped_cases: list[SkippedCase] = []
        self.sample_refs = _build_sample_refs(
            records=self.records,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            min_frames_per_case=self.min_frames_per_case,
            data_root=data_root,
            max_samples=max_samples,
            frame_ratio_range=self.frame_ratio_range,
            fps_range=self.fps_range,
            nyquist_mps_range=self.nyquist_mps_range,
            recording_cache_dir=self.recording_cache_dir,
            recording_cache_read_only=self.recording_cache_read_only,
            recording_cache_include=self.recording_cache_include,
            recording_cache_exclude=self.recording_cache_exclude,
            sample_ref_cache_dir=sample_ref_cache_dir,
            sample_ref_cache_read_only=sample_ref_cache_read_only,
        )
        self.sample_indices_by_case = _sample_indices_by_case(self.sample_refs)

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, index: int) -> Any:  # Sample, or DeferredSample with device_preprocessing
        raw = self.load_raw(self.sample_refs[index])
        if self.device_preprocessing:
            return DeferredSample(raw=raw, finisher=self.finisher)
        return self.finisher.finish(raw, torch.device("cpu"))

    def sample_from_ref(self, ref: SampleRef) -> Sample:
        return self.finisher.finish(self.load_raw(ref), torch.device("cpu"))

    @property
    def finisher(self) -> _Finisher:
        return _Finisher(
            coordinate_space=self.coordinate_space,
            cartesian_height=self.cartesian_height,
            input_spatial_shape=self.input_spatial_shape,
            target_spatial_shape=self.target_spatial_shape,
            include_cartesian_metrics=self.include_cartesian_metrics,
            velocity_loss_bmode_threshold=self.velocity_loss_bmode_threshold,
            velocity_loss_power_threshold=self.velocity_loss_power_threshold,
            velocity_loss_mask_floor=self.velocity_loss_mask_floor,
        )

    def load_raw(self, ref: SampleRef) -> _RawClip:
        """Load one clip's native arrays and scalars; all resampling happens in :meth:`_Finisher.finish`."""
        store = open_recording(
            ref.record,
            root=self.data_root,
            cache_dir=self.recording_cache_dir,
            cache_read_only=self.recording_cache_read_only,
            cache_include=self.recording_cache_include,
            cache_exclude=self.recording_cache_exclude,
        )
        bmode = store.load_stream_slice(_BMODE_PATH, ref.bmode_start, ref.bmode_stop)
        if not isinstance(bmode, BrightnessModeStream):
            raise TypeError(f"{_BMODE_PATH} must be BrightnessModeStream")
        bmode_count = two_dimensional_frame_count(ref.record, self.data_root, _BMODE_PATH)
        velocity_count = two_dimensional_frame_count(ref.record, self.data_root, _VELOCITY_PATH)
        power_count = two_dimensional_frame_count(ref.record, self.data_root, _POWER_PATH)
        temporal_mode = _temporal_mode_from_frame_counts(
            bmode_count=bmode_count,
            doppler_count=min(velocity_count, power_count),
        )
        doppler_slice = _doppler_slice_for_bmode_clip(ref=ref, temporal_mode=temporal_mode)
        velocity = store.load_stream_slice(_VELOCITY_PATH, doppler_slice.start, doppler_slice.stop)
        power = store.load_stream_slice(_POWER_PATH, doppler_slice.start, doppler_slice.stop)
        if not isinstance(velocity, ColorDopplerVelocityStream):
            raise TypeError(f"{_VELOCITY_PATH} must be ColorDopplerVelocityStream")
        if not isinstance(power, ColorDopplerPowerStream):
            raise TypeError(f"{_POWER_PATH} must be ColorDopplerPowerStream")
        bmode_frames = np.asarray(bmode.data)
        fps = ref.record.sample_rate_hz("2d_brightness_mode") or _fps_from_timestamps(bmode.timestamps) or 0.0
        velocity_frames = np.asarray(velocity.data, dtype=np.float32)
        power_frames = np.asarray(power.data, dtype=np.float32)
        source_validity = store.load_validity_mask(_VELOCITY_PATH, doppler_slice.start, doppler_slice.stop)
        frame_nyquist = velocity.metadata.velocity_scales_mps(int(velocity_frames.shape[0]))
        repeat = temporal_mode == "repeat_one_per_interval"
        if repeat and frame_nyquist is not None:
            frame_nyquist = np.repeat(frame_nyquist, _TEMPORAL_UPSAMPLE_FACTOR, axis=0)
        target_count = int(velocity_frames.shape[0]) * (_TEMPORAL_UPSAMPLE_FACTOR if repeat else 1)
        expected_target_time = _TEMPORAL_UPSAMPLE_FACTOR * (int(bmode_frames.shape[0]) - 1)
        if target_count != expected_target_time:
            raise ValueError(
                f"{ref.sample_id} has {target_count} color targets and {bmode_frames.shape[0]} "
                f"B-mode frames; expected {expected_target_time} color targets"
            )
        nyquist = _velocity_limit_from_stream(velocity) if frame_nyquist is None else _clip_nyquist(frame_nyquist, ref)
        frame_timestamps = _slice_timestamps(bmode.timestamps, 0, int(bmode_frames.shape[0]))
        target_timestamps = _target_timestamps_from_bmode(
            bmode_timestamps=frame_timestamps,
            target_count=expected_target_time,
            fps=fps,
        )
        return _RawClip(
            bmode=bmode_frames,
            bmode_geometry=_required_geometry(bmode, role="B-mode reference"),
            velocity=velocity_frames,
            velocity_geometry=_required_geometry(velocity, role="Color Doppler source"),
            power=power_frames,
            power_geometry=_required_geometry(power, role="Color Doppler source"),
            validity=None if source_validity is None else np.asarray(source_validity, dtype=bool),
            repeat_targets=repeat,
            fps=float(fps),
            nyquist=float(nyquist),
            frame_nyquist=None if frame_nyquist is None else np.asarray(frame_nyquist, dtype=np.float32),
            frame_timestamps=np.asarray(frame_timestamps),
            target_timestamps=np.asarray(target_timestamps),
            sample_id=ref.sample_id,
            record=ref.record,
            data_root=self.data_root,
            clip_start=ref.bmode_start,
            clip_stop=ref.bmode_stop,
        )


@dataclass(frozen=True)
class _RawClip:
    """Native-resolution arrays of one Color Doppler clip, as loaded by a data-loader worker."""

    bmode: np.ndarray
    bmode_geometry: SectorGeometry
    velocity: np.ndarray
    velocity_geometry: SectorGeometry
    power: np.ndarray
    power_geometry: SectorGeometry
    validity: np.ndarray | None
    repeat_targets: bool
    fps: float
    nyquist: float
    frame_nyquist: np.ndarray | None
    frame_timestamps: np.ndarray
    target_timestamps: np.ndarray
    sample_id: str
    record: RecordingRecord
    data_root: str | Path | None
    clip_start: int
    clip_stop: int


@dataclass(frozen=True)
class _TargetGrid:
    """The grid targets are resampled onto: B-mode beamspace at ``shape``, or a Cartesian grid resized to it."""

    shape: tuple[int, int]
    cartesian: CartesianGrid | None


@dataclass(frozen=True)
class _Finisher:
    """Builds a :class:`Sample` from a :class:`_RawClip` on any torch device."""

    coordinate_space: str
    cartesian_height: int
    input_spatial_shape: tuple[int, int]
    target_spatial_shape: tuple[int, int]
    include_cartesian_metrics: bool
    velocity_loss_bmode_threshold: float
    velocity_loss_power_threshold: float
    velocity_loss_mask_floor: float

    def finish(self, raw: _RawClip, device: torch.device) -> Sample:
        bmode = as_device_tensor(raw.bmode, device).to(dtype=torch.float32) / 255.0
        velocity = as_device_tensor(raw.velocity, device)
        power = as_device_tensor(raw.power, device)
        validity = None if raw.validity is None else as_device_tensor(raw.validity, device).to(dtype=torch.float32)
        if raw.repeat_targets:
            velocity = torch.repeat_interleave(velocity, _TEMPORAL_UPSAMPLE_FACTOR, dim=0)
            power = torch.repeat_interleave(power, _TEMPORAL_UPSAMPLE_FACTOR, dim=0)
            if validity is not None:
                validity = torch.repeat_interleave(validity, _TEMPORAL_UPSAMPLE_FACTOR, dim=0)
        cartesian_grid = CartesianGrid.from_sector_height(raw.bmode_geometry, self.cartesian_height)
        if self.coordinate_space == "cartesian":
            frames = self._cartesian_bmode(bmode, raw, cartesian_grid, self.input_spatial_shape)
            interval_source = frames
            grid = _TargetGrid(shape=self.target_spatial_shape, cartesian=cartesian_grid)
        else:
            frames = _minmax_normalize(_resize_stack(bmode, self.input_spatial_shape))
            interval_source = bmode
            grid = _TargetGrid(shape=self.target_spatial_shape, cartesian=None)
        target, valid_mask, velocity_loss_mask, color_box = self._targets(
            raw, velocity=velocity, power=power, validity=validity, bmode_intervals=interval_source, grid=grid
        )
        metric: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None
        if self.include_cartesian_metrics:
            metric_grid = _TargetGrid(shape=cartesian_grid.shape, cartesian=cartesian_grid)
            cartesian_bmode = self._cartesian_bmode(bmode, raw, cartesian_grid, cartesian_grid.shape)
            metric_values = self._targets(
                raw,
                velocity=velocity,
                power=power,
                validity=validity,
                bmode_intervals=cartesian_bmode,
                grid=metric_grid,
            )
            if self.coordinate_space == "beamspace":
                sample_grid = _cartesian_grid_sample_grid(
                    raw.bmode_geometry,
                    cartesian_grid,
                    source_shape=self.target_spatial_shape,
                    output_shape=cartesian_grid.shape,
                )
            else:
                sample_grid = _cartesian_image_sample_grid(cartesian_grid)
            metric = (*metric_values, as_device_tensor(sample_grid, device))
        conditioning = torch.tensor([[raw.fps, raw.nyquist]], dtype=torch.float32, device=device)
        return Sample(
            frames=frames[None, None],
            frame_timestamps=as_device_tensor(raw.frame_timestamps[None, :], device),
            target_timestamps=as_device_tensor(raw.target_timestamps[None, :], device),
            conditioning=conditioning,
            doppler_target=target,
            color_box_target=color_box,
            valid_mask=valid_mask,
            velocity_loss_mask=velocity_loss_mask,
            nyquist_mps=torch.tensor([raw.nyquist], dtype=torch.float32, device=device),
            sample_id=raw.sample_id,
            record=raw.record,
            data_root=raw.data_root,
            clip_start=raw.clip_start,
            clip_stop=raw.clip_stop,
            coordinate_space=self.coordinate_space,
            cartesian_metric_doppler_target=None if metric is None else metric[0],
            cartesian_metric_valid_mask=None if metric is None else metric[1],
            cartesian_metric_velocity_loss_mask=None if metric is None else metric[2],
            cartesian_metric_color_box_target=None if metric is None else metric[3],
            cartesian_metric_sample_grid=None if metric is None else metric[4][None],
            nyquist_mps_per_frame=(
                None if raw.frame_nyquist is None else as_device_tensor(raw.frame_nyquist[None, :], device)
            ),
        )

    def _cartesian_bmode(
        self, bmode: Tensor, raw: _RawClip, grid: CartesianGrid, output_shape: tuple[int, int]
    ) -> Tensor:
        sampling, mask = cartesian_sampling_grid(
            raw.bmode_geometry, grid, source_shape=(int(bmode.shape[1]), int(bmode.shape[2])), device=bmode.device
        )
        converted = sample_stack(bmode, sampling, mask)
        return _minmax_normalize(_resize_stack(converted, output_shape))

    def _resample(
        self, values: Tensor, *, geometry: SectorGeometry, raw: _RawClip, grid: _TargetGrid
    ) -> tuple[Tensor, Tensor]:
        """Resample ``[T,R,A]`` source-sector values onto the target grid; returns values and the region mask."""
        source_shape = (int(values.shape[1]), int(values.shape[2]))
        if grid.cartesian is None:
            target_geometry = replace(raw.bmode_geometry, grid_shape=grid.shape)
            sampling, mask = sector_sampling_grid(
                geometry, target_geometry, source_shape=source_shape, output_shape=grid.shape, device=values.device
            )
            return sample_stack(values, sampling, mask), mask
        sampling, mask = cartesian_sampling_grid(
            geometry, grid.cartesian, source_shape=source_shape, device=values.device
        )
        converted = sample_stack(values, sampling, mask)
        region = _resize_stack(mask[None].to(dtype=torch.float32), grid.shape)[0] > 0.5
        return _resize_stack(converted, grid.shape), region

    def _targets(
        self,
        raw: _RawClip,
        *,
        velocity: Tensor,
        power: Tensor,
        validity: Tensor | None,
        bmode_intervals: Tensor,
        grid: _TargetGrid,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``doppler_target``, ``valid_mask``, ``velocity_loss_mask`` and ``color_box_target`` (5-D)."""
        count = int(velocity.shape[0])
        # Velocity and its validity share a source sector and grid, so one sampling pass serves both.
        stacked = velocity if validity is None else torch.cat([velocity, validity], dim=0)
        sampled, velocity_region = self._resample(stacked, geometry=raw.velocity_geometry, raw=raw, grid=grid)
        velocity_values = sampled[:count]
        power_values, power_region = self._resample(power, geometry=raw.power_geometry, raw=raw, grid=grid)
        power_values = torch.clamp(power_values, 0.0, 1.0)
        in_box = velocity_region & power_region
        color_region = in_box.to(dtype=torch.float32)
        velocity_std = velocity_std_3x3(velocity_values, in_box, nyquist=raw.nyquist)
        target = torch.stack([velocity_values, power_values, velocity_std], dim=1)
        valid = torch.isfinite(target).all(dim=1).to(dtype=torch.float32)
        if validity is not None:
            # Samples the exporter marks as having no velocity estimate (power code 0 or Rx = Ry = 0) are already
            # NaN or below the power threshold; this only removes their floor weight from the velocity loss.
            valid_for_velocity = valid * (torch.nan_to_num(sampled[count:], nan=0.0) > 0.5).to(dtype=torch.float32)
        else:
            valid_for_velocity = valid
        bmode_on_target = _bmode_interval_stack(bmode_intervals, grid.shape)
        loss_mask = _velocity_loss_mask(
            bmode=bmode_on_target,
            power=power_values,
            valid_mask=valid_for_velocity,
            bmode_threshold=self.velocity_loss_bmode_threshold,
            power_threshold=self.velocity_loss_power_threshold,
            mask_floor=self.velocity_loss_mask_floor,
        )
        color_box = color_region.expand(count, *grid.shape)
        return (
            target.permute(1, 0, 2, 3)[None],
            valid[None, None],
            loss_mask[None, None],
            color_box[None, None].contiguous(),
        )


def _bmode_interval_stack(frames: Tensor, target_shape: tuple[int, int]) -> Tensor:
    """B-mode on the target clock and grid: the mean of each interval's two frames, once per target frame."""
    values = _minmax_normalize(frames)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError(f"B-mode clip must contain at least two frames, got {tuple(values.shape)}")
    intervals = 0.5 * (values[:-1] + values[1:])
    repeated = torch.repeat_interleave(intervals, _TEMPORAL_UPSAMPLE_FACTOR, dim=0)
    return _resize_stack(repeated, target_shape)


def _velocity_loss_mask(
    *,
    bmode: Tensor,
    power: Tensor,
    valid_mask: Tensor,
    bmode_threshold: float,
    power_threshold: float,
    mask_floor: float,
) -> Tensor:
    finite_valid = torch.isfinite(bmode) & torch.isfinite(power) & torch.isfinite(valid_mask) & (valid_mask > 0.5)
    selected = finite_valid & (bmode < float(bmode_threshold)) & (power > float(power_threshold))
    floor = torch.where(finite_valid, float(mask_floor), 0.0)
    return torch.where(selected, 1.0, floor).to(dtype=torch.float32)


def discover_records(
    *,
    root_dir: str | Path,
    max_cases: int | None = None,
    exam_ids: Iterable[str] | None = None,
    sample_fraction: object | None = None,
    seed: int | None = None,
    frame_ratio_range: tuple[float, float] = _FRAME_RATIO_RANGE,
    fps_range: tuple[float, float] = _FPS_RANGE,
    nyquist_mps_range: tuple[float, float] | None = _NYQUIST_MPS_RANGE,
    sample_ref_cache_dir: str | Path | None = DEFAULT_REF_CACHE_DIR,
    sample_ref_cache_read_only: bool = False,
) -> list[RecordingRecord]:
    filter_cache = _color_filter_cache(
        cache_dir=sample_ref_cache_dir,
        read_only=sample_ref_cache_read_only,
        data_root=root_dir,
        frame_ratio_range=frame_ratio_range,
        fps_range=fps_range,
        nyquist_mps_range=nyquist_mps_range,
    )
    catalog = load_croissant(root=root_dir)
    found = find_recordings(
        croissant=catalog,
        array_paths=_required_array_paths(),
        require_all=True,
        predicate=lambda record: _color_metadata_discovery_predicate(
            record,
            frame_ratio_range=frame_ratio_range,
            fps_range=fps_range,
            nyquist_mps_range=nyquist_mps_range,
            catalog=catalog,
        ),
    )
    return discover_records_from_candidates(
        found,
        exam_ids=exam_ids,
        max_cases=max_cases,
        sample_fraction=sample_fraction,
        seed=seed,
        description="Scanning color Doppler recordings",
        predicate=lambda record: _color_runtime_discovery_predicate(
            record,
            data_root=root_dir,
            frame_ratio_range=frame_ratio_range,
            fps_range=fps_range,
            nyquist_mps_range=nyquist_mps_range,
            filter_cache=filter_cache,
        ),
    )


def _color_filter_cache(
    *,
    cache_dir: str | Path | None,
    read_only: bool,
    data_root: str | Path | None,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
) -> RecordingScanCache:
    """Cache of the per-recording Color Doppler filter verdicts, shared by discovery and every split."""
    return RecordingScanCache(
        cache_dir,
        namespace="color_doppler_filters",
        settings={
            # 2: derived velocity resolves its source sector, and a changing scale passes when any frame is in range.
            "version": 2,
            "frame_ratio_range": [float(value) for value in frame_ratio_range],
            "fps_range": [float(value) for value in fps_range],
            "nyquist_mps_range": None if nyquist_mps_range is None else [float(value) for value in nyquist_mps_range],
        },
        data_root=data_root,
        read_only=read_only,
    )


def _color_metadata_discovery_predicate(
    record: RecordingRecord,
    *,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
    catalog: CroissantCatalog | None = None,
) -> bool:
    if any("tissue" in str(content_type).lower() for content_type in record.content_types):
        return False
    if nyquist_mps_range is not None and catalog is not None:
        # Catalog preflight: a stream whose every calibrated frame lies outside the accepted Nyquist range cannot
        # contribute a clip. A summary without a scale range decides nothing; the store check below still runs.
        summary = catalog.stream_summary(record.recording_id, _RAW_COLOR_PATH)
        if summary is not None and summary.velocity_scale_may_intersect(nyquist_mps_range) is False:
            return False
    return not _record_fails_color_metadata_filters(record, frame_ratio_range=frame_ratio_range, fps_range=fps_range)


def _color_runtime_discovery_predicate(
    record: RecordingRecord,
    *,
    data_root: str | Path,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
    filter_cache: RecordingScanCache | None = None,
) -> bool:
    if not _uses_color_filters(
        frame_ratio_range=frame_ratio_range,
        fps_range=fps_range,
        nyquist_mps_range=nyquist_mps_range,
    ):
        return True

    def matches(candidate: RecordingRecord) -> bool:
        return _record_matches_color_filters(
            candidate,
            data_root=data_root,
            frame_ratio_range=frame_ratio_range,
            fps_range=fps_range,
            nyquist_mps_range=nyquist_mps_range,
        )

    return bool(matches(record) if filter_cache is None else filter_cache.get(record, matches))


def _filter_ranges(
    data_config: Mapping[str, object],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float] | None]:
    frame_ratio_range = _ordered_float_pair(
        data_config.get("frame_ratio_range", _FRAME_RATIO_RANGE),
        name="data.frame_ratio_range",
    )
    fps_range = _ordered_float_pair(data_config.get("fps_range", _FPS_RANGE), name="data.fps_range")
    nyquist_mps_range = _optional_ordered_float_pair(
        data_config.get("nyquist_mps_range", _NYQUIST_MPS_RANGE),
        name="data.nyquist_mps_range",
    )
    return frame_ratio_range, fps_range, nyquist_mps_range


def _discover_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    frame_ratio_range, fps_range, nyquist_mps_range = _filter_ranges(data_config)
    return {
        "frame_ratio_range": frame_ratio_range,
        "fps_range": fps_range,
        "nyquist_mps_range": nyquist_mps_range,
        **_scan_cache_kwargs(data_config),
    }


def _scan_cache_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    return {
        "sample_ref_cache_dir": scan_cache_dir(data_config),
        "sample_ref_cache_read_only": _config_bool(data_config.get("sample_ref_cache_read_only", False)),
    }


def _dataset_kwargs(_config: Any, data_config: Mapping[str, object], root: str | Path) -> dict[str, object]:
    frame_ratio_range, fps_range, nyquist_mps_range = _filter_ranges(data_config)
    return {
        "clip_length": int(cast(Any, data_config.get("clip_length", 32))),
        "clip_stride": int(cast(Any, data_config.get("clip_stride", 1))),
        "min_frames_per_case": int(cast(Any, data_config.get("min_frames_per_case", 32))),
        "input_spatial_shape": _as_spatial_shape(cast(Any, data_config.get("input_spatial_shape", (128, 128)))),
        "target_spatial_shape": _as_spatial_shape(cast(Any, data_config.get("target_spatial_shape", (64, 64)))),
        "data_root": root,
        "frame_ratio_range": frame_ratio_range,
        "fps_range": fps_range,
        "nyquist_mps_range": nyquist_mps_range,
        "velocity_loss_bmode_threshold": _config_float(
            data_config,
            "velocity_loss_bmode_threshold",
            _VELOCITY_LOSS_BMODE_THRESHOLD,
        ),
        "velocity_loss_power_threshold": _config_float(
            data_config,
            "velocity_loss_power_threshold",
            _VELOCITY_LOSS_POWER_THRESHOLD,
        ),
        "velocity_loss_mask_floor": _config_float(
            data_config,
            "velocity_loss_mask_floor",
            _VELOCITY_LOSS_MASK_FLOOR,
        ),
        "coordinate_space": str(data_config.get("coordinate_space", "beamspace")),
        "cartesian_height": _optional_int(data_config.get("cartesian_height")),
        "device_preprocessing": _config_bool(data_config.get("device_preprocessing", False)),
        **_scan_cache_kwargs(data_config),
    }


def _train_dataset_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    return {"training_transform": shared_training_transform_from_config(data_config)}


def _val_dataset_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    del data_config
    return {"include_cartesian_metrics": True}


build_dataloaders = partial(
    build_task_dataloaders,
    task_name="color_doppler",
    dataset_cls=RawDataset,
    discover_records=discover_records,
    empty_message="Color Doppler dataset is empty",
    discover_kwargs_fn=_discover_kwargs,
    dataset_kwargs_fn=_dataset_kwargs,
    train_dataset_kwargs_fn=_train_dataset_kwargs,
    val_dataset_kwargs_fn=_val_dataset_kwargs,
)


def _build_sample_refs(
    *,
    records: list[RecordingRecord],
    clip_length: int,
    clip_stride: int,
    min_frames_per_case: int,
    data_root: str | Path | None,
    max_samples: int | None,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
    recording_cache_dir: str | Path | None = None,
    recording_cache_read_only: bool = False,
    recording_cache_include: tuple[str, ...] = (),
    recording_cache_exclude: tuple[str, ...] = (),
    sample_ref_cache_dir: str | Path | None = DEFAULT_REF_CACHE_DIR,
    sample_ref_cache_read_only: bool = False,
) -> list[SampleRef]:
    # Filter verdicts and usable frame counts are cached per recording (see tasks.utils.ref_cache) and shared with
    # discovery and the other splits; the clip windows are cheap and derived here.
    filter_cache = _color_filter_cache(
        cache_dir=sample_ref_cache_dir,
        read_only=sample_ref_cache_read_only,
        data_root=data_root,
        frame_ratio_range=frame_ratio_range,
        fps_range=fps_range,
        nyquist_mps_range=nyquist_mps_range,
    )
    interval_count = int(clip_length)
    min_interval_count = max(interval_count, int(min_frames_per_case))
    starts_cache = RecordingScanCache(
        sample_ref_cache_dir,
        namespace="color_doppler_clip_starts",
        settings={
            "version": _CLIP_STARTS_CACHE_VERSION,
            "clip_length": interval_count,
            "clip_stride": max(1, int(clip_stride)),
            "min_interval_count": min_interval_count,
            "nyquist_mps_range": None if nyquist_mps_range is None else [float(value) for value in nyquist_mps_range],
        },
        data_root=data_root,
        read_only=sample_ref_cache_read_only,
    )

    def matches(candidate: RecordingRecord) -> bool:
        return _record_matches_color_filters(
            candidate,
            data_root=data_root,
            frame_ratio_range=frame_ratio_range,
            fps_range=fps_range,
            nyquist_mps_range=nyquist_mps_range,
            recording_cache_dir=recording_cache_dir,
            recording_cache_read_only=recording_cache_read_only,
            recording_cache_include=recording_cache_include,
            recording_cache_exclude=recording_cache_exclude,
        )

    def clip_starts(candidate: RecordingRecord) -> list[int]:
        return _valid_clip_starts(
            record=candidate,
            data_root=data_root,
            interval_count=interval_count,
            clip_stride=clip_stride,
            min_interval_count=min_interval_count,
            nyquist_mps_range=nyquist_mps_range,
        )

    refs: list[SampleRef] = []
    input_frame_count = interval_count + 1
    uses_filters = _uses_color_filters(
        frame_ratio_range=frame_ratio_range,
        fps_range=fps_range,
        nyquist_mps_range=nyquist_mps_range,
    )
    for record in progress_iter(
        records,
        description="Scanning color Doppler samples",
        total=len(records),
        unit="case",
    ):
        if uses_filters and not bool(filter_cache.get(record, matches)):
            continue
        for start in starts_cache.get(record, clip_starts):
            start = int(start)
            stop = start + input_frame_count
            refs.append(
                SampleRef(
                    record=record,
                    bmode_start=start,
                    bmode_stop=stop,
                    sample_id=f"{record.exam_id}/{record.recording_id}:clip_{start}_{stop}",
                )
            )
            if max_samples is not None and len(refs) >= int(max_samples):
                return refs
    return refs


def _valid_clip_starts(
    *,
    record: RecordingRecord,
    data_root: str | Path | None,
    interval_count: int,
    clip_stride: int,
    min_interval_count: int,
    nyquist_mps_range: tuple[float, float] | None,
) -> list[int]:
    """Return the B-mode clip starts whose frames and colour targets can be read as one consistent clip.

    A clip's B-mode frames and colour frames must each lie on one strictly increasing clock run, so a source-segment
    clock reset never falls inside a clip. Its colour frames must share one calibrated Nyquist scale, inside
    ``nyquist_mps_range`` when given, so the scalar Nyquist conditioning is exact for every target frame.
    """
    usable_bmode_count = _usable_bmode_count(record=record, data_root=data_root)
    if usable_bmode_count - 1 < min_interval_count:
        return []
    store = open_recording(record, root=data_root)
    bmode = store.load_stream_header(_BMODE_PATH)
    velocity = store.load_stream_header(_VELOCITY_PATH)
    bmode_runs = run_ids(_timestamps_or_index(bmode.timestamps, bmode.frame_count))
    doppler_runs = run_ids(_timestamps_or_index(velocity.timestamps, velocity.frame_count))
    scales = velocity.velocity_scale_mps_per_frame
    bmode_count = two_dimensional_frame_count(record, data_root, _BMODE_PATH)
    doppler_count = min(
        two_dimensional_frame_count(record, data_root, _VELOCITY_PATH),
        two_dimensional_frame_count(record, data_root, _POWER_PATH),
    )
    temporal_mode = _temporal_mode_from_frame_counts(bmode_count=bmode_count, doppler_count=doppler_count)
    input_frame_count = int(interval_count) + 1
    starts: list[int] = []
    rejected: dict[str, int] = {}
    for start in range(0, usable_bmode_count - input_frame_count + 1, max(1, int(clip_stride))):
        ref = SampleRef(record=record, bmode_start=start, bmode_stop=start + input_frame_count, sample_id="")
        doppler = np.arange(*_doppler_slice_for_bmode_clip(ref=ref, temporal_mode=temporal_mode).indices(doppler_count))
        reason = None
        if not within_one_run(np.arange(start, start + input_frame_count), bmode_runs):
            reason = "bmode_clock_break"
        elif not within_one_run(doppler, doppler_runs):
            reason = "doppler_clock_break"
        elif scales is not None:
            clip_scales = np.asarray(scales, dtype=np.float64)[doppler]
            if not np.all(np.isfinite(clip_scales)):
                reason = "uncalibrated_doppler_frame"
            elif np.any(clip_scales != clip_scales[0]):
                reason = "velocity_scale_change"
            elif nyquist_mps_range is not None and not (
                float(nyquist_mps_range[0]) <= float(clip_scales[0]) <= float(nyquist_mps_range[1])
            ):
                reason = "nyquist_out_of_range"
        if reason is None:
            starts.append(start)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    if rejected:
        logging.getLogger(__name__).info(
            "%s/%s: skipped Color Doppler clips %s", record.exam_id, record.recording_id, rejected
        )
    return starts


def _uses_color_filters(
    *,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
) -> bool:
    return (
        tuple(float(value) for value in frame_ratio_range) != (0.0, 1.0e9)
        or tuple(float(value) for value in fps_range) != (0.0, 1.0e9)
        or nyquist_mps_range is not None
    )


def _usable_bmode_count(*, record: RecordingRecord, data_root: str | Path | None) -> int:
    bmode_count = two_dimensional_frame_count(record, data_root, _BMODE_PATH)
    velocity_count = two_dimensional_frame_count(record, data_root, _VELOCITY_PATH)
    power_count = two_dimensional_frame_count(record, data_root, _POWER_PATH)
    doppler_count = min(velocity_count, power_count)
    if bmode_count <= 0 or doppler_count <= 0:
        return 0
    mode = _temporal_mode_from_frame_counts(bmode_count=bmode_count, doppler_count=doppler_count)
    if mode == "two_per_interval":
        return max(0, min(int(bmode_count), int(doppler_count) // _TEMPORAL_UPSAMPLE_FACTOR + 1))
    return max(0, min(int(bmode_count), int(doppler_count) + 1))


def _record_matches_color_filters(
    record: RecordingRecord,
    *,
    data_root: str | Path | None,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
    recording_cache_dir: str | Path | None = None,
    recording_cache_read_only: bool = False,
    recording_cache_include: tuple[str, ...] = (),
    recording_cache_exclude: tuple[str, ...] = (),
) -> bool:
    try:
        return _validate_color_record(
            record,
            data_root=data_root,
            frame_ratio_range=frame_ratio_range,
            fps_range=fps_range,
            nyquist_mps_range=nyquist_mps_range,
            recording_cache_dir=recording_cache_dir,
            recording_cache_read_only=recording_cache_read_only,
            recording_cache_include=recording_cache_include,
            recording_cache_exclude=recording_cache_exclude,
        )
    except KeyError:
        return False


def _validate_color_record(
    record: RecordingRecord,
    *,
    data_root: str | Path | None,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
    recording_cache_dir: str | Path | None = None,
    recording_cache_read_only: bool = False,
    recording_cache_include: tuple[str, ...] = (),
    recording_cache_exclude: tuple[str, ...] = (),
) -> bool:
    bmode_count = _record_frame_count(record, data_root, _BMODE_PATH, "2d_brightness_mode")
    doppler_count = _record_color_doppler_frame_count(record, data_root)
    if bmode_count <= 0 or doppler_count <= 0:
        return False
    ratio = float(bmode_count) / float(doppler_count)
    ratio_min, ratio_max = _ordered_float_pair(frame_ratio_range, name="frame_ratio_range")
    if ratio < ratio_min or ratio > ratio_max:
        return False
    fps = record.sample_rate_hz("2d_brightness_mode")
    if fps is None:
        timestamps = open_recording(
            record,
            root=data_root,
            cache_dir=recording_cache_dir,
            cache_read_only=recording_cache_read_only,
            cache_include=recording_cache_include,
            cache_exclude=recording_cache_exclude,
        ).load_timestamps(_BMODE_PATH)
        fps = _fps_from_timestamps(timestamps)
    fps_min, fps_max = _ordered_float_pair(fps_range, name="fps_range")
    if fps is None or fps < fps_min or fps > fps_max:
        return False
    if not _record_has_required_geometries(
        record,
        data_root=data_root,
        recording_cache_dir=recording_cache_dir,
        recording_cache_read_only=recording_cache_read_only,
        recording_cache_include=recording_cache_include,
        recording_cache_exclude=recording_cache_exclude,
    ):
        return False
    accepted_nyquist_range = _optional_ordered_float_pair(nyquist_mps_range, name="nyquist_mps_range")
    if accepted_nyquist_range is None:
        return True
    nyquist = _record_color_nyquist_mps(
        record,
        data_root=data_root,
        recording_cache_dir=recording_cache_dir,
        recording_cache_read_only=recording_cache_read_only,
        recording_cache_include=recording_cache_include,
        recording_cache_exclude=recording_cache_exclude,
    )
    if nyquist is None:
        return False
    # A recording whose scale changes is kept when any calibrated frame is in range; its clips are then restricted
    # to frames in range (see _valid_clip_starts), so no sample is conditioned on a borrowed scale.
    nyquist_min, nyquist_max = accepted_nyquist_range
    values = np.atleast_1d(np.asarray(nyquist, dtype=np.float64))
    return bool(np.any((values >= nyquist_min) & (values <= nyquist_max)))


def _record_fails_color_metadata_filters(
    record: RecordingRecord,
    *,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
) -> bool:
    bmode_count = record.frame_count("2d_brightness_mode")
    doppler_count = _record_color_doppler_frame_count_from_metadata(record)
    if bmode_count is not None and bmode_count <= 0:
        return True
    if doppler_count is not None and doppler_count <= 0:
        return True
    if bmode_count is not None and doppler_count is not None:
        ratio = float(bmode_count) / float(doppler_count)
        ratio_min, ratio_max = _ordered_float_pair(frame_ratio_range, name="frame_ratio_range")
        if ratio < ratio_min or ratio > ratio_max:
            return True
    fps = record.sample_rate_hz("2d_brightness_mode")
    if fps is not None:
        fps_min, fps_max = _ordered_float_pair(fps_range, name="fps_range")
        if fps < fps_min or fps > fps_max:
            return True
    return False


def _record_has_required_geometries(
    record: RecordingRecord,
    *,
    data_root: str | Path | None,
    recording_cache_dir: str | Path | None = None,
    recording_cache_read_only: bool = False,
    recording_cache_include: tuple[str, ...] = (),
    recording_cache_exclude: tuple[str, ...] = (),
) -> bool:
    try:
        store = open_recording(
            record,
            root=data_root,
            cache_dir=recording_cache_dir,
            cache_read_only=recording_cache_read_only,
            cache_include=recording_cache_include,
            cache_exclude=recording_cache_exclude,
        )
        # Stream headers carry the typed metadata and geometry without loading the samples.
        streams = (
            store.load_stream_header(_BMODE_PATH).stream,
            store.load_stream_header(_VELOCITY_PATH).stream,
            store.load_stream_header(_POWER_PATH).stream,
        )
    except (KeyError, FileNotFoundError, TypeError, ValueError):
        return False
    return all(isinstance(stream.metadata.geometry, SectorGeometry) for stream in streams)


def _record_color_doppler_frame_count(record: RecordingRecord, data_root: str | Path | None) -> int:
    count = _record_color_doppler_frame_count_from_metadata(record)
    if count is not None:
        return int(count)
    velocity_count = _record_frame_count(record, data_root, _VELOCITY_PATH, "2d_color_doppler_velocity")
    power_count = _record_frame_count(record, data_root, _POWER_PATH, "2d_color_doppler_power")
    return min(velocity_count, power_count)


def _record_color_doppler_frame_count_from_metadata(record: RecordingRecord) -> int | None:
    count = record.frame_count("2d_color_doppler")
    if count is not None:
        return int(count)
    velocity_count = record.frame_count("2d_color_doppler_velocity")
    power_count = record.frame_count("2d_color_doppler_power")
    if velocity_count is None or power_count is None:
        return None
    return min(int(velocity_count), int(power_count))


def _record_frame_count(
    record: RecordingRecord,
    data_root: str | Path | None,
    data_path: str,
    content_type: str,
) -> int:
    count = record.frame_count(content_type)
    if count is not None:
        return int(count)
    return two_dimensional_frame_count(record, data_root, data_path)


def _record_color_nyquist_mps(
    record: RecordingRecord,
    *,
    data_root: str | Path | None,
    recording_cache_dir: str | Path | None = None,
    recording_cache_read_only: bool = False,
    recording_cache_include: tuple[str, ...] = (),
    recording_cache_exclude: tuple[str, ...] = (),
) -> float | np.ndarray | None:
    """Return the recording's Nyquist scale: one value, or the calibrated per-frame values when it changes."""
    header = open_recording(
        record,
        root=data_root,
        cache_dir=recording_cache_dir,
        cache_read_only=recording_cache_read_only,
        cache_include=recording_cache_include,
        cache_exclude=recording_cache_exclude,
    ).load_stream_header(_VELOCITY_PATH)
    if not isinstance(header.stream, ColorDopplerVelocityStream):
        return None
    if header.velocity_scale_mps_per_frame is not None:
        scales = np.asarray(header.velocity_scale_mps_per_frame, dtype=np.float64)
        calibrated = scales[np.isfinite(scales)]
        return calibrated if calibrated.size else None
    value = header.stream.metadata.velocity_limit_mps
    return None if value is None else _positive_float_or_raise(value, key="velocity_limit_mps")


def _temporal_mode_from_frame_counts(*, bmode_count: int, doppler_count: int) -> str:
    ratio = int(bmode_count) / max(1, int(doppler_count))
    return "two_per_interval" if ratio <= 0.75 else "repeat_one_per_interval"


def _doppler_slice_for_bmode_clip(*, ref: SampleRef, temporal_mode: str) -> slice:
    interval_count = max(0, int(ref.bmode_stop) - int(ref.bmode_start) - 1)
    if temporal_mode == "two_per_interval":
        start = int(ref.bmode_start) * _TEMPORAL_UPSAMPLE_FACTOR
        return slice(start, start + interval_count * _TEMPORAL_UPSAMPLE_FACTOR)
    if temporal_mode == "repeat_one_per_interval":
        start = int(ref.bmode_start)
        return slice(start, start + interval_count)
    raise ValueError(f"Unsupported Color Doppler temporal mode: {temporal_mode}")


def _required_geometry(
    stream: BrightnessModeStream | ColorDopplerVelocityStream | ColorDopplerPowerStream,
    *,
    role: str,
) -> SectorGeometry:
    geometry = stream.metadata.geometry
    if not isinstance(geometry, SectorGeometry):
        raise ValueError(f"{role} stream {stream.data_path} is missing sector geometry")
    return geometry


def _target_timestamps_from_bmode(*, bmode_timestamps: np.ndarray, target_count: int, fps: float) -> np.ndarray:
    timestamps = np.asarray(bmode_timestamps, dtype=np.float32).reshape(-1)
    if timestamps.size >= 2 and int(target_count) == (timestamps.size - 1) * _TEMPORAL_UPSAMPLE_FACTOR:
        values: list[float] = []
        for start, stop in zip(timestamps[:-1], timestamps[1:]):
            if not (np.isfinite(start) and np.isfinite(stop)):
                break
            delta = float(stop - start)
            if delta <= 0.0:
                break
            for offset in range(_TEMPORAL_UPSAMPLE_FACTOR):
                fraction = (float(offset) + 0.5) / float(_TEMPORAL_UPSAMPLE_FACTOR)
                values.append(float(start) + fraction * delta)
        if len(values) == int(target_count):
            return np.asarray(values, dtype=np.float32)
    rate = float(fps) * float(_TEMPORAL_UPSAMPLE_FACTOR)
    if not np.isfinite(rate) or rate <= 0.0:
        rate = 1.0
    return np.arange(int(target_count), dtype=np.float32) / rate


def _clip_nyquist(frame_nyquist: np.ndarray, ref: SampleRef) -> float:
    values = np.asarray(frame_nyquist, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values != values[0]):
        raise ValueError(f"{ref.sample_id} spans Color Doppler frames without one calibrated Nyquist scale")
    return float(values[0])


def _positive_float_or_raise(value: object, *, key: str) -> float:
    out = float(cast(Any, value))
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{key} must be a positive finite float, got {value!r}")
    return out


def _config_float(data_config: Mapping[str, object], key: str, default: float) -> float:
    value = data_config.get(key, default)
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data.{key} must be a finite float, got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"data.{key} must be a finite float, got {value!r}")
    return result


def _required_array_paths() -> tuple[str, str, str]:
    return (_BMODE_PATH, _VELOCITY_PATH, _POWER_PATH)
