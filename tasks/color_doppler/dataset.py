from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from echoxflow import (
    BrightnessModeStream,
    ColorDopplerPowerStream,
    ColorDopplerVelocityStream,
    RecordingRecord,
    find_recordings,
    open_recording,
)
from echoxflow.scan import CartesianGrid, SectorGeometry, sector_to_cartesian
from tasks.color_doppler.types import Sample
from tasks.utils.dataset import SharedTrainingTransform, SkippedCase
from tasks.utils.dataset import as_spatial_shape as _as_spatial_shape
from tasks.utils.dataset import build_task_dataloaders, discover_records_from_candidates
from tasks.utils.dataset import fps_from_timestamps as _fps_from_timestamps
from tasks.utils.dataset import local_std_3x3 as _local_std_3x3
from tasks.utils.dataset import minmax_normalize as _minmax_normalize
from tasks.utils.dataset import optional_int as _optional_int
from tasks.utils.dataset import optional_ordered_float_pair as _optional_ordered_float_pair
from tasks.utils.dataset import ordered_float_pair as _ordered_float_pair
from tasks.utils.dataset import preprocess_bmode_frames as _preprocess_bmode_frames
from tasks.utils.dataset import progress_iter
from tasks.utils.dataset import resample_to_reference_geometry as _resample_to_reference_geometry
from tasks.utils.dataset import resize_stack as _resize_stack
from tasks.utils.dataset import sample_indices_by_case as _sample_indices_by_case
from tasks.utils.dataset import shared_training_transform_from_config
from tasks.utils.dataset import slice_timestamps as _slice_timestamps
from tasks.utils.dataset import training_transform_from_args as _training_transform_from_args
from tasks.utils.dataset import two_dimensional_frame_count
from tasks.utils.dataset import velocity_limit_from_stream as _velocity_limit_from_stream

__all__ = [
    "RawDataset",
    "SampleRef",
    "SkippedCase",
    "build_dataloaders",
    "discover_case_dirs",
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
        self.recording_cache_dir = None if recording_cache_dir is None else Path(recording_cache_dir)
        self.recording_cache_read_only = bool(recording_cache_read_only)
        self.recording_cache_include = tuple(str(path) for path in recording_cache_include) or (
            (_BMODE_PATH, _VELOCITY_PATH, _POWER_PATH) if self.recording_cache_dir is not None else ()
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
        )
        self.sample_indices_by_case = _sample_indices_by_case(self.sample_refs)

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, index: int) -> Sample:
        return self.sample_from_ref(self.sample_refs[index])

    def sample_from_ref(self, ref: SampleRef) -> Sample:
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
        bmode_frames = bmode.to_float().data
        fps = ref.record.sample_rate_hz("2d_brightness_mode") or _fps_from_timestamps(bmode.timestamps) or 0.0
        if self.coordinate_space == "cartesian":
            frames = _cartesian_bmode_stack(
                bmode_frames,
                reference=bmode,
                transform=self.training_transform,
                cartesian_height=self.cartesian_height,
                output_shape=self.input_spatial_shape,
            )
        else:
            frames = _preprocess_bmode_frames(
                bmode_frames,
                transform=self.training_transform,
                input_spatial_shape=self.input_spatial_shape,
            )
        velocity_frames = np.asarray(velocity.data, dtype=np.float32)
        power_frames = np.asarray(power.data, dtype=np.float32)
        if temporal_mode == "repeat_one_per_interval":
            velocity_frames = np.repeat(velocity_frames, _TEMPORAL_UPSAMPLE_FACTOR, axis=0)
            power_frames = np.repeat(power_frames, _TEMPORAL_UPSAMPLE_FACTOR, axis=0)
        expected_target_time = _TEMPORAL_UPSAMPLE_FACTOR * (int(bmode_frames.shape[0]) - 1)
        if int(velocity_frames.shape[0]) != expected_target_time:
            raise ValueError(
                f"{ref.sample_id} has {velocity_frames.shape[0]} color targets and {bmode_frames.shape[0]} "
                f"B-mode frames; expected {expected_target_time} color targets"
            )
        if self.coordinate_space == "cartesian":
            velocity_values, velocity_region = _cartesian_target_stack(
                velocity_frames,
                source=velocity,
                reference=bmode,
                cartesian_height=self.cartesian_height,
                target_shape=self.target_spatial_shape,
            )
            power_values, power_region = _cartesian_target_stack(
                power_frames,
                source=power,
                reference=bmode,
                cartesian_height=self.cartesian_height,
                target_shape=self.target_spatial_shape,
            )
        else:
            velocity_values, velocity_region = _resample_target_stack(
                velocity_frames,
                source=velocity,
                reference=bmode,
                target_shape=self.target_spatial_shape,
            )
            power_values, power_region = _resample_target_stack(
                power_frames,
                source=power,
                reference=bmode,
                target_shape=self.target_spatial_shape,
            )
        power_values = np.clip(power_values, 0.0, 1.0)
        color_region = _combined_color_region(
            velocity_region=velocity_region,
            power_region=power_region,
            target_shape=self.target_spatial_shape,
            target_count=int(velocity_values.shape[0]),
        )
        velocity_std = _local_std_3x3(velocity_values)
        target = np.stack([velocity_values, power_values, velocity_std], axis=1).astype(np.float32)
        valid_mask = cast(np.ndarray, np.isfinite(target).all(axis=1, keepdims=True).astype(np.float32))
        if color_region is None:
            valid_mask *= np.any(target != 0.0, axis=1, keepdims=True).astype(np.float32)
        color_box_values = _color_box_target(
            color_region=color_region,
            fallback_valid_mask=np.asarray(valid_mask[:, 0], dtype=np.float32),
            target_count=int(target.shape[0]),
            target_shape=self.target_spatial_shape,
        )
        bmode_on_target = _bmode_interval_stack_for_target_grid(
            bmode_frames=frames if self.coordinate_space == "cartesian" else bmode_frames,
            target_shape=self.target_spatial_shape,
        )
        velocity_loss_mask_values = _velocity_loss_mask(
            bmode=bmode_on_target,
            power=power_values,
            valid_mask=np.asarray(valid_mask[:, 0], dtype=np.float32),
            bmode_threshold=self.velocity_loss_bmode_threshold,
            power_threshold=self.velocity_loss_power_threshold,
            mask_floor=self.velocity_loss_mask_floor,
        )
        velocity_loss_mask = velocity_loss_mask_values[:, None]
        nyquist = _velocity_limit(velocity)
        frame_timestamps = _slice_timestamps(bmode.timestamps, 0, int(bmode_frames.shape[0]))
        target_timestamps = _target_timestamps_from_bmode(
            bmode_timestamps=frame_timestamps,
            target_count=expected_target_time,
            fps=fps,
        )
        metric_target = None
        metric_valid_mask = None
        metric_velocity_loss_mask = None
        metric_color_box_target = None
        metric_sample_grid = None
        if self.include_cartesian_metrics:
            (
                metric_target,
                metric_valid_mask,
                metric_velocity_loss_mask,
                metric_color_box_target,
                metric_sample_grid,
            ) = self._cartesian_metric_values(
                bmode=bmode,
                bmode_frames=bmode_frames,
                velocity=velocity,
                velocity_frames=velocity_frames,
                power=power,
                power_frames=power_frames,
            )
        return Sample(
            frames=torch.from_numpy(np.transpose(frames[:, None, :, :], (1, 0, 2, 3))[None, ...]),
            frame_timestamps=torch.from_numpy(frame_timestamps[None, :]),
            target_timestamps=torch.from_numpy(target_timestamps[None, :]),
            conditioning=torch.tensor([[fps, nyquist]], dtype=torch.float32),
            doppler_target=torch.from_numpy(np.transpose(target, (1, 0, 2, 3))[None, ...]),
            color_box_target=torch.from_numpy(np.transpose(color_box_values[:, None], (1, 0, 2, 3))[None, ...]),
            valid_mask=torch.from_numpy(np.transpose(valid_mask, (1, 0, 2, 3))[None, ...]),
            velocity_loss_mask=torch.from_numpy(np.transpose(velocity_loss_mask, (1, 0, 2, 3))[None, ...]),
            nyquist_mps=torch.tensor([nyquist], dtype=torch.float32),
            sample_id=ref.sample_id,
            record=ref.record,
            data_root=self.data_root,
            clip_start=ref.bmode_start,
            clip_stop=ref.bmode_stop,
            coordinate_space=self.coordinate_space,
            cartesian_metric_doppler_target=(
                None
                if metric_target is None
                else torch.from_numpy(np.transpose(metric_target, (1, 0, 2, 3))[None, ...])
            ),
            cartesian_metric_valid_mask=(
                None
                if metric_valid_mask is None
                else torch.from_numpy(np.transpose(metric_valid_mask, (1, 0, 2, 3))[None, ...])
            ),
            cartesian_metric_velocity_loss_mask=(
                None
                if metric_velocity_loss_mask is None
                else torch.from_numpy(np.transpose(metric_velocity_loss_mask, (1, 0, 2, 3))[None, ...])
            ),
            cartesian_metric_color_box_target=(
                None
                if metric_color_box_target is None
                else torch.from_numpy(np.transpose(metric_color_box_target, (1, 0, 2, 3))[None, ...])
            ),
            cartesian_metric_sample_grid=(
                None if metric_sample_grid is None else torch.from_numpy(metric_sample_grid[None, ...])
            ),
        )

    def _cartesian_metric_values(
        self,
        *,
        bmode: BrightnessModeStream,
        bmode_frames: np.ndarray,
        velocity: ColorDopplerVelocityStream,
        velocity_frames: np.ndarray,
        power: ColorDopplerPowerStream,
        power_frames: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        geometry = _required_geometry(bmode, role="B-mode reference")
        grid = CartesianGrid.from_sector_height(geometry, self.cartesian_height)
        metric_shape = grid.shape
        velocity_values, velocity_region = _cartesian_target_stack(
            velocity_frames,
            source=velocity,
            reference=bmode,
            cartesian_height=self.cartesian_height,
            target_shape=metric_shape,
        )
        power_values, power_region = _cartesian_target_stack(
            power_frames,
            source=power,
            reference=bmode,
            cartesian_height=self.cartesian_height,
            target_shape=metric_shape,
        )
        power_values = np.clip(power_values, 0.0, 1.0)
        color_region = _combined_color_region(
            velocity_region=velocity_region,
            power_region=power_region,
            target_shape=metric_shape,
            target_count=int(velocity_values.shape[0]),
        )
        velocity_std = _local_std_3x3(velocity_values)
        target = np.stack([velocity_values, power_values, velocity_std], axis=1).astype(np.float32)
        valid_mask = cast(np.ndarray, np.isfinite(target).all(axis=1, keepdims=True).astype(np.float32))
        if color_region is None:
            valid_mask *= np.any(target != 0.0, axis=1, keepdims=True).astype(np.float32)
        color_box = _color_box_target(
            color_region=color_region,
            fallback_valid_mask=np.asarray(valid_mask[:, 0], dtype=np.float32),
            target_count=int(target.shape[0]),
            target_shape=metric_shape,
        )
        cartesian_bmode = _cartesian_bmode_stack(
            bmode_frames,
            reference=bmode,
            transform=None,
            cartesian_height=self.cartesian_height,
            output_shape=metric_shape,
        )
        bmode_on_target = _bmode_interval_stack_for_target_grid(
            bmode_frames=cartesian_bmode,
            target_shape=metric_shape,
        )
        velocity_loss_mask = _velocity_loss_mask(
            bmode=bmode_on_target,
            power=power_values,
            valid_mask=np.asarray(valid_mask[:, 0], dtype=np.float32),
            bmode_threshold=self.velocity_loss_bmode_threshold,
            power_threshold=self.velocity_loss_power_threshold,
            mask_floor=self.velocity_loss_mask_floor,
        )[:, None]
        if self.coordinate_space == "beamspace":
            sample_grid = _cartesian_grid_sample_grid(
                geometry,
                grid,
                source_shape=self.target_spatial_shape,
                output_shape=metric_shape,
            )
        else:
            sample_grid = _cartesian_image_sample_grid(grid)
        return target, valid_mask, velocity_loss_mask, color_box[:, None], sample_grid


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
) -> list[RecordingRecord]:
    found = find_recordings(
        root=root_dir,
        array_paths=_required_array_paths(),
        require_all=True,
        predicate=lambda record: _color_metadata_discovery_predicate(
            record,
            frame_ratio_range=frame_ratio_range,
            fps_range=fps_range,
            nyquist_mps_range=nyquist_mps_range,
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
        ),
    )


def _color_discovery_predicate(
    record: RecordingRecord,
    *,
    data_root: str | Path,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
) -> bool:
    if not _color_metadata_discovery_predicate(
        record,
        frame_ratio_range=frame_ratio_range,
        fps_range=fps_range,
        nyquist_mps_range=nyquist_mps_range,
    ):
        return False
    return _color_runtime_discovery_predicate(
        record,
        data_root=data_root,
        frame_ratio_range=frame_ratio_range,
        fps_range=fps_range,
        nyquist_mps_range=nyquist_mps_range,
    )


def _color_metadata_discovery_predicate(
    record: RecordingRecord,
    *,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
) -> bool:
    if any("tissue" in str(content_type).lower() for content_type in record.content_types):
        return False
    return not _record_fails_color_metadata_filters(
        record,
        frame_ratio_range=frame_ratio_range,
        fps_range=fps_range,
        nyquist_mps_range=nyquist_mps_range,
    )


def _color_runtime_discovery_predicate(
    record: RecordingRecord,
    *,
    data_root: str | Path,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
) -> bool:
    if not _uses_color_filters(
        frame_ratio_range=frame_ratio_range,
        fps_range=fps_range,
        nyquist_mps_range=nyquist_mps_range,
    ):
        return True
    return _record_matches_color_filters(
        record,
        data_root=data_root,
        frame_ratio_range=frame_ratio_range,
        fps_range=fps_range,
        nyquist_mps_range=nyquist_mps_range,
    )


def discover_case_dirs(*, root_dir: str | Path, max_cases: int | None = None) -> list[Path]:
    return [record.path(root_dir) for record in discover_records(root_dir=root_dir, max_cases=max_cases)]


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
) -> list[SampleRef]:
    refs: list[SampleRef] = []
    interval_count = int(clip_length)
    input_frame_count = interval_count + 1
    min_interval_count = max(interval_count, int(min_frames_per_case))
    for record in progress_iter(
        records,
        description="Scanning color Doppler samples",
        total=len(records),
        unit="case",
    ):
        if _uses_color_filters(
            frame_ratio_range=frame_ratio_range,
            fps_range=fps_range,
            nyquist_mps_range=nyquist_mps_range,
        ) and not _record_matches_color_filters(
            record,
            data_root=data_root,
            frame_ratio_range=frame_ratio_range,
            fps_range=fps_range,
            nyquist_mps_range=nyquist_mps_range,
            recording_cache_dir=recording_cache_dir,
            recording_cache_read_only=recording_cache_read_only,
            recording_cache_include=recording_cache_include,
            recording_cache_exclude=recording_cache_exclude,
        ):
            continue
        usable_bmode_count = _usable_bmode_count(record=record, data_root=data_root)
        if usable_bmode_count - 1 < min_interval_count:
            continue
        last_start = usable_bmode_count - input_frame_count + 1
        for start in range(0, last_start, max(1, int(clip_stride))):
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
    nyquist_min, nyquist_max = accepted_nyquist_range
    if nyquist < nyquist_min or nyquist > nyquist_max:
        return False
    return True


def _record_fails_color_metadata_filters(
    record: RecordingRecord,
    *,
    frame_ratio_range: tuple[float, float],
    fps_range: tuple[float, float],
    nyquist_mps_range: tuple[float, float] | None,
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
    accepted_nyquist_range = _optional_ordered_float_pair(nyquist_mps_range, name="nyquist_mps_range")
    if accepted_nyquist_range is not None:
        nyquist = _record_color_nyquist_mps_from_metadata(record)
        if nyquist is not None:
            nyquist_min, nyquist_max = accepted_nyquist_range
            if nyquist < nyquist_min or nyquist > nyquist_max:
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
        streams = (
            store.load_stream(_BMODE_PATH),
            store.load_stream(_VELOCITY_PATH),
            store.load_stream(_POWER_PATH),
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
) -> float | None:
    value = _record_color_nyquist_mps_from_metadata(record)
    if value is not None:
        return value
    stream = open_recording(
        record,
        root=data_root,
        cache_dir=recording_cache_dir,
        cache_read_only=recording_cache_read_only,
        cache_include=recording_cache_include,
        cache_exclude=recording_cache_exclude,
    ).load_stream(_VELOCITY_PATH)
    if not isinstance(stream, ColorDopplerVelocityStream):
        return None
    value = stream.metadata.velocity_limit_mps
    return None if value is None else _positive_float_or_raise(value, key="velocity_limit_mps")


def _record_color_nyquist_mps_from_metadata(record: RecordingRecord) -> float | None:
    raw = dict(record.raw or {})
    for key in (
        "nyquist_mps",
        "velocity_limit_mps",
        "color_doppler_nyquist_mps",
        "color_doppler_velocity_limit_mps",
        "recordings/nyquist_mps",
        "recordings/velocity_limit_mps",
        "recordings/color_doppler_nyquist_mps",
        "recordings/color_doppler_velocity_limit_mps",
    ):
        value = _optional_positive_float(raw, key)
        if value is not None:
            return value
    for dict_key in (
        "nyquist_mps_by_content_type",
        "velocity_limit_mps_by_content_type",
        "recordings/nyquist_mps_by_content_type",
        "recordings/velocity_limit_mps_by_content_type",
    ):
        mapping = raw.get(dict_key)
        if not isinstance(mapping, Mapping):
            continue
        for content_key in ("2d_color_doppler", "2d_color_doppler_velocity", "color_doppler"):
            value = _optional_positive_float(mapping, content_key)
            if value is not None:
                return value
    return None


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


def _resample_target_stack(
    frames: np.ndarray,
    *,
    source: ColorDopplerVelocityStream | ColorDopplerPowerStream,
    reference: BrightnessModeStream,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    source_geometry = _required_geometry(source, role="Color Doppler source")
    reference_geometry = _required_geometry(reference, role="B-mode reference")
    return cast(
        tuple[np.ndarray, np.ndarray | None],
        _resample_to_reference_geometry(
            frames,
            source_geometry=source_geometry,
            reference_geometry=reference_geometry,
            target_shape=target_shape,
            return_region_mask=True,
        ),
    )


def _cartesian_bmode_stack(
    frames: np.ndarray,
    *,
    reference: BrightnessModeStream,
    transform: SharedTrainingTransform | None,
    cartesian_height: int,
    output_shape: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    del transform
    reference_geometry = _required_geometry(reference, role="B-mode reference")
    converted, _mask = _sector_stack_to_cartesian_values(
        values,
        geometry=reference_geometry,
        cartesian_height=cartesian_height,
        interpolation="linear",
    )
    return _minmax_normalize(_resize_stack(converted, output_shape))


def _cartesian_target_stack(
    frames: np.ndarray,
    *,
    source: ColorDopplerVelocityStream | ColorDopplerPowerStream,
    reference: BrightnessModeStream,
    cartesian_height: int,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    source_geometry = _required_geometry(source, role="Color Doppler source")
    reference_geometry = _required_geometry(reference, role="B-mode reference")
    grid = CartesianGrid.from_sector_height(reference_geometry, int(cartesian_height))
    converted, mask = _sector_stack_to_cartesian_values(
        np.asarray(frames, dtype=np.float32),
        geometry=source_geometry,
        cartesian_height=cartesian_height,
        interpolation="linear",
        grid=grid,
    )
    resized = _resize_stack(converted, target_shape)
    resized_mask = None if mask is None else _resize_stack(mask.astype(np.float32), target_shape) > 0.5
    return resized, resized_mask


def _required_geometry(
    stream: BrightnessModeStream | ColorDopplerVelocityStream | ColorDopplerPowerStream,
    *,
    role: str,
) -> SectorGeometry:
    geometry = stream.metadata.geometry
    if not isinstance(geometry, SectorGeometry):
        raise ValueError(f"{role} stream {stream.data_path} is missing sector geometry")
    return geometry


def _sector_stack_to_cartesian_values(
    frames: np.ndarray,
    *,
    geometry: SectorGeometry | None,
    cartesian_height: int,
    interpolation: str,
    grid: CartesianGrid | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if geometry is None:
        return np.asarray(frames, dtype=np.float32), None
    output_grid = grid or CartesianGrid.from_sector_height(geometry, int(cartesian_height))
    converted = tuple(
        sector_to_cartesian(frame, geometry, grid=output_grid, interpolation=interpolation) for frame in frames
    )
    values = np.stack([image.data for image in converted], axis=0).astype(np.float32, copy=False)
    mask = np.asarray(converted[0].mask, dtype=bool)
    return values, np.broadcast_to(mask[None], (int(values.shape[0]), *mask.shape)).copy()


def _cartesian_grid_sample_grid(
    geometry: SectorGeometry,
    grid: CartesianGrid,
    *,
    source_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> np.ndarray:
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    xs = np.linspace(grid.x_range_m[0], grid.x_range_m[1], out_w, dtype=np.float64)
    ys = np.linspace(grid.y_range_m[0], grid.y_range_m[1], out_h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    radii = np.sqrt(xx**2 + yy**2)
    angles = np.arctan2(xx, yy)
    rows = (radii - geometry.depth_start_m) / geometry.depth_span_m * max(0, src_h - 1)
    cols = (angles - geometry.angle_start_rad) / geometry.angle_span_rad * max(0, src_w - 1)
    x = cols / max(1, src_w - 1) * 2.0 - 1.0
    y = rows / max(1, src_h - 1) * 2.0 - 1.0
    return np.stack([x, y], axis=-1).astype(np.float32)


def _cartesian_image_sample_grid(grid: CartesianGrid) -> np.ndarray:
    out_h, out_w = grid.shape
    x = np.linspace(-1.0, 1.0, out_w, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, out_h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1).astype(np.float32)


def _combined_color_region(
    *,
    velocity_region: np.ndarray | None,
    power_region: np.ndarray | None,
    target_shape: tuple[int, int],
    target_count: int | None = None,
) -> np.ndarray | None:
    if velocity_region is None or power_region is None:
        return None
    region = np.asarray(velocity_region, dtype=bool) & np.asarray(power_region, dtype=bool)
    if region.shape != target_shape:
        expected_temporal_shape = None if target_count is None else (int(target_count), *target_shape)
        if expected_temporal_shape is None or region.shape != expected_temporal_shape:
            expected = (
                target_shape if expected_temporal_shape is None else f"{target_shape} or {expected_temporal_shape}"
            )
            raise ValueError(f"Color Doppler region mask shape {region.shape} does not match target shape {expected}")
    return cast(np.ndarray, region)


def _color_box_target(
    *,
    color_region: np.ndarray | None,
    fallback_valid_mask: np.ndarray,
    target_count: int,
    target_shape: tuple[int, int],
) -> np.ndarray:
    if color_region is not None:
        region = np.asarray(color_region, dtype=np.float32)
        if region.shape == (int(target_count), *target_shape):
            return region.astype(np.float32, copy=True)
        if region.shape != target_shape:
            raise ValueError(
                f"Color Doppler box mask shape {region.shape} does not match target shape "
                f"{target_shape} or {(int(target_count), *target_shape)}"
            )
        return np.broadcast_to(region[None, :, :], (int(target_count), *target_shape)).astype(np.float32, copy=True)
    fallback = np.asarray(fallback_valid_mask, dtype=np.float32)
    if fallback.shape != (int(target_count), *target_shape):
        raise ValueError(
            f"Color Doppler fallback mask shape {fallback.shape} does not match "
            f"{(int(target_count), *target_shape)}"
        )
    return np.asarray(fallback > 0.5, dtype=np.float32)


def _bmode_interval_stack_for_target_grid(
    *,
    bmode_frames: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    arr = _minmax_normalize(np.asarray(bmode_frames, dtype=np.float32))
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError(f"B-mode clip must contain at least two frames, got {arr.shape}")
    intervals = 0.5 * (arr[:-1] + arr[1:])
    repeated = np.repeat(intervals, _TEMPORAL_UPSAMPLE_FACTOR, axis=0)
    return _resize_stack(repeated, target_shape)


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


def _velocity_loss_mask(
    *,
    bmode: np.ndarray,
    power: np.ndarray,
    valid_mask: np.ndarray,
    bmode_threshold: float,
    power_threshold: float,
    mask_floor: float,
) -> np.ndarray:
    finite_valid = (
        np.isfinite(bmode)
        & np.isfinite(power)
        & np.isfinite(valid_mask)
        & (np.asarray(valid_mask, dtype=np.float32) > 0.5)
    )
    selected = (
        finite_valid
        & (np.asarray(bmode, dtype=np.float32) < float(bmode_threshold))
        & (np.asarray(power, dtype=np.float32) > float(power_threshold))
    )
    weights = np.zeros_like(np.asarray(valid_mask, dtype=np.float32), dtype=np.float32)
    weights[finite_valid] = float(mask_floor)
    weights[selected] = 1.0
    return weights


def _velocity_limit(stream: ColorDopplerVelocityStream) -> float:
    return _velocity_limit_from_stream(stream)


def _positive_float_or_raise(value: object, *, key: str) -> float:
    out = float(cast(Any, value))
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{key} must be a positive finite float, got {value!r}")
    return out


def _optional_positive_float(values: Mapping[str, object], key: str) -> float | None:
    if key not in values:
        return None
    return _positive_float_or_raise(values[key], key=key)


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
