from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from echoxflow import BrightnessModeStream, RecordingRecord, TissueDopplerFloatStream, find_recordings, open_recording
from echoxflow.scan import CartesianGrid, SectorGeometry
from echoxflow.schema import current_array_candidates
from echoxflow.timing import run_ids, within_one_run
from tasks.tissue_doppler.types import DenseSectorSample as Sample
from tasks.utils.dataset import (
    SharedTrainingTransform,
    SkippedCase,
    _config_bool,
    build_task_dataloaders,
    discover_records_from_candidates,
    progress_iter,
    shared_training_transform_from_config,
)
from tasks.utils.dataset import as_spatial_shape as _as_spatial_shape
from tasks.utils.dataset import cartesian_grid_sample_grid as _cartesian_grid_sample_grid
from tasks.utils.dataset import cartesian_image_sample_grid as _cartesian_image_sample_grid
from tasks.utils.dataset import cartesian_sector_mask as _cartesian_sector_mask
from tasks.utils.dataset import optional_int as _optional_int
from tasks.utils.dataset import optional_ordered_float_pair as _optional_ordered_float_pair
from tasks.utils.dataset import sample_indices_by_case as _sample_indices_by_case
from tasks.utils.dataset import selected_timestamps as _selected_timestamps
from tasks.utils.dataset import slice_timestamps as _slice_timestamps
from tasks.utils.dataset import spatial_shape_from_config as _spatial_shape_from_config
from tasks.utils.dataset import timestamps_or_index as _timestamps_or_index
from tasks.utils.dataset import training_transform_from_args as _training_transform_from_args
from tasks.utils.dataset import velocity_limit_from_stream as _velocity_limit_from_stream
from tasks.utils.deferred import DeferredSample
from tasks.utils.device_sampling import (
    Interpolation,
    as_device_tensor,
    cartesian_sampling_grid,
    sample_stack,
    sector_sampling_grid,
)
from tasks.utils.device_sampling import minmax_normalize as _minmax_normalize
from tasks.utils.device_sampling import resize_stack as _resize_stack
from tasks.utils.ref_cache import DEFAULT_REF_CACHE_DIR, RecordingScanCache, scan_cache_dir

BMODE_PATH = "data/2d_brightness_mode"
TDI_PATH = "data/tissue_doppler"
CM_PER_METER = 100.0
# 4: clips stay on one strictly increasing B-mode and Doppler clock run and on one calibrated velocity scale.
SAMPLE_REF_CACHE_VERSION = 4
DEFAULT_SAMPLE_REF_CACHE_DIR = DEFAULT_REF_CACHE_DIR


@dataclass(frozen=True)
class SampleRef:
    record: RecordingRecord
    clip_start: int
    clip_stop: int
    doppler_indices: tuple[int, ...]
    sample_id: str
    fps: float | None = None


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
        alignment_slack_factor: float = 1.5,
        doppler_interval_assignment: str = "release_strict",
        sampling_mode: str = "sliding",
        full_case_max_frames: int | None = None,
        accepted_fps_range: tuple[float, float] | None = None,
        coordinate_space: str = "beamspace",
        cartesian_height: int | None = None,
        include_cartesian_metrics: bool = False,
        training_transform: SharedTrainingTransform | None = None,
        recording_cache_dir: str | Path | None = None,
        recording_cache_read_only: bool = False,
        recording_cache_include: tuple[str, ...] = (),
        recording_cache_exclude: tuple[str, ...] = (),
        sample_ref_cache_dir: str | Path | None = DEFAULT_SAMPLE_REF_CACHE_DIR,
        sample_ref_cache_read_only: bool = False,
        sample_ref_cache_version: int = SAMPLE_REF_CACHE_VERSION,
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
        self.alignment_slack_factor = float(alignment_slack_factor)
        self.doppler_interval_assignment = str(doppler_interval_assignment).strip().lower()
        if self.doppler_interval_assignment != "release_strict":
            raise ValueError(
                f"doppler_interval_assignment must be 'release_strict', got {doppler_interval_assignment!r}"
            )
        self.sampling_mode = str(sampling_mode)
        self.full_case_max_frames = None if full_case_max_frames is None else int(full_case_max_frames)
        self.accepted_fps_range = _optional_ordered_float_pair(accepted_fps_range, name="accepted_fps_range")
        self.coordinate_space = str(coordinate_space).strip().lower()
        if self.coordinate_space not in {"beamspace", "cartesian"}:
            raise ValueError(
                f"Tissue Doppler coordinate_space must be 'beamspace' or 'cartesian', got {coordinate_space!r}"
            )
        self.cartesian_height = (
            int(cartesian_height) if cartesian_height is not None else int(self.input_spatial_shape[0])
        )
        self.include_cartesian_metrics = bool(include_cartesian_metrics)
        self.device_preprocessing = bool(device_preprocessing)
        self.recording_cache_dir = None if recording_cache_dir is None else Path(recording_cache_dir)
        self.recording_cache_read_only = bool(recording_cache_read_only)
        self.recording_cache_include = tuple(str(path) for path in recording_cache_include) or (
            _cacheable_array_paths(BMODE_PATH, TDI_PATH) if self.recording_cache_dir is not None else ()
        )
        self.recording_cache_exclude = tuple(str(path) for path in recording_cache_exclude)
        self.sample_ref_cache_dir = None if sample_ref_cache_dir is None else Path(sample_ref_cache_dir)
        self.sample_ref_cache_read_only = bool(sample_ref_cache_read_only)
        self.sample_ref_cache_version = int(sample_ref_cache_version)
        self.training_transform = _training_transform_from_args(training_transform=training_transform)
        self.skipped_cases: list[SkippedCase] = []
        self.sample_refs = _build_sample_refs(
            records=self.records,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            min_frames_per_case=self.min_frames_per_case,
            data_root=data_root,
            max_samples=max_samples,
            alignment_slack_factor=self.alignment_slack_factor,
            doppler_interval_assignment=self.doppler_interval_assignment,
            sampling_mode=self.sampling_mode,
            full_case_max_frames=self.full_case_max_frames,
            accepted_fps_range=self.accepted_fps_range,
            recording_cache_dir=self.recording_cache_dir,
            recording_cache_read_only=self.recording_cache_read_only,
            recording_cache_include=self.recording_cache_include,
            recording_cache_exclude=self.recording_cache_exclude,
            sample_ref_cache_dir=self.sample_ref_cache_dir,
            sample_ref_cache_read_only=self.sample_ref_cache_read_only,
            sample_ref_cache_version=self.sample_ref_cache_version,
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
        bmode = store.load_stream_slice(BMODE_PATH, ref.clip_start, ref.clip_stop)
        if not isinstance(bmode, BrightnessModeStream):
            raise TypeError(f"{BMODE_PATH} must be BrightnessModeStream")
        tdi_indices = np.asarray(ref.doppler_indices, dtype=np.int64)
        target_count = int(tdi_indices.size)
        tdi_slice_start = int(np.min(tdi_indices)) if target_count else 0
        tdi_slice_stop = int(np.max(tdi_indices)) + 1 if target_count else 0
        tdi = store.load_modality_slice(TDI_PATH, tdi_slice_start, tdi_slice_stop).stream
        if not isinstance(tdi, TissueDopplerFloatStream):
            raise TypeError(f"{TDI_PATH} must load as TissueDopplerFloatStream")
        bmode_frames = np.asarray(bmode.data)
        local_stop = int(bmode_frames.shape[0])
        local_tdi_indices = tdi_indices - tdi_slice_start
        if local_stop != target_count + 1:
            raise ValueError(
                f"Dense sector sample {ref.sample_id} has {local_stop} B-mode frames and "
                f"{target_count} Doppler frames; expected exactly one extra B-mode frame"
            )
        frame_limits = _selected_velocity_scales(tdi, local_tdi_indices)
        sector_limit = (
            _velocity_limit_from_stream(tdi) if frame_limits is None else _clip_velocity_limit(frame_limits, ref)
        )
        radial_m_per_bin = (
            _cartesian_y_m_per_bin(
                reference=bmode,
                cartesian_height=self.cartesian_height,
                image_height=self.target_spatial_shape[0],
            )
            if self.coordinate_space == "cartesian"
            else _radial_m_per_bin(reference=bmode, image_height=self.target_spatial_shape[0])
        )
        frame_timestamps = _slice_timestamps(bmode.timestamps, 0, local_stop)
        fps = _resolve_sample_fps(
            ref=ref,
            clip_timestamps=frame_timestamps,
            accepted_fps_range=self.accepted_fps_range,
        )
        bmode_geometry = bmode.metadata.geometry
        tdi_geometry = tdi.metadata.geometry
        return _RawClip(
            bmode=bmode_frames,
            bmode_geometry=bmode_geometry if isinstance(bmode_geometry, SectorGeometry) else None,
            tdi=np.asarray(tdi.data[local_tdi_indices], dtype=np.float32),
            tdi_geometry=tdi_geometry if isinstance(tdi_geometry, SectorGeometry) else None,
            sector_limit=float(sector_limit),
            velocity_scale=float(fps * radial_m_per_bin),
            frame_limits=None if frame_limits is None else np.asarray(frame_limits, dtype=np.float32),
            frame_timestamps=np.asarray(frame_timestamps),
            doppler_timestamps=np.asarray(_selected_timestamps(tdi.timestamps, local_tdi_indices)),
            sample_id=ref.sample_id,
            record=ref.record,
            data_root=self.data_root,
            clip_start=ref.clip_start,
            clip_stop=ref.clip_stop,
        )


@dataclass(frozen=True)
class _RawClip:
    """Native-resolution arrays of one Tissue Doppler clip, as loaded by a data-loader worker."""

    bmode: np.ndarray
    bmode_geometry: SectorGeometry | None
    tdi: np.ndarray
    tdi_geometry: SectorGeometry | None
    sector_limit: float
    velocity_scale: float
    frame_limits: np.ndarray | None
    frame_timestamps: np.ndarray
    doppler_timestamps: np.ndarray
    sample_id: str
    record: RecordingRecord
    data_root: str | Path | None
    clip_start: int
    clip_stop: int


@dataclass(frozen=True)
class _Finisher:
    """Builds a :class:`Sample` from a :class:`_RawClip` on any torch device."""

    coordinate_space: str
    cartesian_height: int
    input_spatial_shape: tuple[int, int]
    target_spatial_shape: tuple[int, int]
    include_cartesian_metrics: bool

    def finish(self, raw: _RawClip, device: torch.device) -> Sample:
        bmode = as_device_tensor(raw.bmode, device).to(dtype=torch.float32) / 255.0
        tdi = as_device_tensor(raw.tdi, device)
        if self.coordinate_space == "cartesian":
            converted = _to_cartesian(bmode, raw.bmode_geometry, self.cartesian_height, interpolation="linear")
            frames = _minmax_normalize(_resize_stack(converted, self.input_spatial_shape))
            target = self._cartesian_target(tdi, raw, self.target_spatial_shape)
        else:
            frames = _minmax_normalize(_resize_stack(bmode, self.input_spatial_shape))
            target = self._beamspace_target(tdi, raw)
        # The Cartesian target is zero fill outside the scan sector. Train on the sector pixels only, like the metric,
        # so both domains learn from the same tissue region (beamspace targets have no out-of-sector pixels).
        loss_valid_mask = None
        target_geometry = raw.tdi_geometry or raw.bmode_geometry
        if self.coordinate_space == "cartesian" and target_geometry is not None:
            _sampling, mask = cartesian_sampling_grid(
                target_geometry,
                CartesianGrid.from_sector_height(target_geometry, int(self.cartesian_height)),
                source_shape=(int(tdi.shape[1]), int(tdi.shape[2])),
                device=device,
            )
            loss_valid_mask = _resize_stack(
                mask[None].to(dtype=torch.float32), self.target_spatial_shape, interpolation="nearest"
            )[0]
        metric_target = None
        metric_sample_grid = None
        metric_valid_mask = None
        if self.include_cartesian_metrics:
            geometry = raw.bmode_geometry
            metric_shape = self.target_spatial_shape
            grid = None
            if geometry is not None:
                grid = CartesianGrid.from_sector_height(geometry, int(self.cartesian_height))
                metric_shape = grid.shape
            metric_target = self._cartesian_target(tdi, raw, metric_shape)
            # The Cartesian target is filled with zeros outside the scan sector. Score only the sector, using the
            # geometry and grid that the target was converted with.
            if target_geometry is not None:
                metric_valid_mask = as_device_tensor(
                    _cartesian_sector_mask(
                        target_geometry,
                        CartesianGrid.from_sector_height(target_geometry, int(self.cartesian_height)),
                        shape=(int(metric_target.shape[-2]), int(metric_target.shape[-1])),
                    ).astype(np.float32, copy=False),
                    device,
                )
            if grid is not None and geometry is not None:
                if self.coordinate_space == "beamspace":
                    sample_grid = _cartesian_grid_sample_grid(
                        geometry,
                        grid,
                        source_shape=self.target_spatial_shape,
                        output_shape=metric_shape,
                    )
                else:
                    sample_grid = _cartesian_image_sample_grid(grid)
                metric_sample_grid = as_device_tensor(sample_grid, device)
        target_count = int(target.shape[0])
        return Sample(
            frames=frames[None, None],
            frame_timestamps=as_device_tensor(raw.frame_timestamps[None, :], device),
            doppler_timestamps=as_device_tensor(raw.doppler_timestamps[None, :], device),
            doppler_target=target[None, None],
            marker_target=torch.zeros(
                (1, 0, target_count, *self.target_spatial_shape), dtype=torch.float32, device=device
            ),
            sector_velocity_limit_mps=torch.tensor([raw.sector_limit], dtype=torch.float32, device=device),
            velocity_scale_mps_per_px_frame=torch.tensor([raw.velocity_scale], dtype=torch.float32, device=device),
            sample_id=raw.sample_id,
            record=raw.record,
            data_root=raw.data_root,
            clip_start=raw.clip_start,
            clip_stop=raw.clip_stop,
            coordinate_space=self.coordinate_space,
            cartesian_metric_doppler_target=None if metric_target is None else metric_target[None, None],
            cartesian_metric_sample_grid=None if metric_sample_grid is None else metric_sample_grid[None],
            cartesian_metric_valid_mask=None if metric_valid_mask is None else metric_valid_mask[None, None, None],
            loss_valid_mask=None if loss_valid_mask is None else loss_valid_mask[None, None, None],
            doppler_velocity_limit_mps_per_frame=(
                None if raw.frame_limits is None else as_device_tensor(raw.frame_limits[None, :], device)
            ),
        )

    def _beamspace_target(self, tdi: Tensor, raw: _RawClip) -> Tensor:
        if raw.tdi_geometry is None or raw.bmode_geometry is None:
            return _resize_stack(tdi, self.target_spatial_shape, interpolation="nearest")
        sampling, mask = sector_sampling_grid(
            raw.tdi_geometry,
            replace(raw.bmode_geometry, grid_shape=self.target_spatial_shape),
            source_shape=(int(tdi.shape[1]), int(tdi.shape[2])),
            output_shape=self.target_spatial_shape,
            device=tdi.device,
        )
        return sample_stack(tdi, sampling, mask, interpolation="nearest")

    def _cartesian_target(self, tdi: Tensor, raw: _RawClip, shape: tuple[int, int]) -> Tensor:
        geometry = raw.tdi_geometry or raw.bmode_geometry
        converted = _to_cartesian(tdi, geometry, self.cartesian_height, interpolation="nearest")
        return _resize_stack(converted, shape, interpolation="nearest")


def _to_cartesian(
    frames: Tensor, geometry: SectorGeometry | None, cartesian_height: int, *, interpolation: Interpolation
) -> Tensor:
    """Scan-convert ``[T,R,A]`` frames onto the ``cartesian_height`` grid of their own sector (as-is without one)."""
    if geometry is None:
        return frames.to(dtype=torch.float32)
    grid = CartesianGrid.from_sector_height(geometry, int(cartesian_height))
    sampling, mask = cartesian_sampling_grid(
        geometry, grid, source_shape=(int(frames.shape[1]), int(frames.shape[2])), device=frames.device
    )
    return sample_stack(frames, sampling, mask, interpolation=interpolation)


def discover_records(
    *,
    root_dir: str | Path,
    max_cases: int | None = None,
    exam_ids: Iterable[str] | None = None,
    sample_fraction: object | None = None,
    seed: int | None = None,
) -> list[RecordingRecord]:
    found = find_recordings(root=root_dir, array_paths=(BMODE_PATH, TDI_PATH), require_all=True)
    return discover_records_from_candidates(
        found,
        exam_ids=exam_ids,
        max_cases=max_cases,
        sample_fraction=sample_fraction,
        seed=seed,
        description="Scanning tissue Doppler recordings",
    )


def _dataset_kwargs(_config: Any, data_config: Mapping[str, object], root: str | Path) -> dict[str, object]:
    return {
        "clip_length": int(cast(Any, data_config.get("clip_length", 32))),
        "clip_stride": int(cast(Any, data_config.get("clip_stride", 1))),
        "min_frames_per_case": int(cast(Any, data_config.get("min_frames_per_case", 32))),
        "input_spatial_shape": _spatial_shape_from_config(data_config.get("input_spatial_shape", (128, 128))),
        "target_spatial_shape": _spatial_shape_from_config(data_config.get("target_spatial_shape", (64, 64))),
        "data_root": root,
        "alignment_slack_factor": float(cast(Any, data_config.get("alignment_slack_factor", 1.5))),
        "doppler_interval_assignment": str(data_config.get("doppler_interval_assignment", "release_strict")),
        "accepted_fps_range": _optional_ordered_float_pair(
            data_config.get("accepted_fps_range"), name="accepted_fps_range"
        ),
        "coordinate_space": str(data_config.get("coordinate_space", "beamspace")),
        "cartesian_height": _optional_int(data_config.get("cartesian_height")),
        "device_preprocessing": _config_bool(data_config.get("device_preprocessing", False)),
        "sample_ref_cache_dir": _sample_ref_cache_dir(data_config),
        "sample_ref_cache_read_only": _config_bool(data_config.get("sample_ref_cache_read_only", False)),
        "sample_ref_cache_version": int(
            cast(Any, data_config.get("sample_ref_cache_version", SAMPLE_REF_CACHE_VERSION))
        ),
    }


def _train_dataset_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    return {
        "sampling_mode": str(data_config.get("train_sampling_mode", data_config.get("sampling_mode", "sliding"))),
        "full_case_max_frames": _optional_int(data_config.get("train_full_case_max_frames")),
        "training_transform": shared_training_transform_from_config(data_config),
    }


def _val_dataset_kwargs(data_config: Mapping[str, object]) -> dict[str, object]:
    return {
        "sampling_mode": str(data_config.get("val_sampling_mode", data_config.get("sampling_mode", "sliding"))),
        "full_case_max_frames": _optional_int(data_config.get("val_full_case_max_frames")),
        "include_cartesian_metrics": True,
    }


def _train_batch_size(data_config: Mapping[str, object]) -> int | None:
    return _batch_size_for_sampling_mode(
        data_config.get("train_batch_size", data_config.get("batch_size")),
        sampling_mode=str(data_config.get("train_sampling_mode", data_config.get("sampling_mode", "sliding"))),
        name="data.train_batch_size",
    )


def _val_batch_size(data_config: Mapping[str, object]) -> int | None:
    return _batch_size_for_sampling_mode(
        data_config.get("val_batch_size", data_config.get("batch_size")),
        sampling_mode=str(data_config.get("val_sampling_mode", data_config.get("sampling_mode", "sliding"))),
        name="data.val_batch_size",
    )


build_dataloaders = partial(
    build_task_dataloaders,
    task_name="tissue_doppler",
    dataset_cls=RawDataset,
    discover_records=discover_records,
    empty_message="Tissue Doppler dataset is empty",
    dataset_kwargs_fn=_dataset_kwargs,
    train_dataset_kwargs_fn=_train_dataset_kwargs,
    val_dataset_kwargs_fn=_val_dataset_kwargs,
    train_batch_size_fn=_train_batch_size,
    val_batch_size_fn=_val_batch_size,
)


def _batch_size_for_sampling_mode(value: object, *, sampling_mode: str, name: str) -> int | None:
    del sampling_mode, name
    batch_size = _optional_int(value)
    return batch_size


def _build_sample_refs(
    *,
    records: list[RecordingRecord],
    clip_length: int,
    clip_stride: int,
    min_frames_per_case: int,
    data_root: str | Path | None,
    max_samples: int | None,
    alignment_slack_factor: float,
    doppler_interval_assignment: str,
    sampling_mode: str,
    full_case_max_frames: int | None,
    accepted_fps_range: tuple[float, float] | None,
    recording_cache_dir: str | Path | None = None,
    recording_cache_read_only: bool = False,
    recording_cache_include: tuple[str, ...] = (),
    recording_cache_exclude: tuple[str, ...] = (),
    sample_ref_cache_dir: str | Path | None = DEFAULT_SAMPLE_REF_CACHE_DIR,
    sample_ref_cache_read_only: bool = False,
    sample_ref_cache_version: int = SAMPLE_REF_CACHE_VERSION,
) -> list[SampleRef]:
    """Sample references of ``records`` in record order, from per-recording scans cached across cells.

    Each recording's references depend only on the recording and the sampling settings, so they are cached per
    recording (see :mod:`tasks.utils.ref_cache`) and shared by every fold, split and method.
    """
    mode = str(sampling_mode).strip().lower()
    if mode not in {"sliding", "full_case"}:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
    cache = RecordingScanCache(
        sample_ref_cache_dir,
        namespace="tissue_doppler_sample_refs",
        settings={
            "version": int(sample_ref_cache_version),
            "data_paths": {"bmode": BMODE_PATH, "tissue_doppler": TDI_PATH},
            "clip_length": int(clip_length),
            "clip_stride": int(clip_stride),
            "min_frames_per_case": int(min_frames_per_case),
            "alignment_slack_factor": float(alignment_slack_factor),
            "doppler_interval_assignment": str(doppler_interval_assignment).strip().lower(),
            "sampling_mode": mode,
            "full_case_max_frames": None if full_case_max_frames is None else int(full_case_max_frames),
            "accepted_fps_range": (
                None if accepted_fps_range is None else [float(accepted_fps_range[0]), float(accepted_fps_range[1])]
            ),
        },
        data_root=data_root,
        read_only=sample_ref_cache_read_only,
    )

    def scan(record: RecordingRecord) -> list[list[Any]]:
        return _scan_record_sample_refs(
            record,
            clip_length=clip_length,
            clip_stride=clip_stride,
            min_frames_per_case=min_frames_per_case,
            data_root=data_root,
            alignment_slack_factor=alignment_slack_factor,
            doppler_interval_assignment=doppler_interval_assignment,
            sampling_mode=mode,
            full_case_max_frames=full_case_max_frames,
            accepted_fps_range=accepted_fps_range,
            recording_cache_dir=recording_cache_dir,
            recording_cache_read_only=recording_cache_read_only,
            recording_cache_include=recording_cache_include,
            recording_cache_exclude=recording_cache_exclude,
        )

    refs: list[SampleRef] = []
    for record in progress_iter(
        records,
        description="Scanning tissue Doppler records",
        total=len(records),
        unit="case",
    ):
        for clip_start, clip_stop, doppler_indices, sample_id, fps in cache.get(record, scan):
            refs.append(
                SampleRef(
                    record=record,
                    clip_start=int(clip_start),
                    clip_stop=int(clip_stop),
                    doppler_indices=tuple(int(value) for value in doppler_indices),
                    sample_id=str(sample_id),
                    fps=None if fps is None else float(fps),
                )
            )
            if max_samples is not None and len(refs) >= int(max_samples):
                return refs
    return refs


def _scan_record_sample_refs(
    record: RecordingRecord,
    *,
    clip_length: int,
    clip_stride: int,
    min_frames_per_case: int,
    data_root: str | Path | None,
    alignment_slack_factor: float,
    doppler_interval_assignment: str,
    sampling_mode: str,
    full_case_max_frames: int | None,
    accepted_fps_range: tuple[float, float] | None,
    recording_cache_dir: str | Path | None = None,
    recording_cache_read_only: bool = False,
    recording_cache_include: tuple[str, ...] = (),
    recording_cache_exclude: tuple[str, ...] = (),
) -> list[list[Any]]:
    """``[clip_start, clip_stop, doppler_indices, sample_id, fps]`` of every clip of one recording.

    Only timestamps, array shapes and stream metadata are read (stream headers), not the B-mode or Doppler samples.
    """
    refs: list[list[Any]] = []
    mode = str(sampling_mode).strip().lower()
    store = open_recording(
        record,
        root=data_root,
        cache_dir=recording_cache_dir,
        cache_read_only=recording_cache_read_only,
        cache_include=recording_cache_include,
        cache_exclude=recording_cache_exclude,
    )
    bmode = store.load_stream_header(BMODE_PATH)
    try:
        tdi = store.load_stream_header(TDI_PATH, display=True)
    except ValueError as exc:
        # Some recordings ship tissue Doppler without the velocity limit needed to scale it.
        logging.getLogger(__name__).warning(
            "skipping %s/%s: tissue Doppler is not loadable (%s)", record.exam_id, record.recording_id, exc
        )
        return refs
    if not isinstance(bmode.stream, BrightnessModeStream):
        raise TypeError(f"{BMODE_PATH} must be BrightnessModeStream")
    if not isinstance(tdi.stream, TissueDopplerFloatStream):
        raise TypeError(f"{TDI_PATH} must load as TissueDopplerFloatStream")
    bmode_timestamps = _timestamps_or_index(bmode.timestamps, int(bmode.shape[0]))
    doppler_timestamps = _timestamps_or_index(tdi.timestamps, int(tdi.shape[0]))
    clip_check = _ClipCheck(
        bmode_runs=run_ids(bmode_timestamps),
        doppler_runs=run_ids(doppler_timestamps),
        doppler_scales=tdi.velocity_scale_mps_per_frame,
    )
    fps = _fps_from_mean_step(_mean_step(bmode_timestamps))
    if not _fps_is_accepted(bmode_timestamps=bmode_timestamps, accepted_fps_range=accepted_fps_range):
        return refs
    required_frames = (
        int(min_frames_per_case) + 1 if mode == "full_case" else max(int(clip_length) + 1, int(min_frames_per_case) + 1)
    )
    if len(bmode_timestamps) < required_frames or len(doppler_timestamps) < required_frames - 1:
        return refs
    tolerance = _alignment_tolerance(
        bmode_timestamps=bmode_timestamps,
        doppler_timestamps=doppler_timestamps,
        slack_factor=alignment_slack_factor,
    )
    if mode == "full_case":
        clip_start, clip_stop = _full_case_clip_bounds(
            frame_count=len(bmode_timestamps),
            max_frames=full_case_max_frames,
        )
        clip_timestamps = bmode_timestamps[clip_start:clip_stop]
        doppler_indices = _contained_doppler_indices(
            bmode_timestamps=clip_timestamps,
            doppler_timestamps=doppler_timestamps,
            tolerance=tolerance,
            assignment=doppler_interval_assignment,
        )
        if doppler_indices is None or len(doppler_indices) != max(0, len(clip_timestamps) - 1):
            return refs
        if clip_check.reject_reason(clip_start, clip_stop, doppler_indices) is not None:
            _log_skipped_clips(record, clip_check)
            return refs
        sample_id = f"{record.exam_id}/{record.recording_id}:clip_full"
        refs.append([clip_start, clip_stop, list(doppler_indices), sample_id, fps])
        return refs
    input_clip_length = int(clip_length) + 1
    last_start = len(bmode_timestamps) - input_clip_length + 1
    for clip_start in range(0, last_start, max(1, int(clip_stride))):
        clip_stop = clip_start + input_clip_length
        clip_timestamps = bmode_timestamps[clip_start:clip_stop]
        doppler_indices = _contained_doppler_indices(
            bmode_timestamps=clip_timestamps,
            doppler_timestamps=doppler_timestamps,
            tolerance=tolerance,
            assignment=doppler_interval_assignment,
        )
        if doppler_indices is None or len(doppler_indices) != int(clip_length):
            continue
        if clip_check.reject_reason(clip_start, clip_stop, doppler_indices) is not None:
            continue
        sample_id = f"{record.exam_id}/{record.recording_id}:clip_{clip_start}_{clip_stop}"
        refs.append([clip_start, clip_stop, list(doppler_indices), sample_id, fps])
    _log_skipped_clips(record, clip_check)
    return refs


class _ClipCheck:
    """Rejects clips that cross a clock reset or a change of the exported Doppler velocity scale.

    A clip's B-mode frames and its Doppler targets must each lie on one strictly increasing clock run, so timestamp
    alignment never compares times from different source segments. Its Doppler targets must share one calibrated
    scale, so the per-sample Nyquist limit used by the wrapped losses and metrics is exact for every target frame.
    """

    def __init__(self, *, bmode_runs: np.ndarray, doppler_runs: np.ndarray, doppler_scales: np.ndarray | None) -> None:
        self.bmode_runs = bmode_runs
        self.doppler_runs = doppler_runs
        self.doppler_scales = None if doppler_scales is None else np.asarray(doppler_scales, dtype=np.float64)
        self.rejected: dict[str, int] = {}

    def reject_reason(self, clip_start: int, clip_stop: int, doppler_indices: Iterable[int]) -> str | None:
        indices = np.asarray(tuple(doppler_indices), dtype=np.int64)
        reason = None
        if not within_one_run(np.arange(int(clip_start), int(clip_stop)), self.bmode_runs):
            reason = "bmode_clock_break"
        elif not within_one_run(indices, self.doppler_runs):
            reason = "doppler_clock_break"
        elif self.doppler_scales is not None:
            scales = self.doppler_scales[indices]
            if not np.all(np.isfinite(scales)):
                reason = "uncalibrated_doppler_frame"
            elif np.any(scales != scales[0]):
                reason = "velocity_scale_change"
        if reason is not None:
            self.rejected[reason] = self.rejected.get(reason, 0) + 1
        return reason


def _log_skipped_clips(record: RecordingRecord, check: _ClipCheck) -> None:
    if check.rejected:
        logging.getLogger(__name__).info(
            "%s/%s: skipped tissue Doppler clips %s", record.exam_id, record.recording_id, check.rejected
        )


def _selected_velocity_scales(stream: TissueDopplerFloatStream, indices: np.ndarray) -> np.ndarray | None:
    """Exported velocity scale of each selected target frame, or ``None`` for streams without per-frame calibration."""
    if stream.metadata.velocity_scale_mps_per_frame is None:
        return None
    scales = stream.metadata.velocity_scales_mps(int(stream.data.shape[0]))
    return None if scales is None else np.asarray(scales, dtype=np.float64)[np.asarray(indices, dtype=np.int64)]


def _clip_velocity_limit(scales: np.ndarray, ref: SampleRef) -> float:
    if scales.size == 0 or not np.all(np.isfinite(scales)) or np.any(scales != scales[0]):
        raise ValueError(f"Dense sector sample {ref.sample_id} spans frames without one calibrated velocity scale")
    return float(scales[0])


def _cacheable_array_paths(*paths: str) -> tuple[str, ...]:
    """Requested array paths plus the stored arrays that satisfy them, so the recording cache matches either name."""
    return tuple(dict.fromkeys(candidate for path in paths for candidate in current_array_candidates(path)))


def _sample_ref_cache_dir(data_config: Mapping[str, object]) -> Path | None:
    return scan_cache_dir(data_config)


def _contained_doppler_indices(
    *,
    bmode_timestamps: np.ndarray,
    doppler_timestamps: np.ndarray,
    tolerance: float,
    assignment: str = "release_strict",
) -> tuple[int, ...] | None:
    bmode = np.asarray(bmode_timestamps, dtype=np.float64)
    doppler = np.asarray(doppler_timestamps, dtype=np.float64)
    if len(bmode) < 2:
        return None
    if str(assignment).strip().lower() != "release_strict":
        raise ValueError(f"Unsupported Doppler interval assignment: {assignment!r}")
    indices: list[int] = []
    padding = max(0.0, min(float(tolerance), np.finfo(np.float64).eps * 16.0))
    for interval_index, (left, right) in enumerate(zip(bmode[:-1], bmode[1:])):
        lo, hi = (float(left), float(right)) if left <= right else (float(right), float(left))
        if hi < lo:
            return None
        if interval_index == len(bmode) - 2:
            mask = (doppler >= lo - padding) & (doppler <= hi + padding)
        else:
            mask = (doppler >= lo - padding) & (doppler < hi - padding)
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return None
        midpoint = 0.5 * (lo + hi)
        indices.append(int(candidates[int(np.argmin(np.abs(doppler[candidates] - midpoint)))]))
    return tuple(indices)


def _alignment_tolerance(*, bmode_timestamps: np.ndarray, doppler_timestamps: np.ndarray, slack_factor: float) -> float:
    return max(_median_step(bmode_timestamps), _median_step(doppler_timestamps)) * max(1.0, float(slack_factor))


def _full_case_clip_bounds(*, frame_count: int, max_frames: int | None) -> tuple[int, int]:
    total = int(frame_count)
    if max_frames is None or max_frames <= 0 or total <= int(max_frames):
        return 0, total
    window = int(max_frames)
    start = max(0, (total - window) // 2)
    return start, start + window


def _median_step(timestamps: np.ndarray) -> float:
    if len(timestamps) <= 1:
        return 0.0
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    positive = diffs[diffs > 0.0]
    return 0.0 if positive.size == 0 else float(np.median(positive))


def _mean_step(timestamps: np.ndarray) -> float:
    if len(timestamps) <= 1:
        return 0.0
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    positive = diffs[diffs > 0.0]
    return 0.0 if positive.size == 0 else float(np.mean(positive))


def _fps_from_mean_step(mean_dt: float) -> float:
    if not np.isfinite(mean_dt) or mean_dt <= 0.0:
        return 1.0
    return float(1.0 / mean_dt)


def _fps_is_accepted(*, bmode_timestamps: np.ndarray, accepted_fps_range: tuple[float, float] | None) -> bool:
    if accepted_fps_range is None:
        return True
    fps = _fps_from_mean_step(_mean_step(bmode_timestamps))
    return float(accepted_fps_range[0]) <= fps <= float(accepted_fps_range[1])


def _resolve_sample_fps(
    *,
    ref: SampleRef,
    clip_timestamps: np.ndarray,
    accepted_fps_range: tuple[float, float] | None,
) -> float:
    fps = ref.fps
    if fps is None:
        fps = _fps_from_mean_step(_mean_step(np.asarray(clip_timestamps, dtype=np.float64)))
        if accepted_fps_range is not None and not (
            float(accepted_fps_range[0]) <= float(fps) <= float(accepted_fps_range[1])
        ):
            raise ValueError(
                f"Dense sector sample {ref.sample_id} has timestamp-derived FPS {float(fps):.6g} "
                f"outside accepted range [{accepted_fps_range[0]:.6g}, {accepted_fps_range[1]:.6g}]"
            )
    if not np.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError(f"Dense sector sample {ref.sample_id} has invalid FPS {fps!r}")
    return float(fps)


def _radial_m_per_bin(*, reference: BrightnessModeStream, image_height: int) -> float:
    geometry = reference.metadata.geometry
    if geometry is None:
        raise ValueError("Missing B-mode geometry for Tissue Doppler physical radial scale")
    depth_span_m = float(geometry.depth_span_m)
    if not np.isfinite(depth_span_m) or depth_span_m <= 0.0:
        raise ValueError(f"Invalid B-mode depth span: {depth_span_m}")
    if int(image_height) <= 0:
        raise ValueError(f"Invalid image height for radial spacing: {image_height}")
    return depth_span_m / float(image_height)


def _cartesian_y_m_per_bin(*, reference: BrightnessModeStream, cartesian_height: int, image_height: int) -> float:
    geometry = reference.metadata.geometry
    if geometry is None:
        raise ValueError("Missing B-mode geometry for Tissue Doppler cartesian physical scale")
    grid = CartesianGrid.from_sector_height(geometry, int(cartesian_height))
    span = abs(float(grid.y_range_m[1]) - float(grid.y_range_m[0]))
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(f"Invalid cartesian y span: {span}")
    if int(image_height) <= 0:
        raise ValueError(f"Invalid image height for cartesian spacing: {image_height}")
    return span / float(image_height)
