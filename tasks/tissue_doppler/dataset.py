from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from echoxflow import BrightnessModeStream, RecordingRecord, TissueDopplerFloatStream, find_recordings, open_recording
from echoxflow.scan import CartesianGrid, SectorGeometry, sector_to_cartesian
from tasks.tissue_doppler.types import DenseSectorSample as Sample
from tasks.utils.dataset import SharedTrainingTransform, SkippedCase
from tasks.utils.dataset import as_spatial_shape as _as_spatial_shape
from tasks.utils.dataset import build_task_dataloaders, discover_records_from_candidates
from tasks.utils.dataset import minmax_normalize as _minmax_normalize
from tasks.utils.dataset import optional_int as _optional_int
from tasks.utils.dataset import optional_ordered_float_pair as _optional_ordered_float_pair
from tasks.utils.dataset import preprocess_bmode_frames as _preprocess_bmode_frames
from tasks.utils.dataset import progress_iter
from tasks.utils.dataset import resample_to_reference_geometry as _resample_to_reference_geometry
from tasks.utils.dataset import resize_stack as _resize_stack
from tasks.utils.dataset import sample_indices_by_case as _sample_indices_by_case
from tasks.utils.dataset import selected_timestamps as _selected_timestamps
from tasks.utils.dataset import shared_training_transform_from_config
from tasks.utils.dataset import slice_timestamps as _slice_timestamps
from tasks.utils.dataset import spatial_shape_from_config as _spatial_shape_from_config
from tasks.utils.dataset import training_transform_from_args as _training_transform_from_args
from tasks.utils.dataset import velocity_limit_from_stream as _velocity_limit_from_stream

BMODE_PATH = "data/2d_brightness_mode"
TDI_PATH = "data/tissue_doppler"
CM_PER_METER = 100.0


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
        self.recording_cache_dir = None if recording_cache_dir is None else Path(recording_cache_dir)
        self.recording_cache_read_only = bool(recording_cache_read_only)
        self.recording_cache_include = tuple(str(path) for path in recording_cache_include) or (
            (BMODE_PATH, TDI_PATH) if self.recording_cache_dir is not None else ()
        )
        self.recording_cache_exclude = tuple(str(path) for path in recording_cache_exclude)
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
            sampling_mode=self.sampling_mode,
            full_case_max_frames=self.full_case_max_frames,
            accepted_fps_range=self.accepted_fps_range,
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
        frames_np = bmode.to_float().data
        if self.coordinate_space == "cartesian":
            frames = _cartesian_bmode_stack(
                frames_np,
                reference=bmode,
                transform=self.training_transform,
                cartesian_height=self.cartesian_height,
                output_shape=self.input_spatial_shape,
            )
        else:
            frames = _preprocess_bmode_frames(
                frames_np,
                transform=self.training_transform,
                input_spatial_shape=self.input_spatial_shape,
            )
        local_start = 0
        local_stop = int(frames_np.shape[0])
        local_tdi_indices = tdi_indices - tdi_slice_start
        if self.coordinate_space == "cartesian":
            target = _cartesian_target_stack(
                np.asarray(tdi.data[local_tdi_indices], dtype=np.float32),
                source=tdi,
                reference=bmode,
                cartesian_height=self.cartesian_height,
                target_shape=self.target_spatial_shape,
            )
        else:
            target = _resample_target_stack(
                np.asarray(tdi.data[local_tdi_indices], dtype=np.float32),
                source=tdi,
                reference=bmode,
                target_shape=self.target_spatial_shape,
            )
        if int(frames.shape[0]) != int(target.shape[0]) + 1:
            raise ValueError(
                f"Dense sector sample {ref.sample_id} has {frames.shape[0]} B-mode frames and "
                f"{target.shape[0]} Doppler frames; expected exactly one extra B-mode frame"
            )
        sector_limit = _velocity_limit(tdi)
        radial_m_per_bin = (
            _cartesian_y_m_per_bin(
                reference=bmode,
                cartesian_height=self.cartesian_height,
                image_height=self.target_spatial_shape[0],
            )
            if self.coordinate_space == "cartesian"
            else _radial_m_per_bin(reference=bmode, image_height=self.target_spatial_shape[0])
        )
        fps = _resolve_sample_fps(
            ref=ref,
            clip_timestamps=_slice_timestamps(bmode.timestamps, local_start, local_stop),
            accepted_fps_range=self.accepted_fps_range,
        )
        velocity_scale = fps * radial_m_per_bin
        metric_target = None
        metric_sample_grid = None
        if self.include_cartesian_metrics:
            geometry = bmode.metadata.geometry
            metric_shape = self.target_spatial_shape
            grid = None
            if isinstance(geometry, SectorGeometry):
                grid = CartesianGrid.from_sector_height(geometry, int(self.cartesian_height))
                metric_shape = grid.shape
            metric_target = _cartesian_target_stack(
                np.asarray(tdi.data[local_tdi_indices], dtype=np.float32),
                source=tdi,
                reference=bmode,
                cartesian_height=self.cartesian_height,
                target_shape=metric_shape,
            )
            if grid is not None and isinstance(geometry, SectorGeometry):
                if self.coordinate_space == "beamspace":
                    metric_sample_grid = _cartesian_grid_sample_grid(
                        geometry,
                        grid,
                        source_shape=self.target_spatial_shape,
                        output_shape=metric_shape,
                    )
                else:
                    metric_sample_grid = _cartesian_image_sample_grid(grid)
        return Sample(
            frames=torch.from_numpy(np.transpose(frames[:, None, :, :], (1, 0, 2, 3))[None, ...]),
            frame_timestamps=torch.from_numpy(_slice_timestamps(bmode.timestamps, local_start, local_stop)[None, :]),
            doppler_timestamps=torch.from_numpy(_selected_timestamps(tdi.timestamps, local_tdi_indices)[None, :]),
            doppler_target=torch.from_numpy(np.transpose(target[:, None, :, :], (1, 0, 2, 3))[None, ...]),
            marker_target=torch.zeros((1, 0, target_count, *self.target_spatial_shape), dtype=torch.float32),
            sector_velocity_limit_mps=torch.tensor([sector_limit], dtype=torch.float32),
            velocity_scale_mps_per_px_frame=torch.tensor([velocity_scale], dtype=torch.float32),
            sample_id=ref.sample_id,
            record=ref.record,
            data_root=self.data_root,
            clip_start=ref.clip_start,
            clip_stop=ref.clip_stop,
            coordinate_space=self.coordinate_space,
            cartesian_metric_doppler_target=(
                None
                if metric_target is None
                else torch.from_numpy(np.transpose(metric_target[:, None, :, :], (1, 0, 2, 3))[None, ...])
            ),
            cartesian_metric_sample_grid=(
                None if metric_sample_grid is None else torch.from_numpy(metric_sample_grid[None, ...])
            ),
        )


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


def discover_case_dirs(*, root_dir: str | Path, max_cases: int | None = None) -> list[Path]:
    return [record.path(root_dir) for record in discover_records(root_dir=root_dir, max_cases=max_cases)]


def _dataset_kwargs(_config: Any, data_config: Mapping[str, object], root: str | Path) -> dict[str, object]:
    return {
        "clip_length": int(cast(Any, data_config.get("clip_length", 32))),
        "clip_stride": int(cast(Any, data_config.get("clip_stride", 1))),
        "min_frames_per_case": int(cast(Any, data_config.get("min_frames_per_case", 32))),
        "input_spatial_shape": _spatial_shape_from_config(data_config.get("input_spatial_shape", (128, 128))),
        "target_spatial_shape": _spatial_shape_from_config(data_config.get("target_spatial_shape", (64, 64))),
        "data_root": root,
        "alignment_slack_factor": float(cast(Any, data_config.get("alignment_slack_factor", 1.5))),
        "accepted_fps_range": _optional_ordered_float_pair(
            data_config.get("accepted_fps_range"), name="accepted_fps_range"
        ),
        "coordinate_space": str(data_config.get("coordinate_space", "beamspace")),
        "cartesian_height": _optional_int(data_config.get("cartesian_height")),
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
    sampling_mode: str,
    full_case_max_frames: int | None,
    accepted_fps_range: tuple[float, float] | None,
    recording_cache_dir: str | Path | None = None,
    recording_cache_read_only: bool = False,
    recording_cache_include: tuple[str, ...] = (),
    recording_cache_exclude: tuple[str, ...] = (),
) -> list[SampleRef]:
    refs: list[SampleRef] = []
    mode = str(sampling_mode).strip().lower()
    if mode not in {"sliding", "full_case"}:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
    for record in progress_iter(
        records,
        description="Scanning tissue Doppler records",
        total=len(records),
        unit="case",
    ):
        store = open_recording(
            record,
            root=data_root,
            cache_dir=recording_cache_dir,
            cache_read_only=recording_cache_read_only,
            cache_include=recording_cache_include,
            cache_exclude=recording_cache_exclude,
        )
        bmode = store.load_stream(BMODE_PATH)
        tdi = store.load_modality(TDI_PATH).stream
        if not isinstance(bmode, BrightnessModeStream):
            raise TypeError(f"{BMODE_PATH} must be BrightnessModeStream")
        if not isinstance(tdi, TissueDopplerFloatStream):
            raise TypeError(f"{TDI_PATH} must load as TissueDopplerFloatStream")
        bmode_timestamps = _timestamps_or_index(bmode.timestamps, int(bmode.data.shape[0]))
        doppler_timestamps = _timestamps_or_index(tdi.timestamps, int(tdi.data.shape[0]))
        fps = _fps_from_mean_step(_mean_step(bmode_timestamps))
        if not _fps_is_accepted(bmode_timestamps=bmode_timestamps, accepted_fps_range=accepted_fps_range):
            continue
        required_frames = (
            int(min_frames_per_case) + 1
            if mode == "full_case"
            else max(int(clip_length) + 1, int(min_frames_per_case) + 1)
        )
        if len(bmode_timestamps) < required_frames or len(doppler_timestamps) < required_frames - 1:
            continue
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
            )
            if doppler_indices is None or len(doppler_indices) != max(0, len(clip_timestamps) - 1):
                continue
            refs.append(
                SampleRef(
                    record=record,
                    clip_start=clip_start,
                    clip_stop=clip_stop,
                    doppler_indices=doppler_indices,
                    sample_id=f"{record.exam_id}/{record.recording_id}:clip_full",
                    fps=fps,
                )
            )
            if max_samples is not None and len(refs) >= int(max_samples):
                return refs
            continue
        input_clip_length = int(clip_length) + 1
        last_start = len(bmode_timestamps) - input_clip_length + 1
        for clip_start in range(0, last_start, max(1, int(clip_stride))):
            clip_stop = clip_start + input_clip_length
            clip_timestamps = bmode_timestamps[clip_start:clip_stop]
            doppler_indices = _contained_doppler_indices(
                bmode_timestamps=clip_timestamps,
                doppler_timestamps=doppler_timestamps,
                tolerance=tolerance,
            )
            if doppler_indices is None or len(doppler_indices) != int(clip_length):
                continue
            refs.append(
                SampleRef(
                    record=record,
                    clip_start=clip_start,
                    clip_stop=clip_stop,
                    doppler_indices=doppler_indices,
                    sample_id=f"{record.exam_id}/{record.recording_id}:clip_{clip_start}_{clip_stop}",
                    fps=fps,
                )
            )
            if max_samples is not None and len(refs) >= int(max_samples):
                return refs
    return refs


def _resample_target_stack(
    frames: np.ndarray,
    *,
    source: TissueDopplerFloatStream,
    reference: BrightnessModeStream,
    target_shape: tuple[int, int],
) -> np.ndarray:
    source_geometry = source.metadata.geometry
    reference_geometry = reference.metadata.geometry
    return cast(
        np.ndarray,
        _resample_to_reference_geometry(
            frames,
            source_geometry=source_geometry,
            reference_geometry=reference_geometry,
            target_shape=target_shape,
            interpolation="nearest",
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
    converted = _sector_stack_to_cartesian_values(
        values,
        geometry=reference.metadata.geometry,
        cartesian_height=cartesian_height,
        interpolation="linear",
    )
    return _minmax_normalize(_resize_stack(converted, output_shape))


def _cartesian_target_stack(
    frames: np.ndarray,
    *,
    source: TissueDopplerFloatStream,
    reference: BrightnessModeStream,
    cartesian_height: int,
    target_shape: tuple[int, int],
) -> np.ndarray:
    geometry = source.metadata.geometry or reference.metadata.geometry
    converted = _sector_stack_to_cartesian_values(
        np.asarray(frames, dtype=np.float32),
        geometry=geometry,
        cartesian_height=cartesian_height,
        interpolation="nearest",
    )
    return _resize_stack(converted, target_shape, interpolation="nearest")


def _sector_stack_to_cartesian_values(
    frames: np.ndarray,
    *,
    geometry: SectorGeometry | None,
    cartesian_height: int,
    interpolation: str,
) -> np.ndarray:
    if geometry is None:
        return np.asarray(frames, dtype=np.float32)
    grid = CartesianGrid.from_sector_height(geometry, int(cartesian_height))
    converted = tuple(sector_to_cartesian(frame, geometry, grid=grid, interpolation=interpolation) for frame in frames)
    return np.stack([image.data for image in converted], axis=0).astype(np.float32, copy=False)


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


def _velocity_limit(stream: TissueDopplerFloatStream) -> float:
    return _velocity_limit_from_stream(stream)


def _timestamps_or_index(timestamps: np.ndarray | None, count: int) -> np.ndarray:
    if timestamps is None:
        return np.arange(int(count), dtype=np.float64)
    return np.asarray(timestamps, dtype=np.float64)


def _contained_doppler_indices(
    *,
    bmode_timestamps: np.ndarray,
    doppler_timestamps: np.ndarray,
    tolerance: float,
) -> tuple[int, ...] | None:
    bmode = np.asarray(bmode_timestamps, dtype=np.float64)
    doppler = np.asarray(doppler_timestamps, dtype=np.float64)
    if len(bmode) < 2:
        return None
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
