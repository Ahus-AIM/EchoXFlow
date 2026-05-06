"""ECG-gated beat stitching helpers for volumetric recordings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StitchedVolumeStack:
    """Prepared volume stack and timestamps for display."""

    volumes: np.ndarray
    timestamps: np.ndarray | None
    stitch_beat_count: int
    was_stitched: bool


def prepare_3d_brightness_for_display(
    volumes: np.ndarray,
    timestamps: np.ndarray | None,
    metadata: Mapping[str, Any] | None,
) -> StitchedVolumeStack:
    """Align 3D timestamps to ECG time and stitch multi-beat volumes when needed."""
    arr = np.asarray(volumes)
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D brightness mode with shape [T,E,A,R], got {arr.shape}")
    source_timestamps = None if timestamps is None else np.asarray(timestamps, dtype=np.float64).reshape(-1)
    relative_timestamps = relative_volume_timestamps(timestamps, metadata)
    beat_count = _stitch_beat_count(metadata)
    qrs = _qrs_trigger_times(metadata)
    if source_timestamps is None or relative_timestamps is None or beat_count <= 1 or qrs.size < 2:
        return StitchedVolumeStack(
            volumes=arr,
            timestamps=relative_timestamps,
            stitch_beat_count=beat_count,
            was_stitched=False,
        )
    stitch_timestamps = _best_stitching_timestamps(source_timestamps, relative_timestamps, qrs, beat_count)
    stitched_volumes, stitched_timestamps = stitch_3d_brightness_beats(
        arr,
        stitch_timestamps,
        qrs,
        stitch_beat_count=beat_count,
        output_timestamps=relative_timestamps,
    )
    was_stitched = stitched_volumes is not arr
    return StitchedVolumeStack(
        volumes=stitched_volumes,
        timestamps=stitched_timestamps if was_stitched else relative_timestamps,
        stitch_beat_count=beat_count,
        was_stitched=was_stitched,
    )


def relative_volume_timestamps(
    timestamps: np.ndarray | None,
    metadata: Mapping[str, Any] | None,
) -> np.ndarray | None:
    """Return volume timestamps on the ECG/QRS-relative axis when metadata provides an origin."""
    if timestamps is None:
        return None
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        return ts
    origin = _volume_time_origin_s(metadata)
    if origin is None:
        return ts
    shifted = ts - origin
    if _looks_absolute(ts, origin):
        return shifted
    qrs = _qrs_trigger_times(metadata)
    if qrs.size < 2:
        return ts
    return shifted if _timeline_overlap_score(shifted, qrs) >= _timeline_overlap_score(ts, qrs) else ts


def relative_qrs_trigger_times(metadata: Mapping[str, Any] | None) -> np.ndarray:
    """Return ECG/QRS trigger times on the same relative axis used for display."""
    qrs = np.unique(np.sort(_qrs_trigger_times(metadata))).astype(np.float64, copy=False)
    origin = _volume_time_origin_s(metadata)
    if origin is None or qrs.size == 0:
        return qrs
    shifted = qrs - origin
    if _looks_absolute(qrs, origin):
        return shifted
    return shifted if _timeline_overlap_score(shifted, qrs) >= _timeline_overlap_score(qrs, qrs) else qrs


def mesh_frame_indices_for_volume_timestamps(
    mesh_timestamps: np.ndarray | None,
    volume_timestamps: np.ndarray | None,
    metadata: Mapping[str, Any] | None,
    *,
    mesh_frame_count: int | None = None,
    target_count: int | None = None,
) -> tuple[int, ...]:
    """Map volume frames to mesh frames using ECG/R-peak phase when available."""
    frame_count = _mesh_frame_count(mesh_timestamps, mesh_frame_count)
    if frame_count <= 0:
        return ()
    requested = _target_frame_count(volume_timestamps, target_count)
    if requested <= 0:
        return ()
    volume_times = relative_volume_timestamps(volume_timestamps, metadata)
    mesh_times = _relative_timestamps(mesh_timestamps, metadata)
    if volume_times is None or volume_times.size == 0 or mesh_times.size == 0:
        return _linear_frame_indices(frame_count, requested)
    volume_times = np.asarray(volume_times, dtype=np.float64).reshape(-1)[:requested]
    if volume_times.size != requested:
        return _linear_frame_indices(frame_count, requested)
    mesh_times = mesh_times[:frame_count]
    if mesh_times.size != frame_count or not np.any(np.isfinite(mesh_times)):
        return _linear_frame_indices(frame_count, requested)
    qrs = relative_qrs_trigger_times(metadata)
    sample_times = np.asarray(
        [
            _mesh_sample_time_for_volume_time(mesh_times=mesh_times, current_t=float(time_s), qrs=qrs)
            for time_s in volume_times
        ],
        dtype=np.float64,
    )
    return tuple(_nearest_mesh_frame_index(mesh_times, sample_time) for sample_time in sample_times)


def stitch_3d_brightness_beats(
    volumes: np.ndarray,
    timestamps: np.ndarray,
    qrs_trigger_times: np.ndarray,
    *,
    stitch_beat_count: int,
    output_timestamps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch gated 3D sub-volumes along elevation using QRS-delimited beats."""
    arr = np.asarray(volumes)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    qrs = np.sort(np.asarray(qrs_trigger_times, dtype=np.float64).reshape(-1))
    qrs = qrs[np.isfinite(qrs)]
    beat_count = max(1, int(stitch_beat_count))
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D brightness mode with shape [T,E,A,R], got {arr.shape}")
    if beat_count <= 1 or qrs.size < 2 or ts.size != arr.shape[0]:
        return arr, ts
    output_ts = None if output_timestamps is None else np.asarray(output_timestamps, dtype=np.float64).reshape(-1)
    if output_ts is None or output_ts.size != ts.size:
        output_ts = ts

    beat_buffers = _beat_buffers(ts, qrs, beat_count)

    if sum(1 for buffer in beat_buffers if buffer) < beat_count:
        return arr, ts
    min_len = min(len(buffer) for buffer in beat_buffers)
    if min_len <= 0:
        return arr, ts

    stitched_frames = [
        np.concatenate([arr[beat_buffers[beat_index][pos]] for beat_index in range(beat_count)], axis=0)
        for pos in range(min_len)
    ]
    stitched_timestamps = [float(output_ts[beat_buffers[0][pos]]) for pos in range(min_len)]
    return np.stack(stitched_frames, axis=0), np.asarray(stitched_timestamps, dtype=np.float64)


def _best_stitching_timestamps(
    source_timestamps: np.ndarray,
    relative_timestamps: np.ndarray,
    qrs: np.ndarray,
    beat_count: int,
) -> np.ndarray:
    source_score = _beat_bucket_score(source_timestamps, qrs, beat_count)
    relative_score = _beat_bucket_score(relative_timestamps, qrs, beat_count)
    if relative_score > source_score:
        return relative_timestamps
    return source_timestamps


def _beat_bucket_score(timestamps: np.ndarray, qrs: np.ndarray, beat_count: int) -> tuple[int, int]:
    buffers = _beat_buffers(timestamps, qrs, beat_count)
    nonempty = sum(1 for buffer in buffers if buffer)
    min_len = min((len(buffer) for buffer in buffers), default=0)
    return nonempty, min_len


def _beat_buffers(timestamps: np.ndarray, qrs: np.ndarray, beat_count: int) -> list[list[int]]:
    beat_buffers: list[list[int]] = [list() for _ in range(beat_count)]
    for qrs_index, start in enumerate(qrs):
        stop = float(qrs[qrs_index + 1]) if qrs_index + 1 < qrs.size else _final_stop(timestamps, float(start))
        frame_indices = np.where((timestamps >= float(start)) & (timestamps < stop))[0]
        beat_buffers[qrs_index % beat_count].extend(int(index) for index in frame_indices)
    return beat_buffers


def _mesh_frame_count(mesh_timestamps: np.ndarray | None, mesh_frame_count: int | None) -> int:
    if mesh_frame_count is not None:
        return max(0, int(mesh_frame_count))
    if mesh_timestamps is None:
        return 0
    return int(np.asarray(mesh_timestamps).reshape(-1).size)


def _target_frame_count(volume_timestamps: np.ndarray | None, target_count: int | None) -> int:
    if target_count is not None:
        return max(0, int(target_count))
    if volume_timestamps is None:
        return 0
    return int(np.asarray(volume_timestamps).reshape(-1).size)


def _relative_timestamps(timestamps: np.ndarray | None, metadata: Mapping[str, Any] | None) -> np.ndarray:
    if timestamps is None:
        return np.asarray([], dtype=np.float64)
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    origin = _volume_time_origin_s(metadata)
    if origin is None:
        return values
    shifted = values - origin
    if _looks_absolute(values, origin):
        return shifted
    qrs = relative_qrs_trigger_times(metadata)
    if qrs.size < 2:
        return values
    return shifted if _timeline_overlap_score(shifted, qrs) >= _timeline_overlap_score(values, qrs) else values


def _mesh_sample_time_for_volume_time(*, mesh_times: np.ndarray, current_t: float, qrs: np.ndarray) -> float:
    finite_mesh_times = np.asarray(mesh_times, dtype=np.float64).reshape(-1)
    finite_mesh_times = finite_mesh_times[np.isfinite(finite_mesh_times)]
    if finite_mesh_times.size == 0:
        return float(current_t)
    mesh_start = float(finite_mesh_times[0])
    mesh_end = float(finite_mesh_times[-1])
    qrs = np.unique(np.sort(np.asarray(qrs, dtype=np.float64).reshape(-1)))
    qrs = qrs[np.isfinite(qrs)]
    if qrs.size >= 2 and mesh_end > mesh_start + 1e-8:
        interval_idx = int(np.searchsorted(qrs, float(current_t), side="right") - 1)
        interval_idx = min(max(interval_idx, 0), int(qrs.size) - 2)
        start = float(qrs[interval_idx])
        end = float(qrs[interval_idx + 1])
        if end > start + 1e-8:
            phase = np.clip((float(current_t) - start) / (end - start), 0.0, 1.0)
            return float(mesh_start + float(phase) * (mesh_end - mesh_start))
    if mesh_end <= mesh_start + 1e-8:
        return mesh_start
    cycle_start = float(qrs[0]) if qrs.size else mesh_start
    cycle_end = float(qrs[-1]) if qrs.size else mesh_end
    if cycle_end <= cycle_start + 1e-8:
        cycle_start, cycle_end = mesh_start, mesh_end
    window_duration = float(cycle_end - cycle_start)
    mesh_duration = float(mesh_end - mesh_start)
    if cycle_start <= float(current_t) <= cycle_end:
        return float(mesh_start + ((float(current_t) - cycle_start) / window_duration) * mesh_duration)
    phase = float(np.mod(float(current_t) - cycle_start, 2.0 * window_duration))
    reflected = phase if phase <= window_duration else (2.0 * window_duration - phase)
    return float(mesh_start + (reflected / window_duration) * mesh_duration)


def _nearest_mesh_frame_index(mesh_times: np.ndarray, sample_time: float) -> int:
    values = np.asarray(mesh_times, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0
    distances = np.full(values.shape, np.inf, dtype=np.float64)
    distances[finite] = np.abs(values[finite] - float(sample_time))
    return int(np.argmin(distances))


def _linear_frame_indices(frame_count: int, target_count: int) -> tuple[int, ...]:
    if frame_count <= 0 or target_count <= 0:
        return ()
    if frame_count >= target_count:
        return tuple(range(target_count))
    if target_count == 1:
        return (0,)
    return tuple(int(index) for index in np.rint(np.linspace(0, frame_count - 1, target_count)))


def _stitch_beat_count(metadata: Mapping[str, Any] | None) -> int:
    value = None if metadata is None else metadata.get("stitch_beat_count")
    try:
        count = int(np.asarray(value).reshape(-1)[0])
    except (IndexError, TypeError, ValueError):
        return 1
    return max(1, count)


def _qrs_trigger_times(metadata: Mapping[str, Any] | None) -> np.ndarray:
    if metadata is None:
        return np.asarray([], dtype=np.float64)
    public_metadata = metadata.get("metadata")
    raw = public_metadata.get("qrs_trigger_times") if isinstance(public_metadata, Mapping) else None
    if raw is None:
        raw = metadata.get("volume_qrs_trigger_times")
    if raw is None:
        return np.asarray([], dtype=np.float64)
    try:
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float64)
    return values[np.isfinite(values)]


def _volume_time_origin_s(metadata: Mapping[str, Any] | None) -> float | None:
    if metadata is None:
        return None
    public_metadata = metadata.get("metadata")
    reference = public_metadata.get("time_reference") if isinstance(public_metadata, Mapping) else None
    if not isinstance(reference, Mapping):
        reference = metadata.get("volume_time_reference")
    if not isinstance(reference, Mapping):
        return None
    raw = reference.get("volume_time_origin_s")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _looks_absolute(timestamps: np.ndarray, origin: float) -> bool:
    finite = timestamps[np.isfinite(timestamps)]
    return bool(finite.size and np.nanmin(finite) >= origin - 1.0)


def _timeline_overlap_score(timestamps: np.ndarray, qrs: np.ndarray) -> float:
    finite = timestamps[np.isfinite(timestamps)]
    qrs_finite = qrs[np.isfinite(qrs)]
    if finite.size == 0 or qrs_finite.size == 0:
        return 0.0
    start = max(float(finite[0]), float(qrs_finite[0]))
    stop = min(float(finite[-1]), float(qrs_finite[-1]))
    return max(0.0, stop - start)


def _final_stop(timestamps: np.ndarray, start: float) -> float:
    if timestamps.size == 0:
        return start
    return max(float(timestamps[-1]), start)
