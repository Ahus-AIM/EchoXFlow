"""ECG-gated beat stitching helpers for volumetric recordings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

FULL_WINDOW_START_FRACTION = 0.10
FULL_WINDOW_END_FRACTION = 0.90


@dataclass(frozen=True)
class StitchedVolumeStack:
    """Prepared volume stack and timestamps for display."""

    volumes: np.ndarray
    timestamps: np.ndarray | None
    stitch_beat_count: int
    was_stitched: bool
    source_timestamps: np.ndarray | None = None


@dataclass(frozen=True)
class _BeatTimeSamples:
    indices: np.ndarray
    elapsed_s: np.ndarray
    duration_s: float


def prepare_3d_brightness_for_display(
    volumes: np.ndarray,
    timestamps: np.ndarray | None,
    metadata: Mapping[str, Any] | None,
) -> StitchedVolumeStack:
    """Align 3D timestamps to ECG time and stitch multi-beat volumes when needed."""
    arr = np.asarray(volumes)
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D brightness mode with shape [T,E,A,R], got {arr.shape}")
    relative_timestamps = relative_volume_timestamps(timestamps, metadata)
    beat_count = _stitch_beat_count(metadata)
    qrs = relative_qrs_trigger_times(metadata)
    if relative_timestamps is None or beat_count <= 1 or qrs.size < 2:
        return StitchedVolumeStack(
            volumes=arr,
            timestamps=relative_timestamps,
            stitch_beat_count=beat_count,
            was_stitched=False,
        )
    stitched_volumes, stitched_timestamps, source_timestamps = _stitch_3d_brightness_beats_with_sources(
        arr,
        relative_timestamps,
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
        source_timestamps=source_timestamps,
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
    if _time_reference_basis(metadata) == "relative_to_volume_origin" and not _starts_at_or_after_origin(ts, origin):
        return ts
    return ts - origin if _starts_at_or_after_origin(ts, origin) else ts


def relative_qrs_trigger_times(metadata: Mapping[str, Any] | None) -> np.ndarray:
    """Return ECG/QRS trigger times on the same relative axis used for display."""
    qrs = np.unique(np.sort(_qrs_trigger_times(metadata))).astype(np.float64, copy=False)
    origin = _volume_time_origin_s(metadata)
    if origin is None or qrs.size == 0:
        return qrs
    if _time_reference_basis(metadata) == "relative_to_volume_origin":
        return qrs
    return qrs - origin if _starts_at_or_after_origin(qrs, origin) else qrs


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
    """Stitch gated 3D sub-volumes along the lowest-resolution spatial axis."""
    stitched, stitched_timestamps, _ = _stitch_3d_brightness_beats_with_sources(
        volumes, timestamps, qrs_trigger_times, stitch_beat_count=stitch_beat_count, output_timestamps=output_timestamps
    )
    return stitched, stitched_timestamps


def _stitch_3d_brightness_beats_with_sources(
    volumes: np.ndarray,
    timestamps: np.ndarray,
    qrs_trigger_times: np.ndarray,
    *,
    stitch_beat_count: int,
    output_timestamps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Stitch volumes and return source ECG times for each displayed slab."""
    arr = np.asarray(volumes)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    qrs = np.sort(np.asarray(qrs_trigger_times, dtype=np.float64).reshape(-1))
    qrs = qrs[np.isfinite(qrs)]
    beat_count = max(1, int(stitch_beat_count))
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D brightness mode with shape [T,E,A,R], got {arr.shape}")
    if beat_count <= 1 or qrs.size < 2 or ts.size != arr.shape[0]:
        return arr, ts, None
    output_ts = None if output_timestamps is None else np.asarray(output_timestamps, dtype=np.float64).reshape(-1)
    if output_ts is None or output_ts.size != ts.size:
        output_ts = ts

    aligned_groups = _time_aligned_groups(ts, qrs, beat_count)
    if not aligned_groups:
        return arr, ts, None

    stitch_axis = _lowest_resolution_spatial_axis(arr)
    stitch_offset = _best_antisymmetric_cyclic_rotation(arr, aligned_groups, stitch_axis, beat_count)
    stitched_frames = []
    stitched_timestamps = []
    source_timestamps = []
    for start_index, indices in aligned_groups:
        beat_order = _antisymmetric_cyclic_order(start_index, beat_count, stitch_offset)
        for pos in range(len(indices[0])):
            source_indices = [indices[beat_index][pos] for beat_index in beat_order]
            stitched_frames.append(
                np.concatenate(
                    [arr[source_index] for source_index in source_indices],
                    axis=stitch_axis,
                )
            )
            stitched_timestamps.append(float(output_ts[indices[0][pos]]))
            source_timestamps.append([float(output_ts[source_index]) for source_index in source_indices])
    return np.stack(stitched_frames, axis=0), np.asarray(stitched_timestamps), np.asarray(source_timestamps)


def _time_aligned_groups(timestamps: np.ndarray, qrs: np.ndarray, beat_count: int):
    return _collect_time_aligned_groups(timestamps, qrs, beat_count, require_full=True)


def _collect_time_aligned_groups(timestamps, qrs, beat_count, *, require_full):
    groups = []
    start_index = 0
    while start_index <= qrs.size - beat_count:
        aligned, is_full = _time_aligned_indices(_beat_time_samples(timestamps, qrs, start_index, beat_count))
        if len(aligned) == beat_count and aligned[0] and (is_full or not require_full):
            groups.append((start_index, aligned))
            start_index += beat_count
            continue
        start_index += 1
    return groups


def _beat_time_samples(
    timestamps: np.ndarray, qrs: np.ndarray, start_index: int, beat_count: int
) -> list[_BeatTimeSamples]:
    samples: list[_BeatTimeSamples] = []
    for beat_index in range(beat_count):
        start = float(qrs[start_index + beat_index])
        stop_index = start_index + beat_index + 1
        stop = float(qrs[stop_index]) if stop_index < qrs.size else _final_stop(timestamps, start)
        duration = stop - start
        if duration <= 0.0:
            return []
        frame_indices = np.where((timestamps >= start) & (timestamps < stop))[0]
        samples.append(
            _BeatTimeSamples(
                indices=frame_indices.astype(np.int64, copy=False),
                elapsed_s=np.asarray(timestamps[frame_indices] - start, dtype=np.float64),
                duration_s=duration,
            )
        )
    return samples


def _time_aligned_indices(samples: list[_BeatTimeSamples]) -> tuple[list[list[int]], bool]:
    if not samples or any(sample.indices.size == 0 for sample in samples):
        return [], False
    elapsed = [sample.elapsed_s for sample in samples]
    lower = max(float(values[0]) for values in elapsed)
    upper = min(float(values[-1]) for values in elapsed)
    if upper + 1e-6 < lower:
        return [[] for _ in samples], False
    in_range = [(values >= lower - 1e-6) & (values <= upper + 1e-6) for values in elapsed]
    target_times = elapsed[int(np.argmin([np.count_nonzero(mask) for mask in in_range]))][
        in_range[int(np.argmin([np.count_nonzero(mask) for mask in in_range]))]
    ]
    if target_times.size == 0:
        return [[] for _ in samples], False
    aligned: list[list[int]] = []
    for sample, values in zip(samples, elapsed, strict=True):
        nearest = np.abs(values[:, None] - target_times[None, :]).argmin(axis=0)
        aligned.append([int(index) for index in sample.indices[nearest]])
    shortest_duration = min(sample.duration_s for sample in samples)
    return aligned, (
        lower <= FULL_WINDOW_START_FRACTION * shortest_duration
        and upper >= FULL_WINDOW_END_FRACTION * shortest_duration
    )


def _lowest_resolution_spatial_axis(volumes: np.ndarray) -> int:
    """Return the current metadata-free proxy for the elevation stitch axis."""
    return int(np.argmin(np.asarray(volumes).shape[1:]))


def _best_antisymmetric_cyclic_rotation(volumes, groups, stitch_axis: int, beat_count: int) -> int:
    if beat_count <= 1:
        return 0
    scores = []
    for offset in range(beat_count):
        group_scores = []
        for start_index, indices in groups:
            parts = [np.asarray(volumes[index_group]) for index_group in indices]
            order = _antisymmetric_cyclic_order(start_index, beat_count, offset)
            group_scores.append(_cyclic_rotation_seam_score(parts, stitch_axis, order))
        scores.append((float(np.median(group_scores)), offset != 0, offset))
    return min(scores)[2]


def _antisymmetric_cyclic_order(start_index: int, beat_count: int, offset: int) -> tuple[int, ...]:
    return tuple(sorted(range(beat_count), key=lambda index: (start_index + index + offset) % beat_count, reverse=True))


def _cyclic_rotation_seam_score(parts: list[np.ndarray], stitch_axis: int, order: tuple[int, ...]) -> float:
    axis = 1 + int(stitch_axis)
    ratios = []
    for left_index, right_index in zip(order[:-1], order[1:], strict=True):
        left_part = parts[left_index]
        right_part = parts[right_index]
        left = np.take(left_part, -1, axis=axis).astype(np.float32)
        left_gradient = (
            np.abs(left - np.take(left_part, -2, axis=axis).astype(np.float32)) if left_part.shape[axis] > 1 else None
        )
        right = np.take(right_part, 0, axis=axis).astype(np.float32)
        gradients = [] if left_gradient is None else [left_gradient]
        if right_part.shape[axis] > 1:
            gradients.append(np.abs(right - np.take(right_part, 1, axis=axis).astype(np.float32)))
        seam = np.abs(left - right)
        reference = np.concatenate([gradient.reshape(-1) for gradient in gradients]) if gradients else seam.reshape(-1)
        ratios.append(float(np.median(seam)) / (float(np.median(reference)) + 1e-6))
    return float(np.median(ratios) + 0.25 * max(ratios))


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
    values = relative_volume_timestamps(timestamps, metadata)
    return np.asarray([], dtype=np.float64) if values is None else values


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
    reference = _time_reference(metadata)
    if reference is None:
        return None
    raw = reference.get("volume_time_origin_s", reference.get("origin_s"))
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _time_reference(metadata: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if metadata is None:
        return None
    public_metadata = metadata.get("metadata")
    reference = public_metadata.get("time_reference") if isinstance(public_metadata, Mapping) else None
    if not isinstance(reference, Mapping):
        reference = metadata.get("volume_time_reference")
    return reference if isinstance(reference, Mapping) else None


def _time_reference_basis(metadata: Mapping[str, Any] | None) -> str | None:
    reference = _time_reference(metadata)
    raw = None if reference is None else reference.get("basis")
    return str(raw).strip().lower() if raw is not None else None


def _starts_at_or_after_origin(timestamps: np.ndarray, origin: float) -> bool:
    finite = timestamps[np.isfinite(timestamps)]
    return bool(finite.size and float(np.nanmin(finite)) >= origin - 1e-6)


def _final_stop(timestamps: np.ndarray, start: float) -> float:
    if timestamps.size == 0:
        return start
    return max(float(timestamps[-1]), start)
