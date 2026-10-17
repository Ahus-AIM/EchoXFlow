"""Clock- and segment-aware helpers for EchoXFlow timestamp arrays.

EchoXFlow keeps every stream on its native source clock, in native order. A clock can step backwards (a reset,
for example where ECG acquisition chunks overlap by a sample) or repeat a time, and a stream can be assembled from
source segments listed in `coordinates/*_segment_frame_offsets`. Interpolating or searching across such a step is
undefined, so these helpers split a stream into strictly increasing runs instead of sorting or deduplicating it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClockRun:
    """A half-open index range ``[start, stop)`` over which a clock strictly increases."""

    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def strictly_increasing_runs(
    timestamps: np.ndarray,
    *,
    segment_offsets: np.ndarray | None = None,
    require_continuity_across_segments: bool = False,
) -> tuple[ClockRun, ...]:
    """Split a clock into maximal runs of finite, strictly increasing samples.

    A run breaks wherever the clock repeats or steps backwards and around non-finite samples. Source-segment
    boundaries only break a run when ``require_continuity_across_segments`` is set; the clock itself already breaks
    it wherever a boundary is a real discontinuity.
    """
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return ()
    breaks = np.zeros(times.size, dtype=bool)
    breaks[0] = True
    breaks[1:] = ~(np.diff(times) > 0.0)
    finite = np.isfinite(times)
    breaks |= ~finite
    breaks[1:] |= ~finite[:-1]
    if require_continuity_across_segments and segment_offsets is not None:
        for boundary in validated_segment_offsets(segment_offsets, times.size)[1:-1]:
            breaks[int(boundary)] = True
    starts = np.flatnonzero(breaks)
    stops = np.append(starts[1:], times.size)
    return tuple(
        ClockRun(int(start), int(stop)) for start, stop in zip(starts, stops) if np.isfinite(times[int(start)])
    )


def validated_segment_offsets(offsets: np.ndarray, sample_count: int) -> np.ndarray:
    """Check that segment offsets start at 0, end at ``sample_count`` and never decrease."""
    values = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if values.size < 2 or int(values[0]) != 0 or int(values[-1]) != int(sample_count) or np.any(np.diff(values) < 0):
        raise ValueError(f"Segment offsets {values.tolist()} do not partition {sample_count} samples")
    return values


def run_ids(timestamps: np.ndarray, **kwargs: object) -> np.ndarray:
    """Return, for every sample, the index of its strictly increasing run (-1 for non-finite samples)."""
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    ids = np.full(times.size, -1, dtype=np.int64)
    for index, run in enumerate(strictly_increasing_runs(times, **kwargs)):  # type: ignore[arg-type]
        ids[run.start : run.stop] = index
    return ids


def within_one_run(indices: np.ndarray, ids: np.ndarray) -> bool:
    """Whether every selected sample lies on the same strictly increasing run."""
    selected = np.asarray(ids)[np.asarray(indices, dtype=np.int64)]
    return bool(selected.size) and bool(np.all(selected >= 0)) and bool(np.all(selected == selected[0]))


def interpolate_on_runs(
    timestamps: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    *,
    maximum_gap_s: float | None = None,
    nearest_tolerance_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a signal sampled on a possibly reset clock.

    Each query is answered from the one strictly increasing run whose span contains it. Queries outside every run,
    or inside the span of several runs (an overlap left by a clock reset, where the time is ambiguous), are invalid.
    With ``maximum_gap_s`` a query between two samples further apart than that is invalid unless it lies within
    ``nearest_tolerance_s`` of one of them. Returns ``(interpolated, valid)``; invalid queries are 0.
    """
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if times.size != signal.size:
        raise ValueError(f"Signal has {signal.size} samples but {times.size} timestamps")
    query = np.asarray(queries, dtype=np.float64)
    flat = query.reshape(-1)
    result = np.zeros(flat.size, dtype=np.float64)
    owners = np.zeros(flat.size, dtype=np.int64)
    accepted = np.zeros(flat.size, dtype=bool)
    for run in strictly_increasing_runs(times):
        run_times = times[run.start : run.stop]
        inside = (flat >= run_times[0]) & (flat <= run_times[-1])
        owners += inside
        if not np.any(inside):
            continue
        selected = flat[inside]
        interpolated = np.interp(selected, run_times, signal[run.start : run.stop])
        valid = np.ones(selected.size, dtype=bool)
        if maximum_gap_s is not None and run_times.size > 1:
            right = np.clip(np.searchsorted(run_times, selected, side="left"), 1, run_times.size - 1)
            left = right - 1
            gap = run_times[right] - run_times[left]
            nearest = np.minimum(np.abs(selected - run_times[left]), np.abs(run_times[right] - selected))
            valid = (gap <= float(maximum_gap_s)) | (nearest <= float(nearest_tolerance_s))
        result[inside] = interpolated
        accepted[inside] = valid
    accepted &= owners == 1
    result[~accepted] = 0.0
    return result.reshape(query.shape), accepted.reshape(query.shape)
