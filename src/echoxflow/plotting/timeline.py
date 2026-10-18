"""Timestamp selection and nearest-sample utilities for plotting."""

from __future__ import annotations

import numpy as np

from echoxflow.plotting.specs import FrameRequest, PanelSpec, RenderTimeline


def temporal_length(data: np.ndarray) -> int:
    arr = np.asarray(data)
    if arr.ndim == 0:
        return 1
    return int(arr.shape[0])


def nearest_index(timestamps: np.ndarray | None, time_s: float, *, count: int, fallback_index: int = 0) -> int:
    if count <= 0:
        return 0
    if timestamps is None:
        return int(np.clip(fallback_index, 0, count - 1))
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size != count or ts.size == 0 or not np.any(np.isfinite(ts)):
        return int(np.clip(fallback_index, 0, count - 1))
    distances = np.abs(ts - float(time_s))
    distances[~np.isfinite(distances)] = np.inf
    return int(np.clip(int(np.argmin(distances)), 0, count - 1))


def fps_from_timestamps(timestamps: np.ndarray) -> float | None:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 2:
        return None
    diffs = np.diff(ts)
    valid = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if valid.size == 0:
        return None
    return float(1.0 / float(np.median(valid)))


def is_timeline_driver(panel: PanelSpec) -> bool:
    name = panel.loaded.name
    if name == "ecg" or name.startswith("1d_"):
        return False
    if panel.kind not in {"image", "matrix"}:
        return False
    count = temporal_length(panel.loaded.data)
    timestamps = panel.loaded.timestamps
    return timestamps is not None and np.asarray(timestamps).reshape(-1).size == count and count > 1


def select_timeline(panels: tuple[PanelSpec, ...], *, max_fps: float = 60.0) -> RenderTimeline:
    candidates: list[tuple[float, np.ndarray]] = []
    for panel in panels:
        if not is_timeline_driver(panel):
            continue
        timestamps = np.asarray(panel.loaded.timestamps, dtype=np.float64).reshape(-1)
        fps = fps_from_timestamps(timestamps)
        if fps is not None and np.isfinite(fps) and fps > 0.0:
            candidates.append((fps, timestamps))
    if not candidates:
        strip = _longest_strip_timestamps(panels)
        if strip is None:
            return RenderTimeline(timestamps=np.asarray([0.0], dtype=np.float64), fps=1.0)
        fps = fps_from_timestamps(strip)
        candidates.append((float(fps) if fps is not None and fps > 0.0 else float(max_fps), strip))
    fps, timestamps = max(candidates, key=lambda item: item[0])
    capped_fps = min(float(fps), float(max_fps))
    if capped_fps >= float(fps):
        return RenderTimeline(timestamps=timestamps, fps=float(fps))
    start = float(timestamps[0])
    end = float(timestamps[-1])
    if end <= start:
        return RenderTimeline(timestamps=np.asarray([start], dtype=np.float64), fps=capped_fps)
    count = max(1, int(np.floor((end - start) * capped_fps)) + 1)
    return RenderTimeline(timestamps=start + np.arange(count, dtype=np.float64) / capped_fps, fps=capped_fps)


def _longest_strip_timestamps(panels: tuple[PanelSpec, ...]) -> np.ndarray | None:
    """Return the timestamps of the longest 1D strip or line panel, used when no 2D stream animates."""
    best: np.ndarray | None = None
    best_span = 0.0
    for panel in panels:
        name = panel.loaded.name
        if not (name.startswith("1d_") or panel.kind == "line") or panel.loaded.timestamps is None:
            continue
        timestamps = np.asarray(panel.loaded.timestamps, dtype=np.float64).reshape(-1)
        if timestamps.size < 2 or timestamps.size != temporal_length(panel.loaded.data):
            continue
        finite = timestamps[np.isfinite(timestamps)]
        span = float(finite[-1] - finite[0]) if finite.size > 1 else 0.0
        if span > best_span:
            best, best_span = finite, span
    return best


def strip_drives_timeline(panels: tuple[PanelSpec, ...]) -> bool:
    """Return whether no 2D stream animates and a 1D strip sets the timeline instead."""
    return not any(is_timeline_driver(panel) for panel in panels) and _longest_strip_timestamps(panels) is not None


def resolve_frame_time(panels: tuple[PanelSpec, ...], request: FrameRequest) -> float:
    if request.time_s is not None:
        return float(request.time_s)
    frame_index = 0 if request.frame_index is None else max(0, int(request.frame_index))
    strip = _longest_strip_timestamps(panels) if strip_drives_timeline(panels) else None
    if strip is not None:
        if request.frame_index is None:
            return float(0.5 * (strip[0] + strip[-1]))
        return float(strip[min(frame_index, strip.size - 1)])
    for panel in (*tuple(panel for panel in panels if is_timeline_driver(panel)), *panels):
        timestamps = panel.loaded.timestamps
        if timestamps is None:
            continue
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.size:
            return float(ts[min(frame_index, ts.size - 1)])
    return float(frame_index)
