from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from echoxflow.croissant import RecordingRecord
from echoxflow.export import RecordingArray, write_recording
from echoxflow.plotting import PlotStyle, PlotViewMode, render_recording_video

DEFAULT_PREVIEW_FPS = 4.0
DEFAULT_PREVIEW_MAX_FPS = 60.0
DEFAULT_PREVIEW_VIEW_MODE: PlotViewMode = "both"


def write_preview_recording_video(
    *,
    record: RecordingRecord,
    preview_dir: Path,
    epoch: int,
    split: str,
    suffix: str,
    arrays: Sequence[RecordingArray],
    attrs: Mapping[str, Any],
    modalities: tuple[str, ...],
    view_mode: PlotViewMode = DEFAULT_PREVIEW_VIEW_MODE,
    max_fps: float = DEFAULT_PREVIEW_MAX_FPS,
    style: PlotStyle | None = None,
) -> None:
    preview_style = style or PlotStyle(width_px=1440, height_px=1080, dpi=100)
    recording_dir = preview_dir / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)
    recording_id = _safe_recording_id(f"{record.recording_id}_{epoch:04d}_{split}_{suffix}")
    recording_path = recording_dir / f"{recording_id}.zarr"
    try:
        zarr_path = str(recording_path.relative_to(preview_dir))
    except ValueError:
        zarr_path = str(recording_path)
    write_recording(
        recording_path,
        exam_id=record.exam_id,
        recording_id=recording_id,
        source_recording_id=record.recording_id,
        arrays=arrays,
        attrs=attrs,
        zarr_path=zarr_path,
        croissant_path=preview_dir / "croissant.json",
        overwrite=True,
    )
    output = preview_dir / f"epoch_{epoch:04d}_{split}_{suffix}.mp4"
    render_recording_video(
        recording_path,
        output,
        modalities=modalities,
        view_mode=view_mode,
        max_fps=max_fps,
        dpi=preview_style.dpi,
        style=preview_style,
    )


def common_preview_arrays(
    store: Any,
    sample: object,
    *,
    bmode_path: str = "data/2d_brightness_mode",
    extra: Sequence[RecordingArray] = (),
    fallback_fps: float = DEFAULT_PREVIEW_FPS,
) -> tuple[RecordingArray, ...]:
    bmode = source_bmode_array(store, sample, data_path=bmode_path)
    ecg = source_ecg_arrays(store) or (
        zero_ecg_array(bmode.timestamps, frame_count=len(bmode.values), fallback_fps=fallback_fps),
    )
    return (bmode, *ecg, *extra)


def write_preview_pair(
    *,
    record: RecordingRecord,
    preview_dir: Path,
    epoch: int,
    split: str,
    common: Sequence[RecordingArray],
    build_modality_arrays: Callable[[str], Sequence[RecordingArray]],
    attrs: Mapping[str, Any],
    modalities: tuple[str, ...],
    view_mode: PlotViewMode = DEFAULT_PREVIEW_VIEW_MODE,
    max_fps: float = DEFAULT_PREVIEW_MAX_FPS,
    style: PlotStyle | None = None,
) -> None:
    for suffix in ("real", "predicted"):
        write_preview_recording_video(
            record=record,
            preview_dir=preview_dir,
            epoch=epoch,
            split=split,
            suffix=suffix,
            arrays=(*common, *build_modality_arrays(suffix)),
            attrs=attrs,
            modalities=modalities,
            view_mode=view_mode,
            max_fps=max_fps,
            style=style,
        )


def source_bmode_array(store: Any, sample: object, data_path: str = "data/2d_brightness_mode") -> RecordingArray:
    start = int(getattr(sample, "clip_start", 0) or 0)
    stop_value = getattr(sample, "clip_stop", None)
    stop = None if stop_value is None else int(stop_value)
    stream = store.load_stream_slice(data_path, start, stop)
    values = np.asarray(stream.data)
    return RecordingArray(
        data_path=stream.data_path,
        values=values,
        timestamps=stream.timestamps,
        timestamps_path=stream.timestamps_path,
        content_type=stream.data_path.removeprefix("data/"),
    )


def source_ecg_arrays(store: Any) -> tuple[RecordingArray, ...]:
    arrays = []
    for path in store.array_paths:
        if not _is_ecg_path(path):
            continue
        loaded = store.load_modality(path)
        arrays.append(
            RecordingArray(
                data_path=loaded.data_path,
                values=np.asarray(loaded.data),
                timestamps=loaded.timestamps,
                timestamps_path=loaded.timestamps_path,
                content_type=loaded.data_path.removeprefix("data/"),
            )
        )
    return tuple(arrays)


def zero_ecg_array(
    timestamps: np.ndarray | None,
    *,
    frame_count: int,
    fallback_fps: float = DEFAULT_PREVIEW_FPS,
) -> RecordingArray:
    typed_timestamps = (
        np.arange(int(frame_count), dtype=np.float32) / float(fallback_fps)
        if timestamps is None
        else np.asarray(timestamps, dtype=np.float32).reshape(-1)
    )
    return RecordingArray(
        data_path="data/ecg",
        values=np.zeros_like(typed_timestamps, dtype=np.float32),
        timestamps=typed_timestamps,
        timestamps_path="timestamps/ecg",
        content_type="ecg",
    )


def _is_ecg_path(path: str) -> bool:
    data_path = str(path).strip("/")
    return data_path == "data/ecg" or (data_path.startswith("data/") and data_path.endswith("_ecg"))


def _safe_recording_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)
