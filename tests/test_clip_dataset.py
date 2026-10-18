from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from echoxflow import RecordingArray, RecordingRecord, write_recording

torch = pytest.importorskip("torch")
from echoxflow.torch import ClipDataset, collate_clips  # noqa: E402


def _record(tmp_path: Path, name: str, frames: int, attrs: dict | None = None) -> RecordingRecord:
    values = np.arange(frames, dtype=np.uint8)[:, None, None] * np.ones((1, 4, 6), dtype=np.uint8)
    return write_recording(
        tmp_path / f"{name}.zarr",
        exam_id="exam",
        recording_id=name,
        arrays=[RecordingArray("2d_brightness_mode", values, timestamps=np.arange(frames) * 0.04)],
        zarr_path=f"{name}.zarr",
        attrs=attrs,
    )


def test_windows_come_from_catalog_frame_counts(tmp_path: Path) -> None:
    records = [_record(tmp_path, "long", 10), _record(tmp_path, "exact", 4), _record(tmp_path, "short", 3)]
    other = replace(records[0], frame_counts_by_content_type={})

    dataset = ClipDataset([*records, other], clip_length=4, stride=3, root=tmp_path)

    assert [(record.recording_id, start) for record, start in dataset.windows] == [
        ("long", 0),
        ("long", 3),
        ("long", 6),
        ("exact", 0),
    ]
    assert dataset.skipped == 2 and len(dataset) == 4
    assert "4 clips" in repr(dataset) and "2 records skipped" in repr(dataset)


def test_getitem_reads_one_window(tmp_path: Path) -> None:
    record = _record(tmp_path, "long", 10)
    dataset = ClipDataset([record], clip_length=4, stride=3, root=tmp_path, transform=lambda clip: clip + 1)

    frames, timestamps, source = dataset[1]

    assert frames.dtype == torch.float32 and tuple(frames.shape) == (4, 4, 6)
    assert frames[:, 0, 0].tolist() == [4.0, 5.0, 6.0, 7.0]  # frames 3..6, plus the transform's +1
    assert np.allclose(timestamps.numpy(), np.arange(3, 7) * 0.04) and source == record


def test_resize_and_collate(tmp_path: Path) -> None:
    dataset = ClipDataset([_record(tmp_path, "a", 8)], clip_length=4, stride=4, resize=(8, 8), root=tmp_path)

    frames, timestamps, records = collate_clips([dataset[0], dataset[1]])

    assert tuple(frames.shape) == (2, 4, 8, 8) and tuple(timestamps.shape) == (2, 4) and len(records) == 2


def test_cartesian_scan_converts_each_window(tmp_path: Path) -> None:
    sector = {
        "semantic_id": "bmode",
        "frames": {"array_path": "data/2d_brightness_mode", "format": "zarr_array"},
        "geometry": {"DepthStart": 0.0, "DepthEnd": 0.1, "Width": 1.0, "Tilt": 0.0},
    }
    record = _record(tmp_path, "sector", 6, attrs={"recording_manifest": {"manifest_type": "2d", "sectors": [sector]}})

    frames, _, _ = ClipDataset([record], clip_length=3, stride=3, cartesian=True, root=tmp_path)[0]

    assert frames.ndim == 3 and frames.shape[0] == 3 and frames.shape[1:] != (4, 6)
