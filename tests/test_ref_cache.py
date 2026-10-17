from __future__ import annotations

from pathlib import Path

from echoxflow import RecordingRecord
from tasks.utils.ref_cache import RecordingScanCache

# Digest of the record below under the cache key scheme of the first public release. If this changes, every cached
# benchmark scan is invalidated: only change it on purpose.
GOLDEN_RECORD_KEY = "31e5494fdeb582dc4e99adb4ddcdc723"


def _row() -> dict[str, object]:
    return {
        "recordings/exam_id": "exam_a",
        "recordings/recording_id": "recording_b",
        "recordings/zarr_path": "exams/exam_a/recording_b.zarr",
        "recordings/modes": ["2d", "aps"],
        "recordings/content_types": ["2d_brightness_mode", "2d_color_doppler"],
        "recordings/frame_counts_by_content_type": {"2d_brightness_mode": 64, "2d_color_doppler": 32},
        "recordings/median_delta_time_by_content_type": {"2d_brightness_mode": 0.025},
        "recordings/array_paths": ["data/2d_brightness_mode", "timestamps/2d_brightness_mode"],
    }


def _key(record: RecordingRecord) -> str:
    cache = RecordingScanCache(None, namespace="x", settings={}, data_root="/nonexistent-root")
    return cache._record_key(record)


def test_record_key_is_stable() -> None:
    assert _key(RecordingRecord.from_row(_row())) == GOLDEN_RECORD_KEY


def test_record_key_ignores_catalog_fields_but_not_the_record() -> None:
    record = RecordingRecord.from_row(_row())
    labelled = RecordingRecord(**{**record.__dict__, "fold": 2, "view_echoprime": "A4C", "annotations": ("lv_mesh",)})
    assert _key(labelled) == GOLDEN_RECORD_KEY
    assert _key(RecordingRecord(**{**record.__dict__, "array_paths": ()})) != GOLDEN_RECORD_KEY


def test_scan_cache_computes_once_per_record(tmp_path: Path) -> None:
    record = RecordingRecord.from_row(_row())
    cache = RecordingScanCache(tmp_path / "cache", namespace="t", settings={"a": 1}, data_root=tmp_path)
    calls: list[str] = []

    def compute(item: RecordingRecord) -> list[str]:
        calls.append(item.recording_id)
        return [item.recording_id]

    assert cache.get(record, compute) == cache.get(record, compute) == ["recording_b"]
    assert calls == ["recording_b"] and (cache.hits, cache.misses) == (1, 1)
