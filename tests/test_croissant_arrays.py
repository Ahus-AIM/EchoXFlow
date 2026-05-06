from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from echoxflow import CroissantArrayRecord, load_croissant


def test_load_croissant_raises_value_error_on_corrupt_json(tmp_path: Path) -> None:
    manifest = tmp_path / "croissant.json"
    manifest.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match=str(manifest)):
        load_croissant(root=tmp_path)


def test_load_croissant_missing_recordings_returns_empty_catalog(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    manifest = tmp_path / "croissant.json"
    manifest.write_text(json.dumps({"recordSet": []}), encoding="utf-8")

    catalog = load_croissant(root=tmp_path)

    assert catalog.recordings == ()
    assert any("no recordings record set" in record.getMessage() for record in caplog.records)


def test_load_croissant_parses_array_record_set(tmp_path: Path) -> None:
    (tmp_path / "croissant.json").write_text(
        json.dumps(
            {
                "recordSet": [
                    {
                        "@id": "recordings",
                        "name": "recordings",
                        "data": [
                            {
                                "recordings/exam_id": "exam",
                                "recordings/recording_id": "recording",
                                "recordings/zarr_path": "recording.zarr",
                                "recordings/modes": ["2d_brightness_mode"],
                                "recordings/content_types": ["2d_brightness_mode"],
                                "recordings/frame_counts_by_content_type": {"2d_brightness_mode": 3},
                                "recordings/median_delta_time_by_content_type": {"2d_brightness_mode": 0.1},
                                "recordings/array_paths": ["data/2d_brightness_mode"],
                            }
                        ],
                    },
                    {
                        "@id": "arrays",
                        "name": "arrays",
                        "data": [
                            {
                                "arrays/recording_id": "recording",
                                "arrays/array_path": "data/2d_brightness_mode",
                                "arrays/content_types": ["2d_brightness_mode"],
                                "arrays/role": "frames",
                                "arrays/dtype": "uint8",
                                "arrays/shape": [3, 8, 8],
                                "arrays/data_sha256": "abc",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    catalog = load_croissant(tmp_path / "croissant.json")

    assert catalog.arrays == (
        CroissantArrayRecord(
            recording_id="recording",
            array_path="data/2d_brightness_mode",
            content_types=("2d_brightness_mode",),
            role="frames",
            dtype="uint8",
            shape=(3, 8, 8),
            data_sha256="abc",
            raw=catalog.arrays[0].raw,
        ),
    )
    assert catalog.arrays_for_recording("recording")[0].temporal_count() == 3
    assert catalog.arrays_by_recording_id()["recording"] == catalog.arrays
