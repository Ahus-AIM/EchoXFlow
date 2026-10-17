"""Helpers shared by several test modules; import them with ``from conftest import ...``."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from echoxflow.croissant import RecordingRecord


@pytest.fixture(autouse=True)
def _isolated_catalog_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests away from the user's real catalog cache."""
    monkeypatch.setenv("ECHOXFLOW_CACHE_DIR", str(tmp_path / "cache"))


def make_record(
    recording_id: str,
    zarr_path: str | None = None,
    *,
    content_types: tuple[str, ...] = ("2d_right_ventricular_strain",),
    array_paths: tuple[str, ...] = (),
    modes: tuple[str, ...] = ("2d", "aps"),
    frame_counts: dict[str, int] | None = None,
) -> RecordingRecord:
    return RecordingRecord(
        exam_id="exam",
        recording_id=recording_id,
        zarr_path=zarr_path or f"{recording_id}.zarr",
        modes=modes,
        content_types=content_types,
        frame_counts_by_content_type={ct: 1 for ct in content_types} if frame_counts is None else frame_counts,
        median_delta_time_by_content_type={},
        array_paths=array_paths,
    )


def strain_document(
    roles: tuple[str, ...],
    *,
    self_recording_id: str | None = None,
    include_qrs: bool = False,
    depth_end_m: float = 0.08,
    curves: bool = True,
    contour_kind: bool = False,
) -> dict[str, object]:
    annotations: list[dict[str, object]] = []
    for role in roles:
        annotations.extend(
            [
                {
                    **({"kind": "contour"} if contour_kind else {}),
                    "target": {
                        "type": "linked_panel",
                        "semantic_id": role,
                        "field": "contour_points",
                    },
                    "target_data": {"zarr_path": "data/2d_brightness_mode", "format": "zarr_array"},
                    "time": {"zarr_path": f"timestamps/{role}_contour", "format": "zarr_array"},
                    "value": {"zarr_path": f"data/{role}_contour", "format": "zarr_array"},
                },
            ]
        )
        if curves:
            annotations.append(
                {
                    "target": {
                        "type": "linked_panel",
                        "semantic_id": role,
                        "field": "strain_curve",
                    },
                    "target_data": {"zarr_path": "data/2d_brightness_mode", "format": "zarr_array"},
                    "time": {"zarr_path": f"timestamps/{role}_curve", "format": "zarr_array"},
                    "value": {"zarr_path": f"data/{role}_curve", "format": "zarr_array"},
                }
            )
        if include_qrs:
            annotations.append(
                {
                    "target": {
                        "type": "linked_panel",
                        "semantic_id": role,
                        "field": "ecg_qrs",
                    },
                    "target_data": {"zarr_path": "data/2d_brightness_mode", "format": "zarr_array"},
                    "time": {"zarr_path": f"timestamps/{role}_ecg_qrs", "format": "zarr_array"},
                    "value": {"zarr_path": f"data/{role}_ecg_qrs", "format": "zarr_array"},
                }
            )
    return {
        "manifest_type": "strain",
        "schema_version": 4,
        "annotation_type": "right_ventricular_strain" if roles == ("rv",) else "left_ventricular_strain",
        "linked_panels": [
            {
                "role_id": role,
                "view_code": role.upper(),
                "geometry": {
                    "depth_start_m": 0.0,
                    "depth_end_m": depth_end_m,
                    "tilt_rad": 0.0,
                    "width_rad": 0.5,
                },
                "frame_timestamps": {"zarr_path": "timestamps/2d_brightness_mode", "format": "zarr_array"},
                "linked_recording": {
                    "recording_id": self_recording_id or f"{role}_source",
                    "recording_zarr_path": "rv.zarr" if self_recording_id is not None else f"{role}.zarr",
                    "relative_zarr_path": "rv.zarr" if self_recording_id is not None else f"{role}.zarr",
                    "frames_array_path": "data/2d_brightness_mode",
                    "timestamps_array_path": "timestamps/2d_brightness_mode",
                },
            }
            for role in roles
        ],
        "annotations": annotations,
    }


def croissant_row(
    recording_id: str,
    *,
    content_types: list[str],
    array_paths: list[str],
    frame_counts: dict[str, int] | None = None,
    source_recording_id: str | None = None,
    stitch_beat_count: int | None = None,
    exam_id: str = "exam",
) -> dict[str, object]:
    row: dict[str, object] = {
        "recordings/exam_id": exam_id,
        "recordings/recording_id": recording_id,
        "recordings/zarr_path": f"{recording_id}.zarr",
        "recordings/modes": content_types,
        "recordings/content_types": content_types,
        "recordings/frame_counts_by_content_type": frame_counts or {},
        "recordings/median_delta_time_by_content_type": {},
        "recordings/array_paths": array_paths,
    }
    if source_recording_id is not None:
        row["recordings/source_recording_id"] = source_recording_id
    if stitch_beat_count is not None:
        row["recordings/stitch_beat_count"] = stitch_beat_count
    return row


def write_croissant(
    root: Path,
    rows: list[dict[str, object]],
    *,
    recording_links: list[dict[str, object]] | None = None,
    splits: dict[str, int] | None = None,
    views: list[dict[str, object]] | None = None,
) -> None:
    """Write `croissant.json`, and `splits.csv` / `view_classifier_probabilities.csv` when given."""
    record_sets: list[dict[str, object]] = [{"@id": "recordings", "name": "recordings", "data": rows}]
    if recording_links is not None:
        record_sets.append({"@id": "recording_links", "name": "recording_links", "data": recording_links})
    (root / "croissant.json").write_text(json.dumps({"recordSet": record_sets}), encoding="utf-8")
    if splits is not None:
        _write_csv(root / "splits.csv", [{"exam_id": exam, "five_fold_split": fold} for exam, fold in splits.items()])
    if views is not None:
        _write_csv(root / "view_classifier_probabilities.csv", views)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
