"""Prediction-export manifest primitives."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from echoxflow.croissant import RecordingRecord
from echoxflow.manifest import normalize_public_manifest_attrs
from echoxflow.streams import temporal_axis_for_path, temporal_sample_count


@dataclass(frozen=True)
class RecordingArrayEntry:
    """One array entry in an EchoXFlow-compatible prediction recording manifest."""

    data_path: str
    shape: tuple[int, ...]
    dtype: str
    timestamps_path: str | None = None
    content_type: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PredictionManifest:
    """Minimal manifest for writing task predictions as an EchoXFlow recording."""

    exam_id: str
    recording_id: str
    source_recording_id: str | None
    arrays: tuple[RecordingArrayEntry, ...]
    attrs: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exam_id": self.exam_id,
            "recording_id": self.recording_id,
            "source_recording_id": self.source_recording_id,
            "arrays": [asdict(entry) for entry in self.arrays],
            "attrs": dict(self.attrs or {}),
        }


@dataclass(frozen=True)
class PredictionArray:
    """One prediction array plus optional timestamps to write into a recording store."""

    data_path: str
    values: np.ndarray
    timestamps: np.ndarray | None = None
    timestamps_path: str | None = None
    content_type: str | None = None
    metadata: Mapping[str, Any] | None = None

    def entry(self) -> RecordingArrayEntry:
        return prediction_array_entry(
            self.data_path,
            self.values,
            timestamps_path=self.resolved_timestamps_path,
            content_type=self.content_type,
            metadata=self.metadata,
        )

    @property
    def resolved_data_path(self) -> str:
        return _normalize_data_path(self.data_path)

    @property
    def resolved_timestamps_path(self) -> str | None:
        if self.timestamps is None:
            return None
        if self.timestamps_path is not None:
            return _normalize_timestamp_path(self.timestamps_path)
        return "timestamps/" + self.resolved_data_path.removeprefix("data/")


@dataclass(frozen=True)
class RecordingArray:
    """One concrete array to write into an EchoXFlow recording dump."""

    data_path: str
    values: np.ndarray
    timestamps: np.ndarray | None = None
    timestamps_path: str | None = None
    content_type: str | None = None
    attrs: Mapping[str, Any] | None = None

    def entry(self) -> RecordingArrayEntry:
        return recording_array_entry(
            self.data_path,
            self.values,
            timestamps_path=self.resolved_timestamps_path,
            content_type=self.content_type,
            metadata=self.attrs,
        )

    @property
    def resolved_data_path(self) -> str:
        return _normalize_data_path(self.data_path)

    @property
    def resolved_timestamps_path(self) -> str | None:
        if self.timestamps is None:
            return None
        if self.timestamps_path is not None:
            return _normalize_timestamp_path(self.timestamps_path)
        return "timestamps/" + self.resolved_data_path.removeprefix("data/")


def prediction_array_entry(
    data_path: str,
    values: np.ndarray,
    *,
    timestamps_path: str | None = None,
    content_type: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RecordingArrayEntry:
    """Build a typed manifest entry from a prediction array."""
    normalized_path = _normalize_data_path(data_path)
    return RecordingArrayEntry(
        data_path=normalized_path,
        shape=tuple(int(dim) for dim in np.asarray(values).shape),
        dtype=str(np.asarray(values).dtype),
        timestamps_path=timestamps_path,
        content_type=content_type or normalized_path.removeprefix("data/"),
        metadata=metadata,
    )


def recording_array_entry(
    data_path: str,
    values: np.ndarray,
    *,
    timestamps_path: str | None = None,
    content_type: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RecordingArrayEntry:
    """Build a typed manifest entry for a concrete recording array."""
    return prediction_array_entry(
        data_path,
        values,
        timestamps_path=timestamps_path,
        content_type=content_type,
        metadata=metadata,
    )


def build_prediction_manifest(
    *,
    exam_id: str,
    recording_id: str,
    arrays: tuple[RecordingArrayEntry, ...] | list[RecordingArrayEntry],
    source_recording_id: str | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> PredictionManifest:
    """Construct a prediction manifest without making task code hand-roll dictionaries."""
    if not arrays:
        raise ValueError("A prediction manifest requires at least one array entry")
    return PredictionManifest(
        exam_id=_non_empty(exam_id, "exam_id"),
        recording_id=_non_empty(recording_id, "recording_id"),
        source_recording_id=source_recording_id,
        arrays=tuple(arrays),
        attrs=normalize_public_manifest_attrs(attrs),
    )


def write_prediction_manifest(manifest: PredictionManifest, path: str | Path) -> Path:
    """Write a prediction manifest JSON document and return its path."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_prediction_recording(
    path: str | Path,
    *,
    exam_id: str,
    recording_id: str,
    arrays: Sequence[PredictionArray],
    source_recording_id: str | None = None,
    attrs: Mapping[str, Any] | None = None,
    zarr_path: str | None = None,
    croissant_path: str | Path | None = None,
    overwrite: bool = True,
) -> PredictionManifest:
    """Write prediction arrays and timestamps into a Zarr recording store."""
    if not arrays:
        raise ValueError("At least one prediction array is required")
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_prediction_manifest(
        exam_id=exam_id,
        recording_id=recording_id,
        source_recording_id=source_recording_id,
        arrays=[array.entry() for array in arrays],
        attrs=attrs,
    )
    zarr_module = import_module("zarr")
    group = zarr_module.open_group(output_path, mode="w" if overwrite else "a")
    group.attrs["exam_id"] = manifest.exam_id
    group.attrs["recording_id"] = manifest.recording_id
    group.attrs["source_recording_id"] = manifest.source_recording_id
    group.attrs["prediction_manifest"] = manifest.to_dict()
    for key, value in dict(manifest.attrs or {}).items():
        group.attrs[key] = value
    for array in arrays:
        _write_prediction_array(group, array, overwrite=overwrite)
    if croissant_path is not None:
        upsert_prediction_croissant_recording(
            croissant_path,
            prediction_croissant_row(manifest, zarr_path=zarr_path or str(output_path)),
        )
    return manifest


def write_recording(
    path: str | Path,
    *,
    exam_id: str,
    recording_id: str,
    arrays: Sequence[RecordingArray],
    source_recording_id: str | None = None,
    attrs: Mapping[str, Any] | None = None,
    zarr_path: str | None = None,
    croissant_path: str | Path | None = None,
    overwrite: bool = True,
) -> RecordingRecord:
    """Write a parseable EchoXFlow recording dump to Zarr and optionally upsert Croissant metadata."""
    if not arrays:
        raise ValueError("At least one recording array is required")
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zarr_module = import_module("zarr")
    group = zarr_module.open_group(output_path, mode="w" if overwrite else "a")
    for key, value in dict(attrs or {}).items():
        group.attrs[key] = value
    group.attrs["exam_id"] = _non_empty(exam_id, "exam_id")
    group.attrs["recording_id"] = _non_empty(recording_id, "recording_id")
    group.attrs["source_recording_id"] = source_recording_id
    for array in arrays:
        _write_recording_array(group, array, overwrite=overwrite)
    row = recording_croissant_row(
        exam_id=exam_id,
        recording_id=recording_id,
        arrays=arrays,
        source_recording_id=source_recording_id,
        zarr_path=zarr_path or str(output_path),
        attrs=attrs,
    )
    if croissant_path is not None:
        upsert_croissant_recording(croissant_path, row)
        replace_croissant_recording_links(
            croissant_path,
            source_recording_id=recording_id,
            rows=recording_link_croissant_rows(exam_id=exam_id, recording_id=recording_id, attrs=attrs),
        )
    return recording_record(row)


def prediction_recording_record(manifest: PredictionManifest, *, zarr_path: str) -> RecordingRecord:
    """Build a `RecordingRecord` for a prediction manifest."""
    row = prediction_croissant_row(manifest, zarr_path=zarr_path)
    return RecordingRecord(
        exam_id=str(row["recordings/exam_id"]),
        recording_id=str(row["recordings/recording_id"]),
        zarr_path=str(row["recordings/zarr_path"]),
        modes=tuple(str(value) for value in row["recordings/modes"]),
        content_types=tuple(str(value) for value in row["recordings/content_types"]),
        frame_counts_by_content_type={
            str(key): int(value) for key, value in row["recordings/frame_counts_by_content_type"].items()
        },
        median_delta_time_by_content_type={},
        array_paths=tuple(str(value) for value in row["recordings/array_paths"]),
        stitch_beat_count=_optional_positive_int(row.get("recordings/stitch_beat_count")),
        raw=row,
    )


def recording_record(row: Mapping[str, Any]) -> RecordingRecord:
    """Build a `RecordingRecord` from a Croissant recordings row."""
    return RecordingRecord(
        exam_id=str(row["recordings/exam_id"]),
        recording_id=str(row["recordings/recording_id"]),
        zarr_path=str(row["recordings/zarr_path"]),
        modes=tuple(str(value) for value in row["recordings/modes"]),
        content_types=tuple(str(value) for value in row["recordings/content_types"]),
        frame_counts_by_content_type={
            str(key): int(value) for key, value in row["recordings/frame_counts_by_content_type"].items()
        },
        median_delta_time_by_content_type={
            str(key): float(value) for key, value in row["recordings/median_delta_time_by_content_type"].items()
        },
        array_paths=tuple(str(value) for value in row["recordings/array_paths"]),
        stitch_beat_count=_optional_positive_int(row.get("recordings/stitch_beat_count")),
        raw=row,
    )


def prediction_croissant_row(manifest: PredictionManifest, *, zarr_path: str) -> dict[str, Any]:
    """Build the recordings row corresponding to a prediction manifest."""
    content_types = tuple(entry.content_type or entry.data_path.removeprefix("data/") for entry in manifest.arrays)
    array_paths: list[str] = []
    frame_counts: dict[str, int] = {}
    for entry in manifest.arrays:
        array_paths.append(entry.data_path)
        if entry.timestamps_path is not None:
            array_paths.append(entry.timestamps_path)
        content_type = entry.content_type or entry.data_path.removeprefix("data/")
        frame_counts[content_type] = int(entry.shape[0]) if entry.shape else 0
    row: dict[str, Any] = {
        "recordings/exam_id": manifest.exam_id,
        "recordings/recording_id": manifest.recording_id,
        "recordings/zarr_path": str(zarr_path),
        "recordings/modes": list(content_types),
        "recordings/content_types": list(content_types),
        "recordings/frame_counts_by_content_type": frame_counts,
        "recordings/median_delta_time_by_content_type": {},
        "recordings/array_paths": list(dict.fromkeys(array_paths)),
        "recordings/stitch_beat_count": None,
        "recordings/prediction_manifest": manifest.to_dict(),
    }
    if manifest.source_recording_id is not None:
        row["recordings/source_recording_id"] = manifest.source_recording_id
    return row


def recording_croissant_row(
    *,
    exam_id: str,
    recording_id: str,
    arrays: Sequence[RecordingArray],
    zarr_path: str,
    source_recording_id: str | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Croissant recordings row for a concrete recording dump."""
    content_types: list[str] = []
    array_paths: list[str] = []
    frame_counts: dict[str, int] = {}
    median_deltas: dict[str, float] = {}
    for array in arrays:
        entry = array.entry()
        content_type = entry.content_type or entry.data_path.removeprefix("data/")
        content_types.append(content_type)
        array_paths.append(entry.data_path)
        frame_counts[content_type] = _temporal_count_from_shape(entry.data_path, entry.shape)
        if entry.timestamps_path is not None:
            array_paths.append(entry.timestamps_path)
        delta = _median_delta_time(array.timestamps)
        if delta is not None:
            median_deltas[content_type] = delta
    row: dict[str, Any] = {
        "recordings/exam_id": _non_empty(exam_id, "exam_id"),
        "recordings/recording_id": _non_empty(recording_id, "recording_id"),
        "recordings/zarr_path": str(zarr_path),
        "recordings/modes": list(dict.fromkeys(content_types)),
        "recordings/content_types": list(dict.fromkeys(content_types)),
        "recordings/frame_counts_by_content_type": frame_counts,
        "recordings/median_delta_time_by_content_type": median_deltas,
        "recordings/array_paths": list(dict.fromkeys(array_paths)),
        "recordings/stitch_beat_count": _recording_stitch_beat_count(arrays=arrays, attrs=attrs),
    }
    if source_recording_id is not None:
        row["recordings/source_recording_id"] = source_recording_id
    return row


def recording_link_croissant_rows(
    *,
    exam_id: str,
    recording_id: str,
    attrs: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build Croissant recording link rows from a public recording manifest."""
    manifest = _recording_manifest(attrs)
    if manifest is None:
        return ()
    rows: list[dict[str, Any]] = []
    linked_volume = manifest.get("linked_volume")
    if isinstance(linked_volume, Mapping):
        row = _recording_link_row(
            exam_id=exam_id,
            source_recording_id=recording_id,
            linked=linked_volume,
            panel_role_id=_optional_str(linked_volume.get("panel_role_id")) or "3d_brightness_mode",
            default_frame_path="data/3d_brightness_mode",
            default_timestamp_path="timestamps/3d_brightness_mode",
        )
        if row is not None:
            rows.append(row)
    linked_panels = manifest.get("linked_panels")
    if isinstance(linked_panels, list):
        for item in linked_panels:
            if not isinstance(item, Mapping):
                continue
            linked = item.get("linked_recording")
            if not isinstance(linked, Mapping):
                continue
            row = _recording_link_row(
                exam_id=exam_id,
                source_recording_id=recording_id,
                linked=linked,
                panel_role_id=_optional_str(item.get("role_id")) or _optional_str(item.get("semantic_id")),
                default_frame_path="data/2d_brightness_mode",
                default_timestamp_path="timestamps/2d_brightness_mode",
            )
            if row is not None:
                rows.append(row)
    return tuple(rows)


def upsert_croissant_recording(path: str | Path, row: Mapping[str, Any]) -> Path:
    """Insert or replace one recording row in a Croissant `recordings` record set."""
    return upsert_prediction_croissant_recording(path, row)


def replace_croissant_recording_links(
    path: str | Path,
    *,
    source_recording_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Replace public Croissant links emitted by one derived recording."""
    metadata_path = Path(path).expanduser()
    if metadata_path.exists():
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        document = {"recordSet": []}
    record_sets = document.setdefault("recordSet", [])
    if not isinstance(record_sets, list):
        raise ValueError("Croissant metadata has invalid `recordSet`")
    links = _record_set(record_sets, "recording_links")
    existing_rows = links.setdefault("data", [])
    if not isinstance(existing_rows, list):
        raise ValueError("Croissant recording_links record set has invalid `data`")
    source_id = _non_empty(source_recording_id, "source_recording_id")
    retained = [
        row
        for row in existing_rows
        if not (isinstance(row, Mapping) and row.get("recording_links/source_recording_id") == source_id)
    ]
    retained.extend(dict(row) for row in rows)
    links["data"] = retained
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def upsert_prediction_croissant_recording(path: str | Path, row: Mapping[str, Any]) -> Path:
    """Insert or replace one prediction row in a Croissant `recordings` record set."""
    metadata_path = Path(path).expanduser()
    if metadata_path.exists():
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        document = {"recordSet": []}
    record_sets = document.setdefault("recordSet", [])
    if not isinstance(record_sets, list):
        raise ValueError("Croissant metadata has invalid `recordSet`")
    recordings = _record_set(record_sets, "recordings")
    rows = recordings.setdefault("data", [])
    if not isinstance(rows, list):
        raise ValueError("Croissant recordings record set has invalid `data`")
    recording_id = str(row["recordings/recording_id"])
    updated = False
    for index, existing in enumerate(rows):
        if isinstance(existing, Mapping) and existing.get("recordings/recording_id") == recording_id:
            rows[index] = dict(row)
            updated = True
            break
    if not updated:
        rows.append(dict(row))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def _recording_link_row(
    *,
    exam_id: str,
    source_recording_id: str,
    linked: Mapping[str, Any],
    panel_role_id: str | None,
    default_frame_path: str,
    default_timestamp_path: str,
) -> dict[str, Any] | None:
    linked_recording_id = (
        _optional_str(linked.get("recording_id"))
        or _optional_str(linked.get("linked_recording_id"))
        or _optional_str(linked.get("source_recording_id"))
    )
    if linked_recording_id is None:
        return None
    frame_path = (
        _optional_str(linked.get("frames_array_path"))
        or _optional_str(linked.get("frame_array_path"))
        or _optional_str(linked.get("data_path"))
        or default_frame_path
    )
    timestamp_path = (
        _optional_str(linked.get("timestamps_array_path"))
        or _optional_str(linked.get("timestamp_array_path"))
        or _optional_str(linked.get("timestamps_path"))
        or default_timestamp_path
    )
    return {
        "recording_links/exam_id": _non_empty(exam_id, "exam_id"),
        "recording_links/source_recording_id": _non_empty(source_recording_id, "source_recording_id"),
        "recording_links/linked_recording_id": linked_recording_id,
        "recording_links/panel_role_id": panel_role_id,
        "recording_links/linked_frame_array_id": _recording_array_id(linked_recording_id, frame_path),
        "recording_links/linked_timestamp_array_id": _recording_array_id(linked_recording_id, timestamp_path),
    }


def _write_prediction_array(group: Any, array: PredictionArray, *, overwrite: bool) -> None:
    values = np.asarray(array.values)
    entry = array.entry()
    if array.timestamps is not None:
        timestamps = np.asarray(array.timestamps, dtype=np.float32).reshape(-1)
        expected = temporal_sample_count(entry.data_path, values)
        if timestamps.size != expected:
            raise ValueError(
                f"{entry.data_path} has {expected} temporal samples but {timestamps.size} prediction timestamps"
            )
        group.create_array(entry.timestamps_path, data=timestamps, overwrite=overwrite)
    group.create_array(entry.data_path, data=values, overwrite=overwrite)
    zarr_array = group[entry.data_path]
    zarr_array.attrs["content_type"] = entry.content_type
    if entry.metadata is not None:
        zarr_array.attrs["metadata"] = dict(entry.metadata)


def _write_recording_array(group: Any, array: RecordingArray, *, overwrite: bool) -> None:
    values = np.asarray(array.values)
    entry = array.entry()
    if array.timestamps is not None:
        timestamps = np.asarray(array.timestamps, dtype=np.float32).reshape(-1)
        expected = temporal_sample_count(entry.data_path, values)
        if timestamps.size != expected:
            raise ValueError(f"{entry.data_path} has {expected} temporal samples but {timestamps.size} timestamps")
        group.create_array(entry.timestamps_path, data=timestamps, overwrite=overwrite)
    group.create_array(entry.data_path, data=values, overwrite=overwrite)
    zarr_array = group[entry.data_path]
    zarr_array.attrs["content_type"] = entry.content_type
    for key, value in dict(array.attrs or {}).items():
        zarr_array.attrs[key] = value


def _normalize_data_path(path: str) -> str:
    text = str(path).strip("/")
    return text if text.startswith("data/") else f"data/{text}"


def _normalize_timestamp_path(path: str) -> str:
    text = str(path).strip("/")
    if text.startswith("data/"):
        raise ValueError(f"Expected a timestamp path, got data path `{path}`")
    return text if text.startswith("timestamps/") else f"timestamps/{text}"


def _non_empty(value: str, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _recording_stitch_beat_count(*, arrays: Sequence[RecordingArray], attrs: Mapping[str, Any] | None) -> int | None:
    if not any(array.entry().content_type == "3d_brightness_mode" for array in arrays):
        return None
    for source in _stitch_beat_count_sources(attrs):
        count = _optional_positive_int(source.get("stitch_beat_count"))
        if count is not None:
            return count
    return None


def _stitch_beat_count_sources(attrs: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = []
    if isinstance(attrs, Mapping):
        sources.append(attrs)
    manifest = _recording_manifest(attrs)
    if manifest is not None:
        sources.append(manifest)
        metadata = manifest.get("metadata")
        if isinstance(metadata, Mapping):
            sources.append(metadata)
    return tuple(sources)


def _temporal_count_from_shape(data_path: str, shape: tuple[int, ...]) -> int:
    if not shape:
        return 0
    axis = temporal_axis_for_path(data_path, len(shape))
    return int(shape[axis])


def _median_delta_time(timestamps: np.ndarray | None) -> float | None:
    if timestamps is None:
        return None
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return None
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _recording_manifest(attrs: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(attrs, Mapping):
        return None
    manifest = attrs.get("recording_manifest")
    return manifest if isinstance(manifest, Mapping) else None


def _recording_array_id(recording_id: str, array_path: str) -> str:
    return f"{recording_id}/{_normalize_array_path(array_path)}"


def _normalize_array_path(path: str) -> str:
    text = str(path).strip("/")
    if text.startswith("data/") or text.startswith("timestamps/"):
        return text
    return _normalize_data_path(text)


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _record_set(record_sets: list[Any], record_set_id: str) -> dict[str, Any]:
    for existing in record_sets:
        if isinstance(existing, dict) and existing.get("@id") == record_set_id:
            return cast(dict[str, Any], existing)
    new_record_set: dict[str, Any] = {
        "@type": "cr:RecordSet",
        "@id": record_set_id,
        "name": record_set_id,
        "data": [],
    }
    record_sets.append(new_record_set)
    return new_record_set
