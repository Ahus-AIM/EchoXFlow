"""Croissant metadata utilities for EchoXFlow recording discovery."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from echoxflow.config import data_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingRelationship:
    """Typed relationship between two recordings in the Croissant catalog."""

    exam_id: str
    source_recording_id: str
    target_recording_id: str
    relationship_type: str
    panel_role_id: str | None = None
    source_array_path: str | None = None
    target_array_path: str | None = None


@dataclass(frozen=True)
class RecordingRecord:
    """One recording entry from the EchoXFlow Croissant manifest."""

    exam_id: str
    recording_id: str
    zarr_path: str
    modes: tuple[str, ...]
    content_types: tuple[str, ...]
    frame_counts_by_content_type: Mapping[str, int]
    median_delta_time_by_content_type: Mapping[str, float]
    array_paths: tuple[str, ...]
    stitch_beat_count: int | None = None
    relationships: tuple[RecordingRelationship, ...] = ()
    raw: Mapping[str, Any] | None = None

    def path(self, root: str | Path | None = None) -> Path:
        return data_root(root) / self.zarr_path

    def has_mode(self, mode: str) -> bool:
        return mode in self.modes

    def has_content_type(self, content_type: str) -> bool:
        return content_type in self.content_types

    def has_array_path(self, array_path: str) -> bool:
        return _normalize_array_path(array_path) in self.array_paths

    def median_delta_time(self, content_type: str) -> float | None:
        return self.median_delta_time_by_content_type.get(content_type)

    def frame_count(self, content_type: str) -> int | None:
        """Return the Croissant frame count for a content type, if present."""
        return self.frame_counts_by_content_type.get(_normalize_content_type(content_type))

    def sample_rate_hz(self, content_type: str) -> float | None:
        delta = self.median_delta_time(content_type)
        if delta is None or delta <= 0.0:
            return None
        return 1.0 / delta

    @property
    def is_stitched_3d(self) -> bool:
        """Return whether this 3D recording advertises multi-beat stitching."""
        return self.stitch_beat_count is not None and self.stitch_beat_count > 1


@dataclass(frozen=True)
class CroissantArrayRecord:
    """One array entry from the EchoXFlow Croissant manifest."""

    recording_id: str
    array_path: str
    content_types: tuple[str, ...]
    role: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    data_sha256: str | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def is_data_array(self) -> bool:
        return self.array_path.startswith("data/")

    @property
    def is_timestamp_array(self) -> bool:
        return self.array_path.startswith("timestamps/")

    def temporal_count(self) -> int | None:
        if not self.shape:
            return None
        return int(self.shape[0])


@dataclass(frozen=True)
class CroissantCatalog:
    """In-memory index of EchoXFlow recordings from `croissant.json`."""

    path: Path
    recordings: tuple[RecordingRecord, ...]
    arrays: tuple[CroissantArrayRecord, ...] = ()

    def filter(
        self,
        *,
        exam_id: str | None = None,
        recording_id: str | None = None,
        modes: Iterable[str] | None = None,
        content_types: Iterable[str] | None = None,
        array_paths: Iterable[str] | None = None,
        min_frame_counts: Mapping[str, int] | None = None,
        max_frame_counts: Mapping[str, int] | None = None,
        stitch_beat_count: int | None = None,
        min_stitch_beat_count: int | None = None,
        max_stitch_beat_count: int | None = None,
        require_all: bool = False,
        predicate: Callable[[RecordingRecord], bool] | None = None,
    ) -> tuple[RecordingRecord, ...]:
        mode_set = set(modes or ())
        content_type_set = set(content_types or ())
        array_path_set = {_normalize_array_path(path) for path in array_paths or ()}
        min_counts = _normalize_frame_count_map(min_frame_counts)
        max_counts = _normalize_frame_count_map(max_frame_counts)
        found: list[RecordingRecord] = []
        for record in self.recordings:
            if exam_id is not None and record.exam_id != exam_id:
                continue
            if recording_id is not None and record.recording_id != recording_id:
                continue
            if mode_set and not _matches(record.modes, mode_set, require_all=require_all):
                continue
            if content_type_set and not _matches(record.content_types, content_type_set, require_all=require_all):
                continue
            if array_path_set and not _matches(record.array_paths, array_path_set, require_all=require_all):
                continue
            if min_counts and not _matches_min_frame_counts(record, min_counts):
                continue
            if max_counts and not _matches_max_frame_counts(record, max_counts):
                continue
            if stitch_beat_count is not None and record.stitch_beat_count != int(stitch_beat_count):
                continue
            if min_stitch_beat_count is not None and (
                record.stitch_beat_count is None or record.stitch_beat_count < int(min_stitch_beat_count)
            ):
                continue
            if max_stitch_beat_count is not None and (
                record.stitch_beat_count is None or record.stitch_beat_count > int(max_stitch_beat_count)
            ):
                continue
            if predicate is not None and not predicate(record):
                continue
            found.append(record)
        return tuple(found)

    def paths(self, root: str | Path | None = None) -> tuple[Path, ...]:
        return tuple(record.path(root) for record in self.recordings)

    def by_recording_id(self, recording_id: str, *, exam_id: str | None = None) -> RecordingRecord | None:
        for record in self.recordings:
            if record.recording_id == recording_id and (exam_id is None or record.exam_id == exam_id):
                return record
        return None

    def arrays_by_recording_id(self) -> dict[str, tuple[CroissantArrayRecord, ...]]:
        grouped: dict[str, list[CroissantArrayRecord]] = {}
        for array in self.arrays:
            grouped.setdefault(array.recording_id, []).append(array)
        return {recording_id: tuple(items) for recording_id, items in grouped.items()}

    def arrays_for_recording(self, recording_id: str) -> tuple[CroissantArrayRecord, ...]:
        return tuple(array for array in self.arrays if array.recording_id == recording_id)

    def relationships(self) -> tuple[RecordingRelationship, ...]:
        explicit = [relationship for record in self.recordings for relationship in record.relationships]
        explicit_pairs = {_relationship_key(relationship) for relationship in explicit}
        explicit_recording_pairs = {
            (relationship.exam_id, relationship.source_recording_id, relationship.target_recording_id)
            for relationship in explicit
        }
        inferred = [
            relationship
            for relationship in _inferred_relationships(self.recordings)
            if _relationship_key(relationship) not in explicit_pairs
            and (relationship.exam_id, relationship.source_recording_id, relationship.target_recording_id)
            not in explicit_recording_pairs
        ]
        return tuple(explicit + inferred)


def croissant_path(root: str | Path | None = None) -> Path:
    """Return the default Croissant manifest path under the configured data root."""
    return data_root(root) / "croissant.json"


def load_croissant(path: str | Path | None = None, *, root: str | Path | None = None) -> CroissantCatalog:
    """Load the EchoXFlow Croissant recording catalog."""
    manifest_path = Path(path).expanduser() if path is not None else croissant_path(root)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse Croissant manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"Croissant manifest must be a JSON object: {manifest_path}")
    record_set = _optional_record_set(raw, "recordings")
    if record_set is None:
        logger.info("Croissant manifest has no recordings record set: %s", manifest_path)
        return CroissantCatalog(path=manifest_path, recordings=(), arrays=_parse_array_record_set(raw))
    rows = record_set.get("data")
    if rows is None:
        logger.info("Croissant recordings record set has no inline data: %s", manifest_path)
        return CroissantCatalog(path=manifest_path, recordings=(), arrays=_parse_array_record_set(raw))
    if not isinstance(rows, list):
        raise ValueError(f"Croissant recordings record set has no inline data: {manifest_path}")
    try:
        recordings = tuple(_parse_recording(row) for row in rows if isinstance(row, dict))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not parse Croissant recordings from {manifest_path}: {exc}") from exc
    relationships = _parse_recording_link_record_set(raw, recordings)
    if relationships:
        by_recording_id = {record.recording_id: record for record in recordings}
        for relationship in relationships:
            record = by_recording_id.get(relationship.source_recording_id)
            if record is None:
                record = by_recording_id.get(relationship.target_recording_id)
            if record is None:
                continue
            by_recording_id[record.recording_id] = replace(record, relationships=record.relationships + (relationship,))
        recordings = tuple(by_recording_id.get(record.recording_id, record) for record in recordings)
    return CroissantCatalog(
        path=manifest_path,
        recordings=recordings,
        arrays=_parse_array_record_set(raw),
    )


def find_recordings(
    *,
    croissant: CroissantCatalog | None = None,
    path: str | Path | None = None,
    root: str | Path | None = None,
    exam_id: str | None = None,
    recording_id: str | None = None,
    mode: str | None = None,
    modes: Iterable[str] | None = None,
    content_type: str | None = None,
    content_types: Iterable[str] | None = None,
    array_path: str | None = None,
    array_paths: Iterable[str] | None = None,
    min_frame_counts: Mapping[str, int] | None = None,
    max_frame_counts: Mapping[str, int] | None = None,
    stitch_beat_count: int | None = None,
    min_stitch_beat_count: int | None = None,
    max_stitch_beat_count: int | None = None,
    require_all: bool = False,
    predicate: Callable[[RecordingRecord], bool] | None = None,
) -> tuple[RecordingRecord, ...]:
    """Find recordings using common Croissant metadata filters."""
    catalog = croissant if croissant is not None else load_croissant(path, root=root)
    return catalog.filter(
        exam_id=exam_id,
        recording_id=recording_id,
        modes=_with_single(modes, mode),
        content_types=_with_single(content_types, content_type),
        array_paths=_with_single(array_paths, array_path),
        min_frame_counts=min_frame_counts,
        max_frame_counts=max_frame_counts,
        stitch_beat_count=stitch_beat_count,
        min_stitch_beat_count=min_stitch_beat_count,
        max_stitch_beat_count=max_stitch_beat_count,
        require_all=require_all,
        predicate=predicate,
    )


def find_linked_recordings(
    record: RecordingRecord,
    *,
    croissant: CroissantCatalog | None = None,
    path: str | Path | None = None,
    root: str | Path | None = None,
    relationship_type: str | None = None,
    direction: str = "both",
) -> tuple[tuple[RecordingRelationship, RecordingRecord], ...]:
    """Find recordings linked to a record and return typed relationship records with matches."""
    catalog = croissant if croissant is not None else load_croissant(path, root=root)
    direction_text = direction.strip().lower()
    if direction_text not in {"both", "source", "derived", "target"}:
        raise ValueError("direction must be one of: both, source, derived, target")
    found: list[tuple[RecordingRelationship, RecordingRecord]] = []
    for relationship in catalog.relationships():
        if relationship_type is not None and relationship.relationship_type != relationship_type:
            continue
        source_match = relationship.source_recording_id == record.recording_id
        target_match = relationship.target_recording_id == record.recording_id
        if direction_text in {"derived", "target"} and source_match:
            linked = catalog.by_recording_id(relationship.target_recording_id, exam_id=record.exam_id)
        elif direction_text == "source" and target_match:
            linked = catalog.by_recording_id(relationship.source_recording_id, exam_id=record.exam_id)
        elif direction_text == "both" and source_match:
            linked = catalog.by_recording_id(relationship.target_recording_id, exam_id=record.exam_id)
        elif direction_text == "both" and target_match:
            linked = catalog.by_recording_id(relationship.source_recording_id, exam_id=record.exam_id)
        else:
            linked = None
        if linked is not None:
            found.append((relationship, linked))
    return tuple(found)


def find_source_recordings(
    record: RecordingRecord,
    *,
    croissant: CroissantCatalog | None = None,
    path: str | Path | None = None,
    root: str | Path | None = None,
    content_type: str | None = None,
) -> tuple[RecordingRecord, ...]:
    """Find source recordings linked to a derived annotation, mesh, or prediction recording."""
    linked = find_linked_recordings(record, croissant=croissant, path=path, root=root, direction="source")
    records = tuple(match for _, match in linked)
    if content_type is None:
        return records
    return tuple(match for match in records if match.has_content_type(content_type))


def find_derived_recordings(
    record: RecordingRecord,
    *,
    croissant: CroissantCatalog | None = None,
    path: str | Path | None = None,
    root: str | Path | None = None,
    content_types: Iterable[str] | None = None,
) -> tuple[RecordingRecord, ...]:
    """Find derived annotation, mesh, or prediction recordings linked from a source recording."""
    content_type_set = set(content_types or ())
    linked = find_linked_recordings(record, croissant=croissant, path=path, root=root, direction="derived")
    records = tuple(match for _, match in linked)
    if not content_type_set:
        return records
    return tuple(match for match in records if content_type_set & set(match.content_types))


def linked_frame_timestamp_paths(record: RecordingRecord, data_path: str) -> tuple[str, ...]:
    """Return timestamp paths explicitly linked to a data path, followed by conventional fallbacks."""
    normalized = _normalize_array_path(data_path)
    paths: list[str] = []
    for relationship in record.relationships:
        if relationship.source_array_path == normalized and relationship.target_array_path:
            paths.append(relationship.target_array_path)
        if relationship.target_array_path == normalized and relationship.source_array_path:
            paths.append(relationship.source_array_path)
    name = normalized.removeprefix("data/")
    paths.append(f"timestamps/{name}")
    for suffix in ("_velocity", "_power"):
        if name.endswith(suffix):
            paths.append(f"timestamps/{name[: -len(suffix)]}")
    return tuple(dict.fromkeys(paths))


def _optional_record_set(raw: Mapping[str, Any], record_set_id: str) -> Mapping[str, Any] | None:
    record_sets = raw.get("recordSet")
    if not isinstance(record_sets, list):
        return None
    for record_set in record_sets:
        if isinstance(record_set, dict) and record_set.get("@id") == record_set_id:
            return record_set
    return None


def _parse_recording(row: Mapping[str, Any]) -> RecordingRecord:
    exam_id = _string(row, "recordings/exam_id")
    recording_id = _string(row, "recordings/recording_id")
    return RecordingRecord(
        exam_id=exam_id,
        recording_id=recording_id,
        zarr_path=_string(row, "recordings/zarr_path"),
        modes=tuple(str(value) for value in row.get("recordings/modes", ())),
        content_types=tuple(str(value) for value in row.get("recordings/content_types", ())),
        frame_counts_by_content_type={
            str(key): int(value) for key, value in _dict(row, "recordings/frame_counts_by_content_type").items()
        },
        median_delta_time_by_content_type={
            str(key): float(value) for key, value in _dict(row, "recordings/median_delta_time_by_content_type").items()
        },
        array_paths=tuple(str(value) for value in row.get("recordings/array_paths", ())),
        stitch_beat_count=_optional_positive_int(row, "recordings/stitch_beat_count", "stitch_beat_count"),
        relationships=_parse_relationships(row, exam_id=exam_id, recording_id=recording_id),
        raw=row,
    )


def _parse_array_record_set(raw: Mapping[str, Any]) -> tuple[CroissantArrayRecord, ...]:
    record_set = _optional_record_set(raw, "arrays")
    if record_set is None:
        return ()
    rows = record_set.get("data")
    if not isinstance(rows, list):
        return ()
    arrays: list[CroissantArrayRecord] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            arrays.append(_parse_array(row))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Could not parse Croissant array row: {exc}") from exc
    return tuple(arrays)


def _parse_array(row: Mapping[str, Any]) -> CroissantArrayRecord:
    return CroissantArrayRecord(
        recording_id=_string_any(row, "arrays/recording_id", "recording_id"),
        array_path=_normalize_array_path(_string_any(row, "arrays/array_path", "array_path", "name")),
        content_types=tuple(str(value) for value in row.get("arrays/content_types", row.get("content_types", ()))),
        role=_optional_string(row, "arrays/role", "role"),
        dtype=_optional_string(row, "arrays/dtype", "dtype"),
        shape=_shape(row.get("arrays/shape", row.get("shape", ()))),
        data_sha256=_optional_string(row, "arrays/data_sha256", "data_sha256"),
        raw=row,
    )


def _parse_relationships(
    row: Mapping[str, Any],
    *,
    exam_id: str,
    recording_id: str,
) -> tuple[RecordingRelationship, ...]:
    relationships: list[RecordingRelationship] = []
    source_id = _optional_string(
        row,
        "recordings/source_recording_id",
        "recordings/derived_from_recording_id",
        "recordings/source_bmode_recording_id",
    )
    if source_id is not None:
        relationships.append(
            RecordingRelationship(
                exam_id=exam_id,
                source_recording_id=source_id,
                target_recording_id=recording_id,
                relationship_type="derived_from",
            )
        )
    for item in row.get("recordings/relationships", ()) or row.get("recordings/links", ()):
        if not isinstance(item, Mapping):
            continue
        source = _optional_string(item, "source_recording_id", "source", "from") or source_id
        target = _optional_string(item, "target_recording_id", "target", "to") or recording_id
        if source is None or target is None:
            continue
        relationships.append(
            RecordingRelationship(
                exam_id=exam_id,
                source_recording_id=source,
                target_recording_id=target,
                relationship_type=_optional_string(item, "relationship_type", "type", "name") or "linked",
                panel_role_id=_optional_string(item, "panel_role_id"),
                source_array_path=_optional_array_path(item, "source_array_path", "source_path", "from_path"),
                target_array_path=_optional_array_path(item, "target_array_path", "target_path", "to_path"),
            )
        )
    return tuple(relationships)


def _parse_recording_link_record_set(
    raw: Mapping[str, Any],
    recordings: tuple[RecordingRecord, ...],
) -> tuple[RecordingRelationship, ...]:
    record_set = _optional_record_set(raw, "recording_links")
    if record_set is None:
        return ()
    rows = record_set.get("data")
    if not isinstance(rows, list):
        return ()
    exam_by_recording = {record.recording_id: record.exam_id for record in recordings}
    relationships: list[RecordingRelationship] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        derived_id = _optional_string(row, "recording_links/source_recording_id", "source_recording_id")
        linked_id = _optional_string(row, "recording_links/linked_recording_id", "linked_recording_id")
        if derived_id is None or linked_id is None:
            continue
        linked_frame_array_id = _optional_string(row, "recording_links/linked_frame_array_id", "linked_frame_array_id")
        linked_timestamp_array_id = _optional_string(
            row,
            "recording_links/linked_timestamp_array_id",
            "linked_timestamp_array_id",
        )
        relationships.append(
            RecordingRelationship(
                exam_id=exam_by_recording.get(derived_id) or exam_by_recording.get(linked_id) or "",
                source_recording_id=linked_id,
                target_recording_id=derived_id,
                relationship_type="linked_recording",
                panel_role_id=_optional_string(row, "recording_links/panel_role_id", "panel_role_id"),
                source_array_path=_array_path_from_id(linked_frame_array_id, recording_id=linked_id),
                target_array_path=_array_path_from_id(linked_timestamp_array_id, recording_id=linked_id),
            )
        )
    return tuple(relationships)


def _inferred_relationships(recordings: tuple[RecordingRecord, ...]) -> tuple[RecordingRelationship, ...]:
    inferred: list[RecordingRelationship] = []
    by_exam: dict[str, list[RecordingRecord]] = {}
    for record in recordings:
        by_exam.setdefault(record.exam_id, []).append(record)
    for exam_id, records in by_exam.items():
        sources = [record for record in records if _looks_like_source_bmode(record)]
        derived = [record for record in records if _looks_like_derived_annotation(record)]
        if len(sources) != 1:
            continue
        source = sources[0]
        for target in derived:
            if target.recording_id == source.recording_id:
                continue
            inferred.append(
                RecordingRelationship(
                    exam_id=exam_id,
                    source_recording_id=source.recording_id,
                    target_recording_id=target.recording_id,
                    relationship_type="inferred_source_bmode",
                )
            )
    return tuple(inferred)


def _looks_like_source_bmode(record: RecordingRecord) -> bool:
    return record.has_content_type("2d_brightness_mode") or record.has_array_path("data/2d_brightness_mode")


def _looks_like_derived_annotation(record: RecordingRecord) -> bool:
    text = " ".join((*record.content_types, *record.array_paths)).lower()
    return any(token in text for token in ("annotation", "mesh", "mask", "segmentation"))


def _relationship_key(relationship: RecordingRelationship) -> tuple[str, str, str, str | None, str | None, str | None]:
    return (
        relationship.exam_id,
        relationship.source_recording_id,
        relationship.target_recording_id,
        relationship.panel_role_id,
        relationship.source_array_path,
        relationship.target_array_path,
    )


def _string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid Croissant field `{key}`")
    return value


def _string_any(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Missing or invalid Croissant field `{keys[0]}`")


def _optional_string(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _optional_array_path(row: Mapping[str, Any], *keys: str) -> str | None:
    value = _optional_string(row, *keys)
    return None if value is None else _normalize_array_path(value)


def _array_path_from_id(value: str | None, *, recording_id: str) -> str | None:
    if value is None:
        return None
    text = value.strip("/")
    prefix = f"{recording_id}/"
    if text.startswith(prefix):
        text = text.removeprefix(prefix)
    for marker in ("/data/", "/timestamps/"):
        if marker in text:
            text = text.split(marker, 1)[1]
            text = f"{marker.strip('/')}/{text}"
            break
    return _normalize_array_path(text)


def _optional_positive_int(row: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _dict(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid Croissant field `{key}`")
    return value


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        return ()
    shape: list[int] = []
    for item in value:
        try:
            dimension = int(item)
        except (TypeError, ValueError):
            return ()
        if dimension < 0:
            return ()
        shape.append(dimension)
    return tuple(shape)


def _normalize_array_path(path: str) -> str:
    path = str(path).strip("/")
    if path.startswith("data/") or path.startswith("timestamps/"):
        return path
    return f"data/{path}"


def _normalize_content_type(content_type: str) -> str:
    text = str(content_type).strip("/")
    for prefix in ("data/", "timestamps/"):
        if text.startswith(prefix):
            return text.removeprefix(prefix)
    return text


def _normalize_frame_count_map(frame_counts: Mapping[str, int] | None) -> dict[str, int]:
    return {_normalize_content_type(key): int(value) for key, value in (frame_counts or {}).items()}


def _matches_min_frame_counts(record: RecordingRecord, min_counts: Mapping[str, int]) -> bool:
    return all((record.frame_count(content_type) or 0) >= minimum for content_type, minimum in min_counts.items())


def _matches_max_frame_counts(record: RecordingRecord, max_counts: Mapping[str, int]) -> bool:
    return all(
        (count := record.frame_count(content_type)) is not None and count <= maximum
        for content_type, maximum in max_counts.items()
    )


def _matches(values: Iterable[str], filters: set[str], *, require_all: bool) -> bool:
    value_set = set(values)
    return filters <= value_set if require_all else bool(filters & value_set)


def _with_single(values: Iterable[str] | None, value: str | None) -> tuple[str, ...]:
    merged = list(values or ())
    if value is not None:
        merged.append(value)
    return tuple(merged)
