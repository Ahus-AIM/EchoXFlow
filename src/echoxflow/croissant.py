"""Croissant metadata utilities for EchoXFlow recording discovery."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import pickle
import secrets
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypedDict, Unpack, cast

from echoxflow.config import data_root
from echoxflow.schema import (
    ARRAY_GROUPS,
    ECHOPRIME_LABELS,
    NON_FRAME_ARRAY_ROLES,
    annotation_names,
    consensus_view,
    is_lookup_table_path,
    normalize_array_path,
    resolve_array_path,
    semantic_name,
)

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
    fold: int | None = None
    view_echoprime: str | None = None
    view_echoprime_prob: float | None = None
    view_echofine: str | None = None
    view_echofine_prob: float | None = None
    annotations: tuple[str, ...] = ()
    source_root: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> RecordingRecord:
        """Build a record from a Croissant `recordings` row (also used for freshly exported recordings)."""
        exam_id = _string(row, "recordings/exam_id")
        recording_id = _string(row, "recordings/recording_id")
        # Interned: the same few hundred names recur in every row, and the cache stores each one once.
        array_paths = tuple(sys.intern(str(value)) for value in row.get("recordings/array_paths", ()))
        content_types = tuple(sys.intern(str(value)) for value in row.get("recordings/content_types", ()))
        return cls(
            exam_id=exam_id,
            recording_id=recording_id,
            zarr_path=_string(row, "recordings/zarr_path"),
            modes=tuple(str(value) for value in row.get("recordings/modes", ())),
            content_types=content_types,
            frame_counts_by_content_type={
                str(key): int(value) for key, value in _dict(row, "recordings/frame_counts_by_content_type").items()
            },
            median_delta_time_by_content_type={
                str(key): float(value)
                for key, value in _dict(row, "recordings/median_delta_time_by_content_type").items()
            },
            array_paths=array_paths,
            stitch_beat_count=_optional_positive_int(row, "recordings/stitch_beat_count"),
            relationships=_parse_relationships(row, exam_id=exam_id, recording_id=recording_id),
            annotations=annotation_names(array_paths, content_types),
        )

    @property
    def view(self) -> str | None:
        """Coarse view (A2C, A3C, A4C, PLAX, PSAX, Other) on which EchoPrime and EchoFine agree, else None."""
        return self.view_for("consensus")

    def view_for(self, source: str = "consensus") -> str | None:
        """View label of one classifier: `echoprime` and `echofine` are their raw labels, `consensus` the coarse one."""
        if source == "echoprime":
            return self.view_echoprime
        if source == "echofine":
            return self.view_echofine
        if source == "consensus":
            return consensus_view(self.view_echofine, self.view_echoprime)
        raise ValueError(f"view source must be consensus, echoprime or echofine, got {source!r}")

    def view_confidence_for(self, source: str = "consensus") -> float | None:
        """Probability of the view label; the consensus confidence is the lower of the two classifiers."""
        if source == "echoprime":
            return self.view_echoprime_prob
        if source == "echofine":
            return self.view_echofine_prob
        if source != "consensus":
            raise ValueError(f"view source must be consensus, echoprime or echofine, got {source!r}")
        probs = (self.view_echoprime_prob, self.view_echofine_prob)
        return None if self.view is None or None in probs else min(cast(tuple[float, float], probs))

    def path(self, root: str | Path | None = None) -> Path:
        return data_root(root if root is not None else self.source_root) / self.zarr_path

    def has_mode(self, mode: str) -> bool:
        return mode in self.modes

    def has_content_type(self, content_type: str) -> bool:
        return content_type in self.content_types

    def has_array_path(self, array_path: str) -> bool:
        """Return whether the recording stores an array, accepting short names such as `data/tissue_doppler`."""
        return resolve_array_path(array_path, self.array_paths) is not None

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
    units: str | None = None
    coordinate_system: str | None = None

    @property
    def is_frame_array(self) -> bool:
        """Whether this array is a stream of images, as opposed to a table, coordinate or calibration vector.

        Per-frame calibration (`derived/*_velocity_scale_per_frame`) shares the time axis of its stream but holds one
        scalar per frame; counting it as frames would double the stream and misreport its length.
        """
        if not self.is_data_array or (self.role or "") in NON_FRAME_ARRAY_ROLES:
            return False
        return len(self.shape) >= 3 if self.shape else True

    @property
    def is_data_array(self) -> bool:
        """Whether this array holds recorded or derived values rather than timestamps or coordinates.

        Derived products such as colour Doppler velocity are published under
        `derived/`, and they carry frames just as `data/` arrays do.
        """
        return self.array_path.startswith(("data/", "derived/")) and not is_lookup_table_path(self.array_path)

    @property
    def is_timestamp_array(self) -> bool:
        return self.array_path.startswith("timestamps/")

    def temporal_count(self) -> int | None:
        if not self.shape:
            return None
        return int(self.shape[0])


@dataclass(frozen=True)
class StreamSummary:
    """Exporter-computed timing, geometry and calibration summary of one stream.

    One summary per stream is published twice: as the catalog `streams` record set, for preflight
    without opening stores, and as the store's root `stream_summaries`, next to the payload. Both describe the
    exporter's view of the stream; they are diagnostics and cheap eligibility hints, not a substitute for
    checking the payload itself. Absent fields are ``None``.
    """

    recording_id: str
    array_path: str
    timestamp_path: str | None = None
    sample_count: int | None = None
    clock_status: str | None = None
    duration_s: float | None = None
    median_delta_s: float | None = None
    largest_gap_s: float | None = None
    large_gap_count: int | None = None
    reset_count: int | None = None
    repeated_time_count: int | None = None
    nonfinite_time_count: int | None = None
    depth_start_m: float | None = None
    depth_end_m: float | None = None
    radial_spacing_m: float | None = None
    azimuth_spacing_rad: float | None = None
    elevation_spacing_rad: float | None = None
    velocity_scale_status: str | None = None
    velocity_scale_min_mps: float | None = None
    velocity_scale_max_mps: float | None = None
    velocity_scale_change_count: int | None = None
    excluded_source_frame_count: int | None = None
    heart_rate_bpm: float | None = None
    calibration_status: str | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.recording_id, self.array_path

    @property
    def has_clock_reset(self) -> bool:
        return bool(self.reset_count) or self.clock_status == "reset"

    @property
    def has_varying_velocity_scale(self) -> bool:
        return self.velocity_scale_status == "varying" or bool(self.velocity_scale_change_count)

    def issues(self) -> tuple[str, ...]:
        """Name the conditions a consumer must handle explicitly: these are flags, not rejections."""
        found: list[str] = []
        if self.has_clock_reset:
            found.append("clock_reset")
        if self.repeated_time_count:
            found.append("repeated_timestamps")
        if self.nonfinite_time_count:
            found.append("nonfinite_timestamps")
        if self.large_gap_count:
            found.append("large_gaps")
        if self.velocity_scale_status in {"varying", "partial"}:
            found.append(f"velocity_scale_{self.velocity_scale_status}")
        if self.calibration_status in {"varying", "partial"}:
            found.append(f"calibration_{self.calibration_status}")
        if self.excluded_source_frame_count:
            found.append("excluded_source_frames")
        return tuple(found)

    def velocity_scale_may_intersect(self, accepted: tuple[float, float]) -> bool | None:
        """Whether any frame's scale can lie within ``accepted``; ``None`` when the summary has no scale range."""
        if self.velocity_scale_min_mps is None or self.velocity_scale_max_mps is None:
            return None
        low, high = float(accepted[0]), float(accepted[1])
        return self.velocity_scale_max_mps >= low and self.velocity_scale_min_mps <= high


#: Summary fields compared by :func:`stream_summary_mismatches`; ``raw`` and identity fields are excluded.
_SUMMARY_VALUE_FIELDS: tuple[str, ...] = (
    "timestamp_path",
    "sample_count",
    "clock_status",
    "reset_count",
    "repeated_time_count",
    "nonfinite_time_count",
    "large_gap_count",
    "velocity_scale_status",
    "velocity_scale_change_count",
    "calibration_status",
    "excluded_source_frame_count",
)


def stream_summary_mismatches(catalog: StreamSummary, store: StreamSummary) -> tuple[str, ...]:
    """Return the fields on which a catalog summary and its store's own summary disagree.

    The store summary sits next to the payload and takes precedence for an opened store; a disagreement means the
    catalog was built from a different export state and its preflight verdicts should not be trusted.
    """
    if catalog.key != store.key:
        raise ValueError(f"Cannot compare summaries of different streams: {catalog.key} and {store.key}")
    return tuple(
        name
        for name in _SUMMARY_VALUE_FIELDS
        if getattr(catalog, name) is not None
        and getattr(store, name) is not None
        and getattr(catalog, name) != getattr(store, name)
    )


def stream_summaries_from_attrs(
    attrs: Mapping[str, Any], *, recording_id: str | None = None
) -> tuple[StreamSummary, ...]:
    """Parse a store's root `stream_summaries` list; stores without one yield an empty tuple."""
    rows = attrs.get("stream_summaries")
    if not isinstance(rows, list):
        return ()
    owner = recording_id or (str(attrs.get("recording_id")) if attrs.get("recording_id") else "")
    return tuple(_parse_stream_summary(row, recording_id=owner) for row in rows if isinstance(row, Mapping))


class RecordingFilters(TypedDict, total=False):
    """Keyword filters of :meth:`CroissantCatalog.filter` and :func:`find_recordings`."""

    exam_id: str
    recording_id: str
    modes: Iterable[str]
    content_types: Iterable[str]
    array_paths: Iterable[str]
    min_frame_counts: Mapping[str, int]
    max_frame_counts: Mapping[str, int]
    stitch_beat_count: int
    min_stitch_beat_count: int
    max_stitch_beat_count: int
    views: Iterable[str]
    view_source: str  # "consensus" (default), "echoprime" or "echofine"
    min_view_confidence: float
    folds: Iterable[int]
    exclude_folds: Iterable[int]
    annotated: Iterable[str]
    require_all: bool
    predicate: Callable[[RecordingRecord], bool]


@dataclass(frozen=True)
class CroissantCatalog:
    """In-memory index of EchoXFlow recordings from `croissant.json`."""

    path: Path
    recordings: tuple[RecordingRecord, ...]
    # `None` when the catalog came from the local cache: `arrays` and `stream_summaries` then re-read the manifest.
    _arrays: tuple[CroissantArrayRecord, ...] | None = field(default=(), repr=False, compare=False)
    _stream_summaries: tuple[StreamSummary, ...] | None = field(default=(), repr=False, compare=False)

    @property
    def arrays(self) -> tuple[CroissantArrayRecord, ...]:
        self._load_full()
        return cast(tuple[CroissantArrayRecord, ...], self._arrays)

    @property
    def stream_summaries(self) -> tuple[StreamSummary, ...]:
        self._load_full()
        return cast(tuple[StreamSummary, ...], self._stream_summaries)

    def _load_full(self) -> None:
        if self._arrays is None or self._stream_summaries is None:
            raw = _read_manifest(self.path)
            object.__setattr__(self, "_arrays", _parse_array_record_set(raw))
            object.__setattr__(self, "_stream_summaries", _parse_stream_record_set(raw))

    def stream_summary(self, recording_id: str, array_path: str) -> StreamSummary | None:
        """Return the catalog summary of one stream, accepting short array names."""
        index = self._stream_summary_index()
        normalized = normalize_array_path(array_path)
        direct = index.get((recording_id, normalized))
        if direct is not None:
            return direct
        available = {path for owner, path in index if owner == recording_id}
        resolved = resolve_array_path(normalized, available)
        return None if resolved is None else index.get((recording_id, resolved))

    def stream_summaries_for_recording(self, recording_id: str) -> tuple[StreamSummary, ...]:
        return tuple(summary for summary in self.stream_summaries if summary.recording_id == recording_id)

    def _stream_summary_index(self) -> Mapping[tuple[str, str], StreamSummary]:
        cached = self.__dict__.get("_summary_index")
        if cached is None:
            cached = {summary.key: summary for summary in self.stream_summaries}
            object.__setattr__(self, "_summary_index", cached)
        return cast(Mapping[tuple[str, str], StreamSummary], cached)

    def filter(self, **filters: Unpack[RecordingFilters]) -> tuple[RecordingRecord, ...]:
        """Select recordings; every keyword in :class:`RecordingFilters` narrows the result.

        ``modes``, ``content_types``, ``array_paths`` and ``annotated`` match any of the given values, or all of them
        with ``require_all``. ``views`` compares case-insensitively against the labels of ``view_source``; recordings
        without a label or fold never pass a view or fold filter.
        """
        f = filters
        require_all = bool(f.get("require_all", False))
        mode_set = set(f.get("modes") or ())
        content_type_set = set(f.get("content_types") or ())
        array_path_set = {normalize_array_path(path) for path in f.get("array_paths") or ()}
        annotated = set(f.get("annotated") or ())
        views = {str(view).lower() for view in f.get("views") or ()}
        folds = {int(fold) for fold in f.get("folds") or ()}
        excluded_folds = {int(fold) for fold in f.get("exclude_folds") or ()}
        source = f.get("view_source", "consensus")
        min_confidence = f.get("min_view_confidence")
        min_counts = _normalize_frame_count_map(f.get("min_frame_counts"))
        max_counts = _normalize_frame_count_map(f.get("max_frame_counts"))
        stitch, min_stitch, max_stitch = (
            f.get("stitch_beat_count"),
            f.get("min_stitch_beat_count"),
            f.get("max_stitch_beat_count"),
        )
        predicate = f.get("predicate")
        found: list[RecordingRecord] = []
        for record in self.recordings:
            if f.get("exam_id") is not None and record.exam_id != f["exam_id"]:
                continue
            if f.get("recording_id") is not None and record.recording_id != f["recording_id"]:
                continue
            if mode_set and not _matches(record.modes, mode_set, require_all=require_all):
                continue
            if content_type_set and not _matches(record.content_types, content_type_set, require_all=require_all):
                continue
            if array_path_set and not _matches_array_paths(record, array_path_set, require_all=require_all):
                continue
            if annotated and not _matches(record.annotations, annotated, require_all=require_all):
                continue
            if min_counts and not _matches_min_frame_counts(record, min_counts):
                continue
            if max_counts and not _matches_max_frame_counts(record, max_counts):
                continue
            if stitch is not None and record.stitch_beat_count != int(stitch):
                continue
            if min_stitch is not None and (
                record.stitch_beat_count is None or record.stitch_beat_count < int(min_stitch)
            ):
                continue
            if max_stitch is not None and (
                record.stitch_beat_count is None or record.stitch_beat_count > int(max_stitch)
            ):
                continue
            label = record.view_for(source)
            if views and (label is None or label.lower() not in views):
                continue
            confidence = record.view_confidence_for(source)
            if min_confidence is not None and (confidence is None or confidence < float(min_confidence)):
                continue
            if (folds or excluded_folds) and (
                record.fold is None or (folds and record.fold not in folds) or record.fold in excluded_folds
            ):
                continue
            if predicate is not None and not predicate(record):
                continue
            found.append(record)
        return tuple(found)

    def to_dataframe(self) -> Any:
        """One row per recording as a pandas DataFrame, for exploring the catalog; pandas is imported lazily.

        Columns: ids and `zarr_path`, `fold`, the consensus `view` and the raw EchoPrime / EchoFine labels with
        their probabilities, `modes`, `content_types` and `annotations` as `|`-joined strings, `n_frames_bmode`,
        `fps_bmode`, `is_3d`, and a boolean `has_<content type>` / `has_<annotation>` column for each name present.
        :meth:`select` maps filtered rows back to records.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped,unused-ignore]
        except ImportError as exc:
            raise ImportError("to_dataframe needs pandas: `uv sync --extra pandas` or `pip install pandas`") from exc
        types = sorted({name for record in self.recordings for name in record.content_types})
        annotations = sorted({name for record in self.recordings for name in record.annotations})
        rows = [
            {
                "exam_id": record.exam_id,
                "recording_id": record.recording_id,
                "zarr_path": record.zarr_path,
                "fold": record.fold,
                "view": record.view,
                "view_echoprime": record.view_echoprime,
                "view_echoprime_prob": record.view_echoprime_prob,
                "view_echofine": record.view_echofine,
                "view_echofine_prob": record.view_echofine_prob,
                "modes": "|".join(record.modes),
                "content_types": "|".join(record.content_types),
                "annotations": "|".join(record.annotations),
                "n_frames_bmode": record.frame_count("2d_brightness_mode"),
                "fps_bmode": record.sample_rate_hz("2d_brightness_mode"),
                "is_3d": record.has_mode("3d"),
                **{f"has_{name}": name in record.content_types for name in types},
                **{f"has_{name}": name in record.annotations for name in annotations},
            }
            for record in self.recordings
        ]
        return pd.DataFrame(rows)

    def select(self, frame: Any) -> tuple[RecordingRecord, ...]:
        """The records of the rows of a (filtered) :meth:`to_dataframe` frame, in row order."""
        by_id = {record.recording_id: record for record in self.recordings}
        return tuple(by_id[recording_id] for recording_id in frame["recording_id"])

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


def load_croissant(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    cache: bool = True,
    splits_csv: str | Path | None = None,
    view_csv: str | Path | None = None,
) -> CroissantCatalog:
    """Load the EchoXFlow Croissant recording catalog.

    The parsed recordings are cached under :func:`cache_dir` (keyed by the manifest and CSV signatures), which makes
    later loads take well under a second; ``arrays`` and ``stream_summaries`` then re-read the manifest on first use.
    ``cache=False`` neither reads nor writes the cache.

    Next to the manifest, ``splits.csv`` fills ``record.fold`` and
    ``view_classifier_probabilities.csv`` fills the view fields; explicit paths override them, relative ones resolve
    against the manifest directory. Missing files leave those fields ``None``.
    """
    manifest_path = Path(path).expanduser() if path is not None else croissant_path(root)
    source_root = Path(root).expanduser().resolve() if root is not None else manifest_path.resolve().parent
    splits = _sidecar(manifest_path, splits_csv, ("splits.csv",))
    views = _sidecar(manifest_path, view_csv, ("view_classifier_probabilities.csv",))
    cache_file = _cache_file(manifest_path, splits, views) if cache else None
    if cache_file is not None and (cached := _read_cache(cache_file)) is not None:
        return CroissantCatalog(
            manifest_path, tuple(replace(record, source_root=source_root) for record in cached), None, None
        )
    raw = _read_manifest(manifest_path)
    folds = {} if splits is None else load_exam_folds(splits, required=splits_csv is not None)
    labels = {} if views is None else _read_view_labels(views)
    recordings = tuple(
        replace(
            record,
            source_root=source_root,
            fold=folds.get(record.exam_id),
            **_view_fields(labels.get(record.recording_id)),
        )
        for record in _parse_recordings(raw, manifest_path)
    )
    if cache_file is not None:
        _write_cache(cache_file, recordings)
    return CroissantCatalog(
        manifest_path,
        recordings,
        _parse_array_record_set(raw),
        _parse_stream_record_set(raw),
    )


def _read_manifest(manifest_path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse Croissant manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"Croissant manifest must be a JSON object: {manifest_path}")
    for activity in _generating_activities(raw):
        check_schema_generation(activity.get("schema_version"), context=f"Croissant manifest {manifest_path}")
    return raw


MINIMUM_SCHEMA_GENERATION = 2
"""Oldest ``EchoXFlow-<n>`` store schema this library reads; the May 2026 release (0.0.0) is EchoXFlow-1."""

RELEASE_URL = "https://huggingface.co/datasets/Ahus-AIM/EchoXFlow"


def schema_generation(value: object) -> int | None:
    """Return ``n`` from an ``EchoXFlow-<n>`` schema version, or ``None`` when absent or not of that form."""
    if not isinstance(value, str):
        return None
    prefix, _, number = value.strip().partition("-")
    return int(number) if prefix == "EchoXFlow" and number.isdigit() else None


def check_schema_generation(value: object, *, context: str) -> None:
    """Reject data written by a schema older than :data:`MINIMUM_SCHEMA_GENERATION`.

    Older schemas open without errors but lack the arrays, documents and attributes this library expects, so they
    would load silently with missing streams and annotations. Newer generations are accepted: they add to the
    layout rather than change it. A missing or foreign value is accepted so hand-built stores keep working.
    """
    generation = schema_generation(value)
    if generation is not None and generation < MINIMUM_SCHEMA_GENERATION:
        raise ValueError(
            f"{context} was written by the {value} schema; this library needs EchoXFlow-{MINIMUM_SCHEMA_GENERATION} "
            f"or newer. Download EchoXFlow 1.0.0 or later from {RELEASE_URL}"
        )


def _generating_activities(raw: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    activities = raw.get("prov:wasGeneratedBy")
    if isinstance(activities, Mapping):
        return (activities,)
    if isinstance(activities, list):
        return tuple(item for item in activities if isinstance(item, Mapping))
    return ()


CACHE_FORMAT = 4  # invalidate caches predating schema validation and record source roots


def cache_dir() -> Path:
    """Catalog cache directory: `ECHOXFLOW_CACHE_DIR`, else `$XDG_CACHE_HOME/echoxflow`, else `~/.cache/echoxflow`."""
    configured = os.environ.get("ECHOXFLOW_CACHE_DIR")
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(configured or Path(base) / "echoxflow").expanduser()


def _cache_file(manifest_path: Path, *sidecars: Path | None) -> Path:
    def signature(path: Path | None) -> list[object] | None:
        try:
            stat = None if path is None else path.stat()
        except OSError:
            return None
        return None if stat is None else [str(path), stat.st_size, stat.st_mtime_ns]

    layout = [f.name for f in fields(RecordingRecord)]
    key = [CACHE_FORMAT, layout, str(manifest_path.resolve()), signature(manifest_path), *map(signature, sidecars)]
    return cache_dir() / f"catalog-{hashlib.sha256(json.dumps(key).encode()).hexdigest()[:32]}.pkl"


def _read_cache(cache_file: Path) -> tuple[RecordingRecord, ...] | None:
    try:
        with cache_file.open("rb") as stream:
            payload = pickle.load(stream)
        if payload["format"] == CACHE_FORMAT:
            return cast(tuple[RecordingRecord, ...], payload["recordings"])
    except FileNotFoundError:
        pass
    except Exception as exc:  # a damaged or foreign cache file only costs a full parse
        logger.warning("ignoring unreadable catalog cache %s: %s", cache_file, exc)
    return None


def _write_cache(cache_file: Path, recordings: tuple[RecordingRecord, ...]) -> None:
    temporary = cache_file.with_name(f".{cache_file.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            pickle.dump({"format": CACHE_FORMAT, "recordings": recordings}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_file)
    except OSError as exc:
        logger.warning("could not write catalog cache %s: %s", cache_file, exc)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def _parse_recordings(raw: Mapping[str, Any], manifest_path: Path) -> tuple[RecordingRecord, ...]:
    record_set = _optional_record_set(raw, "recordings")
    rows = None if record_set is None else record_set.get("data")
    if rows is None:
        logger.info("Croissant manifest has no recordings record set with inline data: %s", manifest_path)
        return ()
    if not isinstance(rows, list):
        raise ValueError(f"Croissant recordings record set has no inline data: {manifest_path}")
    try:
        recordings = tuple(RecordingRecord.from_row(row) for row in rows if isinstance(row, dict))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not parse Croissant recordings from {manifest_path}: {exc}") from exc
    by_recording_id = {record.recording_id: record for record in recordings}
    for relationship in _parse_recording_link_record_set(raw, recordings):
        record = by_recording_id.get(relationship.source_recording_id) or by_recording_id.get(
            relationship.target_recording_id
        )
        if record is not None:
            by_recording_id[record.recording_id] = replace(record, relationships=record.relationships + (relationship,))
    return tuple(by_recording_id.get(record.recording_id, record) for record in recordings)


def _sidecar(manifest_path: Path, explicit: str | Path | None, defaults: tuple[str, ...]) -> Path | None:
    """The explicit CSV (relative to the manifest directory), else the first default file that exists."""
    if explicit is not None:
        return manifest_path.parent / Path(explicit).expanduser()
    return next((candidate for name in defaults if (candidate := manifest_path.parent / name).is_file()), None)


_VIEW_COLUMNS = {
    "echoprime": ("echoprime_top1", "echoprime_top1_prob"),
    "echofine": ("echofine_top1", "echofine_top1_prob"),
}


def _read_view_labels(path: Path) -> dict[str, dict[str, tuple[str | None, float | None]]]:
    """Per recording id, the (label, probability) of each classifier; rows with a `skip_reason` are left out."""
    labels: dict[str, dict[str, tuple[str | None, float | None]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = {name: index for index, name in enumerate(next(reader, ()))}

        def column(row: list[str], name: str) -> str:
            return row[header[name]].strip() if name in header and header[name] < len(row) else ""

        for row in reader:
            if column(row, "skip_reason"):
                continue
            labels[column(row, "recording_id")] = {
                source: (
                    ECHOPRIME_LABELS.get(label := column(row, label_column), label) or None,
                    float(prob) if (prob := column(row, prob_column)) else None,
                )
                for source, (label_column, prob_column) in _VIEW_COLUMNS.items()
            }
    return labels


def _view_fields(labels: Mapping[str, tuple[str | None, float | None]] | None) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for source in _VIEW_COLUMNS:
        fields[f"view_{source}"], fields[f"view_{source}_prob"] = (labels or {}).get(source, (None, None))
    return fields


def load_exam_folds(path: str | Path, *, required: bool = True) -> dict[str, int]:
    """Exam id -> fold from a split CSV; without ``required``, a CSV holding other splits yields no folds."""
    split_path = Path(path).expanduser()
    with split_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "exam_id" not in fields:
            raise ValueError(f"Fold split CSV must contain exam_id: {split_path}")
        if "five_fold_split" not in fields:
            if not required:
                return {}
            raise ValueError(f"Fold split CSV must contain five_fold_split: {split_path}")
        fold_by_exam_id: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            exam_id = str(row.get("exam_id") or "").strip()
            if not exam_id:
                raise ValueError(f"Fold split CSV has an empty exam_id at row {row_number}: {split_path}")
            try:
                fold = int(str(row.get("five_fold_split") or "").strip())
            except ValueError as exc:
                raise ValueError(f"Fold split CSV has a non-integer fold at row {row_number}: {split_path}") from exc
            existing = fold_by_exam_id.get(exam_id)
            if existing is not None and existing != fold:
                raise ValueError(f"Fold split CSV has conflicting folds for {exam_id}: {split_path}")
            fold_by_exam_id[exam_id] = fold
    if not fold_by_exam_id:
        raise ValueError(f"Fold split CSV has no rows: {split_path}")
    return fold_by_exam_id


def _catalog(croissant: CroissantCatalog | None, path: str | Path | None, root: str | Path | None) -> CroissantCatalog:
    return croissant if croissant is not None else load_croissant(path, root=root)


def find_recordings(
    *,
    croissant: CroissantCatalog | None = None,
    path: str | Path | None = None,
    root: str | Path | None = None,
    **filters: Unpack[RecordingFilters],
) -> tuple[RecordingRecord, ...]:
    """Find recordings using common Croissant metadata filters."""
    return _catalog(croissant, path, root).filter(**filters)


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
    catalog = _catalog(croissant, path, root)
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
    normalized = normalize_array_path(data_path)
    paths: list[str] = []
    for relationship in record.relationships:
        if relationship.source_array_path == normalized and relationship.target_array_path:
            paths.append(relationship.target_array_path)
        if relationship.target_array_path == normalized and relationship.source_array_path:
            paths.append(relationship.source_array_path)
    name = semantic_name(normalized)
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
        recording_id=_string(row, "arrays/recording_id"),
        array_path=normalize_array_path(_string(row, "arrays/array_path")),
        content_types=tuple(str(value) for value in row.get("arrays/content_types", ())),
        role=_optional_string(row, "arrays/role"),
        dtype=_optional_string(row, "arrays/dtype"),
        shape=_shape(row.get("arrays/shape", ())),
        data_sha256=_optional_string(row, "arrays/data_sha256"),
        raw=row,
        units=_optional_string(row, "arrays/units"),
        coordinate_system=_optional_string(row, "arrays/coordinate_system"),
    )


def _parse_stream_record_set(raw: Mapping[str, Any]) -> tuple[StreamSummary, ...]:
    record_set = _optional_record_set(raw, "streams")
    rows = None if record_set is None else record_set.get("data")
    if not isinstance(rows, list):
        return ()
    summaries: list[StreamSummary] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            summaries.append(_parse_stream_summary(row))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Could not parse Croissant stream row: {exc}") from exc
    return tuple(summaries)


def _parse_stream_summary(row: Mapping[str, Any], *, recording_id: str | None = None) -> StreamSummary:
    """Parse a catalog `streams/...` row or a store `stream_summaries` entry (unprefixed keys)."""

    def field(name: str) -> Any:
        return row.get(f"streams/{name}", row.get(name))

    owner = _optional_string(row, "streams/recording_id", "recording_id") or recording_id
    array_id = _optional_string(row, "streams/array_id", "array_id")
    array_path = _optional_string(row, "array_path")
    if array_path is None and array_id is not None:
        owner = owner or _recording_id_from_array_id(array_id)
        array_path = _array_path_from_id(array_id, recording_id=owner or "")
    if not owner or array_path is None:
        raise ValueError("Stream summary needs a recording id and an array path")
    timestamp_path = _optional_string(row, "timestamp_path")
    timestamp_id = _optional_string(row, "streams/timestamp_array_id", "timestamp_array_id")
    if timestamp_path is None and timestamp_id is not None:
        timestamp_path = _array_path_from_id(timestamp_id, recording_id=owner)
    return StreamSummary(
        recording_id=owner,
        array_path=normalize_array_path(array_path),
        timestamp_path=None if timestamp_path is None else normalize_array_path(timestamp_path),
        sample_count=_summary_int(field("sample_count")),
        clock_status=_summary_text(field("clock_status")),
        duration_s=_summary_float(field("duration_s")),
        median_delta_s=_summary_float(field("median_delta_s")),
        largest_gap_s=_summary_float(field("largest_gap_s")),
        large_gap_count=_summary_int(field("large_gap_count")),
        reset_count=_summary_int(field("reset_count")),
        repeated_time_count=_summary_int(field("repeated_time_count")),
        nonfinite_time_count=_summary_int(field("nonfinite_time_count")),
        depth_start_m=_summary_float(field("depth_start_m")),
        depth_end_m=_summary_float(field("depth_end_m")),
        radial_spacing_m=_summary_float(field("radial_spacing_m")),
        azimuth_spacing_rad=_summary_float(field("azimuth_spacing_rad")),
        elevation_spacing_rad=_summary_float(field("elevation_spacing_rad")),
        velocity_scale_status=_summary_text(field("velocity_scale_status")),
        velocity_scale_min_mps=_summary_float(field("velocity_scale_min_mps")),
        velocity_scale_max_mps=_summary_float(field("velocity_scale_max_mps")),
        velocity_scale_change_count=_summary_int(field("velocity_scale_change_count")),
        excluded_source_frame_count=_summary_int(field("excluded_source_frame_count")),
        heart_rate_bpm=_summary_float(field("heart_rate_bpm")),
        calibration_status=_summary_text(field("calibration_status")),
        raw=row,
    )


def _summary_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    return int(value)


def _summary_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _summary_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parse_relationships(
    row: Mapping[str, Any],
    *,
    exam_id: str,
    recording_id: str,
) -> tuple[RecordingRelationship, ...]:
    relationships: list[RecordingRelationship] = []
    source_id = _optional_string(row, "recordings/source_recording_id")
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
        derived_id = _optional_string(row, "recording_links/source_recording_id")
        linked_frame_array_id = _optional_string(row, "recording_links/linked_frame_array_id")
        linked_timestamp_array_id = _optional_string(row, "recording_links/linked_timestamp_array_id")
        # The array ids are `<recording_id>/<array_path>`, so the link target is recoverable from them.
        linked_id = _optional_string(row, "recording_links/linked_recording_id") or _recording_id_from_array_id(
            linked_frame_array_id
        )
        if derived_id is None or linked_id is None:
            continue
        relationships.append(
            RecordingRelationship(
                exam_id=exam_by_recording.get(derived_id) or exam_by_recording.get(linked_id) or "",
                source_recording_id=linked_id,
                target_recording_id=derived_id,
                relationship_type="linked_recording",
                panel_role_id=_optional_string(row, "recording_links/panel_role_id"),
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


def _optional_string(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _optional_array_path(row: Mapping[str, Any], *keys: str) -> str | None:
    value = _optional_string(row, *keys)
    return None if value is None else normalize_array_path(value)


def _recording_id_from_array_id(value: str | None) -> str | None:
    """Return the recording id leading a Croissant array id, for example `recording_ab/data/x`."""
    if value is None:
        return None
    head = value.strip("/").split("/", 1)[0]
    return head if head.startswith("recording_") else None


def _array_path_from_id(value: str | None, *, recording_id: str) -> str | None:
    if value is None:
        return None
    text = value.strip("/")
    prefix = f"{recording_id}/"
    if text.startswith(prefix):
        text = text.removeprefix(prefix)
    for group in ARRAY_GROUPS:
        marker = f"/{group}/"
        if marker in text:
            text = f"{group}/{text.split(marker, 1)[1]}"
            break
    return normalize_array_path(text)


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


def _normalize_content_type(content_type: str) -> str:
    return semantic_name(content_type)


def _normalize_frame_count_map(frame_counts: Mapping[str, int] | None) -> dict[str, int]:
    return {_normalize_content_type(key): int(value) for key, value in (frame_counts or {}).items()}


def _matches_array_paths(record: RecordingRecord, wanted: set[str], *, require_all: bool) -> bool:
    available = set(record.array_paths)
    found = (resolve_array_path(path, available) is not None for path in wanted)
    return all(found) if require_all else any(found)


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
