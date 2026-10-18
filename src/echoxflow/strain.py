"""Structured access to exported regional strain sources, in their declared order.

These sources are not an AHA 17-segment product: the current export does not resolve
the anatomical ordering, apex reduction or scanner's strain summary convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from echoxflow.croissant import RecordingRecord
from echoxflow.loading import RecordingStore, open_recording
from echoxflow.manifest import manifest_documents


@dataclass(frozen=True)
class AFIContourResidue:
    """Guarded native contour header retained accidentally in a curve stack.

    This encoding is not a public export contract. Callers must validate the
    retained points against the linked contour before using its view identifier.
    """

    record_index: int
    header_words: tuple[int, ...]
    view_id: int
    view: str
    points_per_wall: int
    wall_rows: int
    xy: np.ndarray


@dataclass(frozen=True)
class SourceStrainMetadata:
    """Validated source view and unmodified reference candidates; no AHA mapping."""

    source_view: str
    source_view_id: int
    declared_view: str | None
    view_conflict: bool
    residue_record_indices: tuple[int, ...]
    biological_record_indices: tuple[int, ...]
    global_record_index: int | None
    global_values: np.ndarray | None
    common_zero_frame_indices: tuple[int, ...]
    first_common_zero_frame_index: int | None
    last_common_zero_frame_index: int | None
    reference_status: str
    provenance: Mapping[str, Any]


def decode_afi_contour_residue(source: np.ndarray) -> AFIContourResidue | None:
    """Return one guarded automated function imaging (AFI) contour residue, or None on any ambiguity.

    Its native view enum is 1=APLAX/A3C, 2=A4C, 3=A2C. This was checked
    against original AFI payload headers, independently of scanner JPEG values.
    """
    if source.ndim != 2 or source.dtype.kind != "f" or source.dtype.itemsize != 4:
        return None
    views = {1: "A3C", 2: "A4C", 3: "A2C"}
    candidates = []
    for index, row in enumerate(source):
        if row.size < 12:
            continue
        raw = np.asarray(row, dtype="<f4")
        words = raw.view("<u4")
        if words[:3].tolist() != [0, 0, 1072693248]:
            continue
        view_id = int(words[3])
        if view_id not in views or int(words[4]) < 1 or words[5:8].tolist() != [0, 5, 0]:
            continue
        count = (row.size - 8) // 4
        records = raw[8 : 8 + 4 * count].reshape(-1, 4)
        if not np.all(records.view("<u4")[:, :2] == np.asarray([2, 1])):
            continue
        xy = records[:, 2:].copy()
        if not np.isfinite(xy).all() or count > int(words[4]) * 5:
            continue
        candidates.append(
            AFIContourResidue(
                index, tuple(int(word) for word in words[:8]), view_id, views[view_id], int(words[4]), 5, xy
            )
        )
    return candidates[0] if len(candidates) == 1 else None


def decode_afi_source_metadata(
    source: np.ndarray,
    first_contour_frame: np.ndarray,
    *,
    source_to_canonical_transform: str | None,
    record_roles: tuple[str, ...] = (),
    declared_view: str | None = None,
    contour_path: str | None = None,
) -> SourceStrainMetadata | None:
    """Decode source facts only after contour dimensions and bits agree.

    Common-zero frames are candidates for ED and cycle closure. The export does
    not retain the original selected-beat/ED/AVC state; no beat is chosen here.
    """
    residue = decode_afi_contour_residue(source)
    if residue is None or first_contour_frame.shape != (residue.points_per_wall * 5, 2):
        return None
    if first_contour_frame.dtype.kind != "f" or first_contour_frame.dtype.itemsize != 4:
        return None
    expected = residue.xy.copy()
    if source_to_canonical_transform == "reflect_x":
        expected[:, 0] *= -1
    elif source_to_canonical_transform not in (None, "identity"):
        return None
    actual = np.asarray(first_contour_frame[: len(expected)], dtype="<f4")
    if not np.array_equal(expected.view("<u4"), actual.view("<u4")):
        return None
    biological = tuple(index for index in range(source.shape[0]) if index != residue.record_index)
    zero_frames = tuple(int(index) for index in np.flatnonzero(np.all(source[list(biological)] == 0, axis=0)))
    global_index = 0 if record_roles and record_roles[0] == "opener" and 0 in biological else None
    declared_raw = declared_view
    if declared_view is not None:
        declared_view = {"4CH": "A4C", "2CH": "A2C", "3CH": "A3C", "APLAX": "A3C"}.get(declared_view, declared_view)
    return SourceStrainMetadata(
        source_view=residue.view,
        source_view_id=residue.view_id,
        declared_view=declared_view,
        view_conflict=declared_view is not None and declared_view != residue.view,
        residue_record_indices=(residue.record_index,),
        biological_record_indices=biological,
        global_record_index=global_index,
        global_values=None if global_index is None else source[global_index],
        common_zero_frame_indices=zero_frames,
        first_common_zero_frame_index=zero_frames[0] if zero_frames else None,
        last_common_zero_frame_index=zero_frames[-1] if zero_frames else None,
        reference_status="common_zero_candidates" if zero_frames else "unresolved",
        provenance={
            "method": "guarded_native_afi_contour_header_and_bitwise_first_contour_points",
            "header_words": residue.header_words,
            "contour_path": contour_path,
            "source_to_canonical_transform": source_to_canonical_transform,
            "declared_view_raw": declared_raw,
            "peak_summary_status": "unresolved",
            "anatomical_segment_labels_status": "unresolved",
        },
    )


@dataclass(frozen=True)
class SegmentalStrainPanel:
    """One declared regional selection and its unmodified source stack and clock.

    ``values`` is [selected record, native time], with rows addressed by
    ``record_indices``. ``header_values`` is an alternative source family, not a
    validated replacement. Units and anatomical status are the export's declarations.
    """

    semantic_id: str
    values: np.ndarray
    timestamps: np.ndarray
    record_indices: tuple[int, ...]
    source_values: np.ndarray
    record_roles: tuple[str, ...]
    header_record_indices: tuple[int, ...]
    header_values: np.ndarray
    units: str | None
    alignment_status: str | None
    anatomical_segment_labels_status: str
    curve_record_order: str | None
    value_path: str
    timestamps_path: str
    linked_panel: Mapping[str, Any] | None
    qrs_times: np.ndarray | None
    qrs_timestamps_path: str | None
    provenance: Mapping[str, Any]
    source_metadata: SourceStrainMetadata | None = None


@dataclass(frozen=True)
class SegmentalStrain:
    """Regional sources from one recording; panels keep their independent clocks."""

    recording_id: str | None
    store_path: Path
    panels: tuple[SegmentalStrainPanel, ...]


def load_segmental_strain(
    record: RecordingStore | RecordingRecord | str | Path,
    *,
    root: str | Path | None = None,
) -> SegmentalStrain:
    """Read explicitly declared ``aps_segmental_curves`` selections.

    An empty panel tuple means no such selection is declared. Malformed or missing
    referenced arrays raise an error rather than guessing from stack dimensions.
    No strain units, AHA labels, clock alignment or peak convention are inferred.
    Caller-owned stores remain open; stores opened here are closed after loading.
    """
    store = record if isinstance(record, RecordingStore) else open_recording(record, root=root)
    try:
        attrs = store.group.attrs
        panels = tuple(
            _load_panel(store, annotation, document)
            for document in manifest_documents(attrs)
            for annotation in document.get("annotations", [])
            if isinstance(annotation, Mapping) and annotation.get("kind") == "aps_segmental_curves"
        )
        recording_id = store.record.recording_id if store.record is not None else attrs.get("recording_id")
        return SegmentalStrain(recording_id=recording_id, store_path=store.path, panels=panels)
    finally:
        if not isinstance(record, RecordingStore):
            store.close()


def _load_panel(
    store: RecordingStore, annotation: Mapping[str, Any], document: Mapping[str, Any]
) -> SegmentalStrainPanel:
    value_path = _reference_path(annotation, "value")
    timestamps_path = _reference_path(annotation, "time")
    links = annotation.get("links", {})
    if not isinstance(links, Mapping) or links.get("curve_record_axis") != 0:
        raise ValueError(f"{value_path}: expected declared curve_record_axis=0")
    source = store.load_array(value_path)
    timestamps = store.load_array(timestamps_path)
    if source.ndim != 2 or timestamps.ndim != 1 or source.shape[1] != timestamps.size:
        raise ValueError(f"{value_path}: expected [record, time] stack matching a 1D native clock")
    indices = _record_indices(links.get("curve_record_indices"), source.shape[0], value_path)
    roles = _record_roles(document, value_path, source.shape[0])
    headers = tuple(index for index, role in enumerate(roles) if role == "aux_header")
    semantic_id = str(annotation.get("target", {}).get("semantic_id", ""))
    panel_id = links.get("linked_panel_semantic_id")
    panel = next(
        (
            item
            for item in document.get("linked_panels", [])
            if isinstance(item, Mapping) and panel_id is not None and item.get("semantic_id") == panel_id
        ),
        None,
    )
    qrs_annotation = next(
        (
            item
            for item in document.get("annotations", [])
            if isinstance(item, Mapping)
            and item.get("target", {}).get("semantic_id") == panel_id
            and item.get("target", {}).get("field") == "ecg_qrs"
            and isinstance(item.get("time"), Mapping)
        ),
        None,
    )
    qrs_path = None if qrs_annotation is None else _reference_path(qrs_annotation, "time")
    qrs = None if qrs_path is None else store.load_array(qrs_path)
    if qrs is not None and qrs.ndim != 1:
        raise ValueError(f"{qrs_path}: expected a 1D QRS timestamp array")
    info = store.array_info(value_path)
    source_metadata = None
    contours = [
        item
        for item in document.get("annotations", [])
        if isinstance(item, Mapping)
        and item.get("kind") == "aps_contour"
        and item.get("target", {}).get("semantic_id") == semantic_id
    ]
    if len(contours) == 1 and decode_afi_contour_residue(source) is not None:
        contour_annotation = contours[0]
        contour_path = _reference_path(contour_annotation, "value")
        contour_shape = store.array_shape(contour_path)
        if len(contour_shape) == 3 and contour_shape[0] > 0:
            first_frame = store.load_array_slice(contour_path, 0, 1)[0]
            declared = links.get("view_code")
            if declared is None and panel is not None:
                declared = panel.get("view_code")
            source_metadata = decode_afi_source_metadata(
                source,
                first_frame,
                source_to_canonical_transform=contour_annotation.get("links", {}).get("source_to_canonical_transform"),
                record_roles=roles,
                declared_view=declared if isinstance(declared, str) else None,
                contour_path=contour_path,
            )
    return SegmentalStrainPanel(
        semantic_id=semantic_id,
        values=source[list(indices)],
        timestamps=timestamps,
        record_indices=indices,
        source_values=source,
        record_roles=roles,
        header_record_indices=headers,
        header_values=source[list(headers)],
        units=None if info is None else info.units,
        alignment_status=links.get("alignment_status"),
        anatomical_segment_labels_status=str(links.get("anatomical_segment_labels_status", "unresolved")),
        curve_record_order=links.get("curve_record_order"),
        value_path=value_path,
        timestamps_path=timestamps_path,
        linked_panel=panel,
        qrs_times=qrs,
        qrs_timestamps_path=qrs_path,
        provenance=dict(annotation),
        source_metadata=source_metadata,
    )


def _reference_path(annotation: Mapping[str, Any], field: str) -> str:
    reference = annotation.get(field)
    path = reference.get("zarr_path") if isinstance(reference, Mapping) else None
    if not isinstance(path, str) or not path:
        raise ValueError(f"Segmental strain requires a {field}.zarr_path reference")
    return path


def _record_indices(raw: Any, count: int, path: str) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{path}: no declared curve_record_indices")
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count for index in raw):
        raise ValueError(f"{path}: invalid curve_record_indices for {count} source records")
    if len(set(raw)) != len(raw):
        raise ValueError(f"{path}: duplicate curve_record_indices")
    return tuple(raw)


def _record_roles(document: Mapping[str, Any], value_path: str, count: int) -> tuple[str, ...]:
    candidates = [
        tuple(item["links"]["record_roles"])
        for item in document.get("annotations", [])
        if isinstance(item, Mapping)
        and item.get("kind") == "aps_curve"
        and isinstance(item.get("value"), Mapping)
        and item["value"].get("zarr_path") == value_path
        and isinstance(item.get("links"), Mapping)
        and "record_roles" in item["links"]
    ]
    if not candidates:
        return ()
    roles = candidates[0]
    if (
        len(roles) != count
        or any(candidate != roles for candidate in candidates)
        or not all(isinstance(role, str) for role in roles)
    ):
        raise ValueError(f"{value_path}: inconsistent source record_roles")
    return roles
