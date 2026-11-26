"""List the annotations of a recording and read its measurements in image coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np

from echoxflow.manifest import annotation_documents
from echoxflow.objects import AnnotationRef, RecordingRef, _annotation_refs
from echoxflow.scan.geometry import SectorGeometry, sector_geometry_from_mapping
from echoxflow.schema import array_group, semantic_name

if TYPE_CHECKING:
    from echoxflow.loading import RecordingStore

FrameStatus = Literal[
    "exact",
    "between_frames",
    "no_matching_frame",
    "no_time",
    "unknown_plane",
    "linked_recording_missing",
    "not_applicable",
]

#: A stored frame counts as the measured one when its timestamp is this close to the measurement time.
FRAME_TIME_TOLERANCE_S = 1e-4

_AXES = {
    "spatial": ("x_m", "y_m"),
    "spectral": ("time_s", "velocity_mps"),
    "time_depth": ("time_s", "depth_m"),
}


@dataclass(frozen=True)
class AnnotationInfo:
    """One annotation a recording carries, described without loading its values.

    Attributes:
        kind: The annotation family, for example `measurement`, `aps_contour`, `aha_segmental_strain` or
            `recovered_la_trace`. Arrays under `annotations/` that no manifest entry describes (the ECG QRS times)
            are listed with kind `array`.
        name: What the annotation is: the measurement kind (`dimension_lvot_diam`), or the name of its array.
        path: Store path of the values, loadable with `store.load_array(path)`.
        shape: Stored shape of the values, or `None` when they are a group of arrays or held by a linked recording.
        target: Path of the data array the annotation belongs to, when the recording names one.
        coordinate_type: `spatial`, `spectral` or `time_depth` for measurements, otherwise `None`.
        ref: The parsed manifest entry, with the unprocessed record in `ref.raw`.
    """

    kind: str
    name: str
    path: str
    shape: tuple[int, ...] | None = None
    target: str | None = None
    coordinate_type: str | None = None
    ref: AnnotationRef | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, eq=False)
class Measurement:
    """One measurement the operator made on the scanner, with its points ready to use.

    Left-atrial volume traces are included: the scanner stored no coordinates for them, so their outline was
    recovered from the graphics burned into the still (`geometry_kind` is `polygon`).

    Attributes:
        kind: Controlled measurement name, for example `dimension_lvot_diam` or `volume_laesv_a4c`.
        measure_id: The scanner's own name, for example `Cardiac/2D/Dimension/LVOT Diam` (withheld for site-defined
            measurements).
        coordinate_type: `spatial` (drawn on a B-mode image), `spectral` (on a Doppler spectrum) or `time_depth`
            (on an M-mode strip).
        geometry_kind: Shape of the measurement, for example `caliper`, `point`, `polyline` or `polygon`.
        points: `[N, 2]` float64 array. Its columns are named by `axes`. Spatial points are in the Cartesian frame
            of the B-mode sector: x is lateral position and y is depth, both in metres, so they can be drawn on a
            scan-converted frame or mapped to beamspace pixels with `echoxflow.scan.BeamspacePixelGrid`. Spectral
            and time-depth points are returned as stored.
        axes: Names of the two point columns.
        time_s: Time of the frame a spatial measurement was drawn on, in the recording's clock.
        data_path: Path of the stream the points belong to, or `None` when the recording does not identify one.
        recording: The recording that holds `data_path` when it is not this one: left-atrial traces are drawn on a
            still and belong to a frame of the loop it was taken from. Open it with `store.open_reference()`.
        frame_index: Index into `data_path` of the measured frame. `None` unless `frame_status` is `exact` or
            `between_frames`.
        frame_status: How `frame_index` was resolved. `exact`: a stored frame has the measurement time.
            `between_frames`: the time falls between two stored frames and the nearest is given.
            `no_matching_frame`: the measured frame is not stored. `no_time`: the measurement has no frame time.
            `unknown_plane`: the recording has no single B-mode plane the points can be assigned to.
            `linked_recording_missing`: the frame is in another recording (`recording`) that is not on disk.
            `not_applicable`: spectral and time-depth points carry time in their coordinates.
        frame_time_offset_s: Timestamp of the frame at `frame_index` minus `time_s`.
        geometry: Sector geometry of the stream at `data_path` for spatial measurements.
        ref: The parsed manifest entry, with the unprocessed record in `ref.raw`.
    """

    kind: str
    measure_id: str | None
    coordinate_type: str
    geometry_kind: str | None
    points: np.ndarray
    axes: tuple[str, str]
    time_s: float | None = None
    data_path: str | None = None
    frame_index: int | None = None
    frame_status: FrameStatus = "not_applicable"
    frame_time_offset_s: float | None = None
    recording: RecordingRef | None = field(default=None, repr=False)
    geometry: SectorGeometry | None = field(default=None, repr=False)
    ref: AnnotationRef | None = field(default=None, repr=False)

    @property
    def length_m(self) -> float | None:
        """Distance between the two points of a spatial measurement in metres, otherwise `None`."""
        if self.coordinate_type != "spatial" or self.points.shape[0] != 2 or self.geometry_kind == "polygon":
            return None
        return float(np.linalg.norm(self.points[1] - self.points[0]))

    @property
    def area_m2(self) -> float | None:
        """Area enclosed by a polygon in square metres, otherwise `None`."""
        if self.geometry_kind != "polygon" or self.points.shape[0] < 3:
            return None
        x, y = self.points[:, 0], self.points[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def list_annotations(store: RecordingStore) -> tuple[AnnotationInfo, ...]:
    """Describe every annotation of an opened recording that has values to load, without loading them."""
    attrs = dict(getattr(store.group, "attrs", {}))
    infos: list[AnnotationInfo] = []
    described: set[str] = set()
    for ref in _annotation_refs(annotation_documents(attrs)):
        raw = ref.raw or {}
        links = _mapping(raw.get("links"))
        path = None if ref.value is None else ref.value.path
        _collect_paths(raw, described)
        if path is None:
            continue
        infos.append(
            AnnotationInfo(
                kind=str(raw.get("kind") or "annotation"),
                name=_annotation_name(links, path),
                path=path,
                shape=_array_shape(store, ref),
                target=_target_data_path(raw),
                coordinate_type=_optional_str(links.get("coordinate_type")),
                ref=ref,
            )
        )
    for path in store.array_paths:
        if array_group(path) != "annotations" or any(path == seen or path.startswith(f"{seen}/") for seen in described):
            continue
        infos.append(AnnotationInfo(kind="array", name=semantic_name(path), path=path, shape=store.array_shape(path)))
    return tuple(infos)


class _Entry(NamedTuple):
    ref: AnnotationRef
    raw: Mapping[str, Any]
    links: Mapping[str, Any]
    document: Mapping[str, Any]
    kind: str
    measure_id: str | None
    coordinate_type: str
    geometry_kind: str | None


def _measurement_entries(group_attrs: Mapping[str, Any]) -> list[_Entry]:
    """Manifest entries that are measurements: the scanner's own, and the recovered left-atrial outlines."""
    located = annotation_documents(group_attrs)
    documents = dict(located)
    entries: list[_Entry] = []
    for ref in _annotation_refs(located):
        raw = ref.raw or {}
        links = _mapping(raw.get("links"))
        document = documents.get(ref.document_path, {})
        if ref.value is None:
            continue
        if raw.get("kind") == "measurement":
            kind = _optional_str(links.get("measurement_kind")) or "unknown"
            measure_id = _optional_str(links.get("measure_id"))
            coordinate_type = _optional_str(links.get("coordinate_type")) or "spatial"
            entries.append(
                _Entry(
                    ref,
                    raw,
                    links,
                    document,
                    kind,
                    measure_id,
                    coordinate_type,
                    _optional_str(links.get("geometry_kind")),
                )
            )
        elif raw.get("kind") == "recovered_la_trace" and links.get("geometry_kind") == "polygon":
            kind, measure_id = _la_trace_identity(document, links)
            entries.append(_Entry(ref, raw, links, document, kind, measure_id, "spatial", "polygon"))
    return entries


def _la_trace_identity(document: Mapping[str, Any], links: Mapping[str, Any]) -> tuple[str, str | None]:
    """Name a recovered left-atrial outline after the measurement the scanner listed for its plane."""
    annotations = document.get("annotations")
    for annotation in annotations if isinstance(annotations, list) else ():
        listed = _mapping(annotation.get("links")) if isinstance(annotation, Mapping) else {}
        if not isinstance(annotation, Mapping) or annotation.get("kind") != "burned_in_trace":
            continue
        ids, kinds = listed.get("measure_ids"), listed.get("measurement_kinds")
        planes = listed.get("planes")
        for plane in planes if isinstance(planes, list) else ():
            if not isinstance(plane, Mapping) or plane.get("plane_index") != links.get("plane_index"):
                continue
            measure_id = _optional_str(plane.get("measure_id"))
            if measure_id and isinstance(ids, list) and isinstance(kinds, list) and measure_id in ids:
                position = ids.index(measure_id)
                kind = _optional_str(kinds[position]) if position < len(kinds) else None
                if kind:
                    return kind, measure_id
    view = _optional_str(links.get("view"))
    return (f"la_trace_{view.lower()}" if view else "la_trace"), None


def measurement_names(group_attrs: Mapping[str, Any]) -> tuple[str, ...]:
    """Names of the measurements a recording holds, from its root attributes, sorted and without repeats."""
    return tuple(sorted({entry.kind for entry in _measurement_entries(group_attrs)}))


def load_measurements(
    store: RecordingStore,
    kind: str | None = None,
    *,
    coordinate_type: str | None = None,
) -> tuple[Measurement, ...]:
    """Load the measurements of an opened recording.

    Args:
        store: An opened recording.
        kind: Keep only this measurement kind, for example `dimension_lvot_diam`.
        coordinate_type: Keep only `spatial`, `spectral` or `time_depth` measurements.

    Returns:
        The measurements in stored order. See `Measurement` for the coordinate conventions.
    """
    measurements: list[Measurement] = []
    for entry in _measurement_entries(dict(getattr(store.group, "attrs", {}))):
        if kind is not None and entry.kind != kind:
            continue
        if coordinate_type is not None and entry.coordinate_type != coordinate_type:
            continue
        ref, path = entry.ref, entry.ref.value.path if entry.ref.value is not None else ""
        if entry.coordinate_type not in _AXES:
            raise ValueError(f"{path} has unsupported measurement coordinate type {entry.coordinate_type!r}")
        points = np.array(store.load_array(path), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"{path} must be shaped [N, 2], got {points.shape}")
        common: dict[str, Any] = {
            "kind": entry.kind,
            "measure_id": entry.measure_id,
            "coordinate_type": entry.coordinate_type,
            "geometry_kind": entry.geometry_kind,
            "axes": _AXES[entry.coordinate_type],
            "ref": ref,
        }
        if entry.raw.get("kind") == "recovered_la_trace":
            measurements.append(_la_trace_measurement(store, entry, points, common))
        elif entry.coordinate_type == "spatial":
            measurements.append(_spatial_measurement(store, ref, entry.links, entry.document, points, common))
        else:
            spectral = entry.coordinate_type == "spectral"
            data_path = _target_data_path(entry.raw) if spectral else _single_motion_mode_path(entry.document)
            measurements.append(Measurement(points=points, data_path=data_path, **common))
    return tuple(measurements)


def _la_trace_measurement(
    store: RecordingStore, entry: _Entry, points: np.ndarray, common: dict[str, Any]
) -> Measurement:
    """A recovered left-atrial outline, placed on the loop frame the still was taken from.

    The outline is stored in the image frame already, unlike the scanner's own measurement points.
    """
    time_s = _first_value(store, entry.ref.time)
    target = _mapping(entry.raw.get("target_data"))
    data_path = _target_data_path(entry.raw)
    linked_path = _optional_str(target.get("recording_zarr_path"))
    recording = None
    if linked_path is not None:
        recording = RecordingRef(
            recording_id=_optional_str(target.get("recording_id")),
            # Catalog paths are relative to the dataset root; without one the loop is looked up beside this store.
            zarr_path=linked_path if store.root is not None else None,
            relative_zarr_path=Path(linked_path).name,
        )
    located: dict[str, Any] = {"points": points, "time_s": time_s, "data_path": data_path, "recording": recording}
    if recording is not None and not store._reference_path(recording).exists():
        # The outline is still returned, so one missing loop does not hide a recording's other measurements.
        return Measurement(frame_status="linked_recording_missing", **located, **common)
    source = store if recording is None else store.open_reference(recording)
    try:
        documents = annotation_documents(dict(getattr(source.group, "attrs", {})))
        planes = [plane for _, document in documents for plane in _brightness_mode_sectors(document)]
        plane = next((plane for plane in planes if plane[0] == data_path), None)
        if plane is None:
            return Measurement(frame_status="unknown_plane", **located, **common)
        index, status, offset = _resolve_frame(source, plane[1], time_s)
    finally:
        if source is not store:
            source.close()
    return Measurement(
        frame_index=index, frame_status=status, frame_time_offset_s=offset, geometry=plane[2], **located, **common
    )


def _spatial_measurement(
    store: RecordingStore,
    ref: AnnotationRef,
    links: Mapping[str, Any],
    document: Mapping[str, Any],
    points: np.ndarray,
    common: dict[str, Any],
) -> Measurement:
    # The scanner stores spatial measurement points with the lateral axis mirrored relative to the sector
    # geometry of the images. EchoXFlow-2 stores publish them unchanged, so they are reflected here. A record
    # that declares `source_to_canonical_transform` has already been brought into the image frame.
    if not links.get("source_to_canonical_transform"):
        points[:, 0] *= -1.0
    time_s = _first_value(store, ref.time)
    planes = _brightness_mode_sectors(document)
    if len(planes) != 1:
        return Measurement(points=points, time_s=time_s, frame_status="unknown_plane", **common)
    data_path, timestamps_path, geometry = planes[0]
    located: dict[str, Any] = {"points": points, "time_s": time_s, "data_path": data_path, "geometry": geometry}
    index, status, offset = _resolve_frame(store, timestamps_path, time_s)
    return Measurement(frame_index=index, frame_status=status, frame_time_offset_s=offset, **located, **common)


def _resolve_frame(
    store: RecordingStore, timestamps_path: str, time_s: float | None
) -> tuple[int | None, FrameStatus, float | None]:
    """Index of the stored frame a measurement was drawn on, how it was found, and its time offset."""
    if time_s is None:
        return None, "no_time", None
    timestamps = np.asarray(store.load_array(timestamps_path), dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        return None, "no_matching_frame", None
    index = int(np.argmin(np.abs(timestamps - time_s)))
    offset = float(timestamps[index] - time_s)
    if abs(offset) <= FRAME_TIME_TOLERANCE_S:
        return index, "exact", offset
    if float(timestamps.min()) <= time_s <= float(timestamps.max()):
        return index, "between_frames", offset
    return None, "no_matching_frame", None


def _brightness_mode_sectors(document: Mapping[str, Any]) -> list[tuple[str, str, SectorGeometry | None]]:
    planes: list[tuple[str, str, SectorGeometry | None]] = []
    sectors = document.get("sectors")
    for sector in sectors if isinstance(sectors, list) else ():
        if not isinstance(sector, Mapping):
            continue
        data_path = _ref_path(sector.get("frames"))
        timestamps_path = _ref_path(sector.get("timestamps"))
        if data_path is None or timestamps_path is None:
            continue
        if not semantic_name(data_path).startswith("2d_brightness_mode"):
            continue
        try:
            geometry = sector_geometry_from_mapping(_mapping(sector.get("geometry")))
        except (KeyError, TypeError, ValueError):
            geometry = None
        planes.append((data_path, timestamps_path, geometry))
    return planes


def _single_motion_mode_path(document: Mapping[str, Any]) -> str | None:
    tracks = document.get("tracks")
    paths = [
        path
        for track in (tracks if isinstance(tracks, list) else ())
        if isinstance(track, Mapping)
        and (path := _ref_path(track.get("data"))) is not None
        and semantic_name(path).startswith("1d_motion_mode")
    ]
    return paths[0] if len(paths) == 1 else None


def _annotation_name(links: Mapping[str, Any], path: str) -> str:
    for value in (links.get("measurement_kind"), links.get("family")):
        text = _optional_str(value)
        if text is not None:
            return text
    return semantic_name(path)


def _target_data_path(raw: Mapping[str, Any]) -> str | None:
    target = _mapping(raw.get("target_data"))
    return _optional_str(target.get("array_path")) or _optional_str(target.get("zarr_path"))


def _array_shape(store: RecordingStore, ref: AnnotationRef) -> tuple[int, ...] | None:
    if ref.value is None or ref.value.format != "zarr_array" or ref.value.recording_zarr_path:
        return None
    return store.array_shape(ref.value.path) if store.has_array(ref.value.path) else None


def _first_value(store: RecordingStore, ref: Any) -> float | None:
    if ref is None:
        return None
    values = np.asarray(store.load_array(ref.path), dtype=np.float64).reshape(-1)
    return float(values[0]) if values.size and np.isfinite(values[0]) else None


def _collect_paths(value: Any, paths: set[str]) -> None:
    if isinstance(value, Mapping):
        path = _optional_str(value.get("zarr_path"))
        if path is not None:
            paths.add(path.strip("/"))
        for item in value.values():
            _collect_paths(item, paths)
    elif isinstance(value, list):
        for item in value:
            _collect_paths(item, paths)


def _ref_path(value: Any) -> str | None:
    path = _optional_str(_mapping(value).get("zarr_path"))
    return None if path is None else path.strip("/")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    text = None if value is None else str(value).strip()
    return text or None
