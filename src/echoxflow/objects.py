"""Recording-level object references parsed from EchoXFlow metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from echoxflow.croissant import RecordingRecord
from echoxflow.manifest import manifest_documents

RecordingObjectKind = Literal["strain", "3d_bmode", "generic"]


@dataclass(frozen=True)
class ArrayRef:
    """Reference to one array or group inside a Zarr recording."""

    path: str
    format: str = "zarr_array"
    recording_zarr_path: str | None = None


@dataclass(frozen=True)
class RecordingRef:
    """Reference to a recording store, either this store or another store."""

    recording_id: str | None
    zarr_path: str | None
    relative_zarr_path: str | None = None
    is_self: bool = False


@dataclass(frozen=True)
class LinkedArray:
    """Array reference qualified by the recording that owns it."""

    recording: RecordingRef
    data_path: str
    timestamps_path: str | None = None


@dataclass(frozen=True)
class AnnotationRef:
    """Reference to annotation values and their target."""

    target_entity: str
    target_semantic_id: str
    field: str
    value: ArrayRef
    time: ArrayRef | None = None
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class StrainPanel:
    """One strain panel and the B-mode data it annotates."""

    role_id: str
    view_code: str | None
    sequence_id: int | None
    bmode: LinkedArray
    geometry: Mapping[str, Any] | None = None
    annotations: tuple[AnnotationRef, ...] = ()
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MeshSequenceRef:
    """Reference to one linked mesh sequence."""

    role_id: str
    mesh_group: ArrayRef
    timestamps: ArrayRef | None = None
    sequence_id: int | None = None
    label: str | None = None
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RecordingObject:
    """Semantic object view over one recording store."""

    record: RecordingRecord | None
    kind: RecordingObjectKind
    panels: tuple[StrainPanel, ...] = ()
    mesh_sequences: tuple[MeshSequenceRef, ...] = ()
    annotations: tuple[AnnotationRef, ...] = ()
    raw_documents: tuple[Mapping[str, Any], ...] = ()


def load_recording_object(
    record: RecordingRecord | str | Path,
    *,
    root: str | Path | None = None,
) -> RecordingObject:
    """Open a recording and return its semantic object references."""
    from echoxflow.loading import open_recording

    return open_recording(record, root=root).load_object()


def recording_object_from_metadata(
    record: RecordingRecord | None,
    group_attrs: Mapping[str, Any],
    *,
    store_path: str | Path | None = None,
) -> RecordingObject:
    """Parse object references from recording metadata without loading referenced arrays."""
    documents = _manifest_documents(group_attrs)
    annotations = _annotation_refs(documents)
    panels = _strain_panels(record, store_path=store_path, documents=documents, annotations=annotations)
    mesh_sequences = _mesh_sequence_refs(documents)
    if panels:
        kind: RecordingObjectKind = "strain"
    elif mesh_sequences or _has_3d_bmode(record, group_attrs):
        kind = "3d_bmode"
    else:
        kind = "generic"
    return RecordingObject(
        record=record,
        kind=kind,
        panels=panels,
        mesh_sequences=mesh_sequences,
        annotations=annotations,
        raw_documents=documents,
    )


def _strain_panels(
    record: RecordingRecord | None,
    *,
    store_path: str | Path | None,
    documents: tuple[Mapping[str, Any], ...],
    annotations: tuple[AnnotationRef, ...],
) -> tuple[StrainPanel, ...]:
    panels: list[StrainPanel] = []
    for document in documents:
        raw_panels = document.get("linked_panels")
        if not isinstance(raw_panels, list):
            continue
        for index, item in enumerate(raw_panels):
            if not isinstance(item, Mapping):
                continue
            linked = item.get("linked_recording")
            if not isinstance(linked, Mapping):
                continue
            role_id = _panel_role_id(item, index=index)
            if role_id is None:
                continue
            frame_path = _optional_str(linked.get("frames_array_path")) or "data/2d_brightness_mode"
            timestamps_path = _panel_timestamps_path(item, linked)
            panel_annotations = tuple(
                annotation for annotation in annotations if annotation.target_semantic_id == role_id
            )
            panels.append(
                StrainPanel(
                    role_id=role_id,
                    view_code=_optional_str(item.get("view_code")),
                    sequence_id=None,
                    bmode=LinkedArray(
                        recording=_recording_ref(record, store_path=store_path, linked=linked),
                        data_path=_normalize_data_path(frame_path),
                        timestamps_path=_normalize_timestamp_path(timestamps_path) if timestamps_path else None,
                    ),
                    geometry=item.get("geometry") if isinstance(item.get("geometry"), Mapping) else None,
                    annotations=panel_annotations,
                    raw=item,
                )
            )
    if not panels:
        return ()
    if len(panels) > 3:
        raise ValueError(f"Strain recordings must have 1-3 linked panels, got {len(panels)}")
    return tuple(panels)


def _panel_timestamps_path(panel: Mapping[str, Any], linked: Mapping[str, Any]) -> str | None:
    frame_timestamps = _array_ref(panel.get("frame_timestamps"))
    if frame_timestamps is not None:
        return frame_timestamps.path
    timestamps_path = _optional_str(linked.get("timestamps_array_path"))
    return _normalize_timestamp_path(timestamps_path) if timestamps_path else None


def _annotation_refs(documents: tuple[Mapping[str, Any], ...]) -> tuple[AnnotationRef, ...]:
    refs: list[AnnotationRef] = []
    for document in documents:
        annotations = document.get("annotations")
        if not isinstance(annotations, list):
            continue
        for item in annotations:
            if not isinstance(item, Mapping):
                continue
            target = item.get("target")
            value = item.get("value")
            if not isinstance(target, Mapping) or not isinstance(value, Mapping):
                continue
            value_ref = _array_ref(value)
            if value_ref is None:
                continue
            field = _optional_str(target.get("field")) or ""
            refs.append(
                AnnotationRef(
                    target_entity=_optional_str(target.get("type")) or "",
                    target_semantic_id=_optional_str(target.get("semantic_id")) or "",
                    field=field,
                    value=value_ref,
                    time=_array_ref(item.get("time")),
                    raw=item,
                )
            )
    return tuple(refs)


def _panel_role_id(item: Mapping[str, Any], *, index: int) -> str | None:
    for value in (
        item.get("role_id"),
        item.get("semantic_id"),
        item.get("view_code"),
        item.get("view"),
    ):
        role_id = _normalized_role_id(value)
        if role_id is not None:
            return role_id
    linked = item.get("linked_recording")
    if isinstance(linked, Mapping):
        for value in (
            linked.get("recording_id"),
            linked.get("relative_zarr_path"),
            linked.get("recording_zarr_path"),
        ):
            role_id = _normalized_role_id(value)
            if role_id is not None and _is_known_strain_panel_role(role_id):
                return role_id
    return None


def _normalized_role_id(value: Any) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    role = Path(text).stem.strip().lower().replace("-", "_").replace(" ", "_")
    role = role.removeprefix("data/").removesuffix("_source").removesuffix("_recording")
    aliases = {
        "2_ch": "2ch",
        "3_ch": "3ch",
        "4_ch": "4ch",
        "2_chamber": "2ch",
        "3_chamber": "3ch",
        "4_chamber": "4ch",
        "apical_two_chamber": "2ch",
        "apical_three_chamber": "3ch",
        "apical_four_chamber": "4ch",
    }
    role = aliases.get(role, role)
    if role in {"2ch", "3ch", "4ch", "rv", "la", "lv"}:
        return role
    for suffix in ("_contour_points", "_contour", "_strain_curve", "_curve", "_points"):
        if role.endswith(suffix) and len(role) > len(suffix):
            return role[: -len(suffix)]
    return role or None


def _is_known_strain_panel_role(role_id: str) -> bool:
    return role_id in {"2ch", "3ch", "4ch", "rv", "la", "lv"}


def _mesh_sequence_refs(documents: tuple[Mapping[str, Any], ...]) -> tuple[MeshSequenceRef, ...]:
    refs: list[MeshSequenceRef] = []
    for document in documents:
        raw_sequences = document.get("linked_mesh_sequences")
        if not isinstance(raw_sequences, list):
            continue
        for item in raw_sequences:
            if not isinstance(item, Mapping):
                continue
            mesh_group = _array_ref(item.get("mesh_data"))
            if mesh_group is None:
                continue
            mesh_key = _optional_str(item.get("mesh_key"))
            refs.append(
                MeshSequenceRef(
                    role_id=_mesh_role_id(mesh_group.path),
                    mesh_group=mesh_group,
                    timestamps=_array_ref(item.get("timestamps")),
                    sequence_id=None,
                    label=mesh_key,
                    raw=item,
                )
            )
    return tuple(refs)


def _recording_ref(
    record: RecordingRecord | None,
    *,
    store_path: str | Path | None,
    linked: Mapping[str, Any],
) -> RecordingRef:
    recording_id = _optional_str(linked.get("recording_id"))
    zarr_path = _optional_str(linked.get("recording_zarr_path"))
    relative_zarr_path = _optional_str(linked.get("relative_zarr_path"))
    is_self = False
    if record is not None:
        is_self = recording_id == record.recording_id or _same_path(zarr_path, record.zarr_path)
    if not is_self and store_path is not None:
        is_self = _same_path(zarr_path, str(store_path)) or _same_path(relative_zarr_path, Path(store_path).name)
    return RecordingRef(
        recording_id=recording_id,
        zarr_path=zarr_path,
        relative_zarr_path=relative_zarr_path,
        is_self=is_self,
    )


def _array_ref(value: Any) -> ArrayRef | None:
    if not isinstance(value, Mapping):
        return None
    path = (
        _optional_str(value.get("array_path"))
        or _optional_str(value.get("zarr_path"))
        or _optional_str(value.get("path"))
    )
    if path is None:
        return None
    return ArrayRef(
        path=path.strip("/"),
        format=_optional_str(value.get("format")) or "zarr_array",
        recording_zarr_path=_optional_str(value.get("recording_zarr_path")),
    )


def _manifest_documents(group_attrs: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return manifest_documents(group_attrs)


def _has_3d_bmode(record: RecordingRecord | None, group_attrs: Mapping[str, Any]) -> bool:
    if record is not None and record.has_array_path("data/3d_brightness_mode"):
        return True
    arrays = group_attrs.get("arrays")
    if not isinstance(arrays, list):
        return False
    return any(isinstance(item, Mapping) and item.get("name") == "data/3d_brightness_mode" for item in arrays)


def _mesh_role_id(path: str) -> str:
    text = path.strip("/")
    if text.startswith("data/"):
        text = text.removeprefix("data/")
    return text.split("/", 1)[0]


def _normalize_data_path(path: str) -> str:
    text = str(path).strip("/")
    return text if text.startswith("data/") else f"data/{text}"


def _normalize_timestamp_path(path: str) -> str:
    text = str(path).strip("/")
    return text if text.startswith("timestamps/") else f"timestamps/{text}"


def _same_path(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    left_text = str(left).strip("/")
    right_text = str(right).strip("/")
    return left_text == right_text or left_text.endswith(f"/{right_text}") or right_text.endswith(f"/{left_text}")


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
