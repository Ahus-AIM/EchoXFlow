"""Annotation overlay preparation for plotting."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np

from echoxflow.loading import LoadedArray, PackedMeshAnnotation, RecordingStore
from echoxflow.plotting.specs import PanelView
from echoxflow.scan import SphericalMosaicLayout, mesh_frame_indices_for_volume_timestamps
from echoxflow.scan.spherical import MosaicCell

logger = logging.getLogger(__name__)


def attach_annotation_overlays(
    store: RecordingStore,
    loaded_arrays: tuple[LoadedArray, ...],
) -> tuple[LoadedArray, ...]:
    """Attach display-ready annotation overlay metadata to loaded arrays."""
    show_sampling_gate = _has_sample_volume_gate(loaded_arrays)
    show_sampling_line = _has_sampling_line_source(loaded_arrays)
    return tuple(
        _attach_annotation_overlay(
            store,
            loaded,
            show_sampling_gate=show_sampling_gate,
            show_sampling_line=show_sampling_line,
        )
        for loaded in loaded_arrays
    )


def _attach_annotation_overlay(
    store: RecordingStore,
    loaded: LoadedArray,
    *,
    show_sampling_gate: bool,
    show_sampling_line: bool,
) -> LoadedArray:
    overlays: list[dict[str, object]] = []
    annotation_records = _annotation_records_by_value_path(store)
    for path in _annotation_paths_for_loaded(store, loaded):
        try:
            values = store.load_array(path)
            points = np.asarray(values, dtype=np.float32)
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("could not load annotation overlay `%s`: %s", path, exc)
            continue
        except (TypeError, ValueError) as exc:
            logger.warning("could not parse annotation overlay `%s`: %s", path, exc)
            continue
        records = annotation_records.get(_normalize_annotation_path(path), ())
        for metadata in records or (None,):
            if metadata is not None and _resolved_annotation_alias(store, loaded, metadata):
                continue
            if (
                metadata is not None
                and loaded.data_path in _SPECTRAL_VELOCITY_STRIPS
                and not _spectral_owner_matches(store, loaded, metadata)
            ):
                continue
            overlay_metadata = _annotation_overlay_metadata(metadata, array_attrs=_array_attrs(store, path))
            if (
                _is_spectral_annotation_path(loaded.data_path, path)
                or _metadata_is_spectral(metadata or {})
                or _is_spectral_trace(metadata)
            ):
                kind = "spectral_trace" if _is_spectral_trace(metadata) else "spectral_points"
            else:
                kind = "physical_points"
            overlays.append({"kind": kind, "points": points, "path": path, **overlay_metadata})
    if show_sampling_gate:
        gate_overlay = _sampling_gate_overlay(store, loaded)
        if gate_overlay is not None:
            overlays.append(gate_overlay)
    if show_sampling_line:
        overlays.extend(_sampling_line_overlays(store, loaded))
    attrs = dict(loaded.attrs)
    if overlays:
        attrs["annotation_overlays"] = tuple(overlays)
    if loaded.data_path == "data/3d_brightness_mode":
        mesh = load_mesh_annotation(store)
        if mesh is not None:
            attrs["mesh_annotation"] = mesh
    return replace(loaded, attrs=attrs) if attrs != dict(loaded.attrs) else loaded


def load_mesh_annotation(store: RecordingStore) -> PackedMeshAnnotation | None:
    """Load the first linked LV mesh sequence of a recording, or None when it has none or cannot be read."""
    mesh_sequences = store.load_object().mesh_sequences
    if not mesh_sequences:
        return None
    try:
        mesh_sequence = mesh_sequences[0]
        mesh = store.load_packed_mesh_annotation(mesh_sequence.mesh_group.path)
        timestamps = mesh_sequence.timestamps
        if timestamps is not None:
            timestamp_values = (
                store.group[timestamps.path][:] if timestamps.path in store.group else store.load_array(timestamps.path)
            )
            mesh = replace(
                mesh,
                timestamps_path=timestamps.path,
                timestamps=np.asarray(timestamp_values, dtype=np.float32).reshape(-1),
            )
        return mesh
    except (FileNotFoundError, KeyError) as exc:
        logger.warning("could not load mesh annotation: %s", exc)
    except (TypeError, ValueError) as exc:
        logger.warning("could not parse mesh annotation: %s", exc)
    return None


def _has_sample_volume_gate(loaded_arrays: tuple[LoadedArray, ...]) -> bool:
    return any(loaded.data_path == "data/1d_pulsed_wave_doppler" for loaded in loaded_arrays)


def _has_sampling_line_source(loaded_arrays: tuple[LoadedArray, ...]) -> bool:
    return any(
        loaded.data_path in {"data/1d_continuous_wave_doppler", "data/1d_motion_mode"} for loaded in loaded_arrays
    )


def _annotation_paths_for_loaded(store: RecordingStore, loaded: LoadedArray) -> tuple[str, ...]:
    name = loaded.data_path.removeprefix("data/")
    if name.startswith("1d_"):
        paths = []
        annotation_records = _annotation_records_by_value_path(store)
        for path in store.array_paths:
            items = annotation_records.get(_normalize_annotation_path(path), ())
            if items:
                if any(_spectral_owner_matches(store, loaded, item) for item in items):
                    paths.append(path)
            elif _is_spectral_annotation_path(loaded.data_path, path):
                paths.append(path)
        return tuple(dict.fromkeys(paths))
    if name.startswith("2d_brightness_mode"):
        return tuple(
            path for path in store.array_paths if path.startswith("data/") and "overlay_physical_points" in path
        )
    return ()


def _is_spectral_annotation_path(data_path: str, annotation_path: str) -> bool:
    """Whether `annotation_path` is a traced-envelope array of the 1D stream `data_path`.

    Exports store these either as `data/<stream>_annotation_NN` (older layout) or as
    `annotations/<stream>_annotation_NN` (optionally inside `annotations/<document>/`).
    """
    name = data_path.removeprefix("data/")
    if not name.startswith("1d_"):
        return False
    parts = annotation_path.split("/")
    if parts[0] not in {"data", "annotations"}:
        return False
    return parts[-1].startswith(f"{name}_annotation_")


def _is_spectral_trace(metadata: Mapping[str, object] | None) -> bool:
    return metadata is not None and metadata.get("kind") == "spectral_trace"


_SPECTRAL_VELOCITY_STRIPS = frozenset({"data/1d_pulsed_wave_doppler", "data/1d_continuous_wave_doppler"})


def _metadata_is_spectral(metadata: Mapping[str, object]) -> bool:
    links = metadata.get("links")
    return isinstance(links, Mapping) and links.get("coordinate_type") == "spectral"


def _spectral_owner_matches(store: RecordingStore, loaded: LoadedArray, metadata: Mapping[str, object]) -> bool:
    if isinstance(metadata.get("target_data"), Mapping):
        return _targets_data_path(metadata, loaded.data_path)
    target = metadata.get("target")
    if isinstance(target, Mapping) and target.get("type") == "track":
        role = target.get("semantic_id")
        owners = set()
        for document in _render_documents(getattr(store.group, "attrs", {})):
            tracks = document.get("tracks")
            if not isinstance(tracks, (list, tuple)):
                continue
            for track in tracks:
                if isinstance(track, Mapping) and _semantic_id(track) == role:
                    path = _annotation_value_path(track.get("data"))
                    if path:
                        owners.add(_normalize_annotation_path(path))
        return owners == {loaded.data_path}
    value_path = _annotation_value_path(metadata.get("value"))
    if not (
        _metadata_is_spectral(metadata)
        or _is_spectral_trace(metadata)
        or (value_path and _is_spectral_annotation_path(loaded.data_path, value_path))
    ):
        return False
    owners = set(store.array_paths) & _SPECTRAL_VELOCITY_STRIPS
    return owners == {loaded.data_path}


def _targets_data_path(metadata: Mapping[str, object], data_path: str) -> bool:
    target_data = metadata.get("target_data")
    if not isinstance(target_data, Mapping):
        return False
    raw = _annotation_value_path(target_data)
    return raw is not None and _normalize_annotation_path(raw) == data_path


def _annotation_records_by_value_path(store: RecordingStore) -> dict[str, tuple[Mapping[str, object], ...]]:
    """Preserve every semantic reference; shared arrays do not imply identical measurements."""
    from echoxflow.manifest import annotation_documents

    records: dict[str, list[Mapping[str, object]]] = {}
    for location, document in annotation_documents(getattr(store.group, "attrs", {})):
        annotations = document.get("annotations")
        if not isinstance(annotations, list):
            continue
        for item in annotations:
            if not isinstance(item, Mapping):
                continue
            path = _annotation_value_path(item.get("value"))
            if path is not None:
                records.setdefault(_normalize_annotation_path(path), []).append({**item, "document_path": location})
    return {path: tuple(items) for path, items in records.items()}


def _resolved_annotation_alias(store: RecordingStore, loaded: LoadedArray, metadata: Mapping[str, object]) -> bool:
    alias = metadata.get("alias_of")
    if not isinstance(alias, str) or not alias:
        return False
    for records in _annotation_records_by_value_path(store).values():
        for canonical in records:
            if (
                canonical.get("document_path") == metadata.get("document_path")
                and canonical.get("annotation_identity") == alias
                and not canonical.get("alias_of")
                and _annotation_value_path(canonical.get("value")) in store.array_paths
                and _spectral_owner_matches(store, loaded, canonical)
            ):
                return True
    return False


def _annotation_value_path(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("array_path", "zarr_path", "path"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw
    return None


def _normalize_annotation_path(path: str) -> str:
    return str(path).strip().strip("/")


def _array_attrs(store: RecordingStore, path: str) -> Mapping[str, object]:
    try:
        attrs = getattr(store.group[_normalize_annotation_path(path)], "attrs", {})
    except (FileNotFoundError, KeyError):
        return {}
    return dict(attrs)


def _annotation_overlay_metadata(
    metadata: Mapping[str, object] | None,
    *,
    array_attrs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if metadata is None and not array_attrs:
        return {}
    overlay: dict[str, object] = {}
    if metadata is not None:
        for key in ("annotation_identity", "alias_of", "document_path", "target_data"):
            if key in metadata:
                overlay[key] = metadata[key]
        label = _non_empty_string(metadata.get("label"))
        measure = metadata.get("links")
        if label is None and isinstance(measure, Mapping):
            label = _non_empty_string(measure.get("measure_id"))
        if label is not None:
            overlay["label"] = label
        target = metadata.get("target")
        if isinstance(target, Mapping):
            overlay["target"] = dict(target)
            field = _non_empty_string(target.get("field"))
            if field is not None:
                overlay["field"] = field
        links = metadata.get("links")
        if isinstance(links, Mapping):
            overlay["links"] = dict(links)
            geometry_kind = _non_empty_string(links.get("geometry_kind"))
            if geometry_kind is not None:
                overlay["geometry_kind"] = geometry_kind
    unit = _annotation_hint_string(
        metadata,
        array_attrs,
        keys=("y_unit", "y_units", "velocity_unit", "velocity_units", "value_unit", "value_units", "unit", "units"),
    )
    if unit is not None:
        overlay["y_unit"] = unit
    coordinate_system = _annotation_hint_string(
        metadata,
        array_attrs,
        keys=("y_coordinate_system", "coordinate_system", "coordinate_kind", "value_coordinate_system"),
    )
    if coordinate_system is not None:
        overlay["y_coordinate_system"] = coordinate_system
    return overlay


def _annotation_hint_string(
    metadata: Mapping[str, object] | None,
    array_attrs: Mapping[str, object] | None,
    *,
    keys: tuple[str, ...],
) -> str | None:
    for source in _annotation_hint_sources(metadata, array_attrs):
        for key in keys:
            value = _non_empty_string(source.get(key))
            if value is not None:
                return value
    return None


def _annotation_hint_sources(
    metadata: Mapping[str, object] | None,
    array_attrs: Mapping[str, object] | None,
) -> tuple[Mapping[str, object], ...]:
    sources: list[Mapping[str, object]] = []
    if metadata is not None:
        sources.append(metadata)
        for key in ("target", "links", "value"):
            value = metadata.get(key)
            if isinstance(value, Mapping):
                sources.append(cast(Mapping[str, object], value))
    if array_attrs:
        sources.append(array_attrs)
    return tuple(sources)


def _non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _sampling_gate_overlay(store: RecordingStore, loaded: LoadedArray) -> dict[str, object] | None:
    if not _supports_sampling_gate_overlay(loaded):
        return None
    gate = _sampling_gate_metadata(store, loaded)
    if gate is None:
        return None
    geometry = None if loaded.stream is None else loaded.stream.metadata.geometry
    beam_points, tick_points = _sampling_gate_points(gate, geometry)
    if beam_points is None or tick_points is None:
        return None
    return {"kind": "sampling_gate", "points": beam_points, "tick_points": tick_points, "metadata": dict(gate)}


def _sampling_line_overlays(store: RecordingStore, loaded: LoadedArray) -> tuple[dict[str, object], ...]:
    if not _supports_sampling_gate_overlay(loaded):
        return ()
    direct = _physical_sampling_line_overlays(store, loaded)
    if direct:
        return direct
    metadata = _sampling_line_metadata(store, loaded)
    geometry = None if loaded.stream is None else loaded.stream.metadata.geometry
    points = _sampling_line_points(metadata, geometry)
    if metadata is None or points is None:
        return ()
    return ({"kind": "sampling_line", "points": points, "metadata": dict(metadata)},)


_SAMPLING_OVERLAY_PANEL_PATHS = frozenset(
    {"data/tissue_doppler", "data/2d_color_doppler_velocity", "data/2d_color_doppler_power"}
)


def _supports_sampling_gate_overlay(loaded: LoadedArray) -> bool:
    return loaded.data_path in _SAMPLING_OVERLAY_PANEL_PATHS or loaded.data_path.startswith("data/2d_brightness_mode")


def _physical_sampling_line_overlays(store: RecordingStore, loaded: LoadedArray) -> tuple[dict[str, object], ...]:
    overlays: list[dict[str, object]] = []
    raw = None if loaded.stream is None else loaded.stream.metadata.raw
    for entry in (*_physical_line_entries_from_document(store), *_physical_line_entries(raw)):
        if not _is_sampling_line_entry(entry):
            continue
        points = _line_points(entry.get("points"))
        if points is None:
            continue
        overlays.append(
            {
                "kind": "sampling_line",
                "points": points,
                "metadata": {"label": str(entry.get("label") or ""), "source": "physical_line_overlay"},
            }
        )
    return tuple(overlays)


def _physical_line_entries_from_document(store: RecordingStore) -> tuple[Mapping[str, object], ...]:
    entries: list[Mapping[str, object]] = []
    for document in _render_documents(getattr(store.group, "attrs", {})):
        entries.extend(_physical_line_entries(document))
    return tuple(entries)


def _physical_line_entries(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Mapping):
        return ()
    overlays = value.get("overlays")
    if not isinstance(overlays, Mapping):
        return ()
    lines = overlays.get("physical_lines")
    if not isinstance(lines, list):
        return ()
    return tuple(line for line in lines if isinstance(line, Mapping))


def _is_sampling_line_entry(entry: Mapping[str, object]) -> bool:
    label = (
        str(entry.get("label") or entry.get("semantic_id") or "").strip().lower().replace("-", "_").replace(" ", "_")
    )
    if not label:
        return False
    tokens = (
        "sampling_line",
        "cursor_line",
        "m_mode_cursor_line",
        "mmode_cursor_line",
        "motion_mode_cursor_line",
        "continuous_wave_cursor_line",
        "cw_cursor_line",
        "line_cf_gate",
    )
    return any(token in label for token in tokens)


def _line_points(value: object) -> np.ndarray | None:
    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    if not np.all(np.isfinite(points[:, :2])):
        return None
    return np.asarray(points[:, :2], dtype=np.float32)


def _sampling_gate_metadata(store: RecordingStore, loaded: LoadedArray) -> Mapping[str, object] | None:
    for document in _render_documents(getattr(store.group, "attrs", {})):
        found = _gate_metadata_from_tracks(document) or _gate_metadata_from_document(document)
        if found is not None:
            return found
    raw = None if loaded.stream is None else loaded.stream.metadata.raw
    if isinstance(raw, Mapping):
        gate = raw.get("sampling_gate_metadata")
        if isinstance(gate, Mapping):
            return gate
    return None


def _has_sampling_gate_pose(metadata: Mapping[object, object]) -> bool:
    return (
        _finite_float(metadata.get("gate_center_depth_m")) is not None
        and _finite_float(metadata.get("gate_tilt_rad")) is not None
    )


def _gate_metadata_from_tracks(document: Mapping[str, object]) -> Mapping[str, object] | None:
    tracks = document.get("tracks")
    if not isinstance(tracks, list):
        return None
    for track in tracks:
        if not isinstance(track, Mapping):
            continue
        derived = track.get("derived_from")
        is_gate = _semantic_id(track) == "tissue_doppler_gate"
        if isinstance(derived, Mapping) and (is_gate or str(derived.get("kind", "")).strip() == "tissue_doppler_gate"):
            if _has_sampling_gate_pose(derived):
                return derived
        if is_gate and _has_sampling_gate_pose(track):
            return track
    return None


def _gate_metadata_from_document(document: Mapping[str, object]) -> Mapping[str, object] | None:
    gate = document.get("sampling_gate_metadata")
    if isinstance(gate, Mapping) and _has_sampling_gate_pose(gate):
        return gate
    sectors = document.get("sectors")
    if isinstance(sectors, list):
        for sector in sectors:
            if isinstance(sector, Mapping):
                gate = sector.get("sampling_gate_metadata")
                if isinstance(gate, Mapping) and _has_sampling_gate_pose(gate):
                    return gate
    return None


def _sampling_line_metadata(store: RecordingStore, loaded: LoadedArray) -> Mapping[str, object] | None:
    for document in _render_documents(getattr(store.group, "attrs", {})):
        for candidate in _line_metadata_candidates(document):
            return candidate
        gate = document.get("sampling_gate_metadata")
        if isinstance(gate, Mapping) and _has_sampling_line_pose(gate):
            return gate
    raw = None if loaded.stream is None else loaded.stream.metadata.raw
    for candidate in _line_metadata_candidates(raw):
        return candidate
    return None


def _line_metadata_candidates(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Mapping):
        return ()
    candidates: list[Mapping[str, object]] = []
    for key in ("sampling_line_metadata", "cursor_line_metadata"):
        metadata = value.get(key)
        if isinstance(metadata, Mapping) and _has_sampling_line_pose(metadata):
            candidates.append(metadata)
    gate_metadata = value.get("sampling_gate_metadata")
    if isinstance(gate_metadata, Mapping) and _is_line_only_gate_metadata(value, gate_metadata):
        candidates.append(gate_metadata)
    tracks = value.get("tracks")
    if isinstance(tracks, list):
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            derived = track.get("derived_from")
            if isinstance(derived, Mapping) and _is_sampling_line_metadata(track, derived):
                candidates.append(derived)
            elif _is_sampling_line_metadata(track, track):
                candidates.append(track)
    sectors = value.get("sectors")
    if isinstance(sectors, list):
        for sector in sectors:
            if isinstance(sector, Mapping):
                candidates.extend(_line_metadata_candidates(sector))
    return tuple(candidates)


def _is_sampling_line_metadata(owner: Mapping[object, object], metadata: Mapping[object, object]) -> bool:
    if not _has_sampling_line_pose(metadata):
        return False
    label = str(
        owner.get("semantic_id")
        or owner.get("track_role_id")
        or owner.get("sector_role_id")
        or owner.get("kind")
        or metadata.get("kind")
        or ""
    ).lower()
    return any(token in label for token in ("mmode", "m_mode", "motion_mode", "continuous_wave", "cw", "cursor_line"))


def _is_line_only_gate_metadata(owner: Mapping[object, object], metadata: Mapping[object, object]) -> bool:
    sample_volume = _finite_float(_first_value(metadata, "gate_sample_volume_m", "sample_volume_m"))
    return sample_volume is None and _is_sampling_line_metadata(owner, metadata)


def _has_sampling_line_pose(metadata: Mapping[object, object]) -> bool:
    return _finite_float(_first_value(metadata, "gate_tilt_rad", "tilt_rad", "tilt")) is not None


def _first_value(mapping: Mapping[Any, object], *keys: str) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _sampling_gate_points(
    gate: Mapping[str, object], geometry: object | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    center_depth = _finite_float(gate.get("gate_center_depth_m"))
    tilt_rad = _finite_float(gate.get("gate_tilt_rad"))
    if center_depth is None or tilt_rad is None:
        return None, None
    sample_volume = _finite_float(gate.get("gate_sample_volume_m"))
    depth_start = _finite_float(getattr(geometry, "depth_start_m", None))
    depth_end = _finite_float(getattr(geometry, "depth_end_m", None))
    start_depth = max(0.0, depth_start if depth_start is not None else 0.0)
    end_depth = center_depth
    if depth_start is not None:
        end_depth = max(depth_start, end_depth)
    if depth_end is not None:
        end_depth = min(depth_end, end_depth)
    if end_depth <= start_depth:
        start_depth = center_depth
        end_depth = center_depth
    beam_depths = np.asarray([start_depth, end_depth], dtype=np.float32)
    beam_points = _polar_sector_points(beam_depths, tilt_rad)

    gate_width_m = sample_volume if sample_volume is not None and sample_volume > 0.0 else 0.006
    half_angle = 0.5 * gate_width_m / max(1e-6, center_depth)
    angle_start = tilt_rad - half_angle
    angle_end = tilt_rad + half_angle
    angle_min = _finite_float(getattr(geometry, "angle_start_rad", None))
    angle_max = _finite_float(getattr(geometry, "angle_end_rad", None))
    if angle_min is not None and angle_max is not None:
        angle_start = float(np.clip(angle_start, min(angle_min, angle_max), max(angle_min, angle_max)))
        angle_end = float(np.clip(angle_end, min(angle_min, angle_max), max(angle_min, angle_max)))
    tick_angles = np.asarray([angle_start, angle_end], dtype=np.float32)
    tick_points = np.column_stack([center_depth * np.sin(tick_angles), center_depth * np.cos(tick_angles)]).astype(
        np.float32
    )
    return beam_points, tick_points


def _sampling_line_points(metadata: Mapping[str, object] | None, geometry: object | None) -> np.ndarray | None:
    if metadata is None:
        return None
    tilt_rad = _finite_float(_first_value(metadata, "gate_tilt_rad", "tilt_rad", "tilt"))
    if tilt_rad is None:
        return None
    depth_start = _finite_float(getattr(geometry, "depth_start_m", None))
    depth_end = _finite_float(getattr(geometry, "depth_end_m", None))
    if depth_start is None or depth_end is None:
        center_depth = _finite_float(_first_value(metadata, "gate_center_depth_m", "center_depth_m", "depth_m"))
        if center_depth is None:
            return None
        depth_start = 0.0
        depth_end = center_depth
    d0 = max(0.0, float(depth_start))
    d1 = max(d0 + 1e-4, float(depth_end))
    return _polar_sector_points(np.asarray([d0, d1], dtype=np.float32), float(tilt_rad))


def _polar_sector_points(depths: np.ndarray, angle_rad: float) -> np.ndarray:
    return np.column_stack([depths * math.sin(angle_rad), depths * math.cos(angle_rad)]).astype(np.float32)


def _render_documents(group_attrs: Mapping[object, object]) -> tuple[Mapping[str, object], ...]:
    documents: list[Mapping[str, object]] = []
    recording_manifest = group_attrs.get("recording_manifest")
    if isinstance(recording_manifest, Mapping):
        documents.append(recording_manifest)
    render_inputs = group_attrs.get("render_inputs")
    if not isinstance(render_inputs, list):
        metadata = group_attrs.get("metadata")
        if isinstance(metadata, Mapping):
            render_inputs = metadata.get("render_inputs")
    if isinstance(render_inputs, list):
        for item in render_inputs:
            if isinstance(item, Mapping) and isinstance(item.get("document"), Mapping):
                documents.append(item["document"])
    for document in documents:
        merged = document.get("merged_documents")
        if isinstance(merged, list):
            documents.extend(item for item in merged if isinstance(item, Mapping))
    return tuple(documents)


def _semantic_id(value: Mapping[object, object]) -> str:
    return str(value.get("semantic_id") or value.get("sector_role_id") or value.get("track_role_id") or "").strip()


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


#: Mesh points are ``physical_xyz_m`` with depth along mesh y. The render frame has depth along
#: z, azimuth along x and elevation along y, so mesh (x, y, z) maps to render (x, -z, y). Checked
#: against the linked volumes of six exams: the contour encloses the dark cavity in the long-axis
#: and short-axis slices, while the 180-degree rotation about depth crosses the septum.
_MESH_RENDER_FRAME_TRANSFORM = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def mesh_mosaic_annotation_lines(
    annotation: PackedMeshAnnotation,
    *,
    frame_count: int,
    layout: SphericalMosaicLayout,
    view: PanelView,
    volume_timestamps: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Build per-frame mesh intersection lines in cartesian 3D mosaic pixel coordinates.

    `layout` is the layout the mosaic pixels were built with, so the lines register with the slices.
    """
    if view != "cartesian":
        return tuple(() for _ in range(max(0, int(frame_count))))
    if frame_count > 1:
        volume_times = np.asarray([] if volume_timestamps is None else volume_timestamps).reshape(-1)
        mesh_times = np.asarray([] if annotation.timestamps is None else annotation.timestamps).reshape(-1)
        if (
            annotation.frame_count <= 1
            or volume_times.size < frame_count
            or mesh_times.size < annotation.frame_count
            or not np.any(np.isfinite(volume_times))
            or not np.any(np.isfinite(mesh_times))
        ):
            return tuple(() for _ in range(max(0, int(frame_count))))
    cells = tuple(cell for cell in layout.cells if cell.plane is not None)
    mesh_indices = mesh_frame_indices_for_volume_timestamps(
        annotation.timestamps,
        volume_timestamps,
        metadata,
        mesh_frame_count=annotation.frame_count,
        target_count=int(frame_count),
    )
    overlays: list[tuple[np.ndarray, ...]] = []
    for frame_index in range(max(0, int(frame_count))):
        mesh_index = mesh_indices[frame_index] if frame_index < len(mesh_indices) else frame_index
        mesh_frame = annotation.frame(min(mesh_index, max(0, annotation.frame_count - 1)))
        points = _mesh_points_to_render_frame(np.asarray(mesh_frame.points, dtype=np.float32))
        faces = _valid_mesh_faces(np.asarray(mesh_frame.faces, dtype=np.int32), n_points=int(points.shape[0]))
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3 or faces.size == 0:
            overlays.append(())
            continue
        lines: list[np.ndarray] = []
        for cell in cells:
            plane = cell.plane
            if plane is None:
                continue
            segments = _mesh_plane_segments(
                points,
                faces,
                plane.plane_point,
                plane.plane_normal,
                plane.basis_u,
                plane.basis_v,
            )
            lines.extend(_segments_to_mosaic_lines(cell, segments))
        overlays.append(tuple(lines))
    return tuple(overlays)


def _mesh_plane_segments(
    points: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> tuple[np.ndarray, ...]:
    pts = np.asarray(points, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return ()
    if tris.ndim != 2 or tris.shape[0] == 0 or tris.shape[1] < 3:
        return ()
    tris = tris[:, :3]
    valid = np.all((tris >= 0) & (tris < int(pts.shape[0])), axis=1)
    if not np.any(valid):
        return ()
    tri_points = pts[tris[valid]]
    plane_point_arr = np.asarray(plane_point, dtype=np.float64)
    signed = np.einsum("mij,j->mi", tri_points - plane_point_arr[None, None, :], plane_normal)
    abs_signed = np.abs(signed)
    eps = 1e-8
    edge_indices = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
    coplanar_edges = (abs_signed[:, edge_indices[:, 0]] <= eps) & (abs_signed[:, edge_indices[:, 1]] <= eps)
    segments: list[np.ndarray] = []
    if np.any(coplanar_edges):
        coplanar_mask = np.any(coplanar_edges, axis=1)
        chosen_edges = edge_indices[np.argmax(coplanar_edges[coplanar_mask], axis=1)]
        coplanar_points = tri_points[coplanar_mask]
        edge_point_a = coplanar_points[np.arange(coplanar_points.shape[0]), chosen_edges[:, 0]]
        edge_point_b = coplanar_points[np.arange(coplanar_points.shape[0]), chosen_edges[:, 1]]
        projected = _project_points_to_plane(
            np.stack([edge_point_a, edge_point_b], axis=1).reshape(-1, 3),
            plane_point_arr,
            np.asarray(basis_u, dtype=np.float64),
            np.asarray(basis_v, dtype=np.float64),
        ).reshape(-1, 2, 2)
        segments.extend(np.asarray(segment, dtype=np.float32) for segment in projected)
        tri_points = tri_points[~coplanar_mask]
        signed = signed[~coplanar_mask]
        abs_signed = abs_signed[~coplanar_mask]
    if tri_points.shape[0] == 0:
        return tuple(segments)
    candidate_points = np.full((tri_points.shape[0], 6, 3), np.nan, dtype=np.float64)
    candidate_counts = np.zeros(tri_points.shape[0], dtype=np.int8)
    for start_idx, end_idx in edge_indices:
        vertex_mask = abs_signed[:, start_idx] <= eps
        if np.any(vertex_mask):
            matched = np.flatnonzero(vertex_mask)
            slots = candidate_counts[matched]
            candidate_points[matched, slots] = tri_points[matched, start_idx]
            candidate_counts[matched] += 1
        cross_mask = signed[:, start_idx] * signed[:, end_idx] < 0.0
        if np.any(cross_mask):
            matched = np.flatnonzero(cross_mask)
            start_signed = signed[matched, start_idx]
            end_signed = signed[matched, end_idx]
            t = start_signed / (start_signed - end_signed)
            intersections = tri_points[matched, start_idx] + t[:, None] * (
                tri_points[matched, end_idx] - tri_points[matched, start_idx]
            )
            slots = candidate_counts[matched]
            candidate_points[matched, slots] = intersections
            candidate_counts[matched] += 1
    for tri_idx in np.flatnonzero(candidate_counts >= 2):
        hits = candidate_points[tri_idx, : int(candidate_counts[tri_idx])]
        unique_hits: list[np.ndarray] = []
        for hit in hits:
            if not np.all(np.isfinite(hit)):
                continue
            if not any(np.allclose(hit, other, atol=1e-7) for other in unique_hits):
                unique_hits.append(np.asarray(hit, dtype=np.float64))
            if len(unique_hits) > 2:
                break
        if len(unique_hits) != 2:
            continue
        segments.append(
            _project_points_to_plane(
                np.asarray(unique_hits, dtype=np.float64),
                plane_point_arr,
                np.asarray(basis_u, dtype=np.float64),
                np.asarray(basis_v, dtype=np.float64),
            ).astype(np.float32)
        )
    return tuple(segments)


def _project_points_to_plane(
    points_xyz: np.ndarray,
    plane_point: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> np.ndarray:
    rel = np.asarray(points_xyz, dtype=np.float64) - plane_point[None, :]
    x = rel @ basis_u
    y = rel @ basis_v
    return np.stack([x, y], axis=-1)


def _segments_to_mosaic_lines(cell: MosaicCell, segments: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    plane = cell.plane
    if plane is None:
        return ()
    x_span = max(plane.x_max - plane.x_min, 1e-8)
    y_span = max(plane.y_max - plane.y_min, 1e-8)
    lines: list[np.ndarray] = []
    for segment in segments:
        pts = np.asarray(segment, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 2:
            continue
        px = (pts[:, 0] - plane.x_min) / x_span * float(cell.image_width - 1) + cell.image_x0
        py = (pts[:, 1] - plane.y_min) / y_span * float(cell.image_height - 1) + cell.image_y0
        lines.append(np.stack([px, py], axis=-1).astype(np.float32))
    return tuple(lines)


def _mesh_points_to_render_frame(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected mesh points with shape (N, 3+), got {arr.shape}")
    return (arr[:, :3] @ _MESH_RENDER_FRAME_TRANSFORM.T).astype(np.float32)


def _valid_mesh_faces(faces: np.ndarray, *, n_points: int) -> np.ndarray:
    tris = np.asarray(faces, dtype=np.int32)
    if tris.ndim != 2 or tris.shape[1] < 3 or tris.shape[0] == 0 or int(n_points) <= 0:
        return np.empty((0, 3), dtype=np.int32)
    tris = tris[:, :3]
    valid = tris[np.all((tris >= 0) & (tris < int(n_points)), axis=1)]
    return valid if valid.size else np.empty((0, 3), dtype=np.int32)
