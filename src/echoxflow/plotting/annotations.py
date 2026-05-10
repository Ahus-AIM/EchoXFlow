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
from echoxflow.scan import SphericalGeometry, mesh_frame_indices_for_volume_timestamps

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
    annotation_metadata = _annotation_metadata_by_value_path(store)
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
        metadata = annotation_metadata.get(_normalize_annotation_path(path))
        overlay_metadata = _annotation_overlay_metadata(metadata, array_attrs=_array_attrs(store, path))
        if _is_spectral_annotation_path(loaded.data_path, path):
            overlays.append({"kind": "spectral_points", "points": points, "path": path, **overlay_metadata})
        else:
            overlays.append({"kind": "physical_points", "points": points, "path": path, **overlay_metadata})
    if show_sampling_gate:
        gate_overlay = _sampling_gate_overlay(store, loaded)
        if gate_overlay is not None:
            overlays.append(gate_overlay)
    if show_sampling_line:
        overlays.extend(_sampling_line_overlays(store, loaded))
    attrs = dict(loaded.attrs)
    if overlays:
        attrs["annotation_overlays"] = tuple(overlays)
    mesh_sequences = store.load_object().mesh_sequences if loaded.data_path == "data/3d_brightness_mode" else ()
    if mesh_sequences:
        try:
            mesh_sequence = mesh_sequences[0]
            mesh = store.load_packed_mesh_annotation(mesh_sequence.mesh_group.path)
            timestamps = mesh_sequence.timestamps
            if timestamps is not None:
                timestamp_values = (
                    store.group[timestamps.path][:]
                    if timestamps.path in store.group
                    else store.load_array(timestamps.path)
                )
                mesh = replace(
                    mesh,
                    timestamps_path=timestamps.path,
                    timestamps=np.asarray(timestamp_values, dtype=np.float32).reshape(-1),
                )
            attrs["mesh_annotation"] = mesh
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("could not load mesh annotation: %s", exc)
        except (TypeError, ValueError) as exc:
            logger.warning("could not parse mesh annotation: %s", exc)
    return replace(loaded, attrs=attrs) if attrs != dict(loaded.attrs) else loaded


def _has_sample_volume_gate(loaded_arrays: tuple[LoadedArray, ...]) -> bool:
    return any(loaded.data_path == "data/1d_pulsed_wave_doppler" for loaded in loaded_arrays)


def _has_sampling_line_source(loaded_arrays: tuple[LoadedArray, ...]) -> bool:
    return any(
        loaded.data_path in {"data/1d_continuous_wave_doppler", "data/1d_motion_mode"} for loaded in loaded_arrays
    )


def _annotation_paths_for_loaded(store: RecordingStore, loaded: LoadedArray) -> tuple[str, ...]:
    paths = tuple(path for path in store.array_paths if path.startswith("data/"))
    name = loaded.data_path.removeprefix("data/")
    if name.startswith("1d_"):
        prefix = f"data/{name}_annotation_"
        return tuple(path for path in paths if path.startswith(prefix))
    if name.startswith("2d_brightness_mode"):
        return tuple(path for path in paths if "overlay_physical_points" in path)
    return ()


def _is_spectral_annotation_path(data_path: str, annotation_path: str) -> bool:
    name = data_path.removeprefix("data/")
    return name.startswith("1d_") and annotation_path.startswith(f"data/{name}_annotation_")


def _annotation_metadata_by_value_path(store: RecordingStore) -> dict[str, Mapping[str, object]]:
    metadata: dict[str, Mapping[str, object]] = {}
    for document in _render_documents(getattr(store.group, "attrs", {})):
        annotations = document.get("annotations")
        if not isinstance(annotations, list):
            continue
        for item in annotations:
            if not isinstance(item, Mapping):
                continue
            path = _annotation_value_path(item.get("value"))
            if path is not None:
                metadata[_normalize_annotation_path(path)] = item
    return metadata


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
        label = _non_empty_string(metadata.get("label"))
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


def _supports_sampling_gate_overlay(loaded: LoadedArray) -> bool:
    return loaded.data_path == "data/tissue_doppler" or loaded.data_path.startswith("data/2d_brightness_mode")


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
        tracks = document.get("tracks")
        if isinstance(tracks, list):
            for track in tracks:
                if not isinstance(track, Mapping):
                    continue
                derived = track.get("derived_from")
                if _semantic_id(track) == "tissue_doppler_gate":
                    return derived if isinstance(derived, Mapping) else track
                if isinstance(derived, Mapping) and str(derived.get("kind", "")).strip() == "tissue_doppler_gate":
                    return derived
        gate = document.get("sampling_gate_metadata")
        if isinstance(gate, Mapping):
            return gate
    raw = None if loaded.stream is None else loaded.stream.metadata.raw
    if isinstance(raw, Mapping):
        gate = raw.get("sampling_gate_metadata")
        if isinstance(gate, Mapping):
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
    if not isinstance(render_inputs, list):
        return tuple(documents)
    for item in render_inputs:
        if isinstance(item, Mapping) and isinstance(item.get("document"), Mapping):
            documents.append(item["document"])
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


_MESH_RENDER_FRAME_TRANSFORM = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def mesh_mosaic_annotation_lines(
    annotation: PackedMeshAnnotation,
    geometry: SphericalGeometry,
    *,
    frame_count: int,
    mosaic_shape: tuple[int, int],
    view: PanelView,
    volume_timestamps: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Build per-frame mesh intersection lines in cartesian 3D mosaic pixel coordinates."""
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
    rows, cols = 3, 4
    cell_h = max(1, int(mosaic_shape[0]) // rows)
    cell_w = max(1, int(mosaic_shape[1]) // cols)
    panel_sizes = _mesh_panel_sizes_from_volume(geometry, height=cell_h, width=cell_h, cover_depth_fraction=1.0)
    plane_specs = _mesh_slice_plane_specs_from_volume(geometry, cover_depth_fraction=1.0)
    panel_count = min(rows * cols, len(panel_sizes), len(plane_specs))
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
        for panel_idx, (plane_spec, panel_size) in enumerate(zip(plane_specs[:panel_count], panel_sizes[:panel_count])):
            segments = _mesh_plane_segments(
                points,
                faces,
                np.asarray(plane_spec["plane_point"], dtype=np.float64),
                np.asarray(plane_spec["plane_normal"], dtype=np.float64),
                np.asarray(plane_spec["basis_u"], dtype=np.float64),
                np.asarray(plane_spec["basis_v"], dtype=np.float64),
            )
            lines.extend(
                _segments_to_mosaic_lines(
                    panel_idx=panel_idx,
                    segments=segments,
                    plane_spec=plane_spec,
                    panel_size_hw=panel_size,
                    cell_shape=(cell_h, cell_w),
                )
            )
        overlays.append(tuple(lines))
    return tuple(overlays)


def _mesh_slice_plane_specs_from_volume(
    geometry: SphericalGeometry,
    *,
    cover_depth_fraction: float,
) -> tuple[dict[str, np.ndarray | float], ...]:
    depth_fractions = np.linspace(4.0 / 15.0, 10.0 / 15.0, 9, dtype=np.float64).tolist()
    radial_axis_angles_deg = (0.0, -60.0, -120.0)
    radial_axis_half_width = max(
        _radial_axis_half_width_m_3d(geometry, cover_depth_fraction, angle) for angle in radial_axis_angles_deg
    )
    cover_depth_m = geometry.depth_start_m + float(np.clip(cover_depth_fraction, 0.0, 1.0)) * (
        geometry.depth_end_m - geometry.depth_start_m
    )
    normal_half_width_x_m = cover_depth_m * math.tan(0.5 * geometry.azimuth_width_rad)
    normal_half_width_y_m = cover_depth_m * math.tan(0.5 * geometry.elevation_width_rad)
    plane_specs: list[dict[str, np.ndarray | float]] = []
    for row_idx, angle_deg in enumerate(radial_axis_angles_deg):
        basis_u, basis_v, plane_normal = _slice_plane_basis_radial_axis(float(angle_deg))
        plane_specs.append(
            {
                "plane_point": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
                "plane_normal": plane_normal,
                "basis_u": basis_u,
                "basis_v": basis_v,
                "x_min": -float(radial_axis_half_width),
                "x_max": float(radial_axis_half_width),
                "y_min": float(geometry.depth_start_m),
                "y_max": float(geometry.depth_end_m),
            }
        )
        start = row_idx * 3
        basis_u, basis_v, plane_normal = _slice_plane_basis_normal_slice()
        for depth_fraction in depth_fractions[start : start + 3]:
            depth_m = geometry.depth_start_m + float(depth_fraction) * (geometry.depth_end_m - geometry.depth_start_m)
            plane_specs.append(
                {
                    "plane_point": np.asarray([0.0, 0.0, depth_m], dtype=np.float64),
                    "plane_normal": plane_normal,
                    "basis_u": basis_u,
                    "basis_v": basis_v,
                    "x_min": -float(normal_half_width_x_m),
                    "x_max": float(normal_half_width_x_m),
                    "y_min": -float(normal_half_width_y_m),
                    "y_max": float(normal_half_width_y_m),
                }
            )
    return tuple(plane_specs)


def _mesh_panel_sizes_from_volume(
    geometry: SphericalGeometry,
    *,
    height: int,
    width: int,
    cover_depth_fraction: float,
) -> tuple[tuple[int, int], ...]:
    radial_axis_angles_deg = (0.0, -60.0, -120.0)
    radial_axis_size = _radial_axis_output_size_3d(
        (height, width),
        geometry,
        cover_depth_fraction,
        radial_axis_angles_deg,
    )
    slice_size = (int(height), int(width))
    row = (radial_axis_size, slice_size, slice_size, slice_size)
    sizes: list[tuple[int, int]] = []
    for _ in range(3):
        sizes.extend(row)
    return tuple(sizes)


def _radial_axis_output_size_3d(
    depth_slice_size: tuple[int, int],
    geometry: SphericalGeometry,
    cover_depth_fraction: float,
    angles_deg: tuple[float, ...],
) -> tuple[int, int]:
    depth_slice_height, _ = depth_slice_size
    depth_span_m = max(geometry.depth_end_m - geometry.depth_start_m, 1e-6)
    lateral_half_width_m = max(
        _radial_axis_half_width_m_3d(geometry, cover_depth_fraction, angle) for angle in angles_deg
    )
    return max(2, int(depth_slice_height)), max(
        2, int(round(max(2, int(depth_slice_height)) * (2.0 * lateral_half_width_m) / depth_span_m))
    )


def _radial_axis_half_width_m_3d(
    geometry: SphericalGeometry,
    cover_depth_fraction: float,
    angle_deg: float,
) -> float:
    depth_span_m = max(geometry.depth_end_m - geometry.depth_start_m, 1e-6)
    cover_depth_m = geometry.depth_start_m + float(np.clip(cover_depth_fraction, 0.0, 1.0)) * depth_span_m
    az_half_width_m = cover_depth_m * math.tan(0.5 * geometry.azimuth_width_rad)
    el_half_width_m = cover_depth_m * math.tan(0.5 * geometry.elevation_width_rad)
    angle_rad = math.radians(float(angle_deg))
    return float(az_half_width_m * abs(math.cos(angle_rad)) + el_half_width_m * abs(math.sin(angle_rad)))


def _slice_plane_basis_radial_axis(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle_rad = math.radians(float(angle_deg))
    basis_u = np.asarray([math.cos(angle_rad), math.sin(angle_rad), 0.0], dtype=np.float64)
    basis_v = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    normal = np.cross(basis_u, basis_v)
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    return basis_u, basis_v, normal


def _slice_plane_basis_normal_slice() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    basis_v = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return basis_u, basis_v, normal


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


def _segments_to_mosaic_lines(
    *,
    panel_idx: int,
    segments: tuple[np.ndarray, ...],
    plane_spec: dict[str, np.ndarray | float],
    panel_size_hw: tuple[int, int],
    cell_shape: tuple[int, int],
) -> tuple[np.ndarray, ...]:
    row = int(panel_idx) // 4
    col = int(panel_idx) % 4
    height = int(panel_size_hw[0])
    width = int(panel_size_hw[1])
    x_min = float(plane_spec["x_min"])
    x_max = float(plane_spec["x_max"])
    y_min = float(plane_spec["y_min"])
    y_max = float(plane_spec["y_max"])
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)
    lines: list[np.ndarray] = []
    for segment in segments:
        pts = np.asarray(segment, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 2:
            continue
        px = (pts[:, 0] - x_min) / x_span * float(width - 1)
        py = (pts[:, 1] - y_min) / y_span * float(height - 1)
        line = np.stack([px, py], axis=-1).astype(np.float32)
        line[:, 0] += col * int(cell_shape[1])
        line[:, 1] += row * int(cell_shape[0])
        lines.append(line)
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
