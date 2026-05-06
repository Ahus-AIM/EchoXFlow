"""Packed mesh annotation loading and validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshFrame:
    """One frame from a packed mesh annotation."""

    points: np.ndarray
    faces: np.ndarray
    point_values: np.ndarray | None = None
    face_values: np.ndarray | None = None
    timestamp: float | None = None


@dataclass(frozen=True)
class PackedMeshAnnotation:
    """Typed packed mesh annotation arrays loaded from one recording."""

    points_path: str
    points: np.ndarray
    faces_path: str
    faces: np.ndarray
    point_frame_offsets_path: str | None = None
    point_frame_offsets: np.ndarray | None = None
    face_frame_offsets_path: str | None = None
    face_frame_offsets: np.ndarray | None = None
    point_component_offsets_path: str | None = None
    point_component_offsets: np.ndarray | None = None
    face_component_offsets_path: str | None = None
    face_component_offsets: np.ndarray | None = None
    point_values_path: str | None = None
    point_values: np.ndarray | None = None
    face_values_path: str | None = None
    face_values: np.ndarray | None = None
    component_names_path: str | None = None
    component_names: tuple[str, ...] = ()
    timestamps_path: str | None = None
    timestamps: np.ndarray | None = None

    @property
    def frame_count(self) -> int:
        if self.point_frame_offsets is not None:
            return max(0, int(self.point_frame_offsets.size) - 1)
        if self.timestamps is not None:
            return int(self.timestamps.size)
        return 1 if self.points.size or self.faces.size else 0

    @property
    def frame_offsets(self) -> np.ndarray | None:
        """Backward-compatible alias for point frame offsets."""
        return self.point_frame_offsets

    def frame(self, index: int) -> MeshFrame:
        """Return points, faces, values, and timestamp for one mesh frame."""
        frame_index = _valid_frame_index(index, self.frame_count)
        point_slice = _offset_slice(self.point_frame_offsets, frame_index, self.points.shape[0])
        face_slice = _offset_slice(self.face_frame_offsets, frame_index, self.faces.shape[0])
        point_values = None if self.point_values is None else np.asarray(self.point_values[point_slice])
        face_values = None if self.face_values is None else np.asarray(self.face_values[face_slice])
        timestamp = None if self.timestamps is None else float(self.timestamps[frame_index])
        return MeshFrame(
            points=np.asarray(self.points[point_slice]),
            faces=np.asarray(self.faces[face_slice]),
            point_values=point_values,
            face_values=face_values,
            timestamp=timestamp,
        )

    def frames(self) -> tuple[MeshFrame, ...]:
        """Return all mesh frames as task-ready views."""
        return tuple(self.frame(index) for index in range(self.frame_count))


def load_packed_mesh_annotation(
    *,
    group: Any,
    store_path: Path,
    timestamp_path: Callable[[str], str | None],
    prefix: str = "3d_left_ventricle_mesh",
) -> PackedMeshAnnotation:
    """Load typed packed mesh arrays using conventional EchoXFlow mesh paths."""
    base = _normalize_data_path(prefix).removeprefix("data/")
    points_path = _first_existing(group, (f"data/{base}/point_values", f"data/{base}_points", f"data/{base}"))
    faces_path = _first_existing(group, (f"data/{base}/face_values", f"data/{base}_faces", f"data/{base}_face_indices"))
    if points_path is None or faces_path is None:
        raise KeyError(f"Could not find packed mesh points and faces for prefix `{prefix}` in {store_path}")
    point_offsets_path = _first_existing(
        group, (f"data/{base}/point_frame_offsets", f"data/{base}_frame_offsets", f"data/{base}_offsets")
    )
    face_offsets_path = _first_existing(group, (f"data/{base}/face_frame_offsets",))
    point_component_offsets_path = _first_existing(group, (f"data/{base}/point_component_offsets",))
    face_component_offsets_path = _first_existing(group, (f"data/{base}/face_component_offsets",))
    values_path = _first_existing(group, (f"data/{base}_point_values", f"data/{base}_values"))
    face_values_path = _first_existing(group, (f"data/{base}_face_values",))
    component_names_path = _first_existing(group, (f"data/{base}/component_names",))
    timestamps_path = timestamp_path(base)
    points = np.asarray(group[points_path][:], dtype=np.float32)
    faces = np.asarray(group[faces_path][:], dtype=np.int32)
    point_frame_offsets = (
        None if point_offsets_path is None else np.asarray(group[point_offsets_path][:], dtype=np.int64)
    )
    face_frame_offsets = None if face_offsets_path is None else np.asarray(group[face_offsets_path][:], dtype=np.int64)
    point_component_offsets = (
        None
        if point_component_offsets_path is None
        else np.asarray(group[point_component_offsets_path][:], dtype=np.int64)
    )
    face_component_offsets = (
        None
        if face_component_offsets_path is None
        else np.asarray(group[face_component_offsets_path][:], dtype=np.int64)
    )
    point_values = None if values_path is None else np.asarray(group[values_path][:])
    face_values = None if face_values_path is None else np.asarray(group[face_values_path][:])
    component_names = (
        ()
        if component_names_path is None
        else tuple(_decode_component_name(value) for value in np.asarray(group[component_names_path][:]).reshape(-1))
    )
    timestamps = None if timestamps_path is None else np.asarray(group[timestamps_path][:], dtype=np.float32)
    _validate_packed_mesh(
        points,
        faces,
        point_frame_offsets,
        face_frame_offsets,
        point_component_offsets,
        face_component_offsets,
        point_values,
        face_values,
        component_names,
        timestamps,
    )
    return PackedMeshAnnotation(
        points_path=points_path,
        points=points,
        faces_path=faces_path,
        faces=faces,
        point_frame_offsets_path=point_offsets_path,
        point_frame_offsets=point_frame_offsets,
        face_frame_offsets_path=face_offsets_path,
        face_frame_offsets=face_frame_offsets,
        point_component_offsets_path=point_component_offsets_path,
        point_component_offsets=point_component_offsets,
        face_component_offsets_path=face_component_offsets_path,
        face_component_offsets=face_component_offsets,
        point_values_path=values_path,
        point_values=point_values,
        face_values_path=face_values_path,
        face_values=face_values,
        component_names_path=component_names_path,
        component_names=component_names,
        timestamps_path=timestamps_path,
        timestamps=timestamps,
    )


def _normalize_data_path(path: str) -> str:
    path = str(path).strip("/")
    if path.startswith("data/"):
        return path
    if path.startswith("timestamps/"):
        raise ValueError(f"Expected a data array path, got timestamp path `{path}`")
    return f"data/{path}"


def _first_existing(group: Any, paths: tuple[str, ...]) -> str | None:
    return next((path for path in paths if path in group), None)


def _validate_packed_mesh(
    points: np.ndarray,
    faces: np.ndarray,
    point_frame_offsets: np.ndarray | None,
    face_frame_offsets: np.ndarray | None,
    point_component_offsets: np.ndarray | None,
    face_component_offsets: np.ndarray | None,
    point_values: np.ndarray | None,
    face_values: np.ndarray | None,
    component_names: tuple[str, ...],
    timestamps: np.ndarray | None,
) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Packed mesh points must have shape [N,3], got {points.shape}")
    if faces.ndim != 2 or faces.shape[1] not in {3, 4}:
        raise ValueError(f"Packed mesh faces must have shape [M,3] or [M,4], got {faces.shape}")
    if faces.size and int(np.min(faces)) < 0:
        raise ValueError("Packed mesh face indices must not be negative")
    _validate_offsets(point_frame_offsets, expected_final=points.shape[0], label="point_frame_offsets")
    _validate_offsets(face_frame_offsets, expected_final=faces.shape[0], label="face_frame_offsets")
    if point_frame_offsets is not None and face_frame_offsets is not None:
        if point_frame_offsets.size != face_frame_offsets.size:
            raise ValueError("Packed mesh point and face frame offsets must have matching lengths")
        for frame_index in range(max(0, point_frame_offsets.size - 1)):
            point_count = int(point_frame_offsets[frame_index + 1] - point_frame_offsets[frame_index])
            face_slice = slice(int(face_frame_offsets[frame_index]), int(face_frame_offsets[frame_index + 1]))
            frame_faces = faces[face_slice]
            if frame_faces.size and int(np.max(frame_faces)) >= point_count:
                raise ValueError("Packed mesh face indices are outside their frame point array")
    elif faces.size and int(np.max(faces)) >= points.shape[0]:
        raise ValueError("Packed mesh face indices are outside the point array")
    frame_count = max(0, int(point_frame_offsets.size) - 1) if point_frame_offsets is not None else 1
    _validate_offsets(point_component_offsets, expected_final=frame_count, label="point_component_offsets")
    _validate_offsets(face_component_offsets, expected_final=frame_count, label="face_component_offsets")
    if component_names:
        expected_component_offsets = len(component_names) + 1
        if point_component_offsets is not None and point_component_offsets.size != expected_component_offsets:
            raise ValueError("Packed mesh point component offsets must have len(component_names) + 1 entries")
        if face_component_offsets is not None and face_component_offsets.size != expected_component_offsets:
            raise ValueError("Packed mesh face component offsets must have len(component_names) + 1 entries")
    if timestamps is not None and timestamps.size != frame_count:
        raise ValueError("Packed mesh timestamps must match the number of mesh frames")
    if point_values is not None and point_values.shape[0] != points.shape[0]:
        raise ValueError("Packed mesh point values must match the point array length")
    if face_values is not None and face_values.shape[0] != faces.shape[0]:
        raise ValueError("Packed mesh face values must match the face array length")


def _validate_offsets(offsets: np.ndarray | None, *, expected_final: int, label: str) -> None:
    if offsets is None:
        return
    if offsets.ndim != 1 or offsets.size == 0 or int(offsets[0]) != 0:
        raise ValueError(f"Packed mesh {label} must be a non-empty 1D array starting at 0")
    if int(offsets[-1]) != expected_final:
        raise ValueError(f"Packed mesh {label} ends at {int(offsets[-1])}, expected {expected_final}")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"Packed mesh {label} must be monotonically increasing")


def _valid_frame_index(index: int, frame_count: int) -> int:
    resolved = int(index)
    if resolved < 0:
        resolved += frame_count
    if resolved < 0 or resolved >= frame_count:
        raise IndexError(f"Mesh frame index {index} is outside 0..{max(0, frame_count - 1)}")
    return resolved


def _offset_slice(offsets: np.ndarray | None, frame_index: int, fallback_stop: int) -> slice:
    if offsets is None:
        return slice(0, fallback_stop)
    return slice(int(offsets[frame_index]), int(offsets[frame_index + 1]))


def _decode_component_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)
