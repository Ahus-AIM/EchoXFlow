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

    def frame(self, index: int) -> MeshFrame:
        """Return points, faces, and timestamp for one mesh frame."""
        frame_index = _valid_frame_index(index, self.frame_count)
        point_slice = _offset_slice(self.point_frame_offsets, frame_index, self.points.shape[0])
        face_slice = _offset_slice(self.face_frame_offsets, frame_index, self.faces.shape[0])
        timestamp = None if self.timestamps is None else float(self.timestamps[frame_index])
        return MeshFrame(
            points=np.asarray(self.points[point_slice]),
            faces=np.asarray(self.faces[face_slice]),
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
    base = _normalize_data_path(prefix).removeprefix("data/").removeprefix("annotations/")
    mesh_root = f"annotations/{base}"
    points_path = _first_existing(group, (f"{mesh_root}/point_values",))
    faces_path = _first_existing(group, (f"{mesh_root}/face_values",))
    if points_path is None or faces_path is None:
        raise KeyError(f"Could not find packed mesh points and faces for prefix `{prefix}` in {store_path}")
    point_offsets_path = _first_existing(group, (f"{mesh_root}/point_frame_offsets",))
    face_offsets_path = _first_existing(group, (f"{mesh_root}/face_frame_offsets",))
    point_component_offsets_path = _first_existing(group, (f"{mesh_root}/point_component_offsets",))
    face_component_offsets_path = _first_existing(group, (f"{mesh_root}/face_component_offsets",))
    component_names_path = _first_existing(group, (f"{mesh_root}/component_names",))
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


@dataclass(frozen=True)
class MeshVolumeCurve:
    """Enclosed volume of each mesh frame in millilitres, with how it was obtained."""

    volumes_ml: np.ndarray
    component: str | None
    method: str  # "divergence" for closed surfaces, "convex_hull" otherwise
    frame_start: int = 0

    @property
    def description(self) -> str:
        part = "" if self.component is None else f" of `{self.component}`"
        how = "divergence theorem" if self.method == "divergence" else "convex hull (faces are not closed)"
        return f"LV volume{part} by {how}"


def mesh_volume_curve_ml(annotation: PackedMeshAnnotation) -> MeshVolumeCurve | None:
    """Volume enclosed by each mesh frame in ml (points are in metres).

    A closed triangle surface is integrated with the divergence theorem (signed tetrahedra to the origin).
    Surfaces with open edges fall back to their convex hull. With several components the endocardial one is used.
    Returns None when the mesh has no usable frames.
    """
    first, stop, component = _volume_frame_range(annotation)
    frames = range(first, min(stop, annotation.frame_count))
    volumes: list[float] = []
    methods: set[str] = set()
    for index in frames:
        frame = annotation.frame(index)
        points = np.asarray(frame.points, dtype=np.float64)
        tris = _triangles(frame.faces, points.shape[0])
        if points.shape[0] < 4 or tris.shape[0] == 0:
            volumes.append(float("nan"))
            continue
        if _is_closed_surface(tris):
            volumes.append(abs(_signed_volume(points, tris)))
            methods.add("divergence")
        else:
            volumes.append(_convex_hull_volume(points))
            methods.add("convex_hull")
    if not volumes or not np.any(np.isfinite(volumes)):
        return None
    method = "convex_hull" if "convex_hull" in methods else "divergence"
    return MeshVolumeCurve(np.asarray(volumes) * 1e6, component, method, first)


def _volume_frame_range(annotation: PackedMeshAnnotation) -> tuple[int, int, str | None]:
    names = annotation.component_names
    offsets = annotation.point_component_offsets
    if len(names) < 2 or offsets is None or offsets.size != len(names) + 1:
        return 0, annotation.frame_count, names[0] if len(names) == 1 else None
    chosen = next((i for i, name in enumerate(names) if "endo" in name.lower()), 0)
    return int(offsets[chosen]), int(offsets[chosen + 1]), names[chosen]


def _triangles(faces: np.ndarray, n_points: int) -> np.ndarray:
    arr = np.asarray(faces, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)
    if arr.shape[1] == 4:
        arr = np.concatenate([arr[:, [0, 1, 2]], arr[:, [0, 2, 3]]])
    arr = arr[:, :3]
    return arr[np.all((arr >= 0) & (arr < n_points), axis=1)]


def _is_closed_surface(tris: np.ndarray) -> bool:
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _signed_volume(points: np.ndarray, tris: np.ndarray) -> float:
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def _convex_hull_volume(points: np.ndarray) -> float:
    """Volume of the convex hull of `points` by incremental construction."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    scale = float(np.ptp(pts, axis=0).max()) if pts.shape[0] else 0.0
    if pts.shape[0] < 4 or scale <= 0.0:
        return 0.0
    eps = 1e-9 * scale
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(pts - pts[i0], axis=1)))
    i2 = int(np.argmax(np.linalg.norm(np.cross(pts - pts[i0], pts[i1] - pts[i0]), axis=1)))
    normal = np.cross(pts[i1] - pts[i0], pts[i2] - pts[i0])
    dist = (pts - pts[i0]) @ normal
    i3 = int(np.argmax(np.abs(dist)))
    if abs(dist[i3]) <= eps * max(float(np.linalg.norm(normal)), 1e-30):
        return 0.0
    centroid = pts[[i0, i1, i2, i3]].mean(axis=0)
    faces: list[tuple[int, int, int]] = []

    def add(a: int, b: int, c: int) -> None:
        n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
        faces.append((a, b, c) if float(n @ (pts[a] - centroid)) > 0 else (a, c, b))

    for tri in ((i0, i1, i2), (i0, i1, i3), (i0, i2, i3), (i1, i2, i3)):
        add(*tri)
    for p in range(pts.shape[0]):
        if p in (i0, i1, i2, i3):
            continue
        arr = np.asarray(faces)
        normals = np.cross(pts[arr[:, 1]] - pts[arr[:, 0]], pts[arr[:, 2]] - pts[arr[:, 0]])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        visible = np.einsum("ij,ij->i", normals, pts[p] - pts[arr[:, 0]]) > eps
        if not visible.any():
            continue
        directed = {(int(f[k]), int(f[(k + 1) % 3])) for f in arr[visible] for k in range(3)}
        faces = [tuple(int(v) for v in f) for f in arr[~visible]]  # type: ignore[misc]
        faces.extend((a, b, p) for a, b in directed if (b, a) not in directed)
    hull = np.asarray(faces, dtype=np.int64)
    return abs(_signed_volume(pts, hull))
