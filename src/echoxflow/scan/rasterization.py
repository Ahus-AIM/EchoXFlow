"""Beamspace rasterization helpers for annotation geometry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from matplotlib.path import Path as MatplotlibPath

from echoxflow.scan.coordinates import BeamspacePixelGrid
from echoxflow.scan.geometry import SectorGeometry
from echoxflow.scan.spherical import SphericalGeometry


def points_to_sector_indices(
    points_m: np.ndarray,
    geometry: SectorGeometry,
    *,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Map physical xy points to beamspace row/column coordinates."""
    return BeamspacePixelGrid(
        geometry=geometry, shape=(int(output_shape[0]), int(output_shape[1]))
    ).physical_to_row_col(points_m)


def rasterize_beamspace_mask(
    points_m: np.ndarray,
    geometry: SectorGeometry,
    *,
    output_shape: tuple[int, int],
    fill: bool = True,
) -> np.ndarray:
    """Rasterize an LV annotation polygon into a beamspace segmentation mask."""
    row_col = points_to_sector_indices(points_m, geometry, output_shape=output_shape)
    height, width = int(output_shape[0]), int(output_shape[1])
    mask = np.zeros((height, width), dtype=bool)
    if row_col.shape[0] == 0:
        return mask
    if fill and row_col.shape[0] >= 3:
        rr, cc = np.mgrid[0:height, 0:width]
        pixel_centers = np.column_stack([rr.reshape(-1), cc.reshape(-1)])
        mask[...] = MatplotlibPath(row_col).contains_points(pixel_centers).reshape(height, width)
    indices = np.rint(row_col).astype(np.int64)
    valid = (indices[:, 0] >= 0) & (indices[:, 0] < height) & (indices[:, 1] >= 0) & (indices[:, 1] < width)
    mask[indices[valid, 0], indices[valid, 1]] = True
    return mask


def points_to_spherical_indices(
    points_m: np.ndarray,
    geometry: SphericalGeometry,
    *,
    output_shape: tuple[int, int, int],
) -> np.ndarray:
    """Map physical xyz points to native 3D beamspace coordinates [elevation, azimuth, radius]."""
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shaped [N,3+] in meters, got {points.shape}")
    elevation_count, azimuth_count, radial_count = _shape3(output_shape)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    depth = np.sqrt(x * x + y * y + z * z)
    azimuth = np.arctan2(x, np.sqrt(y * y + z * z))
    elevation = np.arctan2(y, z)
    radial = (depth - geometry.depth_start_m) / _depth_span(geometry) * max(0, radial_count - 1)
    azimuth_index = (
        (azimuth + 0.5 * geometry.azimuth_width_rad)
        / max(float(geometry.azimuth_width_rad), 1e-12)
        * max(0, azimuth_count - 1)
    )
    elevation_index = (
        (elevation + 0.5 * geometry.elevation_width_rad)
        / max(float(geometry.elevation_width_rad), 1e-12)
        * max(0, elevation_count - 1)
    )
    return np.column_stack([elevation_index, azimuth_index, radial]).astype(np.float32)


def rasterize_beamspace_volume_mask(
    points_m: np.ndarray,
    faces: np.ndarray,
    geometry: SphericalGeometry,
    *,
    output_shape: tuple[int, int, int],
    fill: bool = True,
    dilation_iterations: int = 0,
) -> np.ndarray:
    """Rasterize one 3D mesh frame into a native beamspace volume mask."""
    height, width, depth = _shape3(output_shape)
    mask = np.zeros((height, width, depth), dtype=bool)
    points = np.asarray(points_m, dtype=np.float32)
    face_indices = np.asarray(faces, dtype=np.int64)
    if points.size == 0:
        return mask
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected mesh points shaped [N,3+], got {points.shape}")
    if face_indices.size:
        if face_indices.ndim != 2 or face_indices.shape[1] not in {3, 4}:
            raise ValueError(f"Expected mesh faces shaped [M,3] or [M,4], got {face_indices.shape}")
        if int(np.min(face_indices)) < 0 or int(np.max(face_indices)) >= points.shape[0]:
            raise ValueError("Mesh face indices are outside the point array")
        sampled = _sample_mesh_surface(points, face_indices)
    else:
        sampled = points
    indices = np.rint(points_to_spherical_indices(sampled, geometry, output_shape=output_shape)).astype(np.int64)
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < height)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < width)
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < depth)
    )
    mask[indices[valid, 0], indices[valid, 1], indices[valid, 2]] = True
    if fill:
        mask = _fill_radial_spans(mask)
    return _dilate3(mask, iterations=dilation_iterations)


def rasterize_packed_mesh_volume_masks(
    annotation: object,
    geometry: SphericalGeometry,
    *,
    output_shape: tuple[int, int, int] | None = None,
    frame_indices: tuple[int, ...] | list[int] | None = None,
    fill: bool = True,
    dilation_iterations: int = 0,
) -> np.ndarray:
    """Rasterize a packed mesh annotation into [T,E,A,R] beamspace volume masks."""
    shape = _shape3(output_shape or _geometry_shape(annotation))
    frame_count = int(getattr(annotation, "frame_count"))
    indices = tuple(range(frame_count)) if frame_indices is None else tuple(int(index) for index in frame_indices)
    masks = []
    for frame_index in indices:
        frame = cast(Any, annotation).frame(frame_index)
        masks.append(
            rasterize_beamspace_volume_mask(
                frame.points,
                frame.faces,
                geometry,
                output_shape=shape,
                fill=fill,
                dilation_iterations=dilation_iterations,
            )
        )
    return np.asarray(masks, dtype=bool)


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D output shape, got {shape}")
    resolved = tuple(max(1, int(dim)) for dim in shape)
    return resolved[0], resolved[1], resolved[2]


def _geometry_shape(annotation: object) -> tuple[int, int, int]:
    raw = getattr(annotation, "volume_shape", None)
    if raw is None:
        raise ValueError("output_shape is required when the annotation has no volume_shape attribute")
    return _shape3(tuple(int(dim) for dim in raw))


def _sample_mesh_surface(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    sampled: list[np.ndarray] = [points]
    for face in faces:
        vertices = points[face[:3]]
        sampled.append(_sample_triangle(vertices))
        if face.shape[0] == 4:
            sampled.append(_sample_triangle(points[face[[0, 2, 3]]]))
    return np.concatenate(sampled, axis=0)


def _sample_triangle(vertices: np.ndarray) -> np.ndarray:
    edge_lengths = (
        float(np.linalg.norm(vertices[0] - vertices[1])),
        float(np.linalg.norm(vertices[1] - vertices[2])),
        float(np.linalg.norm(vertices[2] - vertices[0])),
    )
    steps = max(2, min(16, int(np.ceil(max(edge_lengths) / 0.002))))
    samples: list[np.ndarray] = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            a = i / steps
            b = j / steps
            c = 1.0 - a - b
            samples.append(a * vertices[0] + b * vertices[1] + c * vertices[2])
    return np.asarray(samples, dtype=np.float32)


def _fill_radial_spans(mask: np.ndarray) -> np.ndarray:
    filled = mask.copy()
    height, width, _ = filled.shape
    for row in range(height):
        for col in range(width):
            hits = np.flatnonzero(mask[row, col])
            if hits.size >= 2:
                filled[row, col, int(hits[0]) : int(hits[-1]) + 1] = True
    return filled


def _dilate3(mask: np.ndarray, *, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(out)
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):
                    expanded |= padded[dz : dz + out.shape[0], dy : dy + out.shape[1], dx : dx + out.shape[2]]
        out = expanded
    return out


def _depth_span(geometry: SphericalGeometry) -> float:
    return max(1e-9, float(geometry.depth_end_m) - float(geometry.depth_start_m))
