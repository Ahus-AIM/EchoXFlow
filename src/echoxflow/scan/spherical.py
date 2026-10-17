"""Spherical 3D brightness-mode slicing utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from echoxflow.scan.interpolation import Interpolation, sample_image_stack, sample_volume


@dataclass(frozen=True)
class SphericalGeometry:
    """3D sector geometry for volumes shaped as elevation, azimuth, radius."""

    depth_start_m: float
    depth_end_m: float
    azimuth_width_rad: float
    elevation_width_rad: float


@dataclass(frozen=True)
class MosaicSlicePlane:
    """Physical plane of one mosaic slice and the extent of its image in plane coordinates (metres)."""

    plane_point: np.ndarray
    plane_normal: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class MosaicCell:
    """One mosaic cell: its rectangle, the slice image centred inside it and, for Cartesian mosaics, its plane."""

    row: int
    col: int
    y0: int
    x0: int
    height: int
    width: int
    image_height: int
    image_width: int
    kind: Literal["radial_axis", "depth_slice"]
    angle_deg: float | None = None
    depth_fraction: float | None = None
    cover_depth_fraction: float | None = None
    plane: MosaicSlicePlane | None = None

    @property
    def image_y0(self) -> int:
        return self.y0 + (self.height - self.image_height) // 2

    @property
    def image_x0(self) -> int:
        return self.x0 + (self.width - self.image_width) // 2


@dataclass(frozen=True)
class SphericalMosaicLayout:
    """Cell rectangles of a 3x4 spherical mosaic, shared by the pixel builder and annotation overlays."""

    cells: tuple[MosaicCell, ...]
    rows: int = 3
    cols: int = 4

    @property
    def height(self) -> int:
        return max(cell.y0 + cell.height for cell in self.cells)

    @property
    def width(self) -> int:
        return max(cell.x0 + cell.width for cell in self.cells)


@dataclass(frozen=True)
class SphericalMosaic:
    """Twelve 3D slices arranged as a 3-row by 4-column frame stack."""

    frames: np.ndarray
    rows: int = 3
    cols: int = 4
    layout: SphericalMosaicLayout | None = None


def spherical_geometry_from_metadata(metadata: Mapping[str, object] | None) -> SphericalGeometry:
    """Parse native 3D geometry from the public Zarr manifest."""
    geometry = _spherical_sector_geometry(metadata)
    if not isinstance(geometry, Mapping):
        raise ValueError("3D brightness mode is missing sectors[].geometry")
    depth_start = _required_float(geometry, "DepthStart")
    depth_end = _required_float(geometry, "DepthEnd")
    width = _required_float(geometry, "Width")
    elevation_width = _optional_float(geometry, "ElevationWidth", default=width)
    if depth_end <= depth_start:
        raise ValueError(f"Invalid 3D depth range: {depth_start} to {depth_end}")
    return SphericalGeometry(
        depth_start_m=depth_start,
        depth_end_m=depth_end,
        azimuth_width_rad=width,
        elevation_width_rad=elevation_width,
    )


def _spherical_sector_geometry(metadata: Mapping[str, object] | None) -> Mapping[str, object] | None:
    raw = metadata or {}
    sectors = raw.get("sectors")
    if isinstance(sectors, list):
        for sector in sectors:
            if not isinstance(sector, Mapping):
                continue
            geometry = sector.get("geometry")
            if not isinstance(geometry, Mapping):
                continue
            if geometry.get("coordinate_system") == "spherical_sector_3d":
                return cast(Mapping[str, object], geometry)
            frames = sector.get("frames")
            if _array_ref_path(frames) == "data/3d_brightness_mode":
                return cast(Mapping[str, object], geometry)
    render_metadata = raw.get("render_metadata")
    if isinstance(render_metadata, Mapping):
        return cast(Mapping[str, object], render_metadata)
    return None


def _array_ref_path(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    path = value.get("array_path") or value.get("zarr_path") or value.get("path")
    return str(path).strip("/") if path else None


_RADIAL_AXIS_ANGLES_DEG = (0.0, -60.0, -120.0)
_AZIMUTH_FRACTIONS = (0.75, 0.5, 0.25)


def beamspace_spherical_mosaic(
    volumes: np.ndarray,
    *,
    output_size: tuple[int, int] = (120, 120),
    cell_height: int | None = None,
) -> SphericalMosaic:
    """Build a 12-panel mosaic from native beamspace slices, all `cell_height` high with their aspect kept.

    `cell_height` defaults to `output_size[0]`; the renderer passes a height derived from the target panel size.
    """
    arr = _volume_stack(volumes)
    layout = beamspace_mosaic_layout(arr.shape[1:4], output_size=output_size, cell_height=cell_height)
    slices: list[np.ndarray] = []
    for cell in layout.cells:
        if cell.kind == "radial_axis":
            native = _native_radial_axis_slice(arr, az_fraction=_AZIMUTH_FRACTIONS[cell.row])
        else:
            native = _native_radial_normal_slice(arr, depth_fraction=cast(float, cell.depth_fraction))
        slices.append(_resize_stack(native, (cell.image_height, cell.image_width)))
    return SphericalMosaic(frames=_mosaic_frames(layout, slices, fill_value=0.0), layout=layout)


def beamspace_mosaic_layout(
    shape_ear: tuple[int, int, int],
    *,
    output_size: tuple[int, int] = (120, 120),
    cell_height: int | None = None,
) -> SphericalMosaicLayout:
    """Lay out native slices at one cell height (`cell_height`, default `output_size[0]`), widths following aspect."""
    elevation_count, azimuth_count, radial_count = (max(1, int(n)) for n in shape_ear)
    cell_h = _cell_height(output_size, cell_height)
    radial_size = (cell_h, max(2, int(round(cell_h * elevation_count / radial_count))))
    cut_size = (cell_h, max(2, int(round(cell_h * azimuth_count / elevation_count))))
    return _build_layout(
        radial_sizes=(radial_size,) * 3,
        cut_sizes=(cut_size,) * 3,
        planes=None,
        cover_fractions=(None, None, None),
    )


def cartesian_mosaic_layout(
    geometry: SphericalGeometry,
    *,
    output_size: tuple[int, int] = (120, 120),
    cover_depth_fraction: float = 1.0,
    cell_height: int | None = None,
) -> SphericalMosaicLayout:
    """Lay out the Cartesian mosaic: slice planes, physical extents and cell rectangles.

    Radial-axis slices share one physical scale. Each row of depth cuts shares one scale too, covering exactly the
    deepest cut of that row so the cuts fill their cells. Cell widths are allocated per column. Every image cell is
    isotropic: its width follows its physical extent so a pixel spans the same metres in x and in y.
    `cell_height` defaults to `output_size[0]`.
    """
    cell_h = _cell_height(output_size, cell_height)
    span = _depth_span(geometry)
    half_width = max(_radial_axis_half_width_m(geometry, cover_depth_fraction, a) for a in _RADIAL_AXIS_ANGLES_DEG)
    radial_width = max(2, int(round(cell_h * 2.0 * half_width / span)))
    planes: list[tuple[MosaicSlicePlane, tuple[MosaicSlicePlane, ...]]] = []
    cut_sizes: list[tuple[int, int]] = []
    covers: list[float] = []
    for row, angle_deg in enumerate(_RADIAL_AXIS_ANGLES_DEG):
        basis_u, basis_v, normal = slice_plane_basis_radial_axis(angle_deg)
        radial_plane = MosaicSlicePlane(
            plane_point=np.zeros(3, dtype=np.float64),
            plane_normal=normal,
            basis_u=basis_u,
            basis_v=basis_v,
            x_min=-half_width,
            x_max=half_width,
            y_min=float(geometry.depth_start_m),
            y_max=float(geometry.depth_end_m),
        )
        fractions = _depth_fractions_for_row(row)
        row_cover = max(fractions)
        cover_m = geometry.depth_start_m + row_cover * span
        half_x = cover_m * math.tan(0.5 * geometry.azimuth_width_rad)
        half_y = cover_m * math.tan(0.5 * geometry.elevation_width_rad)
        cut_basis_u, cut_basis_v, cut_normal = slice_plane_basis_normal_slice()
        cuts = tuple(
            MosaicSlicePlane(
                plane_point=np.asarray([0.0, 0.0, geometry.depth_start_m + f * span], dtype=np.float64),
                plane_normal=cut_normal,
                basis_u=cut_basis_u,
                basis_v=cut_basis_v,
                x_min=-half_x,
                x_max=half_x,
                y_min=-half_y,
                y_max=half_y,
            )
            for f in fractions
        )
        planes.append((radial_plane, cuts))
        cut_sizes.append((cell_h, max(2, int(round(cell_h * half_x / max(half_y, 1e-9))))))
        covers.append(row_cover)
    return _build_layout(
        radial_sizes=((cell_h, radial_width),) * 3,
        cut_sizes=tuple(cut_sizes),
        planes=tuple(planes),
        cover_fractions=tuple(covers),
    )


def _build_layout(
    *,
    radial_sizes: tuple[tuple[int, int], ...],
    cut_sizes: tuple[tuple[int, int], ...],
    planes: tuple[tuple[MosaicSlicePlane, tuple[MosaicSlicePlane, ...]], ...] | None,
    cover_fractions: tuple[float | None, ...],
) -> SphericalMosaicLayout:
    sizes = [(radial_sizes[row], (cut_sizes[row],) * 3) for row in range(3)]
    row_heights = [max([radial[0], *(c[0] for c in cuts)]) for radial, cuts in sizes]
    col_widths = [
        max(radial[1] for radial, _ in sizes),
        *(max(cuts[i][1] for _, cuts in sizes) for i in range(3)),
    ]
    y_starts = np.concatenate([[0], np.cumsum(row_heights)]).astype(int)
    x_starts = np.concatenate([[0], np.cumsum(col_widths)]).astype(int)
    cells: list[MosaicCell] = []
    for row in range(3):
        radial_size, cut_row_sizes = sizes[row]
        fractions = _depth_fractions_for_row(row)
        for col in range(4):
            image_h, image_w = radial_size if col == 0 else cut_row_sizes[col - 1]
            plane = None
            if planes is not None:
                plane = planes[row][0] if col == 0 else planes[row][1][col - 1]
            cells.append(
                MosaicCell(
                    row=row,
                    col=col,
                    y0=int(y_starts[row]),
                    x0=int(x_starts[col]),
                    height=int(row_heights[row]),
                    width=int(col_widths[col]),
                    image_height=int(image_h),
                    image_width=int(image_w),
                    kind="radial_axis" if col == 0 else "depth_slice",
                    angle_deg=_RADIAL_AXIS_ANGLES_DEG[row] if col == 0 else None,
                    depth_fraction=None if col == 0 else fractions[col - 1],
                    cover_depth_fraction=cover_fractions[row],
                    plane=plane,
                )
            )
    return SphericalMosaicLayout(cells=tuple(cells))


def cartesian_spherical_mosaic(
    volumes: np.ndarray,
    geometry: SphericalGeometry,
    *,
    output_size: tuple[int, int] = (120, 120),
    interpolation: Interpolation = "linear",
    cover_depth_fraction: float = 1.0,
    layout: SphericalMosaicLayout | None = None,
    cell_height: int | None = None,
) -> SphericalMosaic:
    """Build a 12-panel mosaic from Cartesian slices through a spherical volume.

    Pass `layout` (from `cartesian_mosaic_layout`) to share the exact cell rectangles with annotation overlays.
    """
    arr = _volume_stack(volumes)
    if layout is None:
        layout = cartesian_mosaic_layout(
            geometry, output_size=output_size, cover_depth_fraction=cover_depth_fraction, cell_height=cell_height
        )
    slices: list[np.ndarray] = []
    for cell in layout.cells:
        size = (cell.image_height, cell.image_width)
        if cell.kind == "radial_axis":
            plane = cast(MosaicSlicePlane, cell.plane)
            slices.append(
                spherical_radial_axis_stack(
                    arr,
                    geometry,
                    angle_deg=cast(float, cell.angle_deg),
                    output_size=size,
                    interpolation=interpolation,
                    half_width_m=plane.x_max,
                    depth_start_m=plane.y_min,
                )
            )
        else:
            slices.append(
                spherical_depth_slice_stack(
                    arr,
                    geometry,
                    depth_fraction=cast(float, cell.depth_fraction),
                    output_size=size,
                    interpolation=interpolation,
                    cover_depth_fraction=cast(float, cell.cover_depth_fraction),
                )
            )
    return SphericalMosaic(frames=_mosaic_frames(layout, slices, fill_value=np.nan), layout=layout)


def slice_plane_basis_radial_axis(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle_rad = math.radians(float(angle_deg))
    # Display orientation: the scanner shows the volume rotated 180 degrees about the depth axis.
    basis_u = -np.asarray([math.cos(angle_rad), math.sin(angle_rad), 0.0], dtype=np.float64)
    basis_v = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    normal = np.cross(basis_u, basis_v)
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    return basis_u, basis_v, normal


def slice_plane_basis_normal_slice() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Display orientation: both lateral axes are flipped (180 degree rotation about the depth axis).
    basis_u = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    basis_v = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return basis_u, basis_v, normal


def spherical_depth_slice_stack(
    volumes: np.ndarray,
    geometry: SphericalGeometry,
    *,
    depth_fraction: float,
    output_size: tuple[int, int],
    interpolation: Interpolation = "linear",
    cover_depth_fraction: float = 1.0,
    lateral_scale: float = 1.0,
) -> np.ndarray:
    arr = _volume_stack(volumes)
    height, width = _shape2(output_size)
    depth_m = geometry.depth_start_m + float(np.clip(depth_fraction, 0.0, 1.0)) * _depth_span(geometry)
    cover_depth_m = geometry.depth_start_m + float(np.clip(cover_depth_fraction, 0.0, 1.0)) * _depth_span(geometry)
    half_x = cover_depth_m * math.tan(0.5 * geometry.azimuth_width_rad) * max(float(lateral_scale), 1e-6)
    half_y = cover_depth_m * math.tan(0.5 * geometry.elevation_width_rad)
    x = np.linspace(-half_x, half_x, width, dtype=np.float32)
    y = np.linspace(-half_y, half_y, height, dtype=np.float32)
    v_grid, u_grid = np.meshgrid(y, x, indexing="ij")
    basis_u, basis_v, _ = slice_plane_basis_normal_slice()
    xx = (u_grid * basis_u[0] + v_grid * basis_v[0]).astype(np.float32)
    yy = (u_grid * basis_u[1] + v_grid * basis_v[1]).astype(np.float32)
    zz = np.full_like(xx, depth_m)
    coords = _cartesian_to_index(xx, yy, zz, geometry, arr.shape[1:4])
    return _sample_stack(arr, coords, interpolation=interpolation)


def spherical_radial_axis_stack(
    volumes: np.ndarray,
    geometry: SphericalGeometry,
    *,
    angle_deg: float,
    output_size: tuple[int, int],
    interpolation: Interpolation = "linear",
    cover_depth_fraction: float = 1.0,
    half_width_m: float | None = None,
    depth_start_m: float | None = None,
) -> np.ndarray:
    arr = _volume_stack(volumes)
    height, width = _shape2(output_size)
    depth_start = _radial_axis_depth_start(geometry, depth_start_m)
    half_width = (
        float(half_width_m)
        if half_width_m is not None
        else _radial_axis_half_width_m(geometry, cover_depth_fraction, angle_deg)
    )
    depth = np.linspace(depth_start, geometry.depth_end_m, height, dtype=np.float32)
    lateral = np.linspace(-half_width, half_width, width, dtype=np.float32)
    zz, lateral_grid = np.meshgrid(depth, lateral, indexing="ij")
    basis_u, _, _ = slice_plane_basis_radial_axis(angle_deg)
    xx = (lateral_grid * basis_u[0]).astype(np.float32)
    yy = (lateral_grid * basis_u[1]).astype(np.float32)
    coords = _cartesian_to_index(xx, yy, zz, geometry, arr.shape[1:4])
    return _sample_stack(arr, coords, interpolation=interpolation)


def _volume_stack(volumes: np.ndarray) -> np.ndarray:
    arr = np.asarray(volumes)
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D brightness mode with shape [T,E,A,R], got {arr.shape}")
    return arr


def _native_radial_axis_slice(volumes: np.ndarray, *, az_fraction: float) -> np.ndarray:
    az_index = int(round(float(np.clip(az_fraction, 0.0, 1.0)) * max(0, volumes.shape[2] - 1)))
    return np.asarray(np.swapaxes(volumes[:, :, az_index, :], 1, 2))


def _native_radial_normal_slice(volumes: np.ndarray, *, depth_fraction: float) -> np.ndarray:
    radial_index = int(round(float(np.clip(depth_fraction, 0.0, 1.0)) * max(0, volumes.shape[3] - 1)))
    return np.asarray(volumes[:, :, :, radial_index])


def _sample_stack(volumes: np.ndarray, coords: np.ndarray, *, interpolation: Interpolation) -> np.ndarray:
    coverage = _coordinate_coverage(coords, volumes.shape[1:4])
    outside = coverage <= 0.0
    partial = (coverage < 1.0) & ~outside
    frames = []
    for frame in volumes:
        sampled = sample_volume(frame, coords, interpolation=interpolation).astype(np.float32, copy=False)
        if outside.any():
            sampled = sampled.copy()
            sampled[partial] *= coverage[partial]
            sampled[outside] = np.nan
        frames.append(sampled)
    return np.asarray(frames)


def _cartesian_to_index(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    geometry: SphericalGeometry,
    shape_ear: tuple[int, int, int],
) -> np.ndarray:
    elevation_count, azimuth_count, radial_count = shape_ear
    depth = np.sqrt(x * x + y * y + z * z)
    azimuth = np.arctan2(x, np.sqrt(y * y + z * z))
    elevation = np.arctan2(y, z)
    radial_index = (depth - geometry.depth_start_m) / _depth_span(geometry) * max(0, radial_count - 1)
    azimuth_index = (
        (azimuth + 0.5 * geometry.azimuth_width_rad)
        / max(geometry.azimuth_width_rad, 1e-12)
        * max(0, azimuth_count - 1)
    )
    elevation_index = (
        (elevation + 0.5 * geometry.elevation_width_rad)
        / max(geometry.elevation_width_rad, 1e-12)
        * max(0, elevation_count - 1)
    )
    return np.stack([elevation_index, azimuth_index, radial_index], axis=-1)


def _coordinate_margin(coords: np.ndarray, shape_ear: tuple[int, int, int]) -> np.ndarray:
    """Distance (in index units) of each sample from the nearest volume face; negative outside the volume."""
    margin = np.full(coords.shape[:-1], np.inf, dtype=np.float64)
    for axis, size in enumerate(shape_ear):
        c = coords[..., axis]
        margin = np.minimum(margin, np.minimum(c, (size - 1) - c))
    return margin


def _coordinate_coverage(coords: np.ndarray, shape_ear: tuple[int, int, int]) -> np.ndarray:
    """Anti-aliased sector mask in [0, 1]: 1 inside the volume, ramping to 0 over one output pixel outside it.

    The margin is divided by its gradient over the output grid, which turns index units into a distance in output
    pixels, so the ramp is one pixel wide however the slice is scaled.
    """
    margin = _coordinate_margin(coords, shape_ear)
    inside = margin >= 0.0
    if inside.all() or min(margin.shape) < 2:
        return inside.astype(np.float32)
    grad_rows, grad_cols = np.gradient(margin)
    distance_px = margin / np.maximum(np.hypot(grad_rows, grad_cols), 1e-9)
    coverage = np.where(inside, 1.0, np.clip(1.0 + distance_px, 0.0, 1.0))
    return np.nan_to_num(coverage, nan=0.0).astype(np.float32)


def _mosaic_frames(layout: SphericalMosaicLayout, slices: list[np.ndarray], *, fill_value: float) -> np.ndarray:
    if len(slices) != len(layout.cells):
        raise ValueError(f"Expected {len(layout.cells)} panels, got {len(slices)}")
    out = np.full((int(slices[0].shape[0]), layout.height, layout.width), fill_value, dtype=np.float32)
    for cell, panel in zip(layout.cells, slices):
        y, x = cell.image_y0, cell.image_x0
        out[:, y : y + cell.image_height, x : x + cell.image_width] = np.asarray(panel, dtype=np.float32)
    return out


def _resize_stack(frames: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(frames)
    height, width = _shape2(output_size)
    if arr.shape[1:3] == (height, width):
        return arr.astype(np.float32, copy=False)
    rows = np.linspace(0.0, max(0, arr.shape[1] - 1), height) if height > 1 else np.zeros(1)
    cols = np.linspace(0.0, max(0, arr.shape[2] - 1), width) if width > 1 else np.zeros(1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return sample_image_stack(arr.astype(np.float32, copy=False), rr, cc, interpolation="linear").astype(
        np.float32, copy=False
    )


def _depth_fractions_for_row(row: int) -> tuple[float, float, float]:
    fractions = np.linspace(4.0 / 15.0, 10.0 / 15.0, 9, dtype=np.float32)
    start = int(row) * 3
    selected = fractions[start : start + 3]
    return float(selected[0]), float(selected[1]), float(selected[2])


def _radial_axis_half_width_m(geometry: SphericalGeometry, cover_depth_fraction: float, angle_deg: float) -> float:
    cover_depth_m = geometry.depth_start_m + float(np.clip(cover_depth_fraction, 0.0, 1.0)) * _depth_span(geometry)
    az_half = cover_depth_m * math.tan(0.5 * geometry.azimuth_width_rad)
    el_half = cover_depth_m * math.tan(0.5 * geometry.elevation_width_rad)
    angle_rad = math.radians(float(angle_deg))
    return az_half * abs(math.cos(angle_rad)) + el_half * abs(math.sin(angle_rad))


def _radial_axis_depth_start(geometry: SphericalGeometry, depth_start_m: float | None) -> float:
    if depth_start_m is None:
        return float(geometry.depth_start_m)
    return float(np.clip(float(depth_start_m), float(geometry.depth_start_m), float(geometry.depth_end_m)))


def _cell_height(output_size: tuple[int, int], cell_height: int | None) -> int:
    return max(2, int(cell_height)) if cell_height is not None else _shape2(output_size)[0]


def _shape2(shape: tuple[int, int]) -> tuple[int, int]:
    return max(2, int(shape[0])), max(2, int(shape[1]))


def _depth_span(geometry: SphericalGeometry) -> float:
    return max(1e-9, float(geometry.depth_end_m) - float(geometry.depth_start_m))


def _required_float(metadata: Mapping[str, object], key: str) -> float:
    value = _optional_float(metadata, key, default=float("nan"))
    if not np.isfinite(value):
        raise ValueError(f"Missing or invalid 3D metadata value `{key}`")
    return value


def _optional_float(metadata: Mapping[str, object], key: str, *, default: float) -> float:
    value = metadata.get(key, default)
    if isinstance(value, list | tuple):
        value = value[0] if value else default
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default
