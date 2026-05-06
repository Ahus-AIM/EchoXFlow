#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from echoxflow import (
    CroissantCatalog,
    LoadedArray,
    RecordingRecord,
    load_croissant,
    named_listed_colormap,
    open_recording,
)
from echoxflow.plotting.clinical import clinical_loaded_arrays
from echoxflow.plotting.colorbar import colorbar_spec_for_modality, draw_top_right_colorbar
from echoxflow.plotting.style import PlotStyle
from echoxflow.scan.beat_stitching import prepare_3d_brightness_for_display
from echoxflow.scan.geometry import CartesianGrid, SectorGeometry
from echoxflow.scan.matplotlib import SectorDepthRuler, draw_sector_depth_ruler, set_cartesian_extent
from echoxflow.scan.spherical import clinical_spherical_mosaic, spherical_geometry_from_metadata

DEFAULT_OUTPUT = Path("outputs/images")
DEFAULT_PNG_WIDTH = 450
DEFAULT_PNG_HEIGHT = 225
DEFAULT_MAX_STRIP_DURATION_S = 3.0
DEFAULT_3D_PANEL_HEIGHT = 150
DEFAULT_3D_PANEL_WIDTH = 190
DEFAULT_3D_RADIAL_PANEL_SIZE = 150
DEFAULT_3D_RADIAL_DEPTH_START_M = 0.01
DEFAULT_3D_COVER_DEPTH_FRACTION = 0.72
DEFAULT_3D_DEPTH_SLICE_COVER_DEPTH_FRACTION = 0.576
DEFAULT_3D_DEPTH_SLICE_LATERAL_SCALE = DEFAULT_3D_PANEL_WIDTH / DEFAULT_3D_PANEL_HEIGHT
DEFAULT_TARGET_INDICES = {
    "2d_bmode": 40,
    "2d_tissue_doppler": 10,
    "2d_color_doppler": 14,
    "3d_bmode": 1,
    "continuous_wave_doppler": 6,
}
BMODE_PATHS = ("data/2d_brightness_mode", "data/2d_brightness_mode_0")
SPECTRAL_DOPPLER_STRIP_PATHS = {
    "data/1d_continuous_wave_doppler",
    "data/1d_pulsed_wave_doppler",
}


@dataclass(frozen=True)
class StripTarget:
    name: str
    data_path: str
    description: str
    associated_content_types: tuple[str, ...] = ()
    associated_array_paths: tuple[str, ...] = ()
    excluded_array_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateRecording:
    record: RecordingRecord | None
    path: Path | None
    root: Path | None
    array_paths: tuple[str, ...]
    content_types: tuple[str, ...]

    @property
    def source_label(self) -> str:
        if self.record is None:
            return "" if self.path is None else str(self.path)
        return f"{self.record.exam_id}/{self.record.recording_id}"


@dataclass(frozen=True)
class ExtractionResult:
    target: str
    description: str
    data_path: str
    source: str
    array_path: str
    png_path: str
    shape: tuple[int, ...]
    dtype: str
    sample_rate_hz: float | None


STRIP_TARGETS = (
    StripTarget(
        name="2d_bmode",
        data_path="data/2d_brightness_mode",
        description="Scan-converted 2D B-mode frame",
    ),
    StripTarget(
        name="2d_tissue_doppler",
        data_path="data/tissue_doppler",
        description="Scan-converted 2D tissue Doppler frame",
        associated_array_paths=("data/2d_brightness_mode",),
    ),
    StripTarget(
        name="2d_color_doppler",
        data_path="data/2d_color_doppler_velocity",
        description="Scan-converted 2D color Doppler frame",
        associated_array_paths=("data/2d_brightness_mode", "data/2d_color_doppler_power"),
    ),
    StripTarget(
        name="3d_bmode",
        data_path="data/3d_brightness_mode",
        description="3D B-mode 3x4 scan-converted mosaic",
    ),
    StripTarget(
        name="continuous_wave_doppler",
        data_path="data/1d_continuous_wave_doppler",
        description="Continuous wave Doppler strip",
    ),
    StripTarget(
        name="tissue_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        description="Spectral Doppler strip from a tissue Doppler recording",
        associated_array_paths=("data/tissue_doppler",),
    ),
    StripTarget(
        name="pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        description="Pulsed wave Doppler strip from a color Doppler recording",
        associated_array_paths=("data/2d_color_doppler_velocity", "data/2d_color_doppler_power"),
        excluded_array_paths=("data/tissue_doppler",),
    ),
    StripTarget(
        name="m_mode",
        data_path="data/1d_motion_mode",
        description="M-mode strip",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract example 2D B-mode, 2D tissue Doppler, 2D color Doppler, continuous wave Doppler, "
            "tissue-Doppler-associated spectral Doppler, color-Doppler-associated continuous wave Doppler, "
            "and M-mode images from an EchoXFlow dataset."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Dataset root, croissant.json, or a single .zarr recording. "
            "Defaults to the configured EchoXFlow data root."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root for relative zarr paths in an explicit Croissant file.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output directory, default {DEFAULT_OUTPUT}."
    )
    parser.add_argument("--start", type=int, default=None, help="Optional temporal start index.")
    parser.add_argument("--stop", type=int, default=None, help="Optional temporal stop index.")
    parser.add_argument("--png-width", type=int, default=DEFAULT_PNG_WIDTH, help="Output PNG width in pixels.")
    parser.add_argument("--png-height", type=int, default=DEFAULT_PNG_HEIGHT, help="Output PNG height in pixels.")
    parser.add_argument(
        "--max-strip-duration-s",
        type=float,
        default=DEFAULT_MAX_STRIP_DURATION_S,
        help="Crop 1D strip modalities to at most this many seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--target-index",
        action="append",
        default=None,
        metavar="TARGET=INDEX",
        help="Select the INDEX-th matching candidate for one target, for example continuous_wave=1.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Return success even if one target is not found.")
    args = parser.parse_args()

    if args.start is not None and args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.stop is not None and args.start is not None and args.stop <= args.start:
        raise ValueError("--stop must be greater than --start")

    candidates = source_candidates(args.source, root=args.root)
    results, missing = extract_strips(
        candidates,
        output_dir=args.output,
        start=args.start,
        stop=args.stop,
        png_size=(args.png_width, args.png_height),
        max_strip_duration_s=args.max_strip_duration_s,
        target_indices=_parse_target_indices(tuple(args.target_index or ())),
    )
    args.output.mkdir(parents=True, exist_ok=True)

    for result in results:
        print(f"{result.target}: {result.png_path} ({result.shape})")
    if missing:
        print(f"Missing targets: {', '.join(missing)}")
        return 0 if args.allow_missing else 1
    return 0


def source_candidates(source: Path | None, *, root: Path | None = None) -> tuple[CandidateRecording, ...]:
    if source is not None:
        source = source.expanduser()
    if source is not None and _looks_like_zarr(source):
        return (_direct_candidate(source),)
    catalog_root: Path | None

    if source is not None and source.is_dir():
        croissant_path = source / "croissant.json"
        catalog = load_croissant(croissant_path)
        catalog_root = root.expanduser() if root is not None else source
        return _catalog_candidates(catalog, root=catalog_root)

    catalog = load_croissant(source, root=root)
    if root is not None:
        catalog_root = root.expanduser()
    elif source is not None:
        catalog_root = source.parent
    else:
        catalog_root = None
    return _catalog_candidates(catalog, root=catalog_root)


def extract_strips(
    candidates: tuple[CandidateRecording, ...],
    *,
    output_dir: Path,
    start: int | None = None,
    stop: int | None = None,
    png_size: tuple[int, int] = (DEFAULT_PNG_WIDTH, DEFAULT_PNG_HEIGHT),
    max_strip_duration_s: float | None = DEFAULT_MAX_STRIP_DURATION_S,
    target_indices: dict[str, int] | None = None,
) -> tuple[list[ExtractionResult], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExtractionResult] = []
    missing: list[str] = []
    target_indices = {} if target_indices is None else dict(target_indices)
    for target in STRIP_TARGETS:
        matching_candidates = [item for item in candidates if _candidate_matches(item, target)]
        if not matching_candidates:
            missing.append(target.name)
            continue
        matching_candidates = _rank_matching_candidates(matching_candidates, target)
        requested_index = target.name in target_indices
        target_index = int(
            target_indices[target.name] if requested_index else DEFAULT_TARGET_INDICES.get(target.name, 0)
        )
        if target_index >= len(matching_candidates):
            if requested_index:
                raise ValueError(f"{target.name!r} has only {len(matching_candidates)} matching candidate(s)")
            target_index = 0
        candidate = matching_candidates[target_index]
        results.append(
            _extract_target(
                candidate,
                target,
                output_dir=output_dir,
                start=start,
                stop=stop,
                png_size=png_size,
                max_strip_duration_s=max_strip_duration_s,
            )
        )
    return results, missing


def _catalog_candidates(catalog: CroissantCatalog, *, root: Path | None) -> tuple[CandidateRecording, ...]:
    return tuple(
        CandidateRecording(
            record=record,
            path=None,
            root=root,
            array_paths=record.array_paths,
            content_types=record.content_types,
        )
        for record in catalog.recordings
    )


def _direct_candidate(path: Path) -> CandidateRecording:
    store = open_recording(path)
    array_paths = tuple(str(item).strip("/") for item in store.array_paths)
    return CandidateRecording(
        record=None,
        path=path,
        root=None,
        array_paths=array_paths,
        content_types=tuple(path.removeprefix("data/") for path in array_paths if path.startswith("data/")),
    )


def _extract_target(
    candidate: CandidateRecording,
    target: StripTarget,
    *,
    output_dir: Path,
    start: int | None,
    stop: int | None,
    png_size: tuple[int, int],
    max_strip_duration_s: float | None,
) -> ExtractionResult:
    store = (
        open_recording(candidate.record, root=candidate.root)
        if candidate.record is not None
        else open_recording(_path(candidate))
    )
    loaded = _load_target_modality(store, candidate, target, start=start, stop=stop)
    data = np.asarray(loaded.data)
    timestamps = None if loaded.timestamps is None else np.asarray(loaded.timestamps)
    value_range = None if loaded.stream is None else loaded.stream.metadata.value_range
    data, timestamps = _crop_strip_duration(
        data,
        timestamps,
        data_path=loaded.data_path,
        max_duration_s=max_strip_duration_s,
    )
    output_stem = output_dir / target.name
    png_path = output_stem.with_suffix(".png")
    _save_strip_png(data, png_path, value_range=value_range, data_path=loaded.data_path, size=png_size, loaded=loaded)

    return ExtractionResult(
        target=target.name,
        description=target.description,
        data_path=target.data_path,
        source=candidate.source_label,
        array_path=loaded.data_path,
        png_path=str(png_path),
        shape=tuple(int(size) for size in data.shape),
        dtype=str(data.dtype),
        sample_rate_hz=loaded.sample_rate_hz,
    )


def _load_target_modality(
    store: object,
    candidate: CandidateRecording,
    target: StripTarget,
    *,
    start: int | None,
    stop: int | None,
) -> LoadedArray:
    if target.name == "2d_bmode":
        bmode_path = _first_available_path(candidate, BMODE_PATHS)
        bmode = _load_modality(store, bmode_path, start=start, stop=stop)
        return _scan_converted(clinical_loaded_arrays((bmode,)), bmode_path)
    if target.name == "2d_tissue_doppler":
        bmode_path = _first_available_path(candidate, BMODE_PATHS)
        bmode = _load_modality(store, bmode_path, start=start, stop=stop)
        tissue = _load_modality(store, "data/tissue_doppler", start=start, stop=stop)
        return _scan_converted(clinical_loaded_arrays((bmode, tissue)), "data/clinical_tissue_doppler")
    if target.name == "2d_color_doppler":
        bmode_path = _first_available_path(candidate, BMODE_PATHS)
        bmode = _load_modality(store, bmode_path, start=start, stop=stop)
        velocity = _load_modality(store, "data/2d_color_doppler_velocity", start=start, stop=stop)
        power = _load_modality(store, "data/2d_color_doppler_power", start=start, stop=stop)
        converted = _scan_converted(
            clinical_loaded_arrays((bmode, velocity, power)),
            "data/clinical_color_doppler",
        )
        ruler_geometry = None if bmode.stream is None else bmode.stream.metadata.geometry
        return replace(converted, attrs={**dict(converted.attrs), "clinical_ruler_geometry": ruler_geometry})
    if target.name == "3d_bmode":
        loaded = _load_modality(store, "data/3d_brightness_mode", start=start, stop=stop)
        raw = None if loaded.stream is None else loaded.stream.metadata.raw
        prepared = prepare_3d_brightness_for_display(np.asarray(loaded.data), loaded.timestamps, raw)
        geometry = spherical_geometry_from_metadata(raw)
        mosaic = clinical_spherical_mosaic(
            prepared.volumes,
            geometry,
            output_size=(DEFAULT_3D_PANEL_HEIGHT, DEFAULT_3D_PANEL_WIDTH),
            cover_depth_fraction=DEFAULT_3D_COVER_DEPTH_FRACTION,
            depth_slice_cover_depth_fraction=DEFAULT_3D_DEPTH_SLICE_COVER_DEPTH_FRACTION,
            radial_axis_output_size=(DEFAULT_3D_RADIAL_PANEL_SIZE, DEFAULT_3D_RADIAL_PANEL_SIZE),
            radial_axis_depth_start_m=DEFAULT_3D_RADIAL_DEPTH_START_M,
            depth_slice_lateral_scale=DEFAULT_3D_DEPTH_SLICE_LATERAL_SCALE,
        ).frames
        return replace(
            loaded,
            name="clinical_3d_brightness_mode",
            data=mosaic,
            timestamps=prepared.timestamps,
            attrs={**dict(loaded.attrs), "3d_mosaic_rows": 3, "3d_mosaic_cols": 4},
        )
    return _load_modality(store, target.data_path, start=start, stop=stop)


def _load_modality(store: object, path: str, *, start: int | None, stop: int | None) -> LoadedArray:
    if start is not None or stop is not None:
        return store.load_modality_slice(path, start, stop)  # type: ignore[attr-defined]
    return store.load_modality(path)  # type: ignore[attr-defined]


def _scan_converted(loaded_arrays: tuple[LoadedArray, ...], data_path: str) -> LoadedArray:
    for loaded in loaded_arrays:
        if loaded.data_path == data_path:
            if "clinical_grid" not in loaded.attrs:
                raise ValueError(f"{data_path} is missing sector geometry metadata required for scan conversion")
            return loaded
    raise ValueError(f"No scan-converted array was produced for {data_path}")


def _candidate_matches(candidate: CandidateRecording, target: StripTarget) -> bool:
    array_paths = {_normalize_path(path) for path in candidate.array_paths}
    if _normalize_path(target.data_path) not in array_paths:
        return False
    if any(_normalize_path(path) in array_paths for path in target.excluded_array_paths):
        return False
    if not target.associated_content_types and not target.associated_array_paths:
        return True
    content_types = set(candidate.content_types)
    if target.associated_array_paths and not all(
        _normalize_path(path) in array_paths for path in target.associated_array_paths
    ):
        return False
    if not target.associated_content_types:
        return True
    return bool(content_types.intersection(target.associated_content_types)) or bool(
        array_paths.intersection(_normalize_path(path) for path in target.associated_array_paths)
    )


def _rank_matching_candidates(
    candidates: list[CandidateRecording],
    target: StripTarget,
) -> list[CandidateRecording]:
    if target.name != "2d_color_doppler":
        return candidates
    return sorted(candidates, key=_color_doppler_candidate_score, reverse=True)


def _color_doppler_candidate_score(candidate: CandidateRecording) -> tuple[int, int]:
    if candidate.record is None:
        return (0, 0)
    color_frames = candidate.record.frame_count("2d_color_doppler") or 0
    bmode_frames = candidate.record.frame_count("2d_brightness_mode") or 0
    return (int(color_frames), int(bmode_frames))


def _crop_strip_duration(
    data: np.ndarray,
    timestamps: np.ndarray | None,
    *,
    data_path: str,
    max_duration_s: float | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not _is_temporal_strip_path(data_path) or timestamps is None or not max_duration_s or max_duration_s <= 0.0:
        return data, timestamps
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if times.size < 2 or not np.all(np.isfinite(times)):
        return data, timestamps
    if float(times[-1] - times[0]) <= float(max_duration_s):
        return data, timestamps
    temporal_axis = _temporal_axis_from_timestamps(data, times)
    if temporal_axis is None:
        return data, timestamps
    stop_time = float(times[0]) + float(max_duration_s)
    stop_index = int(np.searchsorted(times, stop_time, side="left"))
    stop_index = max(1, min(stop_index, times.size))
    slicer = [slice(None)] * data.ndim
    slicer[temporal_axis] = slice(0, stop_index)
    return np.asarray(data[tuple(slicer)]), times[:stop_index]


def _temporal_axis_from_timestamps(data: np.ndarray, timestamps: np.ndarray) -> int | None:
    matches = [axis for axis, size in enumerate(data.shape) if int(size) == int(timestamps.size)]
    if not matches:
        return None
    return 0 if 0 in matches else matches[0]


def _is_temporal_strip_path(data_path: str) -> bool:
    normalized = _normalize_path(data_path)
    return normalized in {
        "data/1d_pulsed_wave_doppler",
        "data/1d_continuous_wave_doppler",
        "data/1d_motion_mode",
    }


def _is_spectral_doppler_strip_path(data_path: str) -> bool:
    return _normalize_path(data_path) in SPECTRAL_DOPPLER_STRIP_PATHS


def _tinted_spectral_doppler_strip(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    tint = np.asarray([0x22, 0x22, 0x22], dtype=np.float32)
    return np.asarray(np.rint(np.clip((values + tint) * (7.0 / 8.0), 0.0, 255.0)), dtype=np.uint8)


def _save_strip_png(
    data: np.ndarray,
    path: Path,
    *,
    value_range: tuple[float, float] | None,
    data_path: str,
    size: tuple[int, int],
    loaded: LoadedArray | None = None,
) -> None:
    image = _strip_image(data)
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        if not _save_scan_converted_png(image, path, loaded=loaded, size=size):
            _save_rgb_letterboxed_png(image, path, size=size)
        return
    scaled = _normalize_to_uint8(image, value_range=value_range)
    rgb = _apply_colormap(scaled, data_path=data_path)
    if _is_spectral_doppler_strip_path(data_path):
        rgb = _tinted_spectral_doppler_strip(rgb)
    if _is_3d_mosaic(loaded):
        _save_3d_mosaic_png(image, path, rgb=rgb)
        return
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    Image.fromarray(rgb, mode="RGB").resize((int(size[0]), int(size[1])), resample=resampling).save(path)


def _save_scan_converted_png(
    image: np.ndarray,
    path: Path,
    *,
    loaded: LoadedArray | None,
    size: tuple[int, int],
) -> bool:
    if loaded is None:
        return False
    grid = loaded.attrs.get("clinical_grid")
    geometry = _clinical_geometry(loaded)
    if not isinstance(grid, CartesianGrid) or not isinstance(geometry, SectorGeometry):
        return False

    style = PlotStyle.from_config()
    background = _hex_to_rgb255(style.panel_facecolor)
    rgb = _rgb_from_image(image, background=background)
    width_px, height_px = int(size[0]), int(size[1])
    dpi = 100
    figure = Figure(
        figsize=(width_px / dpi, height_px / dpi),
        dpi=dpi,
        facecolor=style.panel_facecolor,
    )
    FigureCanvasAgg(figure)
    ax = figure.add_axes(_scan_converted_axes_bbox(loaded), facecolor=style.panel_facecolor)
    ax.set_axis_off()
    ax.imshow(
        rgb,
        extent=(grid.x_range_m[0], grid.x_range_m[1], grid.y_range_m[1], grid.y_range_m[0]),
        interpolation="nearest",
    )
    set_cartesian_extent(ax, grid)
    draw_sector_depth_ruler(
        ax,
        geometry,
        SectorDepthRuler(
            side="left",
            color=style.ecg_trace_color,
            linewidth=style.clinical_depth_ruler_linewidth,
            border_linewidth=style.clinical_depth_ruler_border_linewidth,
            label_fontsize=style.clinical_depth_ruler_label_fontsize,
            show_border=style.clinical_depth_ruler_show_border,
            omitted_tick_depths_cm=_scan_converted_omitted_ruler_ticks_cm(loaded),
        ),
    )
    _draw_color_doppler_region_outline(ax, loaded=loaded)
    colorbar_ax = figure.add_axes((0.0, 0.0, 1.0, 1.0), facecolor="none")
    colorbar_ax.set_axis_off()
    _draw_scan_converted_colorbar(colorbar_ax, loaded=loaded, style=style)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor())
    return True


def _scan_converted_axes_bbox(loaded: LoadedArray) -> tuple[float, float, float, float]:
    if _is_scan_converted_doppler(loaded):
        return (-0.045, 0.0, 1.0, 1.0)
    return (0.0, 0.0, 1.0, 1.0)


def _draw_color_doppler_region_outline(ax: object, *, loaded: LoadedArray) -> None:
    if "clinical_color_doppler" not in loaded.data_path:
        return
    geometry = loaded.attrs.get("clinical_color_doppler_geometry")
    if not isinstance(geometry, SectorGeometry):
        return

    angles = np.linspace(float(geometry.angle_start_rad), float(geometry.angle_end_rad), 96)
    depths = (float(geometry.depth_start_m), float(geometry.depth_end_m))
    style = {
        "color": "#FFFFFF",
        "linewidth": 1.0,
        "alpha": 0.9,
        "solid_capstyle": "round",
        "zorder": 9.0,
    }
    for depth in depths:
        ax.plot(depth * np.sin(angles), depth * np.cos(angles), **style)  # type: ignore[attr-defined]
    for angle in (float(geometry.angle_start_rad), float(geometry.angle_end_rad)):
        radii = np.asarray(depths, dtype=np.float64)
        ax.plot(radii * np.sin(angle), radii * np.cos(angle), **style)  # type: ignore[attr-defined]


def _draw_scan_converted_colorbar(ax: object, *, loaded: LoadedArray, style: PlotStyle) -> None:
    data_path = str(loaded.attrs.get("clinical_colorbar_data_path") or loaded.data_path)
    value_range = _colorbar_value_range(loaded)
    spec = colorbar_spec_for_modality(data_path, value_range=value_range)
    cmap = named_listed_colormap(data_path)
    if spec is None or cmap is None:
        return
    spec = replace(
        spec,
        label=_scan_converted_colorbar_label(data_path),
        bbox_axes=_scan_converted_colorbar_bbox(data_path, loaded=loaded),
    )
    draw_top_right_colorbar(ax, cmap, spec, style=replace(style, colorbar_fontsize=12.0))  # type: ignore[arg-type]


def _scan_converted_colorbar_bbox(data_path: str, *, loaded: LoadedArray) -> tuple[float, float, float, float]:
    return (0.930, 0.49, 0.032, 0.38)


def _scan_converted_colorbar_label(data_path: str) -> str:
    if "tissue_doppler" in data_path:
        return "cm/s"
    if "color_doppler_velocity" in data_path:
        return "m/s"
    return ""


def _scan_converted_omitted_ruler_ticks_cm(loaded: LoadedArray) -> tuple[float, ...]:
    if "clinical_color_doppler" in loaded.data_path:
        return (5.0,)
    return ()


def _is_scan_converted_doppler(loaded: LoadedArray) -> bool:
    return "clinical_color_doppler" in loaded.data_path or "clinical_tissue_doppler" in loaded.data_path


def _colorbar_value_range(loaded: LoadedArray) -> tuple[float, float] | None:
    raw = loaded.attrs.get("clinical_colorbar_value_range")
    if isinstance(raw, tuple) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    if loaded.stream is not None:
        return loaded.stream.metadata.value_range
    return None


def _clinical_geometry(loaded: LoadedArray) -> SectorGeometry | None:
    ruler_geometry = loaded.attrs.get("clinical_ruler_geometry")
    if isinstance(ruler_geometry, SectorGeometry):
        return ruler_geometry
    if loaded.stream is not None and isinstance(loaded.stream.metadata.geometry, SectorGeometry):
        return loaded.stream.metadata.geometry
    geometry = loaded.attrs.get("clinical_color_doppler_geometry")
    return geometry if isinstance(geometry, SectorGeometry) else None


def _save_rgb_letterboxed_png(image: np.ndarray, path: Path, *, size: tuple[int, int]) -> None:
    background = _default_background_rgb()
    rgb = _rgb_from_image(image, background=background)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    target_size = (int(size[0]), int(size[1]))
    source = Image.fromarray(rgb, mode="RGB")
    source.thumbnail(target_size, resample=resampling)
    canvas = Image.new("RGB", target_size, tuple(int(value) for value in background))
    offset = ((target_size[0] - source.width) // 2, (target_size[1] - source.height) // 2)
    canvas.paste(source, offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_3d_mosaic_png(image: np.ndarray, path: Path, *, rgb: np.ndarray) -> None:
    out = np.asarray(rgb, dtype=np.uint8).copy()
    finite = np.isfinite(np.asarray(image))
    out[~finite] = _default_background_rgb()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGB").save(path)


def _is_3d_mosaic(loaded: LoadedArray | None) -> bool:
    if loaded is None:
        return False
    return int(loaded.attrs.get("3d_mosaic_rows", 0) or 0) == 3 and int(loaded.attrs.get("3d_mosaic_cols", 0) or 0) == 4


def _strip_image(data: np.ndarray) -> np.ndarray:
    values = np.squeeze(np.asarray(data))
    if values.ndim == 0:
        return values.reshape(1, 1)
    if values.ndim == 1:
        return values.reshape(1, -1)
    if values.ndim == 2:
        return values.T
    if values.ndim == 3 and values.shape[-1] in {3, 4}:
        return values
    frame_index = int(values.shape[0] // 2)
    return np.asarray(values[frame_index])


def _apply_colormap(data: np.ndarray, *, data_path: str) -> np.ndarray:
    cmap = named_listed_colormap(data_path, size=256)
    if cmap is None:
        values = np.asarray(data, dtype=np.uint8)
        return np.stack([values, values, values], axis=-1)
    normalized = np.asarray(data, dtype=np.float32) / 255.0
    rgba = cmap(np.clip(normalized, 0.0, 1.0))
    return np.asarray(np.rint(rgba[..., :3] * 255.0), dtype=np.uint8)


def _rgb_from_image(image: np.ndarray, *, background: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    rgb = arr[..., :3]
    if np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    if arr.shape[-1] == 4:
        alpha = arr[..., 3]
        if np.nanmax(alpha) > 1.0:
            alpha = alpha / 255.0
        alpha = np.clip(alpha[..., None], 0.0, 1.0)
        bg = np.asarray(background, dtype=np.float32) / 255.0
        rgb = rgb * alpha + bg * (1.0 - alpha)
    return np.asarray(np.rint(np.clip(np.nan_to_num(rgb), 0.0, 1.0) * 255.0), dtype=np.uint8)


def _default_background_rgb() -> np.ndarray:
    return _hex_to_rgb255(PlotStyle.from_config().panel_facecolor)


def _hex_to_rgb255(value: str) -> np.ndarray:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {value!r}")
    return np.asarray([int(text[index : index + 2], 16) for index in (0, 2, 4)], dtype=np.uint8)


def _normalize_to_uint8(data: np.ndarray, *, value_range: tuple[float, float] | None) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        return np.zeros(values.shape, dtype=np.uint8)
    if value_range is not None and np.isfinite(value_range).all() and value_range[1] > value_range[0]:
        low, high = float(value_range[0]), float(value_range[1])
    else:
        low, high = tuple(float(v) for v in np.percentile(values[finite], [1.0, 99.0]))
        if high <= low:
            low, high = float(np.min(values[finite])), float(np.max(values[finite]))
        if high <= low:
            high = low + 1.0
    normalized = (np.clip(values, low, high) - low) / (high - low)
    normalized = np.where(finite, normalized, 0.0)
    return np.asarray(np.rint(normalized * 255.0), dtype=np.uint8)


def _looks_like_zarr(path: Path) -> bool:
    return path.name.endswith(".zarr")


def _path(candidate: CandidateRecording) -> Path:
    if candidate.path is None:
        raise ValueError("Candidate has no direct zarr path")
    return candidate.path


def _normalize_path(path: str) -> str:
    text = str(path).strip("/")
    return text if text.startswith("data/") else f"data/{text}"


def _first_available_path(candidate: CandidateRecording, paths: tuple[str, ...]) -> str:
    array_paths = {_normalize_path(path) for path in candidate.array_paths}
    for path in paths:
        normalized = _normalize_path(path)
        if normalized in array_paths:
            return normalized
    raise ValueError(f"{candidate.source_label} has none of {', '.join(paths)}")


def _parse_target_indices(values: tuple[str, ...]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--target-index must use TARGET=INDEX, got {value!r}")
        name, raw_index = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--target-index target name is empty: {value!r}")
        index = int(raw_index)
        if index < 0:
            raise ValueError(f"--target-index must be non-negative, got {value!r}")
        parsed[name] = index
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
