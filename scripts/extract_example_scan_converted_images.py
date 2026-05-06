#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from echoxflow import CroissantCatalog, LoadedArray, RecordingRecord, load_croissant, open_recording
from echoxflow.plotting.clinical import clinical_loaded_arrays
from echoxflow.scan import (
    ImageLayer,
    clinical_spherical_mosaic,
    layer_to_rgba,
    prepare_3d_brightness_for_display,
    spherical_geometry_from_metadata,
)
from echoxflow.streams import default_value_range_for_path

DEFAULT_OUTPUT = Path("outputs/scan_converted_images")
DEFAULT_3D_PANEL_SIZE = 120

BMODE_PATHS = ("data/2d_brightness_mode", "data/2d_brightness_mode_0")


@dataclass(frozen=True)
class ScanConvertedTarget:
    name: str
    description: str
    required_path_groups: tuple[tuple[str, ...], ...]


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
    source: str
    png_path: str
    source_array_paths: tuple[str, ...]
    converted_shape: tuple[int, ...]
    frame_index: int


SCAN_CONVERTED_TARGETS = (
    ScanConvertedTarget(
        name="2d_bmode",
        description="2D B-mode scan-converted image",
        required_path_groups=(BMODE_PATHS,),
    ),
    ScanConvertedTarget(
        name="color_doppler",
        description="Color Doppler scan-converted B-mode composite",
        required_path_groups=(
            BMODE_PATHS,
            ("data/2d_color_doppler_velocity",),
            ("data/2d_color_doppler_power",),
        ),
    ),
    ScanConvertedTarget(
        name="tissue_doppler",
        description="Tissue Doppler scan-converted B-mode composite",
        required_path_groups=(BMODE_PATHS, ("data/tissue_doppler",)),
    ),
    ScanConvertedTarget(
        name="3d_bmode",
        description="3D B-mode scan-converted clinical mosaic",
        required_path_groups=(("data/3d_brightness_mode",),),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract scan-converted example PNGs for 2D B-mode, Color Doppler, Tissue Doppler, "
            "and 3D B-mode from an EchoXFlow dataset."
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
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame to save after scan conversion. Defaults to the middle converted frame.",
    )
    parser.add_argument(
        "--3d-panel-size",
        dest="panel_size_3d",
        type=int,
        default=DEFAULT_3D_PANEL_SIZE,
        help="Height and width in pixels for each 3D mosaic depth-slice panel before aspect-preserving layout.",
    )
    parser.add_argument(
        "--target-index",
        action="append",
        default=None,
        metavar="TARGET=INDEX",
        help="Select the INDEX-th matching candidate for one target, for example color_doppler=1.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Return success even if one target is not found.")
    args = parser.parse_args()

    if args.start is not None and args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.stop is not None and args.start is not None and args.stop <= args.start:
        raise ValueError("--stop must be greater than --start")
    if args.frame_index is not None and args.frame_index < 0:
        raise ValueError("--frame-index must be non-negative")
    if args.panel_size_3d <= 1:
        raise ValueError("--3d-panel-size must be greater than 1")

    candidates = source_candidates(args.source, root=args.root)
    results, missing = extract_scan_converted_images(
        candidates,
        output_dir=args.output,
        start=args.start,
        stop=args.stop,
        frame_index=args.frame_index,
        panel_size_3d=args.panel_size_3d,
        target_indices=_parse_target_indices(tuple(args.target_index or ())),
    )

    for result in results:
        print(f"{result.target}: {result.png_path} ({result.converted_shape}, frame={result.frame_index})")
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


def extract_scan_converted_images(
    candidates: tuple[CandidateRecording, ...],
    *,
    output_dir: Path,
    start: int | None = None,
    stop: int | None = None,
    frame_index: int | None = None,
    panel_size_3d: int = DEFAULT_3D_PANEL_SIZE,
    target_indices: dict[str, int] | None = None,
) -> tuple[list[ExtractionResult], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExtractionResult] = []
    missing: list[str] = []
    target_indices = {} if target_indices is None else dict(target_indices)
    for target in SCAN_CONVERTED_TARGETS:
        matching_candidates = [item for item in candidates if _candidate_matches(item, target)]
        if not matching_candidates:
            missing.append(target.name)
            continue
        target_index = int(target_indices.get(target.name, 0))
        if target_index >= len(matching_candidates):
            raise ValueError(f"{target.name!r} has only {len(matching_candidates)} matching candidate(s)")
        results.append(
            _extract_target(
                matching_candidates[target_index],
                target,
                output_dir=output_dir,
                start=start,
                stop=stop,
                frame_index=frame_index,
                panel_size_3d=panel_size_3d,
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
    target: ScanConvertedTarget,
    *,
    output_dir: Path,
    start: int | None,
    stop: int | None,
    frame_index: int | None,
    panel_size_3d: int,
) -> ExtractionResult:
    store = (
        open_recording(candidate.record, root=candidate.root)
        if candidate.record is not None
        else open_recording(_path(candidate))
    )
    converted, source_paths = _converted_target_data(
        store,
        candidate,
        target,
        start=start,
        stop=stop,
        panel_size_3d=panel_size_3d,
    )
    frame, resolved_frame_index = _select_frame(converted.data, frame_index)
    png_path = (output_dir / target.name).with_suffix(".png")
    _save_png(frame, png_path, data_path=converted.data_path, value_range=_value_range(converted, frame))
    return ExtractionResult(
        target=target.name,
        description=target.description,
        source=candidate.source_label,
        png_path=str(png_path),
        source_array_paths=source_paths,
        converted_shape=tuple(int(size) for size in np.asarray(converted.data).shape),
        frame_index=resolved_frame_index,
    )


def _converted_target_data(
    store: Any,
    candidate: CandidateRecording,
    target: ScanConvertedTarget,
    *,
    start: int | None,
    stop: int | None,
    panel_size_3d: int,
) -> tuple[LoadedArray, tuple[str, ...]]:
    if target.name == "2d_bmode":
        bmode_path = _first_available_path(candidate, BMODE_PATHS)
        bmode = _load_modality(store, bmode_path, start=start, stop=stop)
        return _scan_converted(clinical_loaded_arrays((bmode,)), bmode_path), (bmode_path,)

    if target.name == "color_doppler":
        bmode_path = _first_available_path(candidate, BMODE_PATHS)
        paths = (bmode_path, "data/2d_color_doppler_velocity", "data/2d_color_doppler_power")
        bmode, velocity, power = tuple(_load_modality(store, path, start=start, stop=stop) for path in paths)
        converted = clinical_loaded_arrays((bmode, velocity, power))
        return _scan_converted(converted, "data/clinical_color_doppler"), paths

    if target.name == "tissue_doppler":
        bmode_path = _first_available_path(candidate, BMODE_PATHS)
        paths = (bmode_path, "data/tissue_doppler")
        bmode, tissue = tuple(_load_modality(store, path, start=start, stop=stop) for path in paths)
        converted = clinical_loaded_arrays((bmode, tissue))
        return _scan_converted(converted, "data/clinical_tissue_doppler"), paths

    if target.name == "3d_bmode":
        path = "data/3d_brightness_mode"
        loaded = _load_modality(store, path, start=start, stop=stop)
        raw = None if loaded.stream is None else loaded.stream.metadata.raw
        prepared = prepare_3d_brightness_for_display(np.asarray(loaded.data), loaded.timestamps, raw)
        geometry = spherical_geometry_from_metadata(raw)
        mosaic = clinical_spherical_mosaic(
            prepared.volumes,
            geometry,
            output_size=(int(panel_size_3d), int(panel_size_3d)),
        ).frames
        return (
            LoadedArray(
                name="clinical_3d_brightness_mode",
                data_path=path,
                data=mosaic,
                timestamps_path=loaded.timestamps_path,
                timestamps=prepared.timestamps,
                sample_rate_hz=loaded.sample_rate_hz,
                attrs={**dict(loaded.attrs), "3d_mosaic_rows": 3, "3d_mosaic_cols": 4},
                stream=loaded.stream,
            ),
            (path,),
        )

    raise ValueError(f"Unsupported target {target.name!r}")


def _load_modality(store: Any, path: str, *, start: int | None, stop: int | None) -> LoadedArray:
    return (
        store.load_modality_slice(path, start, stop)
        if start is not None or stop is not None
        else store.load_modality(path)
    )


def _scan_converted(loaded_arrays: tuple[LoadedArray, ...], data_path: str) -> LoadedArray:
    for loaded in loaded_arrays:
        if loaded.data_path == data_path:
            if "clinical_grid" not in loaded.attrs:
                raise ValueError(f"{data_path} is missing sector geometry metadata required for scan conversion")
            return loaded
    raise ValueError(f"No scan-converted array was produced for {data_path}")


def _candidate_matches(candidate: CandidateRecording, target: ScanConvertedTarget) -> bool:
    array_paths = {_normalize_path(path) for path in candidate.array_paths}
    return all(
        any(_normalize_path(path) in array_paths for path in path_group) for path_group in target.required_path_groups
    )


def _first_available_path(candidate: CandidateRecording, paths: tuple[str, ...]) -> str:
    array_paths = {_normalize_path(path) for path in candidate.array_paths}
    for path in paths:
        normalized = _normalize_path(path)
        if normalized in array_paths:
            return normalized
    raise ValueError(f"{candidate.source_label} has none of {', '.join(paths)}")


def _select_frame(data: np.ndarray, frame_index: int | None) -> tuple[np.ndarray, int]:
    arr = np.asarray(data)
    if _is_frame_stack(arr):
        resolved = int(arr.shape[0] // 2) if frame_index is None else int(frame_index)
        if resolved >= arr.shape[0]:
            raise ValueError(f"Requested frame {resolved}, but converted data has only {arr.shape[0]} frame(s)")
        return np.asarray(arr[resolved]), resolved
    if frame_index not in (None, 0):
        raise ValueError("Requested a nonzero --frame-index for a single-frame image")
    return arr, 0


def _is_frame_stack(data: np.ndarray) -> bool:
    arr = np.asarray(data)
    return arr.ndim >= 4 or (arr.ndim == 3 and arr.shape[-1] not in {3, 4})


def _save_png(
    frame: np.ndarray,
    path: Path,
    *,
    data_path: str,
    value_range: tuple[float, float] | None,
) -> None:
    rgb = _rgb_from_frame(frame, data_path=data_path, value_range=value_range)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def _rgb_from_frame(
    frame: np.ndarray,
    *,
    data_path: str,
    value_range: tuple[float, float] | None,
) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        rgb = arr[..., :3].astype(np.float32)
        if np.nanmax(rgb) > 1.0:
            rgb /= 255.0
        if arr.shape[-1] == 4:
            alpha = arr[..., 3].astype(np.float32)
            if np.nanmax(alpha) > 1.0:
                alpha /= 255.0
            rgb = rgb * np.clip(alpha[..., None], 0.0, 1.0)
        return np.asarray(np.rint(np.clip(np.nan_to_num(rgb), 0.0, 1.0) * 255.0), dtype=np.uint8)
    if arr.ndim != 2:
        squeezed = np.squeeze(arr)
        if squeezed.ndim != 2:
            raise ValueError(f"Expected a 2D or RGB image frame, got shape {arr.shape}")
        arr = squeezed
    finite = np.isfinite(arr)
    rgba = layer_to_rgba(
        ImageLayer(
            data=np.nan_to_num(arr, nan=0.0),
            cmap=data_path,
            mask=finite,
            value_range=value_range,
        )
    )
    return np.asarray(np.rint(np.clip(rgba[..., :3] * rgba[..., 3:4], 0.0, 1.0) * 255.0), dtype=np.uint8)


def _value_range(loaded: LoadedArray, values: np.ndarray) -> tuple[float, float] | None:
    if loaded.stream is not None and loaded.stream.metadata.value_range is not None:
        return loaded.stream.metadata.value_range
    if "clinical_colorbar_value_range" in loaded.attrs:
        value = loaded.attrs["clinical_colorbar_value_range"]
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]), float(value[1])
    return default_value_range_for_path(loaded.data_path, values)


def _looks_like_zarr(path: Path) -> bool:
    return path.name.endswith(".zarr")


def _path(candidate: CandidateRecording) -> Path:
    if candidate.path is None:
        raise ValueError("Candidate has no direct zarr path")
    return candidate.path


def _normalize_path(path: str) -> str:
    text = str(path).strip("/")
    return text if text.startswith("data/") else f"data/{text}"


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
