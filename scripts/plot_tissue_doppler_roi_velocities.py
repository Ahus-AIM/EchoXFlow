from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echoxflow import (  # noqa: E402
    BrightnessModeStream,
    RecordingRecord,
    TissueDopplerFloatStream,
    find_recordings,
    load_croissant,
    named_listed_colormap,
    open_recording,
)
from echoxflow.manifest import manifest_documents  # noqa: E402
from echoxflow.plotting.clinical import clinical_loaded_arrays  # noqa: E402
from echoxflow.plotting.specs import TraceSpec  # noqa: E402
from echoxflow.plotting.style import PlotStyle  # noqa: E402
from echoxflow.scan import CartesianGrid, SectorGeometry  # noqa: E402
from echoxflow.streams import default_value_range_for_path  # noqa: E402
from tasks.registry import task_load_config  # noqa: E402
from tasks.tissue_doppler.dataset import BMODE_PATH, TDI_PATH, RawDataset  # noqa: E402
from tasks.utils.models.unet import build_model  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

SPECTRAL_PATH = "data/1d_pulsed_wave_doppler"
DEFAULT_OUTPUT = Path("outputs/tissue_doppler_roi_velocities.png")
TRUE_COLOR = "#00D5FF"
PREDICTED_COLOR = "#FF4FB3"
MARKER_COLOR = "#00A087"


@dataclass(frozen=True)
class ResolvedSource:
    record: RecordingRecord
    root: Path | None


@dataclass(frozen=True)
class RoiPixels:
    mask: np.ndarray
    y: np.ndarray
    x: np.ndarray
    depths_m: np.ndarray


@dataclass(frozen=True)
class SpectralBackground:
    data: np.ndarray
    timestamps: np.ndarray
    velocity_axis_mps: np.ndarray | None
    markers: tuple["SpectralMarker", ...] = ()


@dataclass(frozen=True)
class SpectralMarker:
    time_s: float
    velocity_mps: float
    label: str


@dataclass(frozen=True)
class CartesianScan:
    image: np.ndarray
    grid: CartesianGrid
    time_s: float


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter-plot per-pixel tissue Doppler velocities inside the pulsed-wave sample-volume ROI for "
            "ground truth and a temporal U-Net prediction."
        )
    )
    parser.add_argument("source", type=Path, help="A .zarr recording, croissant.json, or dataset root.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Temporal U-Net weights.pt checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training config. Defaults to config.yaml beside --checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, default=None, help="Dataset root for relative Croissant zarr paths.")
    parser.add_argument("--recording-id", default=None, help="Recording ID to select from a Croissant catalog.")
    parser.add_argument("--exam-id", default=None, help="Exam ID to select from a Croissant catalog.")
    parser.add_argument("--sample-index", type=int, default=0, help="Full-case sample index if multiple are available.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum B-mode frames for the full-case clip. Defaults to the validation config.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-width-in", type=float, default=10.8)
    parser.add_argument("--figure-height-in", type=float, default=4.4)
    args = parser.parse_args(argv)

    config_path = _resolve_config_path(args.config, args.checkpoint)
    config = task_load_config("tissue_doppler")(config_path)
    source = _resolve_source(args.source, root=args.root, exam_id=args.exam_id, recording_id=args.recording_id)
    sample = _load_full_case_sample(
        config=config,
        source=source,
        sample_index=int(args.sample_index),
        max_frames=args.max_frames,
    )
    prediction = _predict_sample(
        config=config, checkpoint=args.checkpoint, sample=sample, device=_resolve_device(args.device)
    )

    store = open_recording(source.record, root=source.root)
    gate = _sampling_gate_metadata(store)
    geometry = _reference_sector_geometry(store)
    target = _tensor_video(getattr(sample, "doppler_target"))
    predicted = _tensor_video(prediction[:, :1])
    frame_count = min(int(target.shape[0]), int(predicted.shape[0]))
    target = target[:frame_count]
    predicted = predicted[:frame_count]
    timestamps = _sample_timestamps(sample, frame_count=frame_count)
    roi = _roi_pixels(
        gate,
        geometry=geometry,
        shape=tuple(int(v) for v in target.shape[-2:]),
        coordinate_space=str(getattr(sample, "coordinate_space", "beamspace") or "beamspace"),
        cartesian_height=_optional_int(config.data.get("cartesian_height")),
    )
    spectral = _spectral_background(store)
    reference_time = _highest_roi_velocity_time(target, roi=roi, timestamps=timestamps, spectral=spectral)
    scan = _cartesian_scan(store, reference_time_s=reference_time)
    ecg = _ecg_trace(store)

    output = args.output.expanduser()
    _write_roi_velocity_plot(
        target,
        predicted,
        roi=roi,
        timestamps=timestamps,
        spectral=spectral,
        scan=scan,
        ecg=ecg,
        gate=gate,
        geometry=geometry,
        reference_time_s=reference_time,
        output=output,
        dpi=int(args.dpi),
        figsize=(float(args.figure_width_in), float(args.figure_height_in)),
        title=f"{source.record.exam_id}/{source.record.recording_id}",
        velocity_limit_mps=_sample_scalar(sample, "sector_velocity_limit_mps"),
    )
    print(output)
    return 0


def _resolve_config_path(config: Path | None, checkpoint: Path) -> Path:
    if config is not None:
        return config.expanduser()
    path = checkpoint.expanduser().parent / "config.yaml"
    if not path.exists():
        raise SystemExit(f"missing --config and no config.yaml beside checkpoint: {path}")
    return path


def _resolve_source(
    source: Path, *, root: Path | None, exam_id: str | None, recording_id: str | None
) -> ResolvedSource:
    source = source.expanduser()
    if _looks_like_zarr_recording(source):
        return ResolvedSource(record=_direct_record(source), root=None)

    catalog_path, catalog_root = _catalog_path_and_root(source, root=root)
    catalog = load_croissant(catalog_path, root=catalog_root)
    records = find_recordings(
        croissant=catalog,
        root=catalog_root,
        exam_id=exam_id,
        recording_id=recording_id,
        array_paths=(BMODE_PATH, TDI_PATH, SPECTRAL_PATH),
        require_all=True,
    )
    if not records:
        label = f"{catalog_path}"
        if exam_id is not None or recording_id is not None:
            label += f" exam_id={exam_id!r} recording_id={recording_id!r}"
        raise SystemExit(f"no tissue Doppler recording with a 1D pulsed-wave trace found in {label}")
    return ResolvedSource(record=records[0], root=catalog_root)


def _looks_like_zarr_recording(path: Path) -> bool:
    if path.suffix == ".zarr":
        return True
    if path.is_file() and path.suffix in {".zip", ".zarr.zip"}:
        return True
    return path.is_dir() and (path / "zarr.json").exists()


def _catalog_path_and_root(source: Path, *, root: Path | None) -> tuple[Path, Path | None]:
    if source.is_dir():
        catalog_path = source / "croissant.json"
        catalog_root = source if root is None else root.expanduser()
    else:
        catalog_path = source
        catalog_root = source.parent if root is None else root.expanduser()
    if not catalog_path.exists():
        raise SystemExit(f"source is neither a zarr recording nor a Croissant manifest/root: {source}")
    return catalog_path, catalog_root


def _direct_record(path: Path) -> RecordingRecord:
    store = open_recording(path)
    array_paths = tuple(store.array_paths)
    content_types = tuple(
        _content_type_for_path(array_path) for array_path in array_paths if array_path.startswith("data/")
    )
    frame_counts = {
        content_type: _frame_count(store.group[array_path])
        for array_path in array_paths
        if array_path.startswith("data/")
        for content_type in (_content_type_for_path(array_path),)
    }
    for required in (BMODE_PATH, TDI_PATH, SPECTRAL_PATH):
        if required not in array_paths:
            raise SystemExit(f"direct recording is missing required array {required!r}: {path}")
    return RecordingRecord(
        exam_id=path.parent.name or "direct",
        recording_id=path.stem,
        zarr_path=str(path.resolve()),
        modes=content_types,
        content_types=content_types,
        frame_counts_by_content_type=frame_counts,
        median_delta_time_by_content_type={},
        array_paths=array_paths,
    )


def _content_type_for_path(path: str) -> str:
    return str(path).removeprefix("data/").strip("/")


def _frame_count(array: Any) -> int:
    shape = tuple(getattr(array, "shape", ()))
    return 1 if not shape else int(shape[0])


def _load_full_case_sample(
    *,
    config: object,
    source: ResolvedSource,
    sample_index: int,
    max_frames: int | None,
) -> object:
    data = cast(Mapping[str, object], getattr(config, "data"))
    dataset = RawDataset(
        records=[source.record],
        clip_length=int(data.get("clip_length", 32)),
        clip_stride=int(data.get("clip_stride", 1)),
        min_frames_per_case=int(data.get("min_frames_per_case", 2)),
        input_spatial_shape=_spatial_shape(data.get("input_spatial_shape", (256, 256))),
        target_spatial_shape=_spatial_shape(data.get("target_spatial_shape", (256, 256))),
        data_root=source.root,
        max_samples=None,
        alignment_slack_factor=float(data.get("alignment_slack_factor", 1.5)),
        sampling_mode="full_case",
        full_case_max_frames=(
            max_frames if max_frames is not None else _optional_int(data.get("val_full_case_max_frames"))
        ),
        accepted_fps_range=_optional_float_pair(data.get("accepted_fps_range")),
        coordinate_space=str(data.get("coordinate_space", "beamspace")),
        cartesian_height=_optional_int(data.get("cartesian_height")),
    )
    if len(dataset) <= 0:
        raise SystemExit("the selected recording produced no aligned tissue Doppler full-case samples")
    index = min(max(0, int(sample_index)), int(len(dataset)) - 1)
    return dataset[index]


def _spatial_shape(value: object) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError(f"expected a 2D spatial shape, got {value!r}")
    return int(value[0]), int(value[1])


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(cast(Any, value))


def _optional_float_pair(value: object) -> tuple[float, float] | None:
    if value is None or value == "":
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError(f"expected a numeric pair, got {value!r}")
    return float(value[0]), float(value[1])


def _predict_sample(*, config: object, checkpoint: Path, sample: object, device: torch.device) -> Tensor:
    model = build_model(**cast(Any, getattr(config, "model")))
    model.load_state_dict(_load_model_state(checkpoint))
    model.to(device)
    model.eval()
    frames = cast(Tensor, getattr(sample, "frames")).to(device)
    scale = cast(Tensor, getattr(sample, "velocity_scale_mps_per_px_frame")).to(device)
    with torch.no_grad():
        return cast(Tensor, model(frames, velocity_scale_mps_per_px_frame=scale)).detach().cpu()


def _load_model_state(checkpoint: Path) -> Mapping[str, Tensor]:
    state = torch.load(checkpoint.expanduser(), map_location="cpu")
    if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
        state = state["state_dict"]
    if isinstance(state, Mapping) and isinstance(state.get("model"), Mapping):
        state = state["model"]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint does not contain a model state dict: {checkpoint}")
    return cast(Mapping[str, Tensor], state)


def _resolve_device(value: str) -> torch.device:
    text = value.strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text == "cpu":
        return torch.device("cpu")
    if text == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is not available")
        return torch.device("cuda:0")
    if text.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{value} requested but CUDA is not available")
        return torch.device(text)
    raise ValueError(f"unsupported device {value!r}; expected auto, cpu, cuda, or cuda:N")


def _sampling_gate_metadata(store: object) -> Mapping[str, object]:
    group = getattr(store, "group")
    for document in manifest_documents(dict(getattr(group, "attrs", {}))):
        gate = _sampling_gate_from_document(document)
        if gate is not None:
            return gate
    loaded = cast(Any, store).load_modality(TDI_PATH)
    raw = None if loaded.stream is None else loaded.stream.metadata.raw
    if isinstance(raw, Mapping):
        gate = raw.get("sampling_gate_metadata")
        if isinstance(gate, Mapping):
            return cast(Mapping[str, object], gate)
    raise SystemExit("selected recording has the 1D trace but no tissue Doppler sampling gate metadata")


def _sampling_gate_from_document(document: Mapping[str, object]) -> Mapping[str, object] | None:
    tracks = document.get("tracks")
    if isinstance(tracks, list):
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            derived = track.get("derived_from")
            if _semantic_id(track) == "tissue_doppler_gate":
                return cast(Mapping[str, object], derived if isinstance(derived, Mapping) else track)
            if isinstance(derived, Mapping) and str(derived.get("kind", "")).strip() == "tissue_doppler_gate":
                return cast(Mapping[str, object], derived)
    sectors = document.get("sectors")
    if isinstance(sectors, list):
        for sector in sectors:
            if isinstance(sector, Mapping):
                gate = sector.get("sampling_gate_metadata")
                if isinstance(gate, Mapping):
                    return cast(Mapping[str, object], gate)
    gate = document.get("sampling_gate_metadata")
    if isinstance(gate, Mapping):
        return cast(Mapping[str, object], gate)
    return None


def _semantic_id(value: Mapping[str, object]) -> str:
    return str(value.get("semantic_id") or value.get("track_role_id") or value.get("sector_role_id") or "").strip()


def _reference_sector_geometry(store: object) -> SectorGeometry:
    bmode = cast(Any, store).load_stream(BMODE_PATH)
    tdi = cast(Any, store).load_modality(TDI_PATH).stream
    if not isinstance(bmode, BrightnessModeStream):
        raise TypeError(f"{BMODE_PATH} must be a BrightnessModeStream")
    if not isinstance(tdi, TissueDopplerFloatStream):
        raise TypeError(f"{TDI_PATH} must load as TissueDopplerFloatStream")
    geometry = bmode.metadata.geometry or tdi.metadata.geometry
    if not isinstance(geometry, SectorGeometry):
        raise SystemExit("selected recording has no sector geometry for mapping the gate ROI to pixels")
    return geometry


def _roi_pixels(
    gate: Mapping[str, object],
    *,
    geometry: SectorGeometry,
    shape: tuple[int, int],
    coordinate_space: str,
    cartesian_height: int | None,
) -> RoiPixels:
    center_depth = _metadata_float(gate.get("gate_center_depth_m"))
    tilt_rad = _metadata_float(gate.get("gate_tilt_rad"))
    if center_depth is None or tilt_rad is None:
        raise SystemExit("sampling gate metadata is missing gate_center_depth_m or gate_tilt_rad")
    sample_volume = _metadata_float(gate.get("gate_sample_volume_m")) or 0.006
    depths, angles = _pixel_depths_and_angles(
        geometry=geometry,
        shape=shape,
        coordinate_space=coordinate_space,
        cartesian_height=cartesian_height,
    )
    half = 0.5 * float(sample_volume)
    depth_delta = depths - float(center_depth)
    lateral_delta = depths * np.sin(angles - float(tilt_rad))
    sector_mask = (
        (depths >= min(float(geometry.depth_start_m), float(geometry.depth_end_m)))
        & (depths <= max(float(geometry.depth_start_m), float(geometry.depth_end_m)))
        & (angles >= min(float(geometry.angle_start_rad), float(geometry.angle_end_rad)))
        & (angles <= max(float(geometry.angle_start_rad), float(geometry.angle_end_rad)))
    )
    mask = sector_mask & (np.abs(depth_delta) <= half) & (np.abs(lateral_delta) <= half)
    if not bool(np.any(mask)):
        mask = _fallback_roi_mask(
            center_depth=float(center_depth),
            tilt_rad=float(tilt_rad),
            geometry=geometry,
            shape=shape,
            coordinate_space=coordinate_space,
            cartesian_height=cartesian_height,
        )
    y, x = np.nonzero(mask)
    order = np.lexsort((x, depths[y, x]))
    return RoiPixels(mask=mask, y=y[order], x=x[order], depths_m=depths[y[order], x[order]].astype(np.float32))


def _pixel_depths_and_angles(
    *,
    geometry: SectorGeometry,
    shape: tuple[int, int],
    coordinate_space: str,
    cartesian_height: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    if coordinate_space.strip().lower() == "cartesian":
        base_grid = CartesianGrid.from_sector_height(geometry, int(cartesian_height or height))
        xs = np.linspace(base_grid.x_range_m[0], base_grid.x_range_m[1], width, dtype=np.float64)
        ys = np.linspace(base_grid.y_range_m[0], base_grid.y_range_m[1], height, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        return np.sqrt(xx**2 + yy**2), np.arctan2(xx, yy)
    depths = np.linspace(geometry.depth_start_m, geometry.depth_end_m, height, dtype=np.float64)
    angles = np.linspace(geometry.angle_start_rad, geometry.angle_end_rad, width, dtype=np.float64)
    angle_grid, depth_grid = np.meshgrid(angles, depths)
    return depth_grid, angle_grid


def _fallback_roi_mask(
    *,
    center_depth: float,
    tilt_rad: float,
    geometry: SectorGeometry,
    shape: tuple[int, int],
    coordinate_space: str,
    cartesian_height: int | None,
) -> np.ndarray:
    depths, angles = _pixel_depths_and_angles(
        geometry=geometry,
        shape=shape,
        coordinate_space=coordinate_space,
        cartesian_height=cartesian_height,
    )
    distance = (depths - float(center_depth)) ** 2 + (depths * np.sin(angles - float(tilt_rad))) ** 2
    index = int(np.nanargmin(distance))
    mask = np.zeros(shape, dtype=bool)
    mask[np.unravel_index(index, shape)] = True
    return mask


def _metadata_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _tensor_video(value: Tensor) -> np.ndarray:
    array = value.detach().cpu().float().numpy()
    if array.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W] tensor, got {tuple(array.shape)}")
    return np.asarray(array[0, 0], dtype=np.float32)


def _sample_timestamps(sample: object, *, frame_count: int) -> np.ndarray:
    value = getattr(sample, "doppler_timestamps", None)
    if isinstance(value, Tensor):
        array = value.detach().cpu().numpy().reshape(-1)
        if array.size >= frame_count:
            return np.asarray(array[:frame_count], dtype=np.float64)
    return np.arange(int(frame_count), dtype=np.float64)


def _sample_scalar(sample: object, name: str) -> float | None:
    value = getattr(sample, name, None)
    if isinstance(value, Tensor) and value.numel() > 0:
        scalar = float(value.detach().cpu().reshape(-1)[0])
        return scalar if np.isfinite(scalar) else None
    return None


def _spectral_background(store: object) -> SpectralBackground | None:
    try:
        loaded = cast(Any, store).load_modality(SPECTRAL_PATH)
    except (FileNotFoundError, KeyError):
        return None
    data = _display_matrix(np.asarray(loaded.data, dtype=np.float32))
    if data.size == 0:
        return None
    timestamps = loaded.timestamps
    if timestamps is None or np.asarray(timestamps).size != int(data.shape[0]):
        timestamps = np.arange(int(data.shape[0]), dtype=np.float64)
    metadata = cast(Any, store).spectral_metadata(SPECTRAL_PATH)
    axis = getattr(metadata, "row_velocity_mps", None)
    velocity_axis = None
    if axis is not None:
        velocity_axis = np.asarray(axis, dtype=np.float32).reshape(-1)
        if velocity_axis.size != int(data.shape[1]) or not np.all(np.isfinite(velocity_axis)):
            velocity_axis = None
    return SpectralBackground(
        data=np.asarray(data, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64).reshape(-1),
        velocity_axis_mps=velocity_axis,
        markers=_spectral_markers(cast(Any, store)),
    )


def _reference_time_s(timestamps: np.ndarray, *, spectral: SpectralBackground | None) -> float:
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    finite = times[np.isfinite(times)]
    if finite.size:
        return float(finite[int(finite.size // 2)])
    if spectral is not None and spectral.markers:
        return float(spectral.markers[0].time_s)
    return 0.0


def _highest_roi_velocity_time(
    target: np.ndarray,
    *,
    roi: RoiPixels,
    timestamps: np.ndarray,
    spectral: SpectralBackground | None,
) -> float:
    values = np.asarray(target[:, roi.y, roi.x], dtype=np.float32)
    if values.ndim == 2 and values.shape[0] > 0:
        frame_max = np.max(np.where(np.isfinite(values), values, -np.inf), axis=1)
        finite = np.isfinite(frame_max)
        if np.any(finite):
            frame_index = int(np.argmax(np.where(finite, frame_max, -np.inf)))
            times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
            if frame_index < int(times.size) and np.isfinite(times[frame_index]):
                return float(times[frame_index])
    return _reference_time_s(timestamps, spectral=spectral)


def _cartesian_scan(store: object, *, reference_time_s: float) -> CartesianScan | None:
    try:
        bmode = cast(Any, store).load_modality(BMODE_PATH)
        tissue = cast(Any, store).load_modality(TDI_PATH)
        clinical = clinical_loaded_arrays((bmode, tissue))
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    loaded = next((item for item in clinical if item.data_path == "data/clinical_tissue_doppler"), None)
    if loaded is None:
        loaded = next(iter(clinical), None)
    if loaded is None:
        return None
    grid = loaded.attrs.get("clinical_grid")
    if not isinstance(grid, CartesianGrid):
        return None
    data = np.asarray(loaded.data)
    if data.ndim == 3:
        image = data
        time_s = float(reference_time_s)
    elif data.ndim >= 4 and int(data.shape[0]) > 0:
        frame_index = _nearest_timestamp_index(loaded.timestamps, reference_time_s, frame_count=int(data.shape[0]))
        image = data[frame_index]
        time_s = _timestamp_at_index(loaded.timestamps, frame_index, fallback=float(reference_time_s))
    else:
        return None
    return CartesianScan(image=np.asarray(image, dtype=np.float32), grid=grid, time_s=float(time_s))


def _ecg_trace(store: object) -> TraceSpec | None:
    array_paths = tuple(str(path) for path in getattr(store, "array_paths", ()))
    if "data/ecg" in array_paths:
        path = "data/ecg"
    else:
        path = next((item for item in array_paths if item.startswith("data/") and item.endswith("_ecg")), None)
        if path is None:
            return None
    try:
        loaded = cast(Any, store).load_modality(path)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    signal = np.asarray(loaded.data, dtype=np.float32).reshape(-1)
    timestamps = loaded.timestamps
    if timestamps is None or np.asarray(timestamps).size != signal.size:
        times = np.arange(signal.size, dtype=np.float64)
    else:
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    return TraceSpec(signal=signal, timestamps=times, label="ECG")


def _nearest_timestamp_index(timestamps: object, time_s: float, *, frame_count: int) -> int:
    if frame_count <= 1:
        return 0
    if timestamps is None:
        return min(max(0, int(round(float(time_s)))), frame_count - 1)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    count = min(int(frame_count), int(times.size))
    if count <= 0:
        return 0
    valid = np.isfinite(times[:count])
    if not np.any(valid):
        return 0
    distances = np.where(valid, np.abs(times[:count] - float(time_s)), np.inf)
    return int(np.argmin(distances))


def _timestamp_at_index(timestamps: object, index: int, *, fallback: float) -> float:
    if timestamps is None:
        return float(fallback)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if 0 <= int(index) < int(times.size) and np.isfinite(times[int(index)]):
        return float(times[int(index)])
    return float(fallback)


def _display_matrix(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _write_roi_velocity_plot(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    roi: RoiPixels,
    timestamps: np.ndarray,
    spectral: SpectralBackground | None,
    scan: CartesianScan | None,
    ecg: TraceSpec | None,
    gate: Mapping[str, object],
    geometry: SectorGeometry,
    reference_time_s: float,
    output: Path,
    dpi: int,
    figsize: tuple[float, float],
    title: str,
    velocity_limit_mps: float | None,
) -> None:
    real_values = target[:, roi.y, roi.x].T
    predicted_values = predicted[:, roi.y, roi.x].T
    limit = velocity_limit_mps or _symmetric_limit(real_values, predicted_values)
    style = PlotStyle.from_config()
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    scan_axis = None
    if scan is not None and ecg is not None:
        spec = figure.add_gridspec(2, 2, width_ratios=(0.72, 2.25), height_ratios=(1.0, 0.25))
        scan_axis = figure.add_subplot(spec[:, 0])
        axis = figure.add_subplot(spec[0, 1])
        ecg_axis = figure.add_subplot(spec[1, 1], sharex=axis)
        _draw_cartesian_scan(scan_axis, scan, gate=gate, geometry=geometry, style=style)
    elif scan is not None:
        spec = figure.add_gridspec(1, 2, width_ratios=(0.72, 2.25))
        scan_axis = figure.add_subplot(spec[0, 0])
        axis = figure.add_subplot(spec[0, 1])
        ecg_axis = None
        _draw_cartesian_scan(scan_axis, scan, gate=gate, geometry=geometry, style=style)
    elif ecg is not None:
        spec = figure.add_gridspec(2, 1, height_ratios=(1.0, 0.25))
        axis = figure.add_subplot(spec[0, 0])
        ecg_axis = figure.add_subplot(spec[1, 0], sharex=axis)
    else:
        axis = figure.add_subplot(1, 1, 1)
        ecg_axis = None
    x_true, y_true = _scatter_xy(timestamps, real_values, seed=11, sign=-1.0)
    x_pred, y_pred = _scatter_xy(timestamps, predicted_values, seed=29, sign=1.0)
    backdrop_xlim = _draw_spectral_background(axis, spectral, fallback_limit_mps=float(limit))
    true_handle = axis.scatter(x_true, y_true, s=4, alpha=0.55, linewidths=0.0, color=TRUE_COLOR, label="Ground truth")
    predicted_handle = axis.scatter(
        x_pred, y_pred, s=4, alpha=0.55, linewidths=0.0, color=PREDICTED_COLOR, label="Temporal U-Net"
    )
    marker_handle = _draw_spectral_markers(axis, spectral)
    axis.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.35)
    axis.set_ylim(-float(limit), float(limit))
    if backdrop_xlim is not None:
        axis.set_xlim(*backdrop_xlim)
    if ecg_axis is not None:
        xlim = backdrop_xlim or _time_edges(np.asarray(timestamps, dtype=np.float64))
        _draw_ecg_trace(ecg_axis, ecg, xlim=xlim, reference_time_s=reference_time_s, style=style)
        axis.tick_params(labelbottom=False)
        axis.set_xlabel("")
    else:
        axis.set_xlabel("Time (s)")
    axis.set_ylabel("Velocity (m/s)")
    axis.set_title(f"{title} ROI velocities ({real_values.shape[0]} px)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend_handles = [true_handle, predicted_handle, *(handle for handle in (marker_handle,) if handle is not None)]
    _draw_legend(scan_axis or axis, legend_handles, above=scan_axis is not None)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _draw_cartesian_scan(
    axis: object,
    scan: CartesianScan,
    *,
    gate: Mapping[str, object],
    geometry: SectorGeometry,
    style: PlotStyle,
) -> None:
    ax = cast(Any, axis)
    extent = _cartesian_extent(scan.grid)
    ax.imshow(
        np.asarray(scan.image),
        extent=extent,
        interpolation="nearest",
    )
    _draw_sampling_gate(ax, gate, geometry=geometry, style=style)
    ax.set_title("")
    ax.set_aspect("equal")
    y_span = abs(float(scan.grid.y_range_m[1]) - float(scan.grid.y_range_m[0]))
    ax.set_ylim(float(scan.grid.y_range_m[1]) + 0.10 * y_span, float(scan.grid.y_range_m[0]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_legend(axis: object, handles: Sequence[object], *, above: bool) -> None:
    if not handles:
        return
    ax = cast(Any, axis)
    if above:
        legend = ax.legend(
            handles=handles,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            borderaxespad=0.0,
            scatterpoints=1,
            markerscale=3.0,
            handletextpad=0.35,
            labelspacing=0.35,
            fontsize=8,
        )
    else:
        legend = ax.legend(
            handles=handles,
            frameon=False,
            loc="best",
            scatterpoints=1,
            markerscale=3.0,
            fontsize=8,
        )
    for handle in getattr(legend, "legend_handles", ()):
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)


def _cartesian_extent(grid: CartesianGrid) -> tuple[float, float, float, float]:
    return grid.x_range_m[0], grid.x_range_m[1], grid.y_range_m[1], grid.y_range_m[0]


def _draw_sampling_gate(
    axis: object,
    gate: Mapping[str, object],
    *,
    geometry: SectorGeometry,
    style: PlotStyle,
) -> None:
    layout = _sector_sampling_gate_layout(gate, geometry)
    if layout is None:
        return
    segments, markers = layout
    ax = cast(Any, axis)
    for segment in segments:
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=style.sampling_gate_color,
            linestyle="--",
            linewidth=1.4,
            zorder=5,
        )
    for marker in markers:
        ax.plot(
            marker[:, 0],
            marker[:, 1],
            color=style.sampling_gate_color,
            linestyle="-",
            linewidth=1.8,
            zorder=6,
        )


def _sector_sampling_gate_layout(
    gate: Mapping[str, object],
    geometry: SectorGeometry,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None:
    center_depth = _metadata_float(gate.get("gate_center_depth_m"))
    tilt_rad = _metadata_float(gate.get("gate_tilt_rad"))
    if center_depth is None or tilt_rad is None:
        return None
    sample_volume = _metadata_float(gate.get("gate_sample_volume_m")) or 0.006
    gate_start = float(np.clip(center_depth - 0.5 * sample_volume, geometry.depth_start_m, geometry.depth_end_m))
    gate_end = float(np.clip(center_depth + 0.5 * sample_volume, geometry.depth_start_m, geometry.depth_end_m))
    segments = _split_gate_line_segments(
        _sector_point(float(geometry.depth_start_m), tilt_rad),
        _sector_point(gate_start, tilt_rad),
        _sector_point(gate_end, tilt_rad),
        _sector_point(float(geometry.depth_end_m), tilt_rad),
    )
    markers = (
        _sector_gate_marker(gate_start, tilt_rad, sample_volume),
        _sector_gate_marker(gate_end, tilt_rad, sample_volume),
    )
    return segments, markers


def _sector_point(depth_m: float, angle_rad: float) -> tuple[float, float]:
    return float(depth_m * np.sin(angle_rad)), float(depth_m * np.cos(angle_rad))


def _sector_gate_marker(depth_m: float, angle_rad: float, width_m: float) -> np.ndarray:
    center = np.asarray(_sector_point(depth_m, angle_rad), dtype=np.float32)
    normal = np.asarray([np.cos(angle_rad), -np.sin(angle_rad)], dtype=np.float32)
    half_width = 0.5 * max(float(width_m), 1e-6)
    return np.asarray([center - half_width * normal, center + half_width * normal], dtype=np.float32)


def _split_gate_line_segments(
    start: tuple[float, float],
    gate_start: tuple[float, float],
    gate_end: tuple[float, float],
    end: tuple[float, float],
) -> tuple[np.ndarray, ...]:
    segments = []
    for point0, point1 in ((start, gate_start), (gate_end, end)):
        segment = np.asarray([point0, point1], dtype=np.float32)
        if not np.allclose(segment[0], segment[1]):
            segments.append(segment)
    return tuple(segments)


def _draw_ecg_trace(
    axis: object,
    ecg: TraceSpec | None,
    *,
    xlim: tuple[float, float],
    reference_time_s: float,
    style: PlotStyle,
) -> None:
    ax = cast(Any, axis)
    if ecg is not None:
        signal = _normalized_ecg_signal(ecg.signal)
        timestamps = np.asarray(ecg.timestamps, dtype=np.float64).reshape(-1)
        count = min(int(signal.size), int(timestamps.size))
        if count:
            ax.plot(timestamps[:count], signal[:count], color=style.ecg_trace_color, linewidth=0.8)
            ax.axvline(float(reference_time_s), color=style.ecg_marker_color, linewidth=0.9, alpha=0.85)
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _normalized_ecg_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    centered = values - float(np.median(finite))
    finite_centered = centered[np.isfinite(centered)]
    scale = float(np.percentile(np.abs(finite_centered), 99.0)) if finite_centered.size else 0.0
    if scale <= 1e-8:
        scale = float(np.max(np.abs(finite_centered))) if finite_centered.size else 1.0
    if scale <= 1e-8:
        return np.zeros_like(centered, dtype=np.float32)
    return np.clip(centered / scale, -1.0, 1.0).astype(np.float32, copy=False)


def _draw_spectral_background(
    axis: object,
    spectral: SpectralBackground | None,
    *,
    fallback_limit_mps: float,
) -> tuple[float, float] | None:
    if spectral is None:
        return None
    data = np.asarray(spectral.data, dtype=np.float32)
    if data.ndim != 2 or data.size == 0:
        return None
    times = np.asarray(spectral.timestamps, dtype=np.float64).reshape(-1)
    if times.size != int(data.shape[0]):
        times = np.arange(int(data.shape[0]), dtype=np.float64)
    left, right = _time_edges(times)
    velocity_axis = spectral.velocity_axis_mps
    if velocity_axis is None:
        bottom, top = -float(fallback_limit_mps), float(fallback_limit_mps)
        origin = "lower"
    else:
        bottom, top = float(np.min(velocity_axis)), float(np.max(velocity_axis))
        origin = "upper" if float(velocity_axis[0]) > float(velocity_axis[-1]) else "lower"
    vmin, vmax = _spectral_display_range(data)
    cast(Any, axis).imshow(
        data.T,
        cmap=named_listed_colormap(SPECTRAL_PATH) or "gray",
        vmin=float(vmin),
        vmax=float(vmax),
        aspect="auto",
        interpolation="nearest",
        origin=origin,
        extent=(left, right, bottom, top),
        zorder=0,
    )
    return left, right


def _spectral_markers(store: object) -> tuple[SpectralMarker, ...]:
    labels_by_path: dict[str, str] = {}
    markers: list[SpectralMarker] = []
    for document in manifest_documents(dict(getattr(cast(Any, store).group, "attrs", {}))):
        annotations = document.get("annotations")
        if isinstance(annotations, list):
            for annotation in annotations:
                if not isinstance(annotation, Mapping):
                    continue
                value = annotation.get("value")
                path = _annotation_path(value)
                label = _marker_label(annotation.get("label"))
                if path and label is not None:
                    labels_by_path[path] = label
        tracks = document.get("tracks")
        if isinstance(tracks, list):
            for track in tracks:
                if not isinstance(track, Mapping):
                    continue
                for annotation in _sequence_items(track.get("spectral_annotations")):
                    label = _marker_label(annotation.get("label"))
                    if label is None:
                        continue
                    markers.extend(_spectral_markers_from_points(annotation.get("points"), label=label))

    for path, label in labels_by_path.items():
        try:
            points = np.asarray(cast(Any, store).load_array(path), dtype=np.float32)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        markers.extend(_spectral_markers_from_array(points, label=label))
    return _deduplicate_markers(markers)


def _annotation_path(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("zarr_path", "array_path", "path"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().strip("/")
    return None


def _sequence_items(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _marker_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip()
    if not label:
        return None
    lower = label.lower()
    if not any(token in lower for token in ("eprime", "sprime", "aprime", "e'", "s'", "a'")):
        return None
    if "eprime" in lower or "e'" in lower:
        prefix = "E'"
    elif "sprime" in lower or "s'" in lower:
        prefix = "S'"
    else:
        prefix = "A'"
    if "lateral" in lower:
        return f"{prefix} lat"
    if "septal" in lower:
        return f"{prefix} sep"
    return prefix


def _spectral_markers_from_points(value: object, *, label: str) -> list[SpectralMarker]:
    markers: list[SpectralMarker] = []
    if not isinstance(value, list):
        return markers
    for point in value:
        if not isinstance(point, Mapping):
            continue
        time_s = _metadata_float(point.get("time_s"))
        velocity_mps = _metadata_float(point.get("velocity_mps"))
        if time_s is not None and velocity_mps is not None:
            markers.append(SpectralMarker(time_s=float(time_s), velocity_mps=-float(velocity_mps), label=label))
    return markers


def _spectral_markers_from_array(points: np.ndarray, *, label: str) -> list[SpectralMarker]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return []
    markers = []
    for row in arr:
        if np.all(np.isfinite(row[:2])):
            markers.append(SpectralMarker(time_s=float(row[0]), velocity_mps=-float(row[1]), label=label))
    return markers


def _deduplicate_markers(markers: Sequence[SpectralMarker]) -> tuple[SpectralMarker, ...]:
    seen: set[tuple[str, int, int]] = set()
    deduped: list[SpectralMarker] = []
    for marker in markers:
        key = (marker.label, int(round(marker.time_s * 1000.0)), int(round(marker.velocity_mps * 1000.0)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(marker)
    return tuple(deduped)


def _draw_spectral_markers(axis: object, spectral: SpectralBackground | None) -> object | None:
    if spectral is None or not spectral.markers:
        return None
    ax = cast(Any, axis)
    xs = np.asarray([marker.time_s for marker in spectral.markers], dtype=np.float32)
    ys = np.asarray([marker.velocity_mps for marker in spectral.markers], dtype=np.float32)
    handle = ax.scatter(
        xs,
        ys,
        marker="+",
        s=70,
        linewidths=1.5,
        color=MARKER_COLOR,
        zorder=5,
        label="Operator Annotation",
    )
    for marker in spectral.markers:
        ax.annotate(
            marker.label,
            xy=(marker.time_s, marker.velocity_mps),
            xytext=(4, 5),
            textcoords="offset points",
            color="#111111",
            fontsize=8,
            zorder=6,
            bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.86},
        )
    return handle


def _time_edges(timestamps: np.ndarray) -> tuple[float, float]:
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return -0.5, 0.5
    if times.size == 1:
        return float(times[0]) - 0.5, float(times[0]) + 0.5
    step = _timestamp_step(times)
    return float(times[0]) - 0.5 * step, float(times[-1]) + 0.5 * step


def _finite_range(data: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(data, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _spectral_display_range(data: np.ndarray, *, black_hold_fraction: float = 0.18) -> tuple[float, float]:
    lo, hi = default_value_range_for_path(SPECTRAL_PATH, data) or _finite_range(data)
    span = max(float(hi) - float(lo), 1e-6)
    return float(lo) + max(0.0, float(black_hold_fraction)) * span, float(hi)


def _scatter_xy(timestamps: np.ndarray, values: np.ndarray, *, seed: int, sign: float) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError(f"expected [roi_pixel,time] values, got {values.shape}")
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if times.size != int(values.shape[1]):
        times = np.arange(int(values.shape[1]), dtype=np.float64)
    step = _timestamp_step(times)
    rng = np.random.default_rng(int(seed))
    base = np.broadcast_to(times[None, :], values.shape)
    jitter = rng.uniform(-0.18, 0.18, size=values.shape) * step
    offset = float(sign) * 0.08 * step
    return (base + offset + jitter).reshape(-1), np.asarray(values, dtype=np.float32).reshape(-1)


def _timestamp_step(timestamps: np.ndarray) -> float:
    if timestamps.size <= 1:
        return 1.0
    diffs = np.diff(np.sort(np.asarray(timestamps, dtype=np.float64)))
    positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _symmetric_limit(*arrays: np.ndarray) -> float:
    finite = np.concatenate([np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(np.abs(finite), 99.0)), 1e-6)


if __name__ == "__main__":
    raise SystemExit(main())
