"""Clips of 32 B-mode frames with synchronized ECG windows for ECG-conditioned video diffusion.

Everything is read through the ``echoxflow`` package: recordings are discovered from the Croissant catalog, B-mode
frames and their timestamps come from ``data/2d_brightness_mode``, and the ECG from ``data/ecg`` + ``timestamps/ecg``.

Selection (defaults reproduce the paper runs):

* apical A2C/A3C/A4C recordings (see :mod:`tasks.video_diffusion.views`),
* imaging sector starting at the probe (``depth_start <= 1e-5 m``),
* B-mode frame rate >= 25 Hz; recordings above 50 Hz use every second frame,
* every frame of the clip lies inside the ECG time span, and consecutive clip frames strictly increase in time, so a
  clip never crosses a B-mode clock reset or a gap longer than the frame-rate floor allows,
* exams split by ``five_fold_split`` (training: all folds except ``evaluation_fold``).

Each training epoch visits every recording once and draws a uniformly random valid clip start per visit. Recordings are
ordered round-robin over the views, so the first held-out items cover all views. Validation and evaluation use the
middle valid start.
:class:`ResumableEpochSampler` shuffles each epoch from ``(seed, epoch)`` alone and hands every visit its own seed
for the clip start and ECG polarity, so an interrupted epoch can be resumed at any step with the same data.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from echoxflow import data_root as resolve_data_root
from echoxflow import find_recordings, load_croissant, load_exam_folds, open_recording
from echoxflow.scan import CartesianGrid, SectorGeometry, sector_to_cartesian
from echoxflow.timing import interpolate_on_runs
from tasks.utils.atomic import atomic_write_text
from tasks.video_diffusion.metrics import stable_seed
from tasks.video_diffusion.views import APICAL_VIEWS, ViewSelector, build_view_selector

BMODE_PATH = "data/2d_brightness_mode"
ECG_PATH = "data/ecg"
# 2: the ECG span ignores clock resets (min/max instead of first/last sample) and ECG windows are reset aware.
INDEX_FORMAT_VERSION = 2


@dataclass(frozen=True)
class ClipRecording:
    """One eligible recording and the frame indices where a complete clip may start."""

    exam_id: str
    recording_id: str
    zarr_path: str
    view: str
    frame_stride: int
    candidate_starts: tuple[int, ...]
    depth_start_m: float
    depth_end_m: float
    tilt_rad: float
    width_rad: float


# ---------------------------------------------------------------------------------------------------------------------
# Index construction


def clip_candidate_starts(
    frame_times_s: np.ndarray,
    ecg_times_s: np.ndarray,
    *,
    clip_frames: int,
    frame_stride: int,
    minimum_frame_rate_hz: float,
) -> np.ndarray:
    """Return clip starts whose strided frames are inside the ECG span and meet the frame-rate floor."""
    span = 1 + (clip_frames - 1) * frame_stride
    if len(frame_times_s) < span or len(ecg_times_s) < 2:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(len(frame_times_s) - span + 1, dtype=np.int64)
    frame_indices = starts[:, None] + frame_stride * np.arange(clip_frames, dtype=np.int64)[None, :]
    times = frame_times_s[frame_indices]
    maximum_interval_s = 1.0 / minimum_frame_rate_hz
    intervals = np.diff(times, axis=1)
    valid = np.all((intervals > 0.0) & (intervals <= maximum_interval_s * (1.0 + 1e-6)), axis=1)
    finite_ecg = np.asarray(ecg_times_s, dtype=np.float64)
    finite_ecg = finite_ecg[np.isfinite(finite_ecg)]
    if finite_ecg.size < 2:
        return np.empty(0, dtype=np.int64)
    # A reset ECG clock is not sorted, so its span is its extent, not its first and last sample.
    valid &= (times[:, 0] >= finite_ecg.min()) & (times[:, -1] <= finite_ecg.max())
    return starts[valid]


def _bmode_geometry(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = manifest.get("metadata")
    geometry = metadata.get("bmode_geometry") if isinstance(metadata, Mapping) else None
    if isinstance(geometry, Mapping):
        return geometry
    for sector in manifest.get("sectors") or ():
        if isinstance(sector, Mapping) and sector.get("semantic_id") == "bmode":
            return sector.get("geometry")
    return None


def _index_recording_with_reason(
    record: Any,
    *,
    root: Path,
    selector: ViewSelector,
    settings: Mapping[str, Any],
) -> tuple[ClipRecording | None, str]:
    """Index one recording and name why it was excluded; ``view:*`` reasons are view filtering, the rest eligibility."""
    view = selector.view_from_csv(record.exam_id, record.recording_id)
    if selector.uses_csv and view is None:
        return None, "view:not_in_csv"
    store = open_recording(record, root=root)
    manifest = store.group.attrs.get("recording_manifest") or {}
    if view is None:
        view = selector.view_from_manifest(manifest)
        if view is None:
            return None, "view:no_confident_view"
    geometry = _bmode_geometry(manifest)
    if geometry is None:
        return None, "no_bmode_geometry"
    depth_start_m = float(geometry["depth_start_m"])
    if depth_start_m > float(settings["maximum_depth_start_m"]):
        return None, "depth_start"
    frame_times = np.asarray(store.load_timestamps(BMODE_PATH), dtype=np.float64)
    ecg_times = np.asarray(store.load_timestamps(ECG_PATH), dtype=np.float64)
    if frame_times.ndim != 1 or len(frame_times) < 2 or ecg_times.ndim != 1:
        return None, "missing_timestamps"
    frame_rate_hz = 1.0 / float(np.median(np.diff(frame_times)))
    if not math.isfinite(frame_rate_hz) or frame_rate_hz < float(settings["minimum_frame_rate_hz"]):
        return None, "frame_rate"
    stride = (
        int(settings["high_frame_rate_stride"])
        if frame_rate_hz > float(settings["high_frame_rate_threshold_hz"])
        else 1
    )
    starts = clip_candidate_starts(
        frame_times,
        ecg_times,
        clip_frames=int(settings["clip_frames"]),
        frame_stride=stride,
        minimum_frame_rate_hz=float(settings["minimum_frame_rate_hz"]),
    )
    if not len(starts):
        return None, "no_valid_clip"
    item = ClipRecording(
        exam_id=record.exam_id,
        recording_id=record.recording_id,
        zarr_path=record.zarr_path,
        view=view,
        frame_stride=stride,
        candidate_starts=tuple(int(start) for start in starts),
        depth_start_m=depth_start_m,
        depth_end_m=float(geometry["depth_end_m"]),
        tilt_rad=float(geometry["tilt_rad"]),
        width_rad=float(geometry["width_rad"]),
    )
    views = settings.get("views")
    if views is not None and item.view not in views:
        return None, f"view:{item.view}"
    ecg_reset = bool(np.any(~(np.diff(ecg_times) > 0.0)))
    return item, "eligible_ecg_reset" if ecg_reset else "eligible"


def _index_settings(data_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": INDEX_FORMAT_VERSION,
        "clip_frames": int(data_config.get("clip_frames", 32)),
        "minimum_frame_rate_hz": float(data_config.get("minimum_frame_rate_hz", 25.0)),
        "high_frame_rate_threshold_hz": float(data_config.get("high_frame_rate_threshold_hz", 50.0)),
        "high_frame_rate_stride": int(data_config.get("high_frame_rate_stride", 2)),
        "maximum_depth_start_m": float(data_config.get("maximum_depth_start_m", 1.0e-5)),
        "views": [str(view).upper() for view in data_config.get("views", APICAL_VIEWS)],
        "view_source": str(data_config.get("view_source", "csv")),
        "view_labels_csv": None if not data_config.get("view_labels_csv") else str(data_config["view_labels_csv"]),
        "minimum_view_confidence": float(data_config.get("minimum_view_confidence", 0.85)),
    }


def build_clip_index(
    data_config: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    workers: int = 16,
) -> list[ClipRecording]:
    """Index every eligible recording in the catalog (all folds). Cached as JSON when ``cache_dir`` is given."""
    root_path = resolve_data_root(root if root is not None else data_config.get("root_dir"))
    settings = _index_settings(data_config)
    catalog_path = root_path / "croissant.json"
    cache_path: Path | None = None
    if cache_dir is not None:
        stat = catalog_path.stat()
        key = json.dumps(
            {**settings, "catalog": str(catalog_path.resolve()), "mtime": stat.st_mtime_ns}, sort_keys=True
        )
        cache_path = Path(cache_dir) / f"clip_index_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
        if cache_path.is_file():
            rows = json.loads(cache_path.read_text(encoding="utf-8"))
            return [ClipRecording(**{**row, "candidate_starts": tuple(row["candidate_starts"])}) for row in rows]
    catalog = load_croissant(catalog_path, root=root_path, view_csv=settings["view_labels_csv"])
    records = find_recordings(
        croissant=catalog,
        array_paths=(BMODE_PATH, ECG_PATH),
        require_all=True,
        min_frame_counts={"2d_brightness_mode": settings["clip_frames"]},
    )
    maximum_delta = 1.0 / settings["minimum_frame_rate_hz"]
    records = tuple(
        record
        for record in records
        if (delta := record.median_delta_time("2d_brightness_mode")) is None or delta <= maximum_delta * 1.01
    )
    selector = build_view_selector(settings, catalog=catalog)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = list(
            pool.map(
                lambda record: _index_recording_with_reason(
                    record, root=root_path, selector=selector, settings=settings
                ),
                records,
            )
        )
    reasons = Counter(reason for _item, reason in results)
    logging.getLogger(__name__).info(
        "diffusion clip index: %d candidates; view filtering %s; eligibility %s",
        len(results),
        dict(sorted((reason, count) for reason, count in reasons.items() if reason.startswith("view:"))),
        dict(sorted((reason, count) for reason, count in reasons.items() if not reason.startswith("view:"))),
    )
    index = sorted(
        (item for item, _reason in results if item is not None),
        key=lambda item: (item.exam_id, item.recording_id),
    )
    if cache_path is not None:
        atomic_write_text(cache_path, json.dumps([asdict(item) for item in index]))  # launchers may write it at once
    return index


def split_clip_index(
    index: Sequence[ClipRecording], *, split_csv: str | Path, evaluation_fold: int
) -> tuple[list[ClipRecording], list[ClipRecording]]:
    """Split by exam: training uses every fold except ``evaluation_fold``."""
    folds = load_exam_folds(split_csv)
    missing = sorted({item.exam_id for item in index} - folds.keys())
    if missing:
        raise ValueError(f"split CSV lacks {len(missing)} exams, e.g. {missing[0]}")
    training = [item for item in index if folds[item.exam_id] != int(evaluation_fold)]
    evaluation = [item for item in index if folds[item.exam_id] == int(evaluation_fold)]
    return training, evaluation


# ---------------------------------------------------------------------------------------------------------------------
# ECG


def normalize_ecg(signal: np.ndarray, *, amplitude_clip: float) -> np.ndarray:
    """Robustly centre (median), scale (1.4826 MAD) and clip the ECG to ``[-1, 1]``."""
    signal = np.asarray(signal, dtype=np.float64)
    centre = float(np.median(signal))
    scale = float(1.4826 * np.median(np.abs(signal - centre)))
    if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = float(np.std(signal))
    if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = 1.0
    return np.clip((signal - centre) / scale, -amplitude_clip, amplitude_clip) / amplitude_clip


def ecg_windows(
    ecg_times_s: np.ndarray,
    normalized_ecg: np.ndarray,
    frame_times_s: np.ndarray,
    *,
    sample_rate_hz: float,
    history_seconds: float,
    future_seconds: float,
    maximum_gap_samples: int,
) -> np.ndarray:
    """Return ``[T, 2, W]`` ECG windows (signal, validity mask) resampled at ``sample_rate_hz`` around each frame.

    The ECG stays in its native order. Interpolation runs separately on each strictly increasing stretch of its clock,
    so a clock reset never reaches ``np.interp``; queries where stretches overlap (the time is ambiguous), outside
    every stretch, or across a gap longer than ``maximum_gap_samples`` are masked invalid and set to 0.
    """
    history = round(history_seconds * sample_rate_hz)
    future = round(future_seconds * sample_rate_hz)
    offsets = np.arange(-history, future + 1, dtype=np.float64) / sample_rate_hz
    queries = (np.asarray(frame_times_s, dtype=np.float64)[:, None] + offsets[None, :]).reshape(-1)
    values, valid = interpolate_on_runs(
        ecg_times_s,
        normalized_ecg,
        queries,
        maximum_gap_s=maximum_gap_samples / sample_rate_hz,
        nearest_tolerance_s=0.51 / sample_rate_hz,
    )
    shape = (len(frame_times_s), len(offsets))
    return np.stack((values.reshape(shape), valid.reshape(shape)), axis=1).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# Frames


def square_cartesian_grid(depth_end_m: float, size: int) -> CartesianGrid:
    """Square canvas spanning ``[0, depth_end]`` axially and ``[-depth_end/2, depth_end/2]`` laterally.

    Coordinates are pixel centres.
    """
    half_pixel = 0.5 / size
    return CartesianGrid(
        shape=(size, size),
        x_range_m=(depth_end_m * (half_pixel - 0.5), depth_end_m * (0.5 - half_pixel)),
        y_range_m=(depth_end_m * half_pixel, depth_end_m * (1.0 - half_pixel)),
    )


def prepare_frames(
    frames: np.ndarray,
    *,
    coordinate_space: str,
    size: int,
    geometry: SectorGeometry,
) -> np.ndarray:
    """Map raw uint8 beamspace frames ``[T, R, A]`` to float ``[T, size, size]`` in ``[0, 1]``.

    Frames are first resized to ``size x size`` in beamspace (bilinear). Cartesian frames are then scan converted with
    ``echoxflow.scan.sector_to_cartesian`` onto :func:`square_cartesian_grid`.
    """
    values = np.asarray(frames, dtype=np.float32)
    if values.size and float(values.max()) > 1.0:
        values = values / 255.0
    resized = F.interpolate(torch.from_numpy(np.clip(values, 0.0, 1.0))[:, None], size=(size, size), mode="bilinear")
    beamspace = resized[:, 0].numpy()
    if coordinate_space == "beamspace":
        return np.ascontiguousarray(beamspace, dtype=np.float32)
    if coordinate_space != "cartesian":
        raise ValueError(f"coordinate_space must be 'beamspace' or 'cartesian', got {coordinate_space!r}")
    return scan_convert_stack(beamspace, geometry, size)


def scan_convert_stack(frames: np.ndarray, geometry: SectorGeometry, size: int) -> np.ndarray:
    """Scan convert float beamspace frames ``[T, R, A]`` onto the ``size x size`` square canvas (zero outside)."""
    grid = square_cartesian_grid(geometry.depth_end_m, size)
    stacked = np.moveaxis(np.asarray(frames, dtype=np.float32), 0, -1)
    converted = sector_to_cartesian(stacked, geometry, grid=grid, interpolation="linear").data
    return np.ascontiguousarray(np.moveaxis(converted, -1, 0), dtype=np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# Dataset


class VideoDiffusionDataset(Dataset[dict[str, Any]]):
    """32-frame clips, one item per recording; see the module docstring for the selection rules."""

    def __init__(
        self,
        recordings: Sequence[ClipRecording],
        *,
        root: str | Path,
        coordinate_space: str,
        spatial_size: int,
        clip_frames: int = 32,
        random_start: bool,
        ecg_sign_flip_probability: float = 0.0,
        ecg_sample_rate_hz: float = 600.0,
        ecg_history_seconds: float = 1.2,
        ecg_future_seconds: float = 0.2,
        ecg_maximum_gap_samples: int = 4,
        ecg_amplitude_clip: float = 5.0,
        views: Sequence[str] = APICAL_VIEWS,
    ) -> None:
        self.root = Path(root)
        self.coordinate_space = str(coordinate_space)
        self.spatial_size = int(spatial_size)
        self.clip_frames = int(clip_frames)
        self.random_start = bool(random_start)
        self.ecg_sign_flip_probability = float(ecg_sign_flip_probability)
        self.ecg_options: dict[str, Any] = {
            "sample_rate_hz": float(ecg_sample_rate_hz),
            "history_seconds": float(ecg_history_seconds),
            "future_seconds": float(ecg_future_seconds),
            "maximum_gap_samples": int(ecg_maximum_gap_samples),
        }
        self.ecg_amplitude_clip = float(ecg_amplitude_clip)
        self.views = tuple(view for view in (str(v).upper() for v in views) if any(r.view == view for r in recordings))
        if not self.views:
            raise ValueError("no recordings for the requested views")
        self.grouped = {view: [r for r in recordings if r.view == view] for view in self.views}
        rounds = itertools.zip_longest(*self.grouped.values())
        self.recordings = tuple(r for group in rounds for r in group if r is not None)

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        """Clip ``index``; an ``(index, seed)`` pair (from :class:`ResumableEpochSampler`) seeds its random choices."""
        generator = None
        if isinstance(index, tuple):
            position, item_seed = int(index[0]), int(index[1])
            generator = torch.Generator().manual_seed(item_seed)
        else:
            position = int(index)
        recording = self.recordings[position]
        candidates = recording.candidate_starts
        if self.random_start and len(candidates) > 1:
            start = candidates[int(torch.randint(len(candidates), (), generator=generator).item())]
        else:
            start = candidates[len(candidates) // 2]
        flip = self.random_start and float(torch.rand((), generator=generator).item()) < self.ecg_sign_flip_probability
        return self.load_clip(recording, start, ecg_polarity=-1.0 if flip else 1.0)

    def load_clip(self, recording: ClipRecording, start: int, *, ecg_polarity: float = 1.0) -> dict[str, Any]:
        store = open_recording(self.root / recording.zarr_path, root=self.root)
        span = 1 + (self.clip_frames - 1) * recording.frame_stride
        bmode = store.load_stream_slice(BMODE_PATH, start, start + span)
        positions = np.arange(self.clip_frames) * recording.frame_stride
        frame_times = np.asarray(bmode.timestamps, dtype=np.float64)[positions]
        geometry = SectorGeometry.from_center_width(
            depth_start_m=recording.depth_start_m,
            depth_end_m=recording.depth_end_m,
            tilt_rad=recording.tilt_rad,
            width_rad=recording.width_rad,
        )
        frames = prepare_frames(
            np.asarray(bmode.data)[positions],
            coordinate_space=self.coordinate_space,
            size=self.spatial_size,
            geometry=geometry,
        )
        ecg = store.load_stream(ECG_PATH)
        ecg_times = np.asarray(ecg.timestamps, dtype=np.float64)
        normalized = normalize_ecg(np.asarray(ecg.data).reshape(-1), amplitude_clip=self.ecg_amplitude_clip)
        windows = ecg_windows(ecg_times, normalized, frame_times, **self.ecg_options)[:, :1] * ecg_polarity
        return {
            "frames": torch.from_numpy(frames).unsqueeze(1),
            "elapsed_time_s": torch.from_numpy((frame_times - frame_times[0]).astype(np.float32)),
            "ecg_waveform": torch.from_numpy(windows),
            "depth_start_cm": torch.tensor(recording.depth_start_m * 100.0, dtype=torch.float32),
            "depth_end_cm": torch.tensor(recording.depth_end_m * 100.0, dtype=torch.float32),
            "sector_width_rad": torch.tensor(recording.width_rad, dtype=torch.float32),
            "sector_tilt_rad": torch.tensor(recording.tilt_rad, dtype=torch.float32),
            "start_frame": torch.tensor(int(start), dtype=torch.int64),
            "frame_stride": torch.tensor(recording.frame_stride, dtype=torch.int64),
            "view": recording.view,
            "exam_id": recording.exam_id,
            "recording_id": recording.recording_id,
        }


class ResumableEpochSampler(Sampler[tuple[int, int]]):
    """Shuffled ``(index, item_seed)`` pairs whose order and seeds depend only on ``(seed, epoch)``.

    ``set_position(epoch, start)`` makes the next iteration yield that epoch's items from position ``start`` on, which
    resumes an interrupted epoch exactly, independent of data-loader workers.
    """

    def __init__(self, length: int, *, seed: int) -> None:
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = 0
        self.start = 0

    def set_position(self, epoch: int, start: int = 0) -> None:
        if not 0 <= int(start) <= self.length:
            raise ValueError(f"sampler start {start} outside [0, {self.length}]")
        self.epoch = int(epoch)
        self.start = int(start)

    def __len__(self) -> int:
        return self.length - self.start

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator().manual_seed(stable_seed(self.seed, "epoch", self.epoch))
        order = torch.randperm(self.length, generator=generator).tolist()
        item_seeds = torch.randint(2**62, (self.length,), generator=generator).tolist()
        for position in range(self.start, self.length):
            yield int(order[position]), int(item_seeds[position])


def dataset_options(config: Mapping[str, Any], *, training: bool) -> dict[str, Any]:
    data = config["data"]
    ecg = config.get("ecg", {})
    return {
        "coordinate_space": str(data.get("coordinate_space", "beamspace")),
        "spatial_size": int(data.get("spatial_size", 128)),
        "clip_frames": int(data.get("clip_frames", 32)),
        "random_start": training,
        "ecg_sign_flip_probability": float(ecg.get("sign_flip_probability", 0.5)) if training else 0.0,
        "ecg_sample_rate_hz": float(ecg.get("sample_rate_hz", 600.0)),
        "ecg_history_seconds": float(ecg.get("history_seconds", 1.2)),
        "ecg_future_seconds": float(ecg.get("future_seconds", 0.2)),
        "ecg_maximum_gap_samples": int(ecg.get("maximum_gap_samples", 4)),
        "ecg_amplitude_clip": float(ecg.get("amplitude_clip", 5.0)),
        "views": tuple(data.get("views", APICAL_VIEWS)),
    }


def build_datasets(
    config: Mapping[str, Any], *, data_root: str | Path | None = None
) -> tuple[VideoDiffusionDataset, VideoDiffusionDataset]:
    """Return (training, held-out evaluation-fold) datasets for a ``train.yaml``-style mapping."""
    data = config["data"]
    root = resolve_data_root(data_root if data_root is not None else data.get("root_dir"))
    split_csv = data.get("split_csv") or root / "splits.csv"
    index = build_clip_index(
        data, root=root, cache_dir=data.get("index_cache_dir"), workers=data.get("index_workers", 16)
    )
    training, evaluation = split_clip_index(
        index, split_csv=split_csv, evaluation_fold=int(data.get("evaluation_fold", 4))
    )
    limit = data.get("max_recordings")
    if limit:
        training, evaluation = training[: int(limit)], evaluation[: int(limit)]
    return (
        VideoDiffusionDataset(training, root=root, **dataset_options(config, training=True)),
        VideoDiffusionDataset(evaluation, root=root, **dataset_options(config, training=False)),
    )


def build_dataloaders(*, config: Any, data_root: str | Path | None = None) -> tuple[DataLoader, DataLoader]:
    """Task-registry entry point: training and validation loaders."""
    training, validation = build_datasets(config, data_root=data_root)
    data = config["data"]
    workers = int(data.get("num_workers", 4))
    common: dict[str, Any] = {"num_workers": workers, "pin_memory": True, "persistent_workers": workers > 0}
    return (
        DataLoader(training, batch_size=int(data.get("batch_size", 1)), shuffle=True, drop_last=True, **common),
        DataLoader(validation, batch_size=int(data.get("batch_size", 1)), shuffle=False, **common),
    )
