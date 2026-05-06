#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from extract_example_strips import (
    CandidateRecording,
    StripTarget,
    _apply_colormap,
    _candidate_matches,
    _default_background_rgb,
    _is_spectral_doppler_strip_path,
    _load_target_modality,
    _normalize_to_uint8,
    _rgb_from_image,
    _strip_image,
    _tinted_spectral_doppler_strip,
    source_candidates,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from echoxflow import LoadedArray, open_recording

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT = Path("outputs/images/ecg_alignment_concept")
DEFAULT_EXCLUDED_EXAMS = ("exam_0068f33be3ce3c27",)
LATEX_PALETTE = {
    "tissue_doppler": "#F0F0F0",  # gray!12, with LaTeX xcolor's gray mixed with white
    "continuous_wave_doppler": "#FFEBEB",  # red!8
    "three_d_bmode": "#F0F0F0",  # gray!12
}


@dataclass(frozen=True)
class FigureTarget:
    key: str
    label: str
    color: str
    strip_target: StripTarget


@dataclass(frozen=True)
class RecordingConcept:
    key: str
    label: str
    color: str
    candidate: CandidateRecording
    image: np.ndarray
    ecg_signal: np.ndarray
    ecg_timestamps: np.ndarray
    r_peak_indices: np.ndarray
    acquisition_window_s: tuple[float, float] | None
    volume_curve_ml: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def source_label(self) -> str:
        return self.candidate.source_label


FIGURE_TARGETS = (
    FigureTarget(
        key="tissue_doppler",
        label="Tissue Doppler",
        color="#BBD7F4",
        strip_target=StripTarget(
            name="tissue_doppler",
            data_path="data/1d_pulsed_wave_doppler",
            description="Spectral Doppler strip from a tissue Doppler recording",
            associated_array_paths=("data/tissue_doppler",),
        ),
    ),
    FigureTarget(
        key="continuous_wave_doppler",
        label="CW Doppler",
        color="#F5DE96",
        strip_target=StripTarget(
            name="continuous_wave_doppler",
            data_path="data/1d_continuous_wave_doppler",
            description="Continuous wave Doppler strip",
        ),
    ),
    FigureTarget(
        key="three_d_bmode",
        label="3D B-mode",
        color="#D8C6EC",
        strip_target=StripTarget(
            name="3d_bmode",
            data_path="data/3d_brightness_mode",
            description="3D B-mode 3x4 scan-converted mosaic",
        ),
    ),
)


def main() -> None:
    args = _parse_args()
    candidates = source_candidates(args.source, root=args.root)
    records = _load_same_exam_concepts(
        candidates,
        exam_id=args.exam_id,
        excluded_exam_ids=tuple(args.exclude_exam or ()),
    )
    records = _apply_palette(records, args.palette)
    figure = build_figure(records, figsize=(float(args.width_in), float(args.height_in)))
    output_stem = Path(args.output)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    try:
        figure.savefig(
            png_path, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.0, facecolor=figure.get_facecolor()
        )
        figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.0, facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)
    exam = records[0].candidate.record.exam_id if records[0].candidate.record is not None else "direct_zarr"
    print(f"exam: {exam}")
    for record in records:
        print(f"{record.key}: {record.source_label}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a conceptual ECG R-peak alignment figure from one EchoXFlow exam. "
            "All modality panels are loaded from recordings in the selected exam, and each ECG trace "
            "comes from the same recording as its displayed modality."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help="Dataset root, croissant.json, or a single .zarr recording. Defaults to the configured data root.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Dataset root for relative paths in Croissant.")
    parser.add_argument("--exam-id", default=None, help="Use a specific exam. It must contain all required targets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--palette",
        choices=("default", "latex"),
        default="default",
        help="Use the default pastel palette or a LaTeX-style gray!12/red!8 alternate palette.",
    )
    parser.add_argument("--width-in", type=float, default=15.2)
    parser.add_argument("--height-in", type=float, default=5.2)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--exclude-exam",
        action="append",
        default=list(DEFAULT_EXCLUDED_EXAMS),
        help="Exclude an exam from automatic selection. Can be repeated.",
    )
    return parser.parse_args()


def _apply_palette(records: tuple[RecordingConcept, ...], palette: str) -> tuple[RecordingConcept, ...]:
    if palette == "default":
        return records
    if palette != "latex":
        raise ValueError(f"Unknown palette {palette!r}")
    return tuple(replace(record, color=LATEX_PALETTE.get(record.key, record.color)) for record in records)


def _load_same_exam_concepts(
    candidates: Sequence[CandidateRecording],
    *,
    exam_id: str | None,
    excluded_exam_ids: tuple[str, ...],
) -> tuple[RecordingConcept, ...]:
    usable = [candidate for candidate in candidates if candidate.record is not None and _has_ecg(candidate)]
    by_exam: dict[str, list[CandidateRecording]] = defaultdict(list)
    for candidate in usable:
        assert candidate.record is not None
        by_exam[candidate.record.exam_id].append(candidate)

    if exam_id is not None:
        if exam_id not in by_exam:
            raise ValueError(f"Exam {exam_id!r} was not found with ECG-bearing recordings")
        selected_exam = exam_id
    else:
        selected_exam = _first_complete_exam(by_exam, excluded_exam_ids=set(excluded_exam_ids))

    selected_candidates = by_exam[selected_exam]
    concepts: list[RecordingConcept] = []
    for figure_target in FIGURE_TARGETS:
        candidate = _select_candidate(selected_candidates, figure_target.strip_target)
        concepts.append(
            _load_recording_concept(
                figure_target,
                candidate,
            )
        )
    return tuple(concepts)


def _first_complete_exam(by_exam: dict[str, list[CandidateRecording]], *, excluded_exam_ids: set[str]) -> str:
    for exam_id in sorted(by_exam):
        if exam_id in excluded_exam_ids:
            continue
        if all(
            any(_candidate_matches(candidate, target.strip_target) for candidate in by_exam[exam_id])
            for target in FIGURE_TARGETS
        ):
            return exam_id
    raise ValueError("No single ECG-bearing exam contains all requested targets")


def _select_candidate(candidates: Sequence[CandidateRecording], target: StripTarget) -> CandidateRecording:
    matches = [candidate for candidate in candidates if _candidate_matches(candidate, target)]
    if not matches:
        raise ValueError(f"Selected exam is missing {target.name}")
    return max(matches, key=_candidate_score)


def _candidate_score(candidate: CandidateRecording) -> tuple[int, int, str]:
    assert candidate.record is not None
    ecg_count = int(candidate.record.frame_count("ecg") or 0)
    bmode_count = int(candidate.record.frame_count("2d_brightness_mode") or 0)
    return ecg_count, bmode_count, candidate.record.recording_id


def _load_recording_concept(
    figure_target: FigureTarget,
    candidate: CandidateRecording,
) -> RecordingConcept:
    store = open_recording(candidate.record, root=candidate.root)
    loaded = _load_target_modality(store, candidate, figure_target.strip_target, start=None, stop=None)
    image, acquisition_window_s = _loaded_to_rgb_and_window(loaded)
    ecg = store.load_modality(_ecg_data_path(candidate))
    signal = _orient_ecg(np.asarray(ecg.data, dtype=np.float32).reshape(-1))
    timestamps = _ecg_timestamps(ecg, signal)
    peaks = _r_peak_indices(signal, timestamps)
    if peaks.size < 2:
        raise ValueError(f"{candidate.source_label} ECG did not contain at least two detectable R-peaks")
    volume_curve_ml = _mesh_volume_curve_ml(store) if figure_target.key == "three_d_bmode" else None
    return RecordingConcept(
        key=figure_target.key,
        label=figure_target.label,
        color=figure_target.color,
        candidate=candidate,
        image=image,
        ecg_signal=signal,
        ecg_timestamps=timestamps,
        r_peak_indices=peaks,
        acquisition_window_s=acquisition_window_s,
        volume_curve_ml=volume_curve_ml,
    )


def _mesh_volume_curve_ml(store: object) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        mesh = store.load_packed_mesh_annotation()  # type: ignore[attr-defined]
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    if mesh.frame_count <= 0:
        return None
    volumes_ml: list[float] = []
    times: list[float] = []
    for index in range(mesh.frame_count):
        frame = mesh.frame(index)
        volume_m3 = _closed_surface_volume_m3(frame.points, frame.faces)
        if not np.isfinite(volume_m3):
            continue
        volumes_ml.append(float(volume_m3) * 1.0e6)
        times.append(float(index if frame.timestamp is None else frame.timestamp))
    if len(volumes_ml) < 2:
        return None
    return np.asarray(times, dtype=np.float64), np.asarray(volumes_ml, dtype=np.float64)


def _closed_surface_volume_m3(points: np.ndarray, faces: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    idx = np.asarray(faces, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] < 3 or idx.ndim != 2 or idx.shape[1] < 3:
        return float("nan")
    triangles = idx[:, :3]
    if idx.shape[1] == 4:
        triangles = np.concatenate([triangles, idx[:, [0, 2, 3]]], axis=0)
    valid = np.all((triangles >= 0) & (triangles < pts.shape[0]), axis=1)
    if not bool(np.any(valid)):
        return float("nan")
    tri_points = pts[triangles[valid], :3]
    signed = np.einsum("ij,ij->i", tri_points[:, 0], np.cross(tri_points[:, 1], tri_points[:, 2]))
    return abs(float(np.sum(signed)) / 6.0)


def _loaded_to_rgb_and_window(loaded: LoadedArray) -> tuple[np.ndarray, tuple[float, float] | None]:
    data = np.asarray(loaded.data)
    timestamps = None if loaded.timestamps is None else np.asarray(loaded.timestamps)
    acquisition_window = _timestamp_window(timestamps)
    image = _strip_image(data)
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        return (
            _rgb_from_image(image, background=_default_background_rgb()).astype(np.float32) / 255.0,
            acquisition_window,
        )
    value_range = None if loaded.stream is None else loaded.stream.metadata.value_range
    scaled = _normalize_to_uint8(image, value_range=value_range)
    rgb = _apply_colormap(scaled, data_path=loaded.data_path)
    if _is_spectral_doppler_strip_path(loaded.data_path):
        rgb = _tinted_spectral_doppler_strip(rgb)
    return np.asarray(rgb, dtype=np.float32) / 255.0, acquisition_window


def _timestamp_window(timestamps: np.ndarray | None) -> tuple[float, float] | None:
    if timestamps is None:
        return None
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    start, stop = float(finite[0]), float(finite[-1])
    return (start, stop) if stop > start else None


def _has_ecg(candidate: CandidateRecording) -> bool:
    return "data/ecg" in candidate.array_paths or any(
        path.startswith("data/") and path.endswith("_ecg") for path in candidate.array_paths
    )


def _ecg_data_path(candidate: CandidateRecording) -> str:
    if "data/ecg" in candidate.array_paths:
        return "ecg"
    fallback = next(path for path in candidate.array_paths if path.startswith("data/") and path.endswith("_ecg"))
    return fallback


def _ecg_timestamps(loaded: LoadedArray, signal: np.ndarray) -> np.ndarray:
    if loaded.timestamps is None:
        return np.arange(signal.size, dtype=np.float64)
    timestamps = np.asarray(loaded.timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size != signal.size:
        return np.arange(signal.size, dtype=np.float64)
    return timestamps


def _orient_ecg(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    values = values - np.nanmedian(values)
    if abs(float(np.nanmin(values))) > abs(float(np.nanmax(values))) * 1.15:
        values = -values
    return np.asarray(np.nan_to_num(values), dtype=np.float32)


def _r_peak_indices(signal: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 5:
        return np.asarray([], dtype=np.int64)
    centered = values - np.median(values)
    spread = np.percentile(centered, 98.0) - np.percentile(centered, 10.0)
    if spread <= 0.0:
        return np.asarray([], dtype=np.int64)
    normalized = centered / spread
    candidates = (
        np.flatnonzero(
            (normalized[1:-1] > normalized[:-2])
            & (normalized[1:-1] >= normalized[2:])
            & (normalized[1:-1] > np.percentile(normalized, 96.5))
        )
        + 1
    )
    if candidates.size == 0:
        return np.asarray([], dtype=np.int64)
    min_distance = _min_peak_distance_samples(timestamps, values.size)
    selected: list[int] = []
    for index in candidates[np.argsort(normalized[candidates])[::-1]]:
        if all(abs(int(index) - existing) >= min_distance for existing in selected):
            selected.append(int(index))
    return np.asarray(sorted(selected), dtype=np.int64)


def _min_peak_distance_samples(timestamps: np.ndarray, count: int) -> int:
    times = np.asarray(timestamps, dtype=np.float64)
    if times.size >= 2 and np.isfinite(times).all():
        duration = float(times[-1] - times[0])
        if duration > 0.0:
            sample_rate = (times.size - 1) / duration
            return max(1, int(round(0.35 * sample_rate)))
    return max(1, int(round(count / 30)))


def build_figure(records: tuple[RecordingConcept, ...], *, figsize: tuple[float, float]) -> Figure:
    figure = plt.figure(figsize=figsize, facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.18, 1.0), wspace=0.16)
    left_ax = figure.add_subplot(grid[0, 0])
    right_ax = figure.add_subplot(grid[0, 1])
    _draw_acquisition_panel(left_ax, records)
    _draw_alignment_panel(right_ax, records)
    return figure


def _draw_acquisition_panel(ax: Axes, records: tuple[RecordingConcept, ...]) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    ax.text(0.0, 0.98, "Exam with sequential recordings", fontsize=15, ha="left", va="bottom")

    n = len(records)
    margin = 0.030
    block_w = (1.0 - 2 * margin) / n + 0.015
    overlap = 0.018
    y0, y1 = 0.005, 0.92
    reference_duration_s = _left_reference_duration_s(records)
    for index, record in enumerate(records):
        x0 = margin + index * (block_w - overlap)
        x1 = min(0.985, x0 + block_w)
        top_shift = 0.028 if index % 2 == 0 else -0.010
        polygon = Polygon(
            [(x0, y0), (x1, y0), (x1 + 0.012, y1 + top_shift), (x0 + 0.030, y1 - top_shift)],
            closed=True,
            facecolor=record.color,
            edgecolor="white",
            linewidth=1.2,
            zorder=index,
        )
        ax.add_patch(polygon)
        content_x0 = x0 + 0.018
        content_x1 = x1 - 0.016
        _draw_segment_ecg(
            ax,
            record,
            content_x0,
            content_x1,
            0.695,
            0.825,
            clip=polygon,
            zorder=20,
            reference_duration_s=reference_duration_s,
        )
        ax.text(
            0.5 * (x0 + x1),
            0.625,
            _display_label(record),
            ha="center",
            va="center",
            fontsize=12.4 if record.key == "three_d_bmode" else 13.7,
            color="#222222",
            zorder=30,
        )
        _draw_image(ax, record.image, x0 + 0.018, x1 - 0.018, 0.365, 0.545, zorder=20, clip=polygon)
        if record.volume_curve_ml is not None:
            window_start, window_stop = _left_ecg_window(record, reference_duration_s=reference_duration_s)
            peak_start, peak_stop = _left_ecg_peak_times(record)
            peak_x0 = content_x0 + (peak_start - window_start) / max(1e-9, window_stop - window_start) * (
                content_x1 - content_x0
            )
            peak_x1 = content_x0 + (peak_stop - window_start) / max(1e-9, window_stop - window_start) * (
                content_x1 - content_x0
            )
            _draw_volume_curve(
                ax,
                record.volume_curve_ml,
                peak_x0,
                peak_x1,
                0.195,
                0.285,
                zorder=25,
                clip=polygon,
            )


def _draw_alignment_panel(ax: Axes, records: tuple[RecordingConcept, ...]) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    ax.text(0.0, 0.98, "ECG Alignment", fontsize=15, ha="left", va="bottom")

    r1, r2 = 0.34, 0.58
    r3 = r2 + (r2 - r1)
    for x in (r1, r2):
        ax.axvline(x, ymin=0.005, ymax=0.925, color="#333333", linestyle=(0, (4, 3)), linewidth=1.1, zorder=40)

    strip_h = 0.088
    step = 0.112
    top = 0.780
    for index, record in enumerate(records):
        y = top - index * step
        if record.key == "three_d_bmode":
            x0 = r1 - 0.135
            x1 = r2 + 0.135
        else:
            x0 = r1 - 0.24 - 0.025 * index
            x1 = min(0.985, max(r2 + 0.20, x0 + 0.74 + 0.035 * index))
        patch = Rectangle(
            (x0, y),
            x1 - x0,
            strip_h,
            facecolor=record.color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5 + index,
        )
        ax.add_patch(patch)
        _draw_aligned_ecg(
            ax,
            record,
            x0 + 0.012,
            x1 - 0.012,
            y - 0.004,
            y + strip_h + 0.004,
            r1=r1,
            r2=r2,
            clip=patch,
            zorder=20,
        )

    _draw_aligned_measurement_rows(ax, records, r1=r1, r2=r2, r3=r3)


def _draw_aligned_measurement_rows(
    ax: Axes,
    records: tuple[RecordingConcept, ...],
    *,
    r1: float,
    r2: float,
    r3: float,
) -> None:
    by_key = {record.key: record for record in records}
    x0 = max(0.135, r1 - 0.08)
    x1 = min(0.965, r3 + 0.055)
    row_h = 0.150
    volume_h = 0.150
    rows = (
        ("tissue_doppler", 0.368, row_h),
        ("continuous_wave_doppler", 0.198, row_h),
        ("three_d_bmode", 0.028, volume_h),
    )
    for key, y0, height in rows:
        record = by_key.get(key)
        if record is None:
            continue
        patch = Rectangle(
            (x0, y0),
            x1 - x0,
            height,
            facecolor=record.color,
            edgecolor="white",
            linewidth=0.9,
            zorder=9,
        )
        ax.add_patch(patch)
        if key == "three_d_bmode" and record.volume_curve_ml is not None:
            _draw_phase_aligned_volume_curve(
                ax,
                record,
                x0,
                x1,
                y0 + 0.010,
                y0 + height - 0.010,
                r1=r1,
                r2=r2,
                zorder=25,
                clip=patch,
            )
        elif record.acquisition_window_s is not None:
            _draw_phase_aligned_image(ax, record, x0, x1, y0 + 0.008, y0 + height - 0.008, r1=r1, r2=r2, clip=patch)


def _draw_segment_ecg(
    ax: Axes,
    record: RecordingConcept,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    clip: Polygon,
    zorder: int,
    reference_duration_s: float | None,
) -> None:
    start, stop = _left_ecg_window(record, reference_duration_s=reference_duration_s)
    mask = (record.ecg_timestamps >= start) & (record.ecg_timestamps <= stop)
    times = record.ecg_timestamps[mask]
    signal = record.ecg_signal[mask]
    if times.size < 2:
        times = record.ecg_timestamps[: min(600, record.ecg_timestamps.size)]
        signal = record.ecg_signal[: times.size]
    x = x0 + (times - start) / max(1e-9, stop - start) * (x1 - x0)
    y = _scale_signal(signal, y0, y1)
    line = ax.plot(x, y, color="#202020", linewidth=1.0, zorder=zorder)[0]
    line.set_clip_path(clip)


def _left_reference_duration_s(records: tuple[RecordingConcept, ...]) -> float | None:
    durations: list[float] = []
    for record in records:
        if record.key == "three_d_bmode" or record.acquisition_window_s is None:
            continue
        start, stop = record.acquisition_window_s
        if stop > start:
            durations.append(stop - start)
    if not durations:
        return None
    return float(np.median(np.asarray(durations, dtype=np.float64)))


def _left_ecg_window(record: RecordingConcept, *, reference_duration_s: float | None = None) -> tuple[float, float]:
    if record.acquisition_window_s is not None and record.key != "three_d_bmode":
        return record.acquisition_window_s
    p1, p2 = _left_ecg_peak_times(record)
    cycle = max(1e-6, p2 - p1)
    if reference_duration_s is not None and reference_duration_s > cycle:
        margin = 0.5 * (reference_duration_s - cycle)
        return p1 - margin, p2 + margin
    return p1 - 0.35 * cycle, p2 + 0.65 * cycle


def _left_ecg_peak_times(record: RecordingConcept) -> tuple[float, float]:
    peaks = record.r_peak_indices
    peak_index = int(min(max(0, len(peaks) // 3), len(peaks) - 2))
    p1 = float(record.ecg_timestamps[peaks[peak_index]])
    p2 = float(record.ecg_timestamps[peaks[peak_index + 1]])
    return p1, p2


def _draw_aligned_ecg(
    ax: Axes,
    record: RecordingConcept,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    r1: float,
    r2: float,
    clip: Rectangle,
    zorder: int,
) -> None:
    p1_index, p2_index = _alignment_peak_indices(record)
    p1 = float(record.ecg_timestamps[p1_index])
    p2 = float(record.ecg_timestamps[p2_index])
    cycle = max(1e-6, p2 - p1)
    start = p1 - 0.45 * cycle
    stop = p2 + (0.95 if record.key == "three_d_bmode" else 0.45) * cycle
    mask = (record.ecg_timestamps >= start) & (record.ecg_timestamps <= stop)
    times = record.ecg_timestamps[mask]
    signal = record.ecg_signal[mask]
    x = r1 + (times - p1) / cycle * (r2 - r1)
    keep = (x >= x0) & (x <= x1)
    line = ax.plot(x[keep], _scale_signal(signal[keep], y0, y1), color="#202020", linewidth=1.0, zorder=zorder)[0]
    line.set_clip_path(clip)


def _draw_phase_aligned_image(
    ax: Axes,
    record: RecordingConcept,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    r1: float,
    r2: float,
    clip: Rectangle,
) -> None:
    if record.acquisition_window_s is None:
        return
    p1, p2 = _alignment_peak_times(record)
    cycle = max(1e-6, p2 - p1)
    start, stop = record.acquisition_window_s
    width = r2 - r1
    image_x0 = r1 + ((start - p1) / cycle) * width
    image_x1 = r1 + ((stop - p1) / cycle) * width
    if image_x1 <= x0 or image_x0 >= x1:
        return
    image = ax.imshow(
        record.image,
        extent=(image_x0, image_x1, y0, y1),
        aspect="auto",
        interpolation="bilinear",
        zorder=18,
    )
    image.set_clip_path(clip)


def _draw_phase_aligned_volume_curve(
    ax: Axes,
    record: RecordingConcept,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    r1: float,
    r2: float,
    zorder: int,
    clip: Rectangle,
) -> None:
    if record.volume_curve_ml is None:
        return
    times, volumes = record.volume_curve_ml
    order = np.argsort(times)
    times = np.asarray(times, dtype=np.float64)[order]
    volumes = np.asarray(volumes, dtype=np.float64)[order]
    p1, p2 = _alignment_peak_times(record)
    cycle = max(1e-6, p2 - p1)
    width = r2 - r1
    y = y0 + (volumes - float(np.min(volumes))) / max(1e-9, float(np.max(volumes) - np.min(volumes))) * (y1 - y0)
    phase_x = r1 + ((times - p1) / cycle) * width
    for offset, linestyle in ((0.0, "-"), (width, (0, (4, 3)))):
        x = phase_x + offset
        keep = (x >= x0) & (x <= x1)
        if np.count_nonzero(keep) >= 2:
            line = ax.plot(x[keep], y[keep], color="#1E1E1E", linewidth=1.15, linestyle=linestyle, zorder=zorder)[0]
            line.set_clip_path(clip)


def _alignment_peak_indices(record: RecordingConcept) -> tuple[int, int]:
    peaks = record.r_peak_indices
    center = int(len(peaks) // 2)
    center = max(0, min(center, len(peaks) - 2))
    return int(peaks[center]), int(peaks[center + 1])


def _alignment_peak_times(record: RecordingConcept) -> tuple[float, float]:
    p1_index, p2_index = _alignment_peak_indices(record)
    return float(record.ecg_timestamps[p1_index]), float(record.ecg_timestamps[p2_index])


def _scale_signal(values: np.ndarray, y0: float, y1: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    arr = (arr - np.min(arr)) / max(1e-9, float(np.max(arr) - np.min(arr)))
    return y0 + arr * (y1 - y0)


def _draw_image(
    ax: Axes,
    image: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    zorder: int,
    clip: Polygon | Rectangle | None = None,
) -> None:
    artist = ax.imshow(image, extent=(x0, x1, y0, y1), aspect="auto", interpolation="bilinear", zorder=zorder + 1)
    if clip is not None:
        artist.set_clip_path(clip)


def _draw_volume_curve(
    ax: Axes,
    curve: tuple[np.ndarray, np.ndarray],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    zorder: int,
    clip: Polygon | Rectangle | None = None,
) -> None:
    times, volumes = curve
    start, stop = float(np.min(times)), float(np.max(times))
    x = x0 + (times - start) / max(1e-9, stop - start) * (x1 - x0)
    y = y0 + (volumes - float(np.min(volumes))) / max(1e-9, float(np.max(volumes) - np.min(volumes))) * (y1 - y0)
    line = ax.plot(x, y, color="#1E1E1E", linewidth=1.2, zorder=zorder)[0]
    scatter = ax.scatter(x[[0, -1]], y[[0, -1]], s=5, color="#1E1E1E", zorder=zorder + 1)
    if clip is not None:
        line.set_clip_path(clip)
        scatter.set_clip_path(clip)
    ax.text(x0, y1 + 0.012, "LV volume", ha="left", va="bottom", fontsize=12.5, color="#333333", zorder=zorder)


def _display_label(record: RecordingConcept) -> str:
    if record.key == "tissue_doppler":
        return "Tissue movement"
    if record.key == "pulsed_wave_doppler":
        return "Blood velocity"
    if record.key == "continuous_wave_doppler":
        return "Blood velocity"
    if record.key == "color_doppler":
        return "Blood velocity"
    if record.key == "three_d_bmode":
        return "3D B-mode"
    return record.label


if __name__ == "__main__":
    main()
