#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echoxflow import RecordingRecord, load_croissant, open_recording  # noqa: E402
from echoxflow.config import data_root  # noqa: E402
from echoxflow.scan import (  # noqa: E402
    CartesianGrid,
    SectorDepthRuler,
    contour_group_layout_for_metadata,
    draw_sector_depth_ruler,
)
from tasks.segmentation.dataset import (  # noqa: E402
    RawDataset,
    SampleRef,
    _annotated_target_frame_matches,
    _contour_paths,
    _load_bmode_clip,
    _load_timestamps,
    _panel_frame_count,
    _panel_geometry,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

BACKGROUND = "#FFFFFF"
AXIS_COLOR = "#222222"
GRID_COLOR = "#D8D8D8"
OUTPUT_ROOT = Path("outputs/images/segmentation_masks")
SEGMENTATION_STEM = "paper_segmentation_3patients_6views_qrs_myo_fill_aspect_bg"
STRAIN_STEM = "paper_segmentation_3patients_6views_rr_strain"
STRAIN_POINTS = 200
RULER_LABEL_FONTSIZE = 8.25
STRAIN_TICK_FONTSIZE = 10.5


@dataclass(frozen=True)
class ViewSpec:
    key: str
    label: str
    chamber: str
    content_type: str
    role_id: str
    linestyle: str


@dataclass(frozen=True)
class PatientViews:
    exam_id: str
    records_by_view: dict[str, RecordingRecord]


@dataclass(frozen=True)
class SegmentationPanel:
    image: np.ndarray
    grid: CartesianGrid
    geometry: Any
    dots_xy_m: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True)
class StrainCurve:
    x_percent: np.ndarray
    y_percent: np.ndarray
    manifest: dict[str, Any]


VIEW_SPECS: tuple[ViewSpec, ...] = (
    ViewSpec(
        key="LV_4ch",
        label="LV 4CH",
        chamber="LV",
        content_type="2d_left_ventricular_strain",
        role_id="4ch",
        linestyle="-",
    ),
    ViewSpec(
        key="LV_2ch",
        label="LV 2CH",
        chamber="LV",
        content_type="2d_left_ventricular_strain",
        role_id="2ch",
        linestyle="--",
    ),
    ViewSpec(
        key="LV_3ch",
        label="LV ALAX",
        chamber="LV",
        content_type="2d_left_ventricular_strain",
        role_id="3ch",
        linestyle=":",
    ),
    ViewSpec(
        key="LA_4ch",
        label="LA 4CH",
        chamber="LA",
        content_type="2d_left_atrial_strain",
        role_id="4ch",
        linestyle="-",
    ),
    ViewSpec(
        key="LA_2ch",
        label="LA 2CH",
        chamber="LA",
        content_type="2d_left_atrial_strain",
        role_id="2ch",
        linestyle="--",
    ),
    ViewSpec(
        key="RV_rv",
        label="RV 4CH",
        chamber="RV",
        content_type="2d_right_ventricular_strain",
        role_id="rv",
        linestyle="-",
    ),
)

CHAMBER_COLORS = {
    "LV": "#0072B2",
    "LA": "#009E73",
    "RV": "#D55E00",
}

MYOCARDIUM_CHANNELS = {
    "LV": 1,
    "LA": 3,
    "RV": 5,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-ready segmentation and R-R strain figures for three exams with "
            "LV 4/2/3CH, LA 4/2CH, and RV strain segmentations."
        )
    )
    parser.add_argument("source", nargs="?", type=Path, default=None, help="Dataset root or croissant.json.")
    parser.add_argument("--root", type=Path, default=None, help="Dataset root for an explicit croissant.json.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--patients", type=int, default=3)
    parser.add_argument(
        "--sets", type=int, default=1, help="Number of matched segmentation/strain figure sets to write."
    )
    parser.add_argument("--cartesian-height", type=int, default=384)
    parser.add_argument("--background", default=BACKGROUND)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    parser.add_argument("--target-frame-policy", choices=("annotated", "all"), default="annotated")
    parser.add_argument("--strain-points", type=int, default=STRAIN_POINTS)
    parser.add_argument("--strain-ymin", type=float, default=-20.0)
    parser.add_argument("--strain-ymax", type=float, default=30.0)
    parser.add_argument(
        "--keep-positive-strain",
        action="store_true",
        help="Do not flip positive-going ventricular strain curves.",
    )
    args = parser.parse_args(argv)

    catalog_path, root = _resolve_source(args.source, root=args.root)
    catalog = load_croissant(catalog_path)
    all_patients = [
        patient for patient in _find_patient_views(catalog.recordings) if _patient_is_loadable(patient, root)
    ]
    patient_sets = _patient_sets(all_patients, patients_per_set=int(args.patients), set_count=int(args.sets))
    if not patient_sets:
        raise SystemExit(f"Found only {len(all_patients)} qualifying exam(s); requested {int(args.patients)} per set.")
    requested_unique = int(args.patients) * int(args.sets)
    if len(all_patients) < requested_unique:
        print(
            f"warning: only {len(all_patients)} loadable qualifying exam(s), "
            f"so {int(args.sets)} non-overlapping set(s) of {int(args.patients)} are impossible; "
            "using the next available exam composition for later sets.",
            file=sys.stderr,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_cache: dict[tuple[str, str], SegmentationPanel] = {}
    for index, patients in enumerate(patient_sets, start=1):
        suffix = f"_{index}"
        segmentation_manifest = _write_segmentation_figure(
            patients,
            root=root,
            output_dir=args.output_dir,
            stem=SEGMENTATION_STEM + suffix,
            cartesian_height=int(args.cartesian_height),
            background=str(args.background),
            overlay_alpha=float(args.overlay_alpha),
            dpi=int(args.dpi),
            panel_cache=panel_cache,
            frame_variant=index - 1,
            target_frame_policy=str(args.target_frame_policy),
        )
        strain_manifest = _write_strain_figure(
            patients,
            root=root,
            output_dir=args.output_dir,
            stem=STRAIN_STEM + suffix,
            background=str(args.background),
            dpi=int(args.dpi),
            points=int(args.strain_points),
            invert_positive_ventricular=not bool(args.keep_positive_strain),
            ymin=float(args.strain_ymin),
            ymax=float(args.strain_ymax),
            interval_variant=index - 1,
        )

        print(segmentation_manifest["png"])
        print(strain_manifest["png"])
    return 0


def _resolve_source(source: Path | None, *, root: Path | None) -> tuple[Path, Path]:
    if source is None:
        resolved_root = data_root(root)
        return resolved_root / "croissant.json", resolved_root
    source = source.expanduser()
    if source.is_dir():
        return source / "croissant.json", source
    resolved_root = root.expanduser() if root is not None else source.parent
    return source, resolved_root


def _find_patient_views(records: Sequence[RecordingRecord]) -> list[PatientViews]:
    by_exam: dict[str, list[RecordingRecord]] = {}
    for record in sorted(records, key=lambda item: (item.exam_id, item.recording_id)):
        by_exam.setdefault(record.exam_id, []).append(record)

    patients: list[PatientViews] = []
    for exam_id, exam_records in sorted(by_exam.items()):
        records_by_view: dict[str, RecordingRecord] = {}
        for spec in VIEW_SPECS:
            record = next((item for item in exam_records if _record_matches_view(item, spec)), None)
            if record is None:
                break
            records_by_view[spec.key] = record
        if len(records_by_view) == len(VIEW_SPECS):
            patients.append(PatientViews(exam_id=exam_id, records_by_view=records_by_view))
    return patients


def _patient_sets(
    patients: Sequence[PatientViews],
    *,
    patients_per_set: int,
    set_count: int,
) -> list[list[PatientViews]]:
    count = max(1, int(patients_per_set))
    sets = max(1, int(set_count))
    if len(patients) < count:
        return []
    if len(patients) >= count * sets:
        return [list(patients[index * count : (index + 1) * count]) for index in range(sets)]
    if len(patients) >= count + sets - 1:
        return [list(patients[index : index + count]) for index in range(sets)]
    return [[patients[(index + offset) % len(patients)] for offset in range(count)] for index in range(sets)]


def _record_matches_view(record: RecordingRecord, spec: ViewSpec) -> bool:
    if spec.content_type not in record.content_types:
        return False
    paths = set(record.array_paths)
    return f"data/{spec.role_id}_contour" in paths and f"data/{spec.role_id}_curve" in paths


def _patient_is_loadable(patient: PatientViews, root: Path) -> bool:
    for spec in VIEW_SPECS:
        record = patient.records_by_view[spec.key]
        try:
            store = open_recording(record, root=root)
            panel = next(item for item in store.load_object().panels if item.role_id == spec.role_id)
            if _panel_frame_count(store, panel) is None:
                return False
        except (FileNotFoundError, KeyError, StopIteration, ValueError):
            return False
    return True


def _write_segmentation_figure(
    patients: Sequence[PatientViews],
    *,
    root: Path,
    output_dir: Path,
    stem: str,
    cartesian_height: int,
    background: str,
    overlay_alpha: float,
    dpi: int,
    panel_cache: dict[tuple[str, str], SegmentationPanel] | None = None,
    frame_variant: int = 0,
    target_frame_policy: str = "annotated",
) -> dict[str, Any]:
    fig, axes = plt.subplots(
        len(patients),
        len(VIEW_SPECS),
        figsize=(11.2, 6.1),
        squeeze=False,
        facecolor=background,
    )
    panels: list[dict[str, Any]] = []
    for row, patient in enumerate(patients):
        for col, spec in enumerate(VIEW_SPECS):
            record = patient.records_by_view[spec.key]
            cache_key = (record.recording_id, spec.key, str(int(frame_variant)), target_frame_policy)
            panel = None if panel_cache is None else panel_cache.get(cache_key)
            if panel is None:
                panel = _segmentation_panel(
                    record,
                    spec,
                    root=root,
                    cartesian_height=cartesian_height,
                    background=background,
                    overlay_alpha=overlay_alpha,
                    frame_variant=frame_variant,
                    target_frame_policy=target_frame_policy,
                )
                if panel_cache is not None:
                    panel_cache[cache_key] = panel
            _plot_segmentation_panel(axes[row, col], panel, background=background)
            panels.append({"exam_id": patient.exam_id, "column": spec.key, **panel.manifest})

    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(background)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.945, bottom=0.01, wspace=0.02, hspace=0.02)
    for col, spec in enumerate(VIEW_SPECS):
        bbox = axes[0, col].get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            0.972,
            spec.label,
            ha="center",
            va="center",
            color=AXIS_COLOR,
            fontsize=9.5,
        )
    manifest = {
        "patients": [patient.exam_id for patient in patients],
        "columns": [spec.key for spec in VIEW_SPECS],
        "alignment": f"target_frame_policy={target_frame_policy} via RawDataset.sample_from_ref",
        "frame_selection": (
            "largest annotated myocardium area; alternate sets use a distinct annotated frame when possible"
        ),
        "aspect": "panel-specific cartesian grids; imshow aspect equal; axes padded, not stretched",
        "background": background,
        "panels": panels,
    }
    return _save_figure(fig, output_dir / stem, manifest=manifest, dpi=dpi, background=background)


def _segmentation_panel(
    record: RecordingRecord,
    spec: ViewSpec,
    *,
    root: Path,
    cartesian_height: int,
    background: str,
    overlay_alpha: float,
    frame_variant: int,
    target_frame_policy: str,
) -> SegmentationPanel:
    store = open_recording(record, root=root)
    panel = next(item for item in store.load_object().panels if item.role_id == spec.role_id)
    source_store = store.open_reference(panel.bmode.recording)
    source_shape = tuple(int(value) for value in source_store.group[panel.bmode.data_path].shape[-2:])
    geometry = _panel_geometry(panel, source_shape=source_shape)
    if geometry is None:
        raise ValueError(f"{record.exam_id}/{record.recording_id}/{spec.role_id} has no sector geometry.")
    grid = CartesianGrid.from_sector_height(geometry, int(cartesian_height))
    frame_count = _panel_frame_count(store, panel)
    if frame_count is None or frame_count <= 0:
        raise ValueError(f"{record.exam_id}/{record.recording_id}/{spec.role_id} has no linked B-mode frames.")

    dataset = RawDataset(
        records=[],
        clip_length=1,
        input_spatial_shape=grid.shape,
        data_root=root,
        target_mask_channels=6,
        coordinate_space="cartesian",
        cartesian_height=cartesian_height,
        target_frame_policy=target_frame_policy,
        sampling_mode="full_recording",
    )
    sample = dataset.sample_from_ref(
        SampleRef(
            record=record,
            role_id=spec.role_id,
            start=0,
            stop=frame_count,
            sample_id=f"{record.exam_id}/{record.recording_id}/{spec.role_id}:frames_full",
            content_type=spec.content_type,
            view_code=spec.role_id,
        )
    )

    frames = sample.frames[0, 0].detach().cpu().numpy()
    masks = sample.target_masks[0].detach().cpu().numpy()
    valid = sample.valid_mask[0, 0].detach().cpu().numpy() if sample.valid_mask is not None else np.ones_like(frames)
    myo_channel = MYOCARDIUM_CHANNELS[spec.chamber]
    myo = masks[myo_channel]
    areas = np.sum(myo * valid, axis=(1, 2))
    frame_index = _frame_index_from_areas(areas, variant=int(frame_variant))
    image = _compose_myo_overlay(
        frames[frame_index],
        myo[frame_index],
        valid[frame_index],
        color=CHAMBER_COLORS[spec.chamber],
        background=background,
        alpha=overlay_alpha,
    )
    qrs = _qrs_times(store, spec.role_id)
    dots_xy_m, contour_index = _segment_midpoint_dots(
        store,
        panel=panel,
        source_store=source_store,
        content_type=spec.content_type,
        target_frame_index=frame_index,
        frame_count=frame_count,
    )
    return SegmentationPanel(
        image=image,
        grid=grid,
        geometry=geometry,
        dots_xy_m=dots_xy_m,
        manifest={
            "sample_id": sample.sample_id,
            "frame_idx": frame_index,
            "frame_variant": int(frame_variant),
            "target_frame_policy": target_frame_policy,
            "contour_idx": contour_index,
            "segment_dot_count": int(dots_xy_m.shape[0]),
            "qrs_count": int(qrs.size),
            "area": float(areas[frame_index]) if areas.size else 0.0,
            "grid_shape": tuple(int(value) for value in grid.shape),
        },
    )


def _segment_midpoint_dots(
    store: Any,
    *,
    panel: Any,
    source_store: Any,
    content_type: str,
    target_frame_index: int,
    frame_count: int,
) -> tuple[np.ndarray, int | None]:
    contour_path, contour_timestamps_path = _contour_paths(panel)
    if contour_path is None or contour_path not in store.group:
        return np.zeros((0, 2), dtype=np.float32), None
    contours = np.asarray(store.group[contour_path][:], dtype=np.float32)
    if contours.ndim != 3 or contours.shape[-1] != 2 or int(contours.shape[0]) <= 0:
        return np.zeros((0, 2), dtype=np.float32), None
    _frames, target_timestamps, _metadata = _load_bmode_clip(
        source_store,
        data_path=panel.bmode.data_path,
        timestamps_path=panel.bmode.timestamps_path,
        start=0,
        stop=frame_count,
    )
    contour_timestamps = _load_timestamps(store, contour_timestamps_path, start=0, stop=int(contours.shape[0]))
    matches = _annotated_target_frame_matches(
        contour_timestamps=contour_timestamps,
        target_timestamps=target_timestamps,
        contour_count=int(contours.shape[0]),
        target_count=int(frame_count),
    )
    contour_index_by_target = {int(target): int(contour) for contour, target in matches}
    contour_index = contour_index_by_target.get(int(target_frame_index))
    if contour_index is None:
        return np.zeros((0, 2), dtype=np.float32), None
    points = np.asarray(contours[contour_index], dtype=np.float32)
    group_layout = contour_group_layout_for_metadata(content_type=content_type, point_count=int(points.shape[0]))
    row_count = int(points.shape[0]) // int(group_layout.group_size)
    if row_count <= 0 or row_count * int(group_layout.group_size) != int(points.shape[0]):
        return np.zeros((0, 2), dtype=np.float32), contour_index
    grid = points.reshape(row_count, int(group_layout.group_size), 2)
    mid_col = int(group_layout.group_size) // 2
    dots = np.asarray(grid[:, mid_col, :], dtype=np.float32)
    return dots[np.isfinite(dots).all(axis=1)], contour_index


def _frame_index_from_areas(areas: np.ndarray, *, variant: int) -> int:
    values = np.asarray(areas, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0
    finite = np.isfinite(values) & (values > 0.0)
    if not bool(np.any(finite)):
        return int(values.size // 2)
    order = np.argsort(np.where(finite, values, -np.inf))[::-1]
    if int(variant) <= 0:
        return int(order[0])
    min_distance = max(1, int(round(values.size * 0.20)))
    selected = [int(order[0])]
    for index in order[1:]:
        candidate = int(index)
        if all(abs(candidate - previous) >= min_distance for previous in selected):
            selected.append(candidate)
            if len(selected) > int(variant):
                return candidate
    return int(order[min(int(variant), len(order) - 1)])


def _compose_myo_overlay(
    frame: np.ndarray,
    myocardium: np.ndarray,
    valid: np.ndarray,
    *,
    color: str,
    background: str,
    alpha: float,
) -> np.ndarray:
    values = np.nan_to_num(np.asarray(frame, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    rgb = np.repeat(values[..., None], 3, axis=-1)
    valid_mask = np.asarray(valid, dtype=np.float32) > 0.5
    bg = np.asarray(to_rgb(background), dtype=np.float32)
    rgb[~valid_mask] = bg
    overlay_alpha = np.clip(np.asarray(myocardium, dtype=np.float32), 0.0, 1.0) * float(alpha)
    overlay_alpha *= valid_mask.astype(np.float32)
    overlay = np.asarray(to_rgb(color), dtype=np.float32)
    rgb = rgb * (1.0 - overlay_alpha[..., None]) + overlay * overlay_alpha[..., None]
    rgb[~valid_mask] = bg
    return np.clip(rgb, 0.0, 1.0)


def _plot_segmentation_panel(ax: Axes, panel: SegmentationPanel, *, background: str) -> None:
    grid = panel.grid
    x0, x1 = grid.x_range_m
    y0, y1 = grid.y_range_m
    ax.imshow(panel.image, extent=(x0, x1, y1, y0), interpolation="nearest", aspect="equal")
    if panel.dots_xy_m.size:
        ax.scatter(
            panel.dots_xy_m[:, 0],
            panel.dots_xy_m[:, 1],
            s=2.2,
            c="#FFFFFF",
            edgecolors="none",
            linewidths=0.0,
            zorder=9.0,
        )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(background)
    draw_sector_depth_ruler(
        ax,
        panel.geometry,
        SectorDepthRuler(
            side="left",
            tick_interval_cm=1.0,
            label_interval_cm=5.0,
            color=AXIS_COLOR,
            linewidth=0.65,
            border_linewidth=0.0,
            label_fontsize=RULER_LABEL_FONTSIZE,
            show_border=False,
            include_boundary_ticks=False,
            zorder=5.0,
        ),
    )


def _write_strain_figure(
    patients: Sequence[PatientViews],
    *,
    root: Path,
    output_dir: Path,
    stem: str,
    background: str,
    dpi: int,
    points: int,
    invert_positive_ventricular: bool,
    ymin: float,
    ymax: float,
    interval_variant: int = 0,
) -> dict[str, Any]:
    fig, axes = plt.subplots(
        len(patients),
        1,
        figsize=(7.6, 5.4),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor=background,
    )
    panel_manifest: list[dict[str, Any]] = []
    handles = []
    labels = []
    for row, patient in enumerate(patients):
        ax = axes[row, 0]
        _style_strain_axis(ax, background=background)
        for spec in VIEW_SPECS:
            curve = _rr_strain_curve(
                patient.records_by_view[spec.key],
                spec,
                root=root,
                points=points,
                invert_positive_ventricular=invert_positive_ventricular,
                interval_variant=interval_variant,
            )
            (line,) = ax.plot(
                curve.x_percent,
                curve.y_percent,
                color=CHAMBER_COLORS[spec.chamber],
                linestyle=spec.linestyle,
                linewidth=1.6,
                label=spec.label,
            )
            panel_manifest.append({"exam_id": patient.exam_id, "column": spec.key, **curve.manifest})
            if row == 0:
                handles.append(line)
                labels.append(spec.label)
        ax.set_ylim(float(ymin), float(ymax))
        ax.set_xlim(0.0, 100.0)
        ax.set_ylabel("Strain (%)", color=AXIS_COLOR, fontsize=8)
    axes[-1, 0].set_xlabel("R-R interval (%)", color=AXIS_COLOR, fontsize=8)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(VIEW_SPECS),
        frameon=False,
        labelcolor=AXIS_COLOR,
        fontsize=7,
        handlelength=2.4,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.915, bottom=0.10, hspace=0.18)
    manifest = {
        "patients": [patient.exam_id for patient in patients],
        "columns": [spec.key for spec in VIEW_SPECS],
        "normalization": (
            "x: selected R-R interval mapped to 0-100%; "
            f"y: zeroed at R, LA kept native positive, positive-going LV/RV curves inverted, "
            f"displayed from {float(ymin):g} to {float(ymax):g}%"
        ),
        "interval_variant": int(interval_variant),
        "y_range_percent": [float(ymin), float(ymax)],
        "positive_ventricular_strain_inverted": bool(invert_positive_ventricular),
        "atrial_strain_flipped": False,
        "panels": panel_manifest,
    }
    return _save_figure(fig, output_dir / stem, manifest=manifest, dpi=dpi, background=background)


def _style_strain_axis(ax: Axes, *, background: str) -> None:
    ax.set_facecolor(background)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.tick_params(axis="both", colors=AXIS_COLOR, labelsize=STRAIN_TICK_FONTSIZE, length=3, width=0.7)
    ax.set_yticks([-20, -10, 0, 10, 20, 30])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.4, alpha=0.75)


def _rr_strain_curve(
    record: RecordingRecord,
    spec: ViewSpec,
    *,
    root: Path,
    points: int,
    invert_positive_ventricular: bool,
    interval_variant: int,
) -> StrainCurve:
    store = open_recording(record, root=root)
    curve_path = f"data/{spec.role_id}_curve"
    timestamp_path = f"timestamps/{spec.role_id}_curve"
    if curve_path not in store.group:
        raise ValueError(f"Missing {curve_path} in {record.exam_id}/{record.recording_id}.")
    values = _mean_curve(np.asarray(store.group[curve_path][:], dtype=np.float64))
    timestamps = _curve_timestamps(store, timestamp_path, count=values.size)
    qrs = _qrs_times(store, spec.role_id)
    rr_start, rr_stop = _select_rr_interval(timestamps, values, qrs, variant=int(interval_variant))
    target_timestamps = np.linspace(rr_start, rr_stop, max(2, int(points)), dtype=np.float64)
    finite = np.isfinite(timestamps) & np.isfinite(values)
    order = np.argsort(timestamps[finite])
    interp = np.interp(target_timestamps, timestamps[finite][order], values[finite][order])
    interp = interp - float(interp[0])
    raw_min = float(np.nanmin(interp)) if interp.size else 0.0
    raw_max = float(np.nanmax(interp)) if interp.size else 0.0
    inverted = False
    if spec.chamber != "LA" and invert_positive_ventricular and abs(raw_max) > abs(raw_min):
        interp = -interp
        inverted = True
    display = interp
    x_percent = (target_timestamps - rr_start) / max(1e-12, rr_stop - rr_start) * 100.0
    return StrainCurve(
        x_percent=x_percent,
        y_percent=display,
        manifest={
            "sample": f"{record.exam_id}/{record.recording_id}/{spec.role_id}",
            "rr_start": float(rr_start),
            "rr_stop": float(rr_stop),
            "interval_variant": int(interval_variant),
            "qrs_count": int(qrs.size),
            "points": int(display.size),
            "strain_min": float(np.nanmin(display)) if display.size else 0.0,
            "strain_max": float(np.nanmax(display)) if display.size else 0.0,
            "raw_zeroed_min": raw_min,
            "raw_zeroed_max": raw_max,
            "chamber": spec.chamber,
            "inverted": inverted,
        },
    )


def _mean_curve(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    while arr.ndim > 1:
        arr = np.nanmean(arr, axis=0)
    return arr.reshape(-1)


def _curve_timestamps(store: Any, path: str, *, count: int) -> np.ndarray:
    if path in store.group:
        timestamps = np.asarray(store.group[path][:], dtype=np.float64).reshape(-1)
        if timestamps.size == int(count):
            return timestamps
    return np.arange(int(count), dtype=np.float64)


def _qrs_times(store: Any, role_id: str) -> np.ndarray:
    path = f"timestamps/{role_id}_ecg_qrs"
    if path not in store.group:
        return np.asarray([], dtype=np.float64)
    values = np.asarray(store.group[path][:], dtype=np.float64).reshape(-1)
    return np.sort(values[np.isfinite(values)])


def _select_rr_interval(
    timestamps: np.ndarray,
    values: np.ndarray,
    qrs: np.ndarray,
    *,
    variant: int,
) -> tuple[float, float]:
    finite = np.isfinite(timestamps) & np.isfinite(values)
    if not bool(np.any(finite)):
        return 0.0, 1.0
    times = timestamps[finite]
    vals = values[finite]
    intervals: list[tuple[float, float, int, float]] = []
    for start, stop in zip(qrs[:-1], qrs[1:]):
        if not np.isfinite(start) or not np.isfinite(stop) or float(stop) <= float(start):
            continue
        mask = (times >= float(start)) & (times <= float(stop))
        count = int(np.count_nonzero(mask))
        if count < 2:
            continue
        amplitude = float(np.nanmax(vals[mask]) - np.nanmin(vals[mask]))
        intervals.append((float(start), float(stop), count, amplitude))
    if intervals:
        ranked = sorted(intervals, key=lambda item: (item[2], item[3]), reverse=True)
        start, stop, _count, _amplitude = ranked[min(int(variant), len(ranked) - 1)]
        return start, stop
    return float(np.nanmin(times)), float(np.nanmax(times))


def _save_figure(
    fig: Any,
    stem: Path,
    *,
    manifest: dict[str, Any],
    dpi: int,
    background: str,
) -> dict[str, Any]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    manifest_path = stem.with_suffix(".manifest.json")
    fig.savefig(png, dpi=dpi, facecolor=background, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, dpi=dpi, facecolor=background, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    payload = {**manifest, "png": str(png), "pdf": str(pdf)}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
