#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echoxflow import RecordingRecord, load_croissant, open_recording  # noqa: E402
from echoxflow.config import data_root  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

DEFAULT_OUTPUT = Path("outputs/images/segmentation_masks/paper_3d_lv_volume_curve_fill_2")
BACKGROUND = "#FFFFFF"
AXIS_COLOR = "#222222"
GRID_COLOR = "#D8D8D8"
LINE_COLOR = "#FF6666"
FIGURE_WIDTH_IN = 7.6
AXIS_LABEL_FONTSIZE = 8
TICK_FONTSIZE = 10.5


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot a paper-style LV mesh volume curve over one R-R interval.")
    parser.add_argument("source", nargs="?", type=Path, default=None, help="Dataset root or croissant.json.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exam-id", default=None)
    parser.add_argument("--recording-id", default=None)
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    catalog_path, root = _resolve_source(args.source, root=args.root)
    catalog = load_croissant(catalog_path)
    record = _select_recording(
        catalog.recordings,
        exam_id=args.exam_id,
        recording_id=args.recording_id,
        case_index=int(args.case_index),
    )
    curve = _rr_volume_curve(
        record,
        root=root,
    )
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_IN, 2.25), facecolor=BACKGROUND)
    _style_axis(ax)
    ax.plot(curve["x_percent"], curve["volume_ml"], color=LINE_COLOR, linewidth=1.8)
    ax.set_xlim(0.0, 100.0)
    volumes = np.asarray(curve["volume_ml"], dtype=np.float64)
    margin = max(5.0, 0.08 * float(np.nanmax(volumes) - np.nanmin(volumes)))
    ax.set_ylim(float(np.nanmin(volumes) - margin), float(np.nanmax(volumes) + margin))
    ax.set_ylabel("LV Volume (ml)", color=AXIS_COLOR, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("R-R interval (%)", color=AXIS_COLOR, fontsize=AXIS_LABEL_FONTSIZE)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.24)

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    manifest = output.with_suffix(".manifest.json")
    try:
        fig.savefig(png, dpi=int(args.dpi), facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(pdf, dpi=int(args.dpi), facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(fig)
    manifest.write_text(
        json.dumps(
            {
                "png": str(png),
                "pdf": str(pdf),
                "exam_id": record.exam_id,
                "recording_id": record.recording_id,
                **curve["manifest"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(png)
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


def _select_recording(
    records: Sequence[RecordingRecord],
    *,
    exam_id: str | None,
    recording_id: str | None,
    case_index: int,
) -> RecordingRecord:
    candidates = [
        record
        for record in records
        if "data/3d_brightness_mode" in record.array_paths
        and "data/3d_left_ventricle_mesh/point_values" in record.array_paths
        and "data/3d_left_ventricle_mesh/face_values" in record.array_paths
    ]
    if exam_id is not None:
        candidates = [record for record in candidates if record.exam_id == exam_id]
    if recording_id is not None:
        candidates = [record for record in candidates if record.recording_id == recording_id]
    if not candidates:
        raise SystemExit("No 3D B-mode recording with LV mesh was found.")
    index = int(np.clip(int(case_index), 0, len(candidates) - 1))
    return sorted(candidates, key=lambda item: (item.exam_id, item.recording_id))[index]


def _rr_volume_curve(
    record: RecordingRecord,
    *,
    root: Path,
) -> dict[str, object]:
    store = open_recording(record, root=root)
    times, volumes = _mesh_volume_curve_ml(store)
    x_percent, displayed_volumes = _full_interval_samples(volumes)
    return {
        "x_percent": x_percent,
        "volume_ml": displayed_volumes,
        "manifest": {
            "raw_mesh_points": int(volumes.size),
            "displayed_points": int(displayed_volumes.size),
            "repeated_first_volume_at_100_percent": True,
            "source_frame_start": 0,
            "source_frame_stop": int(volumes.size - 1),
            "source_time_start": float(times[0]),
            "source_time_stop": float(times[-1]),
            "mesh_times_stretched_to_rr": False,
            "volume_min_ml": float(np.nanmin(displayed_volumes)),
            "volume_max_ml": float(np.nanmax(displayed_volumes)),
            "volume_start_ml": float(displayed_volumes[0]),
            "volume_end_ml": float(displayed_volumes[-1]),
        },
    }


def _full_interval_samples(volumes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if source.size < 2:
        raise ValueError("The selected recording does not contain enough LV mesh volume samples.")
    displayed = np.concatenate([source, source[:1]])
    x_percent = np.linspace(0.0, 100.0, displayed.size, dtype=np.float64)
    return x_percent, displayed


def _mesh_volume_curve_ml(store: object) -> tuple[np.ndarray, np.ndarray]:
    mesh = store.load_packed_mesh_annotation()  # type: ignore[attr-defined]
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
        raise ValueError("The selected recording does not contain enough LV mesh volume samples.")
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


def _style_axis(ax: Axes) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.tick_params(axis="both", colors=AXIS_COLOR, labelsize=TICK_FONTSIZE, length=3, width=0.7)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.4, alpha=0.75)


if __name__ == "__main__":
    raise SystemExit(main())
